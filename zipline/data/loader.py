#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from os.path import expanduser
from collections import OrderedDict
from datetime import timedelta

import logbook

import pandas as pd
from pandas.io.data import DataReader
import pytz

from . treasuries import get_treasury_data
from . import benchmarks
from . benchmarks import get_benchmark_returns

from zipline.protocol import DailyReturn
from zipline.utils.tradingcalendar import trading_days

logger = logbook.Logger('Loader')

# TODO: Make this path customizable.
DATA_PATH = os.path.join(
    expanduser("~"),
    '.zipline',
    'data'
)


def get_datafile(name, mode='r'):
    """
    Returns a handle to data file.

    Creates containing directory, if needed.
    """

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    return open(os.path.join(DATA_PATH, name), mode)


def dump_treasury_curves():
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    tr_data = {}

    for curve in get_treasury_data():
        # Not ideal but massaging data into expected format
        tr_data[curve['date']] = curve

    curves = pd.DataFrame(tr_data).T

    datafile = get_datafile('treasury_curves.csv', mode='wb')
    curves.to_csv(datafile)
    datafile.close()


def dump_benchmarks(symbol):
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    benchmark_data = []
    for daily_return in get_benchmark_returns(symbol):
        # Not ideal but massaging data into expected format
        benchmark = (daily_return.date, daily_return.returns)
        benchmark_data.append(benchmark)

    datafile = get_datafile(get_benchmark_filename(symbol), mode='wb')
    benchmark_returns = pd.Series(dict(benchmark_data))
    benchmark_returns.to_csv(datafile)
    datafile.close()


def update_treasury_curves(last_date):
    """
    Updates data in the zipline treasury curves message pack

    last_date should be a datetime object of the most recent data

    Puts source treasury and data into zipline.
    """
    datafile = get_datafile('treasury_curves.csv', mode='rb')
    curves = pd.DataFrame.from_csv(datafile).T
    datafile.close()

    for curve in get_treasury_data():
        curves[curve['date']] = curve

    updated_curves = curves.T

    datafile = get_datafile('treasury_curves.csv', mode='wb')
    updated_curves.T.to_csv(datafile)
    datafile.close()

    return updated_curves


def update_benchmarks(symbol, last_date):
    """
    Updates data in the zipline message pack

    last_date should be a datetime object of the most recent data

    Puts source benchmark into zipline.
    """
    datafile = get_datafile(get_benchmark_filename(symbol), mode='rb')
    saved_benchmarks = pd.Series.from_csv(datafile)
    datafile.close()

    try:
        start = last_date + timedelta(days=1)
        for daily_return in get_benchmark_returns(symbol, start_date=start):
            # Not ideal but massaging data into expected format
            benchmark = pd.Series({daily_return.date: daily_return.returns})
            saved_benchmarks.append(benchmark)

        datafile = get_datafile(get_benchmark_filename(symbol), mode='wb')
        saved_benchmarks.to_csv(datafile)
        datafile.close()
    except benchmarks.BenchmarkDataNotFoundError as exc:
        logger.warn(exc)
    return saved_benchmarks


def get_benchmark_filename(symbol):
    return "%s_benchmark.csv" % symbol


def load_market_data(bm_symbol='^GSPC'):
    try:
        fp_bm = get_datafile(get_benchmark_filename(bm_symbol), "rb")
    except IOError:
        print("""
data files aren't distributed with source.
Fetching data from Yahoo Finance.
""").strip()
        dump_benchmarks(bm_symbol)
        fp_bm = get_datafile(get_benchmark_filename(bm_symbol), "rb")

    saved_benchmarks = pd.Series.from_csv(fp_bm)
    fp_bm.close()

    # Find the offset of the last date for which we have trading data in our
    # list of valid trading days
    last_bm_date = saved_benchmarks.index[-1]
    last_bm_date_offset = trading_days.searchsorted(
        last_bm_date.strftime('%Y/%m/%d'))

    # If more than 1 trading days has elapsed since the last day where
    # we have data,then we need to update
    if len(trading_days) - last_bm_date_offset > 1:
        benchmark_returns = update_benchmarks(bm_symbol, last_bm_date)
    else:
        benchmark_returns = saved_benchmarks

    benchmark_returns = benchmark_returns.tz_localize('UTC')

    bm_returns = []
    for dt, returns in benchmark_returns.iterkv():
        daily_return = DailyReturn(date=dt, returns=returns)
        bm_returns.append(daily_return)

    try:
        fp_tr = get_datafile('treasury_curves.csv', "rb")
    except IOError:
        print("""
data msgpacks aren't distributed with source.
Fetching data from data.treasury.gov
""").strip()
        dump_treasury_curves()
        fp_tr = get_datafile('treasury_curves.csv', "rb")

    saved_curves = pd.DataFrame.from_csv(fp_tr)

    # Find the offset of the last date for which we have trading data in our
    # list of valid trading days
    last_tr_date = saved_curves.index[-1]
    last_tr_date_offset = trading_days.searchsorted(
        last_tr_date.strftime('%Y/%m/%d'))

    # If more than 1 trading days has elapsed since the last day where
    # we have data,then we need to update
    if len(trading_days) - last_tr_date_offset > 1:
        treasury_curves = update_treasury_curves(last_tr_date)
    else:
        treasury_curves = saved_curves

    treasury_curves = treasury_curves.tz_localize('UTC')

    tr_curves = {}
    for tr_dt, curve in treasury_curves.T.iterkv():
        # tr_dt = tr_dt.replace(hour=0, minute=0, second=0, microsecond=0,
        #                       tzinfo=pytz.utc)
        tr_curves[tr_dt] = curve.to_dict()

    fp_tr.close()

    tr_curves = OrderedDict(sorted(
                            ((dt, c) for dt, c in tr_curves.iteritems()),
                            key=lambda t: t[0]))

    return bm_returns, tr_curves


def _load_raw_yahoo_data(indexes=None, stocks=None, start=None, end=None):
    """Load closing prices from yahoo finance.

    :Optional:
        indexes : dict (Default: {'SPX': '^GSPC'})
            Financial indexes to load.
        stocks : list (Default: ['AAPL', 'GE', 'IBM', 'MSFT',
                                 'XOM', 'AA', 'JNJ', 'PEP', 'KO'])
            Stock closing prices to load.
        start : datetime (Default: datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc))
            Retrieve prices from start date on.
        end : datetime (Default: datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc))
            Retrieve prices until end date.

    :Note:
        This is based on code presented in a talk by Wes McKinney:
        http://wesmckinney.com/files/20111017/notebook_output.pdf
    """

    assert indexes is not None or stocks is not None, """
must specify stocks or indexes"""

    if start is None:
        start = pd.datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)

    if not start is None and not end is None:
        assert start < end, "start date is later than end date."

    data = OrderedDict()

    if stocks is not None:
        for stock in stocks:
            print stock
            stkd = DataReader(stock, 'yahoo', start, end).sort_index()
            data[stock] = stkd

    if indexes is not None:
        for name, ticker in indexes.iteritems():
            print name
            stkd = DataReader(ticker, 'yahoo', start, end).sort_index()
            data[name] = stkd

    return data


def load_from_yahoo(indexes=None,
                    stocks=None,
                    start=None,
                    end=None,
                    adjusted=True):
    """
    Loads price data from Yahoo into a dataframe for each of the indicated
    securities.  By default, 'price' is taken from Yahoo's 'Adjusted Close',
    which removes the impact of splits and dividends. If the argument
    'adjusted' is False, then the non-adjusted 'close' field is used instead.

    :param indexes: Financial indexes to load.
    :type indexes: dict
    :param stocks: Stock closing prices to load.
    :type stocks: list
    :param start: Retrieve prices from start date on.
    :type start: datetime
    :param end: Retrieve prices until end date.
    :type end: datetime
    :param adjusted: Adjust the price for splits and dividends.
    :type adjusted: bool

    """
    data = _load_raw_yahoo_data(indexes, stocks, start, end)
    if adjusted:
        close_key = 'Adj Close'
    else:
        close_key = 'Close'
    df = pd.DataFrame({key: d[close_key] for key, d in data.iteritems()})
    df.index = df.index.tz_localize(pytz.utc)
    return df


def load_bars_from_yahoo(indexes=None,
                         stocks=None,
                         start=None,
                         end=None,
                         adjusted=True):
    """
    Loads data from Yahoo into a panel with the following
    column names for each indicated security:

        - open
        - high
        - low
        - close
        - volume
        - price

    Note that 'price' is Yahoo's 'Adjusted Close', which removes the
    impact of splits and dividends. If the argument 'adjusted' is True, then
    the open, high, low, and close values are adjusted as well.

    :param indexes: Financial indexes to load.
    :type indexes: dict
    :param stocks: Stock closing prices to load.
    :type stocks: list
    :param start: Retrieve prices from start date on.
    :type start: datetime
    :param end: Retrieve prices until end date.
    :type end: datetime
    :param adjusted: Adjust open/high/low/close for splits and dividends.
        The 'price' field is always adjusted.
    :type adjusted: bool

    """
    data = _load_raw_yahoo_data(indexes, stocks, start, end)
    panel = pd.Panel(data)
    # Rename columns
    panel.minor_axis = ['open', 'high', 'low', 'close', 'volume', 'price']
    panel.major_axis = panel.major_axis.tz_localize(pytz.utc)
    # Adjust data
    if adjusted:
        adj_cols = ['open', 'high', 'low', 'close']
        for ticker in panel.items:
            ratio = (panel[ticker]['price'] / panel[ticker]['close'])
            ratio_filtered = ratio.fillna(0).values
            for col in adj_cols:
                panel[ticker][col] *= ratio_filtered
    return panel
