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


import importlib
import os
from os.path import expanduser
from collections import OrderedDict
from datetime import timedelta

import logbook

import pandas as pd
from pandas.io.data import DataReader
import pytz

from six import iteritems

from . import benchmarks
from . benchmarks import get_benchmark_returns

from zipline.utils.tradingcalendar import trading_day as trading_day_nyse
from zipline.utils.tradingcalendar import trading_days as trading_days_nyse

logger = logbook.Logger('Loader')

# TODO: Make this path customizable.
DATA_PATH = os.path.join(
    expanduser("~"),
    '.zipline',
    'data'
)

CACHE_PATH = os.path.join(
    expanduser("~"),
    '.zipline',
    'cache'
)

# Mapping from index symbol to appropriate bond data
INDEX_MAPPING = {
    '^GSPC':
    ('treasuries', 'treasury_curves.csv', 'data.treasury.gov'),
    '^GSPTSE':
    ('treasuries_can', 'treasury_curves_can.csv', 'bankofcanada.ca'),
    '^FTSE':  # use US treasuries until UK bonds implemented
    ('treasuries', 'treasury_curves.csv', 'data.treasury.gov'),
}


def get_data_filepath(name):
    """
    Returns a handle to data file.

    Creates containing directory, if needed.
    """

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    return os.path.join(DATA_PATH, name)


def get_cache_filepath(name):
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    return os.path.join(CACHE_PATH, name)


def dump_treasury_curves(module='treasuries', filename='treasury_curves.csv'):
    """
    Dumps data to be used with zipline.

    Puts source treasury and data into zipline.
    """
    try:
        m = importlib.import_module("." + module, package='zipline.data')
    except ImportError:
        raise NotImplementedError(
            'Treasury curve {0} module not implemented'.format(module))

    tr_data = {}

    for curve in m.get_treasury_data():
        # Not ideal but massaging data into expected format
        tr_data[curve['date']] = curve

    curves = pd.DataFrame(tr_data).T

    data_filepath = get_data_filepath(filename)
    curves.to_csv(data_filepath)

    return curves


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

    data_filepath = get_data_filepath(get_benchmark_filename(symbol))
    benchmark_returns = pd.Series(dict(benchmark_data))
    benchmark_returns.to_csv(data_filepath)


def update_benchmarks(symbol, last_date):
    """
    Updates data in the zipline message pack

    last_date should be a datetime object of the most recent data

    Puts source benchmark into zipline.
    """
    datafile = get_data_filepath(get_benchmark_filename(symbol))
    saved_benchmarks = pd.Series.from_csv(datafile)

    try:
        start = last_date + timedelta(days=1)
        for daily_return in get_benchmark_returns(symbol, start_date=start):
            # Not ideal but massaging data into expected format
            benchmark = pd.Series({daily_return.date: daily_return.returns})
            saved_benchmarks = saved_benchmarks.append(benchmark)

        datafile = get_data_filepath(get_benchmark_filename(symbol))
        saved_benchmarks.to_csv(datafile)
    except benchmarks.BenchmarkDataNotFoundError as exc:
        logger.warn(exc)
    return saved_benchmarks


def get_benchmark_filename(symbol):
    return "%s_benchmark.csv" % symbol


def load_market_data(trading_day=trading_day_nyse,
                     trading_days=trading_days_nyse, bm_symbol='^GSPC'):
    bm_filepath = get_data_filepath(get_benchmark_filename(bm_symbol))
    try:
        saved_benchmarks = pd.Series.from_csv(bm_filepath)
    except (OSError, IOError, ValueError):
        logger.info(
            "No cache found at {path}. "
            "Downloading benchmark data for '{symbol}'.",
            symbol=bm_symbol,
            path=bm_filepath,
        )

        dump_benchmarks(bm_symbol)
        saved_benchmarks = pd.Series.from_csv(bm_filepath)

    saved_benchmarks = saved_benchmarks.tz_localize('UTC')

    most_recent = pd.Timestamp('today', tz='UTC') - trading_day
    most_recent_index = trading_days.searchsorted(most_recent)
    days_up_to_now = trading_days[:most_recent_index + 1]

    # Find the offset of the last date for which we have trading data in our
    # list of valid trading days
    last_bm_date = saved_benchmarks.index[-1]
    last_bm_date_offset = days_up_to_now.searchsorted(
        last_bm_date.strftime('%Y/%m/%d'))

    # If more than 1 trading days has elapsed since the last day where
    # we have data,then we need to update
    # We're doing "> 2" rather than "> 1" because we're subtracting an array
    # _length_ from an array _index_, and therefore even if we had data up to
    # and including the current day, the difference would still be 1.
    if len(days_up_to_now) - last_bm_date_offset > 2:
        benchmark_returns = update_benchmarks(bm_symbol, last_bm_date)
        if benchmark_returns.index.tz is None or \
           benchmark_returns.index.tz.zone != 'UTC':
            benchmark_returns = benchmark_returns.tz_localize('UTC')
    else:
        benchmark_returns = saved_benchmarks
        if benchmark_returns.index.tz is None or\
           benchmark_returns.index.tz.zone != 'UTC':
            benchmark_returns = benchmark_returns.tz_localize('UTC')

    # Get treasury curve module, filename & source from mapping.
    # Default to USA.
    module, filename, source = INDEX_MAPPING.get(
        bm_symbol, INDEX_MAPPING['^GSPC'])

    tr_filepath = get_data_filepath(filename)
    try:
        saved_curves = pd.DataFrame.from_csv(tr_filepath)
    except (OSError, IOError, ValueError):
        logger.info(
            "No cache found at {path}. "
            "Downloading treasury data from {source}.",
            path=tr_filepath,
            source=source,
        )

        dump_treasury_curves(module, filename)
        saved_curves = pd.DataFrame.from_csv(tr_filepath)

    # Find the offset of the last date for which we have trading data in our
    # list of valid trading days
    last_tr_date = saved_curves.index[-1]
    last_tr_date_offset = days_up_to_now.searchsorted(
        last_tr_date.strftime('%Y/%m/%d'))

    # If more than 1 trading days has elapsed since the last day where
    # we have data,then we need to update
    # Comment above explains why this is "> 2".
    if len(days_up_to_now) - last_tr_date_offset > 2:
        treasury_curves = dump_treasury_curves(module, filename)
    else:
        treasury_curves = saved_curves.tz_localize('UTC')

    return benchmark_returns, treasury_curves


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

    if start is not None and end is not None:
        assert start < end, "start date is later than end date."

    data = OrderedDict()

    if stocks is not None:
        for stock in stocks:
            print(stock)
            stock_pathsafe = stock.replace(os.path.sep, '--')
            cache_filename = "{stock}-{start}-{end}.csv".format(
                stock=stock_pathsafe,
                start=start,
                end=end).replace(':', '-')
            cache_filepath = get_cache_filepath(cache_filename)
            if os.path.exists(cache_filepath):
                stkd = pd.DataFrame.from_csv(cache_filepath)
            else:
                stkd = DataReader(stock, 'yahoo', start, end).sort_index()
                stkd.to_csv(cache_filepath)
            data[stock] = stkd

    if indexes is not None:
        for name, ticker in iteritems(indexes):
            print(name)
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
    assets.  By default, 'price' is taken from Yahoo's 'Adjusted Close',
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
    df = pd.DataFrame({key: d[close_key] for key, d in iteritems(data)})
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


def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
    data = pd.read_csv(filepath, index_col=identifier_col)
    data.index = pd.DatetimeIndex(data.index, tz=tz)
    data.sort_index(inplace=True)
    return data


def load_prices_from_csv_folder(folderpath, identifier_col, tz='UTC'):
    data = None
    for file in os.listdir(folderpath):
        if '.csv' not in file:
            continue
        raw = load_prices_from_csv(os.path.join(folderpath, file),
                                   identifier_col, tz)
        if data is None:
            data = raw
        else:
            data = pd.concat([data, raw], axis=1)
    return data
