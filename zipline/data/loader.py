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
from collections import OrderedDict

import logbook

import pandas as pd
from pandas.io.data import DataReader
import pytz

from six import iteritems

from . benchmarks import get_benchmark_returns
from .paths import (
    cache_root,
    data_root,
)

from zipline.utils.tradingcalendar import (
    trading_day as trading_day_nyse,
    trading_days as trading_days_nyse,
)

logger = logbook.Logger('Loader')

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
    dr = data_root()

    if not os.path.exists(dr):
        os.makedirs(dr)

    return os.path.join(dr, name)


def get_cache_filepath(name):
    cr = cache_root()
    if not os.path.exists(cr):
        os.makedirs(cr)

    return os.path.join(cr, name)


def get_benchmark_filename(symbol):
    return "%s_benchmark.csv" % symbol


def has_data_for_dates(series_or_df, first_date, last_date):
    """
    Does `series_or_df` have data on or before first_date and on or after
    last_date?
    """
    dts = series_or_df.index
    if not isinstance(dts, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex, but got %s." % type(dts))
    first, last = dts[[0, -1]]
    return (first <= first_date) and (last >= last_date)


def load_market_data(trading_day=trading_day_nyse,
                     trading_days=trading_days_nyse, bm_symbol='^GSPC'):
    first_date = trading_days[0]

    # We expect to have benchmark and treasury data that's current up until
    # two full trading days prior to the most recently completed trading day.
    # Example:
    # On Thu Oct 22 2015, the previous completed trading day is Wed Oct 21.
    # However, data for Oct 21 doesn't become available until the early morning
    # hours of Oct 22.  This means that there are times on the 22nd at which we
    # cannot reasonably expect to have data for the 21st available.  To be
    # conservative, we instead expect that at any time on the 22nd, we can
    # download data for Tuesday the 20th, which is two full trading days prior
    # to the date on which we're running a test.

    # We'll attempt to download new data if the latest entry in our cache is
    # before this date.
    last_date = (
        pd.Timestamp('now', tz='UTC').normalize() - (2 * trading_day)
    )
    benchmark_returns = ensure_benchmark_data(
        bm_symbol,
        first_date,
        last_date,
    )
    treasury_curves = ensure_treasury_data(
        bm_symbol,
        first_date,
        last_date,
    )
    return benchmark_returns, treasury_curves


def ensure_benchmark_data(symbol, first_date, last_date):
    """
    Ensure we have benchmark data for `symbol` from `first_date` to `last_date`

    Parameters
    ----------
    symbol : str
        The symbol for the benchmark to load.
    first_date : pd.Timestamp
        First required date for the cache.
    last_date : pd.Timestamp
        Last required date for the cache.

    We attempt to download data unless we already have data stored at the data
    cache for `symbol` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.
    """
    path = get_data_filepath(get_benchmark_filename(symbol))
    try:
        data = pd.Series.from_csv(path).tz_localize('UTC')
        if has_data_for_dates(data, first_date, last_date):
            return data
    except (OSError, IOError, ValueError) as e:
        # These can all be raised by various versions of pandas on various
        # classes of malformed input.  Treat them all as cache misses.
        logger.info(
            "Loading data for {path} failed with error [{error}].".format(
                path=path, error=e,
            )
        )
    logger.info(
        "Cache at {path} does not have data from {start} to {end}.\n"
        "Downloading benchmark data for '{symbol}'.",
        start=first_date,
        end=last_date,
        symbol=symbol,
        path=path,
    )

    data = get_benchmark_returns(symbol, first_date, last_date)
    data.to_csv(path)
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn("Still don't have expected data after redownload!")
    return data


def ensure_treasury_data(bm_symbol, first_date, last_date):
    """
    Ensure we have treasury data from treasury module associated with
    `bm_symbol`.

    Parameters
    ----------
    bm_symbol : str
        Benchmark symbol for which we're loading associated treasury curves.
    first_date : pd.Timestamp
        First date required to be in the cache.
    last_date : pd.Timestamp
        Last date required to be in the cache.

    We attempt to download data unless we already have data stored in the cache
    for `module_name` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.
    """
    module_name, filename, source = INDEX_MAPPING.get(
        bm_symbol, INDEX_MAPPING['^GSPC']
    )
    path = get_data_filepath(filename)
    try:
        data = pd.DataFrame.from_csv(path).tz_localize('UTC')
        if has_data_for_dates(data, first_date, last_date):
            return data
    except (OSError, IOError, ValueError) as e:
        # These can all be raised by various versions of pandas on various
        # classes of malformed input.  Treat them all as cache misses.
        logger.info(
            "Loading data for {path} failed with error [{error}].".format(
                path=path, error=e,
            )
        )

    try:
        m = importlib.import_module("." + module_name, package='zipline.data')
    except ImportError:
        raise NotImplementedError(
            'Treasury curve {0} module not implemented'.format(module_name))

    data = m.get_treasury_data()
    data.to_csv(path)
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn("Still don't have expected data after redownload!")
    return data


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
