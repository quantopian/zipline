#
# Copyright 2016 Quantopian, Inc.
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
from collections import OrderedDict

import logbook
import pandas as pd
from pandas_datareader.data import DataReader
import pytz
from six import iteritems
from six.moves.urllib_error import HTTPError

from .benchmarks import get_benchmark_returns
from . import treasuries, treasuries_can
from ..utils.paths import (
    cache_root,
    data_root,
)
from zipline.utils.calendars import get_calendar


logger = logbook.Logger('Loader')

# Mapping from index symbol to appropriate bond data
INDEX_MAPPING = {
    'SPY':
    (treasuries, 'treasury_curves.csv', 'www.federalreserve.gov'),
    '^GSPTSE':
    (treasuries_can, 'treasury_curves_can.csv', 'bankofcanada.ca'),
    '^FTSE':  # use US treasuries until UK bonds implemented
    (treasuries, 'treasury_curves.csv', 'www.federalreserve.gov'),
}

ONE_HOUR = pd.Timedelta(hours=1)


def last_modified_time(path):
    """
    Get the last modified time of path as a Timestamp.
    """
    return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')


def get_data_filepath(name, environ=None):
    """
    Returns a handle to data file.

    Creates containing directory, if needed.
    """
    dr = data_root(environ)

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


def load_market_data(trading_day=None, trading_days=None, bm_symbol='SPY',
                     environ=None):
    """
    Load benchmark returns and treasury yield curves for the given calendar and
    benchmark symbol.

    Benchmarks are downloaded as a Series from Google Finance.  Treasury curves
    are US Treasury Bond rates and are downloaded from 'www.federalreserve.gov'
    by default.  For Canadian exchanges, a loader for Canadian bonds from the
    Bank of Canada is also available.

    Results downloaded from the internet are cached in
    ~/.zipline/data. Subsequent loads will attempt to read from the cached
    files before falling back to redownload.

    Parameters
    ----------
    trading_day : pandas.CustomBusinessDay, optional
        A trading_day used to determine the latest day for which we
        expect to have data.  Defaults to an NYSE trading day.
    trading_days : pd.DatetimeIndex, optional
        A calendar of trading days.  Also used for determining what cached
        dates we should expect to have cached. Defaults to the NYSE calendar.
    bm_symbol : str, optional
        Symbol for the benchmark index to load.  Defaults to 'SPY', the Google
        ticker for the S&P 500.

    Returns
    -------
    (benchmark_returns, treasury_curves) : (pd.Series, pd.DataFrame)

    Notes
    -----

    Both return values are DatetimeIndexed with values dated to midnight in UTC
    of each stored date.  The columns of `treasury_curves` are:

    '1month', '3month', '6month',
    '1year','2year','3year','5year','7year','10year','20year','30year'
    """
    if trading_day is None:
        trading_day = get_calendar('NYSE').trading_day
    if trading_days is None:
        trading_days = get_calendar('NYSE').all_sessions

    first_date = trading_days[0]
    now = pd.Timestamp.utcnow()

    # We expect to have benchmark and treasury data that's current up until
    # **two** full trading days prior to the most recently completed trading
    # day.
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
    last_date = trading_days[trading_days.get_loc(now, method='ffill') - 2]

    br = ensure_benchmark_data(
        bm_symbol,
        first_date,
        last_date,
        now,
        # We need the trading_day to figure out the close prior to the first
        # date so that we can compute returns for the first date.
        trading_day,
        environ,
    )
    tc = ensure_treasury_data(
        bm_symbol,
        first_date,
        last_date,
        now,
        environ,
    )
    benchmark_returns = br[br.index.slice_indexer(first_date, last_date)]
    treasury_curves = tc[tc.index.slice_indexer(first_date, last_date)]
    return benchmark_returns, treasury_curves


def ensure_benchmark_data(symbol, first_date, last_date, now, trading_day,
                          environ=None):
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
    now : pd.Timestamp
        The current time.  This is used to prevent repeated attempts to
        re-download data that isn't available due to scheduling quirks or other
        failures.
    trading_day : pd.CustomBusinessDay
        A trading day delta.  Used to find the day before first_date so we can
        get the close of the day prior to first_date.

    We attempt to download data unless we already have data stored at the data
    cache for `symbol` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.

    If we perform a download and the cache criteria are not satisfied, we wait
    at least one hour before attempting a redownload.  This is determined by
    comparing the current time to the result of os.path.getmtime on the cache
    path.
    """
    filename = get_benchmark_filename(symbol)
    data = _load_cached_data(filename, first_date, last_date, now, 'benchmark',
                             environ)
    if data is not None:
        return data

    # If no cached data was found or it was missing any dates then download the
    # necessary data.
    logger.info(
        ('Downloading benchmark data for {symbol!r} '
            'from {first_date} to {last_date}'),
        symbol=symbol,
        first_date=first_date - trading_day,
        last_date=last_date
    )

    try:
        data = get_benchmark_returns(
            symbol,
            first_date - trading_day,
            last_date,
        )
        data.to_csv(get_data_filepath(filename, environ))
    except (OSError, IOError, HTTPError):
        logger.exception('Failed to cache the new benchmark returns')
        raise
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn("Still don't have expected data after redownload!")
    return data


def ensure_treasury_data(symbol, first_date, last_date, now, environ=None):
    """
    Ensure we have treasury data from treasury module associated with
    `symbol`.

    Parameters
    ----------
    symbol : str
        Benchmark symbol for which we're loading associated treasury curves.
    first_date : pd.Timestamp
        First date required to be in the cache.
    last_date : pd.Timestamp
        Last date required to be in the cache.
    now : pd.Timestamp
        The current time.  This is used to prevent repeated attempts to
        re-download data that isn't available due to scheduling quirks or other
        failures.

    We attempt to download data unless we already have data stored in the cache
    for `module_name` whose first entry is before or on `first_date` and whose
    last entry is on or after `last_date`.

    If we perform a download and the cache criteria are not satisfied, we wait
    at least one hour before attempting a redownload.  This is determined by
    comparing the current time to the result of os.path.getmtime on the cache
    path.
    """
    loader_module, filename, source = INDEX_MAPPING.get(
        symbol, INDEX_MAPPING['SPY'],
    )
    first_date = max(first_date, loader_module.earliest_possible_date())

    data = _load_cached_data(filename, first_date, last_date, now, 'treasury',
                             environ)
    if data is not None:
        return data

    # If no cached data was found or it was missing any dates then download the
    # necessary data.
    logger.info('Downloading treasury data for {symbol!r}.', symbol=symbol)

    try:
        data = loader_module.get_treasury_data(first_date, last_date)
        data.to_csv(get_data_filepath(filename, environ))
    except (OSError, IOError, HTTPError):
        logger.exception('failed to cache treasury data')
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn("Still don't have expected data after redownload!")
    return data


def _load_cached_data(filename, first_date, last_date, now, resource_name,
                      environ=None):
    if resource_name == 'benchmark':
        from_csv = pd.Series.from_csv
    else:
        from_csv = pd.DataFrame.from_csv

    # Path for the cache.
    path = get_data_filepath(filename, environ)

    # If the path does not exist, it means the first download has not happened
    # yet, so don't try to read from 'path'.
    if os.path.exists(path):
        try:
            data = from_csv(path)
            data.index = data.index.to_datetime().tz_localize('UTC')
            if has_data_for_dates(data, first_date, last_date):
                return data

            # Don't re-download if we've successfully downloaded and written a
            # file in the last hour.
            last_download_time = last_modified_time(path)
            if (now - last_download_time) <= ONE_HOUR:
                logger.warn(
                    "Refusing to download new {resource} data because a "
                    "download succeeded at {time}.",
                    resource=resource_name,
                    time=last_download_time,
                )
                return data

        except (OSError, IOError, ValueError) as e:
            # These can all be raised by various versions of pandas on various
            # classes of malformed input.  Treat them all as cache misses.
            logger.info(
                "Loading data for {path} failed with error [{error}].",
                path=path,
                error=e,
            )

    logger.info(
        "Cache at {path} does not have data from {start} to {end}.\n",
        start=first_date,
        end=last_date,
        path=path,
    )
    return None


def _load_raw_yahoo_data(indexes=None, stocks=None, start=None, end=None):
    """Load closing prices from yahoo finance.

    :Optional:
        indexes : dict (Default: {'SPX': '^SPY'})
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
            logger.info('Loading stock: {}'.format(stock))
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
            logger.info('Loading index: {} ({})'.format(name, ticker))
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
