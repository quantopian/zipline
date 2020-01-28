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

import logbook
import pandas as pd
from six.moves.urllib_error import HTTPError
from trading_calendars import get_calendar

from .benchmarks import get_benchmark_returns
from ..utils.paths import (
    cache_root,
    data_root,
)


logger = logbook.Logger('Loader')

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


def load_benchmark_data(trading_day=None,
                        trading_days=None,
                        bm_symbol='SPY',
                        environ=None):
    """
    Load benchmark returns for the given calendar and benchmark symbol.

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
        Symbol for the benchmark index to load. Defaults to 'SPY', the ticker
        for the S&P 500, provided by IEX Trading.

    Returns
    -------
    benchmark_returns : pd.Series

    Notes
    -----
    Return values are DatetimeIndexed with values dated to midnight in UTC of
    each stored date.
    """
    if trading_day is None:
        trading_day = get_calendar('XNYS').day
    if trading_days is None:
        trading_days = get_calendar('XNYS').all_sessions

    first_date = trading_days[0]
    now = pd.Timestamp.utcnow()

    # we will fill missing benchmark data through latest trading date
    last_date = trading_days[trading_days.get_loc(now, method='ffill')]

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

    return br[br.index.slice_indexer(first_date, last_date)]


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
    data = _load_cached_data(filename, first_date, last_date, now, environ)
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
        data = get_benchmark_returns(symbol)
        data.to_csv(get_data_filepath(filename, environ))
    except (OSError, IOError, HTTPError):
        logger.exception('Failed to cache the new benchmark returns')
        raise
    if not has_data_for_dates(data, first_date, last_date):
        logger.warn(
            ("Still don't have expected benchmark data for {symbol!r} "
                "from {first_date} to {last_date} after redownload!"),
            symbol=symbol,
            first_date=first_date - trading_day,
            last_date=last_date
        )
    return data


def _load_cached_data(filename, first_date, last_date, now, environ=None):
    # Path for the cache.
    path = get_data_filepath(filename, environ)

    # If the path does not exist, it means the first download has not happened
    # yet, so don't try to read from 'path'.
    if os.path.exists(path):
        try:
            data = pd.read_csv(
                path,
                parse_dates=[0],
                index_col=0,
                header=None,
                # Pass squeeze=True so that we get a series instead of a frame.
                squeeze=True,
            ).tz_localize('UTC')

            if has_data_for_dates(data, first_date, last_date):
                return data

            # Don't re-download if we've successfully downloaded and written a
            # file in the last hour.
            last_download_time = last_modified_time(path)
            if (now - last_download_time) <= ONE_HOUR:
                logger.warn(
                    "Refusing to download new benchmark data because a "
                    "download succeeded at {time}.",
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
