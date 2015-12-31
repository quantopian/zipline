# Copyright 2015 Quantopian, Inc.
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

from abc import (
    ABCMeta,
    abstractmethod,
)
import bcolz
import json
import os
from bcolz import ctable
from datetime import datetime
import numpy as np
from numpy import float64
from os.path import join
import pandas as pd
from pandas import read_csv
from six import with_metaclass

from zipline.finance.trading import TradingEnvironment
from zipline.utils import tradingcalendar

MINUTES_PER_DAY = 390

_writer_env = TradingEnvironment()

_NANOS_IN_MINUTE = 60000000000

METADATA_FILENAME = 'metadata.json'


def write_metadata(directory, first_trading_day):
    metadata_path = os.path.join(directory, METADATA_FILENAME)

    metadata = {
        'first_trading_day': str(first_trading_day.date())
    }

    with open(metadata_path, 'w') as fp:
        json.dump(metadata, fp)


def _bcolz_minute_index(trading_days):
    minutes = np.zeros(len(trading_days) * MINUTES_PER_DAY,
                       dtype='datetime64[ns]')
    market_opens = tradingcalendar.open_and_closes.market_open
    mask = market_opens.index.slice_indexer(start=trading_days[0],
                                            end=trading_days[-1])
    opens = market_opens[mask]

    deltas = np.arange(0, MINUTES_PER_DAY, dtype='timedelta64[m]')
    for i, market_open in enumerate(opens):
        start = market_open.asm8
        minute_values = start + deltas
        start_ix = MINUTES_PER_DAY * i
        end_ix = start_ix + MINUTES_PER_DAY
        minutes[start_ix:end_ix] = minute_values
    return pd.to_datetime(minutes, utc=True, box=True)


class BcolzMinuteBarWriter(with_metaclass(ABCMeta)):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.
    """
    @property
    def first_trading_day(self):
        return self._first_trading_day

    @abstractmethod
    def gen_frames(self, assets):
        """
        Return an iterator of pairs of (asset_id, pd.dataframe).
        """
        raise NotImplementedError()

    def write(self, directory, assets, sid_path_func=None):
        _iterator = self.gen_frames(assets)

        return self._write_internal(directory, _iterator,
                                    sid_path_func=sid_path_func)

    def full_minutes_for_days(self, dt1, dt2):
        start_date = _writer_env.normalize_date(dt1)
        end_date = _writer_env.normalize_date(dt2)

        trading_days = _writer_env.days_in_range(start_date, end_date)
        return _bcolz_minute_index(trading_days)

    def _write_internal(self, directory, iterator, sid_path_func=None):
        first_trading_day = self.first_trading_day

        write_metadata(directory, first_trading_day)

        first_open = pd.Timestamp(
            datetime(
                year=first_trading_day.year,
                month=first_trading_day.month,
                day=first_trading_day.day,
                hour=9,
                minute=31
            ), tz='US/Eastern').tz_convert('UTC')

        all_minutes = None

        for asset_id, df in iterator:
            if sid_path_func is None:
                path = join(directory, "{0}.bcolz".format(asset_id))
            else:
                path = sid_path_func(directory, asset_id)

            os.makedirs(path)

            last_dt = df.index[-1]

            if all_minutes is None:
                all_minutes = \
                    self.full_minutes_for_days(first_open, last_dt)
                minutes = all_minutes
            else:
                if df.index[-1] in all_minutes:
                    mask = all_minutes.slice_indexer(end=last_dt)
                    minutes = all_minutes[mask]
                else:
                    # Need to extend all minutes from open after last value
                    # in all_minutes to the last_dt.
                    next_open, _ = _writer_env.next_open_and_close(
                        all_minutes[-1])
                    to_append = self.full_minutes_for_days(next_open, last_dt)
                    all_minutes = all_minutes.append(to_append)
                    minutes = all_minutes

            minutes_count = len(minutes)

            open_col = np.zeros(minutes_count, dtype=np.uint32)
            high_col = np.zeros(minutes_count, dtype=np.uint32)
            low_col = np.zeros(minutes_count, dtype=np.uint32)
            close_col = np.zeros(minutes_count, dtype=np.uint32)
            vol_col = np.zeros(minutes_count, dtype=np.uint32)

            opens = df.open.values.astype(np.uint32)
            highs = df.high.values.astype(np.uint32)
            lows = df.low.values.astype(np.uint32)
            closes = df.close.values.astype(np.uint32)
            volumes = df.volume.values.astype(np.uint32)

            dt_ixs = np.searchsorted(minutes.values, df.index.values)

            for i, dt_ix in enumerate(dt_ixs):
                # Each day has 390 slots, where 9:31 is the first
                # slot (ix=0) of the day, and each slot represents a
                # minute's data.
                #
                # Get the difference in seconds between the current
                # minute and market open and then divide by 60 to get
                # the index into which to write, while still writing
                    # from the same row aligned with the dt in the from the
                # CSV.
                open_col[dt_ix] = opens[i]
                high_col[dt_ix] = highs[i]
                low_col[dt_ix] = lows[i]
                close_col[dt_ix] = closes[i]
                vol_col[dt_ix] = volumes[i]

            ctable(
                columns=[
                    open_col,
                    high_col,
                    low_col,
                    close_col,
                    vol_col,
                ],
                names=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
                rootdir=path,
                mode='w'
            )


class MinuteBarWriterFromDataFrames(BcolzMinuteBarWriter):
    _csv_dtypes = {
        'open': float64,
        'high': float64,
        'low': float64,
        'close': float64,
        'volume': float64,
    }

    def __init__(self, first_trading_day):
        self._first_trading_day = first_trading_day

    def gen_frames(self, assets):
        for asset in assets:
            df = assets[asset]
            yield asset, df.set_index("minute")


class MinuteBarWriterFromCSVs(BcolzMinuteBarWriter):
    """
    BcolzMinuteBarWriter constructed from a map of CSVs to assets.

    Parameters
    ----------
    asset_map: dict
        A map from asset_id -> path to csv with data for that asset.

    CSVs should have the following columns:
        minute : datetime64
        open : float64
        high : float64
        low : float64
        close : float64
        volume : int64
    """
    _csv_dtypes = {
        'open': float64,
        'high': float64,
        'low': float64,
        'close': float64,
        'volume': float64,
    }

    def __init__(self, asset_map, first_trading_day):
        self._asset_map = asset_map
        self._first_trading_day = first_trading_day

    def gen_frames(self, assets):
        """
        Read CSVs as DataFrames from our asset map.
        """
        dtypes = self._csv_dtypes

        for asset in assets:
            path = self._asset_map.get(asset)
            if path is None:
                raise KeyError("No path supplied for asset %s" % asset)
            df = read_csv(path, parse_dates=['minute'], dtype=dtypes)
            df = df.set_index("minute").tz_localize("UTC")

            yield asset, df


class BcolzMinuteBarReader(object):

    def __init__(self, rootdir, sid_path_func=None):
        self.rootdir = rootdir

        metadata = self._get_metadata()

        self.first_trading_day = pd.Timestamp(
            metadata['first_trading_day'], tz='UTC')
        mask = tradingcalendar.trading_days.slice_indexer(
            self.first_trading_day)
        # TODO: Read/write calendar to match daily, so that calendar is not
        # 'hardcoded'.
        self.trading_days = tradingcalendar.trading_days[mask]
        self._sid_path_func = sid_path_func

        self._carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {},
            'sid': {},
            'dt': {},
        }

        self._minute_index = _bcolz_minute_index(self.trading_days)

    def _get_metadata(self):
        with open(os.path.join(self.rootdir, METADATA_FILENAME)) as fp:
            return json.load(fp)

    def _get_ctable(self, asset):
        sid = int(asset)
        if self._sid_path_func is not None:
            path = self._sid_path_func(self.rootdir, sid)
        else:
            path = "{0}/{1}.bcolz".format(self.rootdir, sid)

        return bcolz.open(path, mode='r')

    def get_last_traded_dt(self, asset, dt):
        minute_pos = self._find_last_traded_position(asset, dt)
        if minute_pos == -1:
            return pd.NaT
        return self._minute_index[minute_pos]

    def _find_last_traded_position(self, asset, dt):
        volumes = self._open_minute_file('volume', asset)
        start_date = asset.start_date
        _minute_index = self._minute_index

        minute_pos = self._find_position_of_minute(dt)

        while True:
            dt = _minute_index[minute_pos]
            if dt < start_date:
                return -1
            if minute_pos == 0 or volumes[minute_pos] != 0:
                return minute_pos
            minute_pos -= 1

    def _find_position_of_minute(self, minute_dt):
        """
        Internal method that returns the position of the given minute in the
        list of every trading minute since market open of the first trading
        day.

        IMPORTANT: This method assumes every day is 390 minutes long, even
        early closes.  Our minute bcolz files are generated like this to
        support fast lookup.

        ex. this method would return 2 for 1/2/2002 9:32 AM Eastern, if
        1/2/2002 is the first trading day of the dataset.

        Parameters
        ----------
        minute_dt: pd.Timestamp
            The minute whose position should be calculated.

        Returns
        -------
        The position of the given minute in the list of all trading minutes
        since market open on the first trading day.
        """
        return self._minute_index.get_loc(minute_dt)

    def _open_minute_file(self, field, asset):
        sid_str = str(int(asset))

        try:
            carray = self._carrays[field][sid_str]
        except KeyError:
            carray = self._carrays[field][sid_str] = \
                self._get_ctable(asset)[field]

        return carray
