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
from textwrap import dedent

import bcolz
from bcolz import ctable
from intervaltree import IntervalTree
from numpy import nan_to_num
from os.path import join
import json
import os
import numpy as np
import pandas as pd
from zipline.gens.sim_engine import NANOS_IN_MINUTE

from zipline.data._minute_bar_internal import (
    minute_value,
    find_position_of_minute,
    find_last_traded_position_internal
)

from zipline.utils.memoize import lazyval

US_EQUITIES_MINUTES_PER_DAY = 390

DEFAULT_EXPECTEDLEN = US_EQUITIES_MINUTES_PER_DAY * 252 * 15

OHLC_RATIO = 1000


class BcolzMinuteOverlappingData(Exception):
    pass


class BcolzMinuteWriterColumnMismatch(Exception):
    pass


def _calc_minute_index(market_opens, minutes_per_day):
    minutes = np.zeros(len(market_opens) * minutes_per_day,
                       dtype='datetime64[ns]')
    deltas = np.arange(0, minutes_per_day, dtype='timedelta64[m]')
    for i, market_open in enumerate(market_opens):
        start = market_open.asm8
        minute_values = start + deltas
        start_ix = minutes_per_day * i
        end_ix = start_ix + minutes_per_day
        minutes[start_ix:end_ix] = minute_values
    return pd.to_datetime(minutes, utc=True, box=True)


def _sid_subdir_path(sid):
    """
    Format subdir path to limit the number directories in any given
    subdirectory to 100.

    The number in each directory is designed to support at least 100000
    equities.

    Parameters:
    -----------
    sid : int
        Asset identifier.

    Returns:
    --------
    out : string
        A path for the bcolz rootdir, including subdirectory prefixes based on
        the padded string representation of the given sid.

        e.g. 1 is formatted as 00/00/000001.bcolz
    """
    padded_sid = format(sid, '06')
    return os.path.join(
        # subdir 1 00/XX
        padded_sid[0:2],
        # subdir 2 XX/00
        padded_sid[2:4],
        "{0}.bcolz".format(str(padded_sid))
    )


class BcolzMinuteBarMetadata(object):

    METADATA_FILENAME = 'metadata.json'

    @classmethod
    def metadata_path(cls, rootdir):
        return os.path.join(rootdir, cls.METADATA_FILENAME)

    @classmethod
    def read(cls, rootdir):
        path = cls.metadata_path(rootdir)
        with open(path) as fp:
            raw_data = json.load(fp)

            first_trading_day = pd.Timestamp(
                raw_data['first_trading_day'], tz='UTC')
            market_opens = pd.to_datetime(raw_data['market_opens'],
                                          unit='m',
                                          utc=True)
            market_closes = pd.to_datetime(raw_data['market_closes'],
                                           unit='m',
                                           utc=True)
            ohlc_ratio = raw_data['ohlc_ratio']
            return cls(first_trading_day,
                       market_opens,
                       market_closes,
                       ohlc_ratio)

    def __init__(self, first_trading_day,
                 market_opens,
                 market_closes,
                 ohlc_ratio):
        """
        Parameters:
        -----------
        first_trading_day : datetime-like
            UTC midnight of the first day available in the dataset.
        minute_index : pd.DatetimeIndex
            The minutes which act as an index into the corresponding values
            written into each sid's ctable.
        market_opens : pd.DatetimeIndex
            The market opens for each day in the data set. (Not yet required.)
        market_closes : pd.DatetimeIndex
            The market closes for each day in the data set. (Not yet required.)
        ohlc_ratio : int
             The factor by which the pricing data is multiplied so that the
             float data can be stored as an integer.
        """
        self.first_trading_day = first_trading_day
        self.market_opens = market_opens
        self.market_closes = market_closes
        self.ohlc_ratio = ohlc_ratio

    def write(self, rootdir):
        """
        Write the metadata to a JSON file in the rootdir.

        Values contained in the metadata are:
        first_trading_day : string
            'YYYY-MM-DD' formatted representation of the first trading day
             available in the dataset.
        minute_index : list of integers
             nanosecond integer representation of the minutes, the enumeration
             of which corresponds to the values in each bcolz carray.
        ohlc_ratio : int
             The factor by which the pricing data is multiplied so that the
             float data can be stored as an integer.
        """
        metadata = {
            'first_trading_day': str(self.first_trading_day.date()),
            'market_opens': self.market_opens.values.
            astype('datetime64[m]').
            astype(np.int64).tolist(),
            'market_closes': self.market_closes.values.
            astype('datetime64[m]').
            astype(np.int64).tolist(),
            'ohlc_ratio': self.ohlc_ratio,
        }
        with open(self.metadata_path(rootdir), 'w+') as fp:
            json.dump(metadata, fp)


class BcolzMinuteBarWriter(object):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.

    Writes a bcolz directory for each individual sid, all contained within
    a root directory which also contains metadata about the entire dataset.

    Each individual asset's data is stored as a bcolz table with a column for
    each pricing field: (open, high, low, close, volume)

    The open, high, low, and close columns are integers which are 1000 times
    the quoted price, so that the data can represented and stored as an
    np.uint32, supporting market prices quoted up to the thousands place.

    volume is a np.uint32 with no mutation of the tens place.

    The 'index' for each individual asset are a repeating period of minutes of
    length `minutes_per_day` starting from each market open.
    The file format does not account for half-days.
    e.g.:
    2016-01-19 14:31
    2016-01-19 14:32
    ...
    2016-01-19 20:59
    2016-01-19 21:00
    2016-01-20 14:31
    2016-01-20 14:32
    ...
    2016-01-20 20:59
    2016-01-20 21:00

    All assets are written with a common 'index', sharing a common first
    trading day. Assets that do not begin trading until after the first trading
    day will have zeros for all pricing data up and until data is traded.

    'index' is in quotations, because bcolz does not provide an index. The
    format allows index-like behavior by writing each minute's data into the
    corresponding position of the enumeration of the aforementioned datetime
    index.

    The datetimes which correspond to each position are written in the metadata
    as integer nanoseconds since the epoch into the `minute_index` key.
    """
    COL_NAMES = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self,
                 first_trading_day,
                 rootdir,
                 market_opens,
                 market_closes,
                 minutes_per_day,
                 ohlc_ratio=OHLC_RATIO,
                 expectedlen=DEFAULT_EXPECTEDLEN):
        """
        Parameters:
        -----------
        first_trading_day : datetime-like
            The first trading day in the data set.

        rootdir : string
            Path to the root directory into which to write the metadata and
            bcolz subdirectories.

        market_opens : pd.Series
            The market opens used as a starting point for each periodic span of
            minutes in the index.

            The index of the series is expected to be a DatetimeIndex of the
            UTC midnight of each trading day.

            The values are datetime64-like UTC market opens for each day in the
            index.

        market_closes : pd.Series
            The market closes that correspond with the market opens,

            The index of the series is expected to be a DatetimeIndex of the
            UTC midnight of each trading day.

            The values are datetime64-like UTC market opens for each day in the
            index.

            The closes are written so that the reader can filter out non-market
            minutes even though the tail end of early closes are written in
            the data arrays to keep a regular shape.

        minutes_per_day : int
            The number of minutes per each period. Defaults to 390, the mode
            of minutes in NYSE trading days.

        ohlc_ratio : int
            The ratio by which to multiply the pricing data to convert the
            floats from floats to an integer to fit within the np.uint32.

            The default is 1000 to support pricing data which comes in to the
            thousands place.

        expectedlen : int
            The expected length of the dataset, used when creating the initial
            bcolz ctable.

            If the expectedlen is not used, the chunksize and corresponding
            compression ratios are not ideal.

            Defaults to supporting 15 years of NYSE equity market data.

            see: http://bcolz.blosc.org/opt-tips.html#informing-about-the-length-of-your-carrays # noqa
        """
        self._rootdir = rootdir
        self._first_trading_day = first_trading_day
        self._market_opens = market_opens[
            market_opens.index.slice_indexer(start=self._first_trading_day)]
        self._market_closes = market_closes[
            market_closes.index.slice_indexer(start=self._first_trading_day)]
        self._trading_days = market_opens.index
        self._minutes_per_day = minutes_per_day
        self._expectedlen = expectedlen
        self._ohlc_ratio = ohlc_ratio

        self._minute_index = _calc_minute_index(
            self._market_opens, self._minutes_per_day)

        metadata = BcolzMinuteBarMetadata(
            self._first_trading_day,
            self._market_opens,
            self._market_closes,
            self._ohlc_ratio,
        )
        metadata.write(self._rootdir)

    @property
    def first_trading_day(self):
        return self._first_trading_day

    def sidpath(self, sid):
        """
        Parameters:
        -----------
        sid : int
            Asset identifier.

        Returns:
        --------
        out : string
            Full path to the bcolz rootdir for the given sid.
        """
        sid_subdir = _sid_subdir_path(sid)
        return join(self._rootdir, sid_subdir)

    def last_date_in_output_for_sid(self, sid):
        """
        Parameters:
        -----------
        sid : int
            Asset identifier.

        Returns:
        --------
        out : pd.Timestamp
            The midnight of the last date written in to the output for the
            given sid.
        """
        sizes_path = "{0}/close/meta/sizes".format(self.sidpath(sid))
        if not os.path.exists(sizes_path):
            return pd.NaT
        with open(sizes_path, mode='r') as f:
            sizes = f.read()
        data = json.loads(sizes)
        num_days = data['shape'][0] / self._minutes_per_day
        if num_days == 0:
            # empty container
            return pd.NaT
        return self._trading_days[num_days - 1]

    def _init_ctable(self, path):
        """
        Create empty ctable for given path.

        Parameters:
        -----------
        path : string
            The path to rootdir of the new ctable.
        """
        # Only create the containing subdir on creation.
        # This is not to be confused with the `.bcolz` directory, but is the
        # directory up one level from the `.bcolz` directories.
        sid_containing_dirname = os.path.dirname(path)
        if not os.path.exists(sid_containing_dirname):
            # Other sids may have already created the containing directory.
            os.makedirs(sid_containing_dirname)
        initial_array = np.empty(0, np.uint32)
        table = ctable(
            rootdir=path,
            columns=[
                initial_array,
                initial_array,
                initial_array,
                initial_array,
                initial_array,
            ],
            names=[
                'open',
                'high',
                'low',
                'close',
                'volume'
            ],
            expectedlen=self._expectedlen,
            mode='w',
        )
        table.flush()
        return table

    def _ensure_ctable(self, sid):
        """Ensure that a ctable exists for ``sid``, then return it."""
        sidpath = self.sidpath(sid)
        if not os.path.exists(sidpath):
            return self._init_ctable(sidpath)
        return bcolz.ctable(rootdir=sidpath, mode='a')

    def _zerofill(self, table, numdays):
        num_to_prepend = numdays * self._minutes_per_day
        prepend_array = np.zeros(num_to_prepend, np.uint32)
        # Fill all OHLCV with zeros.
        table.append([prepend_array] * 5)
        table.flush()

    def pad(self, sid, date):
        """
        Fill sid container with empty data through the specified date.

        e.g. if the date is two days after the last date in the sid's existing
        output, 2 x `minute_per_day` worth of zeros will be added to the
        output.

        Parameters:
        -----------
        sid : int
            The asset identifier for the data being written.
        date : datetime-like
            The date used to calculate how many slots to be pad.
            The padding is done through the date, i.e. after the padding is
            done the `last_date_in_output_for_sid` will be equal to `date`
        """
        table = self._ensure_ctable(sid)

        last_date = self.last_date_in_output_for_sid(sid)

        tds = self._trading_days

        if date <= last_date or date < tds[0]:
            # No need to pad.
            return

        if last_date == pd.NaT:
            # If there is no data, determine how many days to add so that
            # desired days are written to the correct slots.
            days_to_zerofill = tds[tds.slice_indexer(end=date)]
        else:
            days_to_zerofill = tds[tds.slice_indexer(
                start=last_date + tds.freq,
                end=date)]

        self._zerofill(table, len(days_to_zerofill))

        new_last_date = self.last_date_in_output_for_sid(sid)
        assert new_last_date == date, "new_last_date={0} != date={1}".format(
            new_last_date, date)

    def write(self, sid, df):
        """
        Write the OHLCV data for the given sid.
        If there is no bcolz ctable yet created for the sid, create it.
        If the length of the bcolz ctable is not exactly to the date before
        the first day provided, fill the ctable with 0s up to that date.
        Writes in blocks of the size of the days times minutes per day.
        Parameters:
        -----------
        sid : int
            The asset identifer for the data being written.
        df : pd.DataFrame
            DataFrame of market data with the following characteristics.
            columns : ('open', 'high', 'low', 'close', 'volume')
                open : float64
                high : float64
                low  : float64
                close : float64
                volume : float64|int64
            index : DatetimeIndex of market minutes.
        """
        cols = {
            'open': df.open.values,
            'high': df.high.values,
            'low': df.low.values,
            'close': df.close.values,
            'volume': df.volume.values,
        }
        dts = df.index.values
        # Call internal method, since DataFrame has already ensured matching
        # index and value lengths.
        self._write_cols(sid, dts, cols)

    def write_cols(self, sid, dts, cols):
        """
        Write the OHLCV data for the given sid.
        If there is no bcolz ctable yet created for the sid, create it.
        If the length of the bcolz ctable is not exactly to the date before
        the first day provided, fill the ctable with 0s up to that date.
        Writes in blocks of the size of the days times minutes per day.
        Parameters:
        -----------
        sid : int
            The asset identifier for the data being written.
        dts : datetime64 array
            The dts corresponding to values in cols.
        cols : dict of str -> np.array
            dict of market data with the following characteristics.
            keys are ('open', 'high', 'low', 'close', 'volume')
                open : float64
                high : float64
                low  : float64
                close : float64
                volume : float64|int64
        """
        if not all(len(dts) == len(cols[name]) for name in self.COL_NAMES):
            raise BcolzMinuteWriterColumnMismatch(
                "Length of dts={0} should match cols: {1}".format(
                    len(dts),
                    " ".join("{0}={1}".format(name, len(cols[name]))
                             for name in self.COL_NAMES)))
        self._write_cols(sid, dts, cols)

    def _write_cols(self, sid, dts, cols):
        """
        Internal method for `write_cols` and `write`.

        Parameters:
        -----------
        sid : int
            The asset identifier for the data being written.
        dts : datetime64 array
            The dts corresponding to values in cols.
        cols : dict of str -> np.array
            dict of market data with the following characteristics.
            keys are ('open', 'high', 'low', 'close', 'volume')
                open : float64
                high : float64
                low  : float64
                close : float64
                volume : float64|int64
        """
        table = self._ensure_ctable(sid)

        tds = self._trading_days
        input_first_day = pd.Timestamp(dts[0].astype('datetime64[D]'),
                                       tz='UTC')
        input_last_day = pd.Timestamp(dts[-1].astype('datetime64[D]'),
                                      tz='UTC')

        last_date = self.last_date_in_output_for_sid(sid)

        if last_date >= input_first_day:
            raise BcolzMinuteOverlappingData(dedent("""
            Data with last_date={0} already includes input start={1} for
            sid={2}""".strip()).format(last_date, input_first_day, sid))

        day_before_input = input_first_day - tds.freq

        self.pad(sid, day_before_input)
        table = self._ensure_ctable(sid)

        days_to_write = tds[tds.slice_indexer(start=input_first_day,
                                              end=input_last_day)]

        minutes_count = len(days_to_write) * self._minutes_per_day

        all_minutes = self._minute_index
        indexer = all_minutes.slice_indexer(start=days_to_write[0])
        all_minutes_in_window = all_minutes[indexer]

        open_col = np.zeros(minutes_count, dtype=np.uint32)
        high_col = np.zeros(minutes_count, dtype=np.uint32)
        low_col = np.zeros(minutes_count, dtype=np.uint32)
        close_col = np.zeros(minutes_count, dtype=np.uint32)
        vol_col = np.zeros(minutes_count, dtype=np.uint32)

        dt_ixs = np.searchsorted(all_minutes_in_window.values,
                                 dts.astype('datetime64[ns]'))

        ohlc_ratio = self._ohlc_ratio
        open_col[dt_ixs] = (nan_to_num(cols['open']) * ohlc_ratio).\
            astype(np.uint32)
        high_col[dt_ixs] = (nan_to_num(cols['high']) * ohlc_ratio).\
            astype(np.uint32)
        low_col[dt_ixs] = (nan_to_num(cols['low']) * ohlc_ratio).\
            astype(np.uint32)
        close_col[dt_ixs] = (nan_to_num(cols['close']) * ohlc_ratio).\
            astype(np.uint32)
        vol_col[dt_ixs] = cols['volume'].astype(np.uint32)

        table.append([
            open_col,
            high_col,
            low_col,
            close_col,
            vol_col
        ])
        table.flush()


class BcolzMinuteBarReader(object):

    def __init__(self, rootdir):
        """
        Reader for data written by BcolzMinuteBarWriter

        Parameters:
        -----------
        rootdir : string
            The root directory containing the metadata and asset bcolz
            directories.
        """
        self._rootdir = rootdir

        metadata = self._get_metadata()

        self._first_trading_day = metadata.first_trading_day

        self._market_opens = metadata.market_opens
        self._market_open_values = metadata.market_opens.values.\
            astype('datetime64[m]').astype(np.int64)
        self._market_closes = metadata.market_closes
        self._market_close_values = metadata.market_closes.values.\
            astype('datetime64[m]').astype(np.int64)

        self._ohlc_inverse = 1.0 / metadata.ohlc_ratio

        self._carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {},
        }

        self._last_get_value_dt_position = None
        self._last_get_value_dt_value = None

    def _get_metadata(self):
        return BcolzMinuteBarMetadata.read(self._rootdir)

    @lazyval
    def last_available_dt(self):
        return self._market_closes[-1]

    @property
    def first_trading_day(self):
        return self._first_trading_day

    def _minutes_to_exclude(self):
        """
        Calculate the minutes which should be excluded when a window
        occurs on days which had an early close, i.e. days where the close
        based on the regular period of minutes per day and the market close
        do not match.

        Returns:
        --------
        List of DatetimeIndex representing the minutes to exclude because
        of early closes.
        """
        market_opens = self._market_opens.values.astype('datetime64[m]')
        market_closes = self._market_closes.values.astype('datetime64[m]')
        minutes_per_day = (market_closes - market_opens).astype(np.int64)
        early_indices = np.where(
            minutes_per_day != US_EQUITIES_MINUTES_PER_DAY - 1)[0]
        early_opens = self._market_opens[early_indices]
        early_closes = self._market_closes[early_indices]
        minutes = [(market_open, early_close)
                   for market_open, early_close
                   in zip(early_opens, early_closes)]
        return minutes

    @lazyval
    def _minute_exclusion_tree(self):
        """
        Build an interval tree keyed by the start and end of each range
        of positions should be dropped from windows. (These are the minutes
        between an early close and the minute which would be the close based
        on the regular period if there were no early close.)
        The value of each node is the same start and end position stored as
        a tuple.

        The data is stored as such in support of a fast answer to the question,
        does a given start and end position overlap any of the exclusion spans?

        Returns
        -------
        IntervalTree containing nodes which represent the minutes to exclude
        because of early closes.
        """
        itree = IntervalTree()
        for market_open, early_close in self._minutes_to_exclude():
            start_pos = self._find_position_of_minute(early_close) + 1
            end_pos = (
                self._find_position_of_minute(market_open)
                +
                US_EQUITIES_MINUTES_PER_DAY
                -
                1
            )
            data = (start_pos, end_pos)
            itree[start_pos:end_pos + 1] = data
        return itree

    def _exclusion_indices_for_range(self, start_idx, end_idx):
        """
        Returns
        -------
        List of tuples of (start, stop) which represent the ranges of minutes
        which should be excluded when a market minute window is requested.
        """
        itree = self._minute_exclusion_tree
        if itree.overlaps(start_idx, end_idx):
            ranges = []
            intervals = itree[start_idx:end_idx]
            for interval in intervals:
                ranges.append(interval.data)
            return sorted(ranges)
        else:
            return None

    def _get_carray_path(self, sid, field):
        sid_subdir = _sid_subdir_path(sid)
        # carrays are subdirectories of the sid's rootdir
        return os.path.join(self._rootdir, sid_subdir, field)

    def _open_minute_file(self, field, sid):
        sid = int(sid)

        try:
            carray = self._carrays[field][sid]
        except KeyError:
            carray = self._carrays[field][sid] = \
                bcolz.carray(rootdir=self._get_carray_path(sid, field),
                             mode='r')

        return carray

    def get_value(self, sid, dt, field):
        """
        Retrieve the pricing info for the given sid, dt, and field.

        Parameters:
        -----------
        sid : int
            Asset identifier.
        dt : datetime-like
            The datetime at which the trade occurred.
        field : string
            The type of pricing data to retrieve.
            ('open', 'high', 'low', 'close', 'volume')

        Returns:
        --------
        out : float|int

        The market data for the given sid, dt, and field coordinates.

        For OHLC:
            Returns a float if a trade occurred at the given dt.
            If no trade occurred, a np.nan is returned.

        For volume:
            Returns the integer value of the volume.
            (A volume of 0 signifies no trades for the given dt.)
        """
        if self._last_get_value_dt_value == dt.value:
            minute_pos = self._last_get_value_dt_position
        else:
            minute_pos = self._find_position_of_minute(dt)
            self._last_get_value_dt_value = dt.value
            self._last_get_value_dt_position = minute_pos

        value = self._open_minute_file(field, sid)[minute_pos]
        if value == 0:
            if field == 'volume':
                return 0
            else:
                return np.nan
        if field != 'volume':
            value *= self._ohlc_inverse
        return value

    def get_last_traded_dt(self, asset, dt):
        minute_pos = self._find_last_traded_position(asset, dt)
        if minute_pos == -1:
            return pd.NaT
        return self._pos_to_minute(minute_pos)

    def _find_last_traded_position(self, asset, dt):
        volumes = self._open_minute_file('volume', asset)
        start_date_minutes = asset.start_date.value / NANOS_IN_MINUTE
        dt_minutes = dt.value / NANOS_IN_MINUTE

        if dt_minutes < start_date_minutes:
            return -1

        return find_last_traded_position_internal(
            self._market_open_values,
            self._market_close_values,
            dt_minutes,
            start_date_minutes,
            volumes,
            US_EQUITIES_MINUTES_PER_DAY
        )

    def _pos_to_minute(self, pos):
        minute_epoch = minute_value(
            self._market_open_values,
            pos,
            US_EQUITIES_MINUTES_PER_DAY
        )

        return pd.Timestamp(minute_epoch, tz='UTC', unit="m")

    def _find_position_of_minute(self, minute_dt):
        """
        Internal method that returns the position of the given minute in the
        list of every trading minute since market open of the first trading
        day. Adjusts non market minutes to the last close.

        ex. this method would return 1 for 2002-01-02 9:32 AM Eastern, if
        2002-01-02 is the first trading day of the dataset.

        Parameters
        ----------
        minute_dt: pd.Timestamp
            The minute whose position should be calculated.

        Returns
        -------
        int: The position of the given minute in the list of all trading
        minutes since market open on the first trading day.
        """
        return find_position_of_minute(
            self._market_open_values,
            self._market_close_values,
            minute_dt.value / NANOS_IN_MINUTE,
            US_EQUITIES_MINUTES_PER_DAY,
        )

    def unadjusted_window(self, fields, start_dt, end_dt, sids):
        """
        Parameters
        ----------
        fields : list of str
           'open', 'high', 'low', 'close', or 'volume'
        start_dt: Timestamp
           Beginning of the window range.
        end_dt: Timestamp
           End of the window range.
        sids : list of int
           The asset identifiers in the window.

        Returns
        -------
        list of np.ndarray
            A list with an entry per field of ndarrays with shape
            (sids, minutes in range) with a dtype of float64, containing the
            values for the respective field over start and end dt range.
        """
        start_idx = self._find_position_of_minute(start_dt)
        end_idx = self._find_position_of_minute(end_dt)

        num_minutes = (end_idx - start_idx + 1)

        results = []

        indices_to_exclude = self._exclusion_indices_for_range(
            start_idx, end_idx)
        if indices_to_exclude is not None:
            for excl_start, excl_stop in indices_to_exclude:
                length = excl_stop - excl_start + 1
                num_minutes -= length

        shape = (len(sids), num_minutes)

        for field in fields:
            if field != 'volume':
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.uint32)

            for i, sid in enumerate(sids):
                carray = self._open_minute_file(field, sid)
                values = carray[start_idx:end_idx + 1]
                if indices_to_exclude is not None:
                    for excl_start, excl_stop in indices_to_exclude[::-1]:
                        excl_slice = np.s_[
                            excl_start - start_idx:excl_stop - start_idx + 1]
                        values = np.delete(values, excl_slice)
                where = values != 0
                out[i, where] = values[where]
            if field != 'volume':
                out *= self._ohlc_inverse
            results.append(out)
        return results
