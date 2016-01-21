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
import numpy as np
from os.path import join
import json
import os
import pandas as pd
from pandas.core.datetools import normalize_date

MINUTES_PER_DAY = 390
DEFAULT_EXPECTEDLEN = 390 * 252 * 15
OHLC_RATIO = 1000


class BcolzMinuteOverlappingData(Exception):
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
            minute_index = pd.to_datetime(raw_data['minute_index'],
                                          utc=True)
            ohlc_ratio = raw_data['ohlc_ratio']
            return cls(first_trading_day, minute_index, ohlc_ratio)

    def __init__(self, first_trading_day, minute_index, ohlc_ratio):
        """
        Parameters:
        -----------
        first_trading_day : datetime-like
            UTC midnight of the first day available in the dataset.
        minute_index : pd.DatetimeIndex
            The minutes which act as an index into the corresponding values
            written into each sid's ctable.
        ohlc_ratio : int
             The factor by which the pricing data is multiplied so that the
             float data can be stored as an integer.
        """
        self.first_trading_day = first_trading_day
        self.minute_index = minute_index
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
            'minute_index': self.minute_index.asi8.tolist(),
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
    def __init__(self,
                 first_trading_day,
                 rootdir,
                 market_opens,
                 minutes_per_day=MINUTES_PER_DAY,
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
        self._trading_days = market_opens.index
        self._minutes_per_day = minutes_per_day
        self._expectedlen = expectedlen
        self._ohlc_ratio = ohlc_ratio

        self._minute_index = _calc_minute_index(
            self._market_opens, self._minutes_per_day)

        metadata = BcolzMinuteBarMetadata(
            self._first_trading_day,
            self._minute_index,
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
        # Only create the subdir on container creation.
        sid_dirname = os.path.dirname(path)
        os.makedirs(sid_dirname)
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
        return bcolz.ctable(rootdir=sidpath, mode='r')

    def _zerofill(self, table, numdays):
        num_to_prepend = numdays * self._minutes_per_day
        prepend_array = np.zeros(num_to_prepend, np.uint32)
        # Fill all OHLCV with zeros.
        table.append([prepend_array] * 5)
        table.flush()

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
        days : pd.DatetimeIndex
            The days for which to write data from the given df.
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
        table = self._ensure_ctable(sid)

        last_date = self.last_date_in_output_for_sid(sid)
        tds = self._trading_days
        days = tds[tds.slice_indexer(start=normalize_date(df.index[0]),
                                     end=normalize_date(df.index[-1]))]
        input_first_day = days[0]

        if last_date is pd.NaT:
            # If there is no data, determine how many days to add so that
            # desired days are written to the correct slots.
            days_to_zerofill = tds[tds.slice_indexer(end=input_first_day)]
            # Chop off the input first day.
            days_to_zerofill = days_to_zerofill[:-1]
        else:
            next_date = last_date + 1
            if next_date < input_first_day:
                # If last_date and input_first_day are not adjacent need to
                # fill in between.
                days_to_zerofill = tds[tds.slice_indexer(
                    start=last_date + 1,
                    end=input_first_day)]
                # Chop off the input first day.
                days_to_zerofill = days_to_zerofill[:-1]
            elif next_date >= input_first_day:
                raise BcolzMinuteOverlappingData(dedent("""
                window start={0} is before expected write date={1} for
                sid={2}""".strip()).format(days[0], input_first_day, sid))
            else:
                days_to_zerofill = None

        if days_to_zerofill is not None and len(days_to_zerofill):
            self._zerofill(table, len(days_to_zerofill))

        days_to_write = tds[tds.slice_indexer(start=input_first_day,
                                              end=days[-1])]
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
                                 df.index.values)

        ohlc_ratio = self._ohlc_ratio
        open_col[dt_ixs] = (df.open.values * ohlc_ratio).astype(np.uint32)
        high_col[dt_ixs] = (df.high.values * ohlc_ratio).astype(np.uint32)
        low_col[dt_ixs] = (df.low.values * ohlc_ratio).astype(np.uint32)
        close_col[dt_ixs] = (df.close.values * ohlc_ratio).astype(
            np.uint32)
        vol_col[dt_ixs] = df.volume.values.astype(np.uint32)

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
        self.rootdir = rootdir

        metadata = self._get_metadata()

        self._first_trading_day = metadata.first_trading_day
        self._minute_index = metadata.minute_index
        self._ohlc_inverse = 1.0 / metadata.ohlc_ratio

        self._carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {},
        }

    def _get_metadata(self):
        return BcolzMinuteBarMetadata.read(self.rootdir)

    def _get_carray_path(self, sid, field):
        sid_subdir = _sid_subdir_path(sid)
        # carrays are subdirectories of the sid's rootdir
        return os.path.join(self.rootdir, sid_subdir, field)

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
        minute_pos = self._find_position_of_minute(dt)
        value = self._open_minute_file(field, sid)[minute_pos]
        if value == 0:
            if field != 'volume':
                return np.nan
            else:
                return 0
        if field != 'volume':
            value *= self._ohlc_inverse
        return value

    def _find_position_of_minute(self, minute_dt):
        """
        Internal method that returns the position of the given minute in the
        list of every trading minute since market open of the first trading
        day.

        ex. this method would return 1 for 2002-01-02 9:32 AM Eastern, if
        2002-01-02 is the first trading day of the dataset.

        Parameters
        ----------
        minute_dt: pd.Timestamp
            The minute whose position should be calculated.

        Returns
        -------
        out : int

        The position of the given minute in the list of all trading minutes
        since market open on the first trading day.
        """
        return self._minute_index.get_loc(minute_dt)
