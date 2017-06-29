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
from abc import ABCMeta, abstractmethod
import json
import os
from glob import glob
from os.path import join
from textwrap import dedent

from lru import LRU
import bcolz
from bcolz import ctable
from intervaltree import IntervalTree
import logbook
import numpy as np
import pandas as pd
from pandas import HDFStore
import tables
from six import with_metaclass
from toolz import keymap, valmap

from zipline.data._minute_bar_internal import (
    minute_value,
    find_position_of_minute,
    find_last_traded_position_internal
)

from zipline.gens.sim_engine import NANOS_IN_MINUTE

from zipline.data.bar_reader import BarReader, NoDataForSid, NoDataOnDate
from zipline.data.us_equity_pricing import check_uint32_safe
from zipline.utils.calendars import get_calendar
from zipline.utils.cli import maybe_show_progress
from zipline.utils.memoize import lazyval


logger = logbook.Logger('MinuteBars')

US_EQUITIES_MINUTES_PER_DAY = 390
FUTURES_MINUTES_PER_DAY = 1440

DEFAULT_EXPECTEDLEN = US_EQUITIES_MINUTES_PER_DAY * 252 * 15

OHLC_RATIO = 1000


class BcolzMinuteOverlappingData(Exception):
    pass


class BcolzMinuteWriterColumnMismatch(Exception):
    pass


class MinuteBarReader(BarReader):
    @property
    def data_frequency(self):
        return "minute"


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

    Parameters
    ----------
    sid : int
        Asset identifier.

    Returns
    -------
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


def convert_cols(cols, scale_factor, sid, invalid_data_behavior):
    """Adapt OHLCV columns into uint32 columns.

    Parameters
    ----------
    cols : dict
        A dict mapping each column name (open, high, low, close, volume)
        to a float column to convert to uint32.
    scale_factor : int
        Factor to use to scale float values before converting to uint32.
    sid : int
        Sid of the relevant asset, for logging.
    invalid_data_behavior : str
        Specifies behavior when data cannot be converted to uint32.
        If 'raise', raises an exception.
        If 'warn', logs a warning and filters out incompatible values.
        If 'ignore', silently filters out incompatible values.
    """
    scaled_opens = np.nan_to_num(cols['open']) * scale_factor
    scaled_highs = np.nan_to_num(cols['high']) * scale_factor
    scaled_lows = np.nan_to_num(cols['low']) * scale_factor
    scaled_closes = np.nan_to_num(cols['close']) * scale_factor

    exclude_mask = np.zeros_like(scaled_opens, dtype=bool)

    for col_name, scaled_col in [
        ('open', scaled_opens),
        ('high', scaled_highs),
        ('low', scaled_lows),
        ('close', scaled_closes),
    ]:
        max_val = scaled_col.max()

        try:
            check_uint32_safe(max_val, col_name)
        except ValueError:
            if invalid_data_behavior == 'raise':
                raise

            if invalid_data_behavior == 'warn':
                logger.warn(
                    'Values for sid={}, col={} contain some too large for '
                    'uint32 (max={}), filtering them out',
                    sid, col_name, max_val,
                )

            # We want to exclude all rows that have an unsafe value in
            # this column.
            exclude_mask &= (scaled_col >= np.iinfo(np.uint32).max)

    # Convert all cols to uint32.
    opens = scaled_opens.astype(np.uint32)
    highs = scaled_highs.astype(np.uint32)
    lows = scaled_lows.astype(np.uint32)
    closes = scaled_closes.astype(np.uint32)
    volumes = cols['volume'].astype(np.uint32)

    # Exclude rows with unsafe values by setting to zero.
    opens[exclude_mask] = 0
    highs[exclude_mask] = 0
    lows[exclude_mask] = 0
    closes[exclude_mask] = 0
    volumes[exclude_mask] = 0

    return opens, highs, lows, closes, volumes


class BcolzMinuteBarMetadata(object):
    """
    Parameters
    ----------
    ohlc_ratio : int
         The factor by which the pricing data is multiplied so that the
         float data can be stored as an integer.
    calendar :  zipline.utils.calendars.trading_calendar.TradingCalendar
        The TradingCalendar on which the minute bars are based.
    start_session : datetime
        The first trading session in the data set.
    end_session : datetime
        The last trading session in the data set.
    minutes_per_day : int
        The number of minutes per each period.
    """
    FORMAT_VERSION = 3

    METADATA_FILENAME = 'metadata.json'

    @classmethod
    def metadata_path(cls, rootdir):
        return os.path.join(rootdir, cls.METADATA_FILENAME)

    @classmethod
    def read(cls, rootdir):
        path = cls.metadata_path(rootdir)
        with open(path) as fp:
            raw_data = json.load(fp)

            try:
                version = raw_data['version']
            except KeyError:
                # Version was first written with version 1, assume 0,
                # if version does not match.
                version = 0

            default_ohlc_ratio = raw_data['ohlc_ratio']

            if version >= 1:
                minutes_per_day = raw_data['minutes_per_day']
            else:
                # version 0 always assumed US equities.
                minutes_per_day = US_EQUITIES_MINUTES_PER_DAY

            if version >= 2:
                calendar = get_calendar(raw_data['calendar_name'])
                start_session = pd.Timestamp(
                    raw_data['start_session'], tz='UTC')
                end_session = pd.Timestamp(raw_data['end_session'], tz='UTC')
            else:
                # No calendar info included in older versions, so
                # default to NYSE.
                calendar = get_calendar('NYSE')

                start_session = pd.Timestamp(
                    raw_data['first_trading_day'], tz='UTC')
                end_session = calendar.minute_to_session_label(
                    pd.Timestamp(
                        raw_data['market_closes'][-1], unit='m', tz='UTC')
                )

            if version >= 3:
                ohlc_ratios_per_sid = raw_data['ohlc_ratios_per_sid']
                if ohlc_ratios_per_sid is not None:
                    ohlc_ratios_per_sid = keymap(int, ohlc_ratios_per_sid)
            else:
                ohlc_ratios_per_sid = None

            return cls(
                default_ohlc_ratio,
                ohlc_ratios_per_sid,
                calendar,
                start_session,
                end_session,
                minutes_per_day,
                version=version,
            )

    def __init__(
        self,
        default_ohlc_ratio,
        ohlc_ratios_per_sid,
        calendar,
        start_session,
        end_session,
        minutes_per_day,
        version=FORMAT_VERSION,
    ):
        self.calendar = calendar
        self.start_session = start_session
        self.end_session = end_session
        self.default_ohlc_ratio = default_ohlc_ratio
        self.ohlc_ratios_per_sid = ohlc_ratios_per_sid
        self.minutes_per_day = minutes_per_day
        self.version = version

    def write(self, rootdir):
        """
        Write the metadata to a JSON file in the rootdir.

        Values contained in the metadata are:

        version : int
            The value of FORMAT_VERSION of this class.
        ohlc_ratio : int
            The default ratio by which to multiply the pricing data to
            convert the floats from floats to an integer to fit within
            the np.uint32. If ohlc_ratios_per_sid is None or does not
            contain a mapping for a given sid, this ratio is used.
        ohlc_ratios_per_sid : dict
             A dict mapping each sid in the output to the factor by
             which the pricing data is multiplied so that the float data
             can be stored as an integer.
        minutes_per_day : int
            The number of minutes per each period.
        calendar_name : str
            The name of the TradingCalendar on which the minute bars are
            based.
        start_session : datetime
            'YYYY-MM-DD' formatted representation of the first trading
            session in the data set.
        end_session : datetime
            'YYYY-MM-DD' formatted representation of the last trading
            session in the data set.

        Deprecated, but included for backwards compatibility:

        first_trading_day : string
            'YYYY-MM-DD' formatted representation of the first trading day
             available in the dataset.
        market_opens : list
            List of int64 values representing UTC market opens as
            minutes since epoch.
        market_closes : list
            List of int64 values representing UTC market closes as
            minutes since epoch.
        """

        calendar = self.calendar
        slicer = calendar.schedule.index.slice_indexer(
            self.start_session,
            self.end_session,
        )
        schedule = calendar.schedule[slicer]
        market_opens = schedule.market_open
        market_closes = schedule.market_close

        metadata = {
            'version': self.version,
            'ohlc_ratio': self.default_ohlc_ratio,
            'ohlc_ratios_per_sid': self.ohlc_ratios_per_sid,
            'minutes_per_day': self.minutes_per_day,
            'calendar_name': self.calendar.name,
            'start_session': str(self.start_session.date()),
            'end_session': str(self.end_session.date()),
            # Write these values for backwards compatibility
            'first_trading_day': str(self.start_session.date()),
            'market_opens': (
                market_opens.values.astype('datetime64[m]').
                astype(np.int64).tolist()),
            'market_closes': (
                market_closes.values.astype('datetime64[m]').
                astype(np.int64).tolist()),
        }
        with open(self.metadata_path(rootdir), 'w+') as fp:
            json.dump(metadata, fp)


class BcolzMinuteBarWriter(object):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.

    Parameters
    ----------
    rootdir : string
        Path to the root directory into which to write the metadata and
        bcolz subdirectories.
    calendar : zipline.utils.calendars.trading_calendar.TradingCalendar
        The trading calendar on which to base the minute bars. Used to
        get the market opens used as a starting point for each periodic
        span of minutes in the index, and the market closes that
        correspond with the market opens.
    minutes_per_day : int
        The number of minutes per each period. Defaults to 390, the mode
        of minutes in NYSE trading days.
    start_session : datetime
        The first trading session in the data set.
    end_session : datetime
        The last trading session in the data set.
    default_ohlc_ratio : int, optional
        The default ratio by which to multiply the pricing data to
        convert from floats to integers that fit within np.uint32. If
        ohlc_ratios_per_sid is None or does not contain a mapping for a
        given sid, this ratio is used. Default is OHLC_RATIO (1000).
    ohlc_ratios_per_sid : dict, optional
        A dict mapping each sid in the output to the ratio by which to
        multiply the pricing data to convert the floats from floats to
        an integer to fit within the np.uint32.
    expectedlen : int, optional
        The expected length of the dataset, used when creating the initial
        bcolz ctable.

        If the expectedlen is not used, the chunksize and corresponding
        compression ratios are not ideal.

        Defaults to supporting 15 years of NYSE equity market data.
        see: http://bcolz.blosc.org/opt-tips.html#informing-about-the-length-of-your-carrays # noqa
    write_metadata : bool, optional
        If True, writes the minute bar metadata (on init of the writer).
        If False, no metadata is written (existing metadata is
        retained). Default is True.

    Notes
    -----
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

    See Also
    --------
    zipline.data.minute_bars.BcolzMinuteBarReader
    """
    COL_NAMES = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self,
                 rootdir,
                 calendar,
                 start_session,
                 end_session,
                 minutes_per_day,
                 default_ohlc_ratio=OHLC_RATIO,
                 ohlc_ratios_per_sid=None,
                 expectedlen=DEFAULT_EXPECTEDLEN,
                 write_metadata=True):

        self._rootdir = rootdir
        self._start_session = start_session
        self._end_session = end_session
        self._calendar = calendar
        slicer = (
            calendar.schedule.index.slice_indexer(start_session, end_session))
        self._schedule = calendar.schedule[slicer]
        self._session_labels = self._schedule.index
        self._minutes_per_day = minutes_per_day
        self._expectedlen = expectedlen
        self._default_ohlc_ratio = default_ohlc_ratio
        self._ohlc_ratios_per_sid = ohlc_ratios_per_sid

        self._minute_index = _calc_minute_index(
            self._schedule.market_open, self._minutes_per_day)

        if write_metadata:
            metadata = BcolzMinuteBarMetadata(
                self._default_ohlc_ratio,
                self._ohlc_ratios_per_sid,
                self._calendar,
                self._start_session,
                self._end_session,
                self._minutes_per_day,
            )
            metadata.write(self._rootdir)

    @classmethod
    def open(cls, rootdir, end_session=None):
        """
        Open an existing ``rootdir`` for writing.

        Parameters
        ----------
        end_session : Timestamp (optional)
            When appending, the intended new ``end_session``.
        """
        metadata = BcolzMinuteBarMetadata.read(rootdir)
        return BcolzMinuteBarWriter(
            rootdir,
            metadata.calendar,
            metadata.start_session,
            end_session if end_session is not None else metadata.end_session,
            metadata.minutes_per_day,
            metadata.default_ohlc_ratio,
            metadata.ohlc_ratios_per_sid,
            write_metadata=end_session is not None
        )

    @property
    def first_trading_day(self):
        return self._start_session

    def ohlc_ratio_for_sid(self, sid):
        if self._ohlc_ratios_per_sid is not None:
            try:
                return self._ohlc_ratios_per_sid[sid]
            except KeyError:
                pass

        # If no ohlc_ratios_per_sid dict is passed, or if the specified
        # sid is not in the dict, fallback to the general ohlc_ratio.
        return self._default_ohlc_ratio

    def sidpath(self, sid):
        """
        Parameters
        ----------
        sid : int
            Asset identifier.

        Returns
        -------
        out : string
            Full path to the bcolz rootdir for the given sid.
        """
        sid_subdir = _sid_subdir_path(sid)
        return join(self._rootdir, sid_subdir)

    def last_date_in_output_for_sid(self, sid):
        """
        Parameters
        ----------
        sid : int
            Asset identifier.

        Returns
        -------
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
        # use integer division so that the result is an int
        # for pandas index later https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/base.py#L247 # noqa
        num_days = data['shape'][0] // self._minutes_per_day
        if num_days == 0:
            # empty container
            return pd.NaT
        return self._session_labels[num_days - 1]

    def _init_ctable(self, path):
        """
        Create empty ctable for given path.

        Parameters
        ----------
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
        # Compute the number of minutes to be filled, accounting for the
        # possibility of a partial day's worth of minutes existing for
        # the previous day.
        minute_offset = len(table) % self._minutes_per_day
        num_to_prepend = numdays * self._minutes_per_day - minute_offset

        prepend_array = np.zeros(num_to_prepend, np.uint32)
        # Fill all OHLCV with zeros.
        table.append([prepend_array] * 5)
        table.flush()

    def pad(self, sid, date):
        """
        Fill sid container with empty data through the specified date.

        If the last recorded trade is not at the close, then that day will be
        padded with zeros until its close. Any day after that (up to and
        including the specified date) will be padded with `minute_per_day`
        worth of zeros

        Parameters
        ----------
        sid : int
            The asset identifier for the data being written.
        date : datetime-like
            The date used to calculate how many slots to be pad.
            The padding is done through the date, i.e. after the padding is
            done the `last_date_in_output_for_sid` will be equal to `date`
        """
        table = self._ensure_ctable(sid)

        last_date = self.last_date_in_output_for_sid(sid)

        tds = self._session_labels

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

    def set_sid_attrs(self, sid, **kwargs):
        """Write all the supplied kwargs as attributes of the sid's file.
        """
        table = self._ensure_ctable(sid)
        for k, v in kwargs.items():
            table.attrs[k] = v

    def write(self, data, show_progress=False, invalid_data_behavior='warn'):
        """Write a stream of minute data.

        Parameters
        ----------
        data : iterable[(int, pd.DataFrame)]
            The data to write. Each element should be a tuple of sid, data
            where data has the following format:
              columns : ('open', 'high', 'low', 'close', 'volume')
                  open : float64
                  high : float64
                  low  : float64
                  close : float64
                  volume : float64|int64
              index : DatetimeIndex of market minutes.
            A given sid may appear more than once in ``data``; however,
            the dates must be strictly increasing.
        show_progress : bool, optional
            Whether or not to show a progress bar while writing.
        """
        ctx = maybe_show_progress(
            data,
            show_progress=show_progress,
            item_show_func=lambda e: e if e is None else str(e[0]),
            label="Merging minute equity files:",
        )
        write_sid = self.write_sid
        with ctx as it:
            for e in it:
                write_sid(*e, invalid_data_behavior=invalid_data_behavior)

    def write_sid(self, sid, df, invalid_data_behavior='warn'):
        """
        Write the OHLCV data for the given sid.
        If there is no bcolz ctable yet created for the sid, create it.
        If the length of the bcolz ctable is not exactly to the date before
        the first day provided, fill the ctable with 0s up to that date.

        Parameters
        ----------
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
        self._write_cols(sid, dts, cols, invalid_data_behavior)

    def write_cols(self, sid, dts, cols, invalid_data_behavior='warn'):
        """
        Write the OHLCV data for the given sid.
        If there is no bcolz ctable yet created for the sid, create it.
        If the length of the bcolz ctable is not exactly to the date before
        the first day provided, fill the ctable with 0s up to that date.

        Parameters
        ----------
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
        self._write_cols(sid, dts, cols, invalid_data_behavior)

    def _write_cols(self, sid, dts, cols, invalid_data_behavior):
        """
        Internal method for `write_cols` and `write`.

        Parameters
        ----------
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

        tds = self._session_labels
        input_first_day = self._calendar.minute_to_session_label(
            pd.Timestamp(dts[0]), direction='previous')

        last_date = self.last_date_in_output_for_sid(sid)

        day_before_input = input_first_day - tds.freq

        self.pad(sid, day_before_input)
        table = self._ensure_ctable(sid)

        # Get the number of minutes already recorded in this sid's ctable
        num_rec_mins = table.size

        all_minutes = self._minute_index
        # Get the latest minute we wish to write to the ctable
        last_minute_to_write = pd.Timestamp(dts[-1], tz='UTC')

        # In the event that we've already written some minutely data to the
        # ctable, guard against overwriting that data.
        if num_rec_mins > 0:
            last_recorded_minute = all_minutes[num_rec_mins - 1]
            if last_minute_to_write <= last_recorded_minute:
                raise BcolzMinuteOverlappingData(dedent("""
                Data with last_date={0} already includes input start={1} for
                sid={2}""".strip()).format(last_date, input_first_day, sid))

        latest_min_count = all_minutes.get_loc(last_minute_to_write)

        # Get all the minutes we wish to write (all market minutes after the
        # latest currently written, up to and including last_minute_to_write)
        all_minutes_in_window = all_minutes[num_rec_mins:latest_min_count + 1]

        minutes_count = all_minutes_in_window.size

        open_col = np.zeros(minutes_count, dtype=np.uint32)
        high_col = np.zeros(minutes_count, dtype=np.uint32)
        low_col = np.zeros(minutes_count, dtype=np.uint32)
        close_col = np.zeros(minutes_count, dtype=np.uint32)
        vol_col = np.zeros(minutes_count, dtype=np.uint32)

        dt_ixs = np.searchsorted(all_minutes_in_window.values,
                                 dts.astype('datetime64[ns]'))

        ohlc_ratio = self.ohlc_ratio_for_sid(sid)

        (
            open_col[dt_ixs],
            high_col[dt_ixs],
            low_col[dt_ixs],
            close_col[dt_ixs],
            vol_col[dt_ixs],
        ) = convert_cols(cols, ohlc_ratio, sid, invalid_data_behavior)

        table.append([
            open_col,
            high_col,
            low_col,
            close_col,
            vol_col
        ])
        table.flush()

    def data_len_for_day(self, day):
        """
        Return the number of data points up to and including the
        provided day.
        """
        day_ix = self._session_labels.get_loc(day)
        # Add one to the 0-indexed day_ix to get the number of days.
        num_days = day_ix + 1
        return num_days * self._minutes_per_day

    def truncate(self, date):
        """Truncate data beyond this date in all ctables."""
        truncate_slice_end = self.data_len_for_day(date)

        glob_path = os.path.join(self._rootdir, "*", "*", "*.bcolz")
        sid_paths = sorted(glob(glob_path))

        for sid_path in sid_paths:
            file_name = os.path.basename(sid_path)

            try:
                table = bcolz.open(rootdir=sid_path)
            except IOError:
                continue
            if table.len <= truncate_slice_end:
                logger.info("{0} not past truncate date={1}.", file_name, date)
                continue

            logger.info(
                "Truncating {0} at end_date={1}", file_name, date.date()
            )

            table.resize(truncate_slice_end)

        # Update end session in metadata.
        metadata = BcolzMinuteBarMetadata.read(self._rootdir)
        metadata.end_session = date
        metadata.write(self._rootdir)


class BcolzMinuteBarReader(MinuteBarReader):
    """
    Reader for data written by BcolzMinuteBarWriter

    Parameters
    ----------
    rootdir : string
        The root directory containing the metadata and asset bcolz
        directories.

    See Also
    --------
    zipline.data.minute_bars.BcolzMinuteBarWriter
    """
    FIELDS = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self, rootdir, sid_cache_size=1000):
        self._rootdir = rootdir

        metadata = self._get_metadata()

        self._start_session = metadata.start_session
        self._end_session = metadata.end_session

        self.calendar = metadata.calendar
        slicer = self.calendar.schedule.index.slice_indexer(
            self._start_session,
            self._end_session,
        )
        self._schedule = self.calendar.schedule[slicer]
        self._market_opens = self._schedule.market_open
        self._market_open_values = self._market_opens.values.\
            astype('datetime64[m]').astype(np.int64)
        self._market_closes = self._schedule.market_close
        self._market_close_values = self._market_closes.values.\
            astype('datetime64[m]').astype(np.int64)

        self._default_ohlc_inverse = 1.0 / metadata.default_ohlc_ratio
        ohlc_ratios = metadata.ohlc_ratios_per_sid
        if ohlc_ratios:
            self._ohlc_inverses_per_sid = (
                valmap(lambda x: 1.0 / x, ohlc_ratios))
        else:
            self._ohlc_inverses_per_sid = None

        self._minutes_per_day = metadata.minutes_per_day

        self._carrays = {
            field: LRU(sid_cache_size)
            for field in self.FIELDS
        }

        self._last_get_value_dt_position = None
        self._last_get_value_dt_value = None

        # This is to avoid any bad data or other performance-killing situation
        # where there a consecutive streak of 0 (no volume) starting at an
        # asset's start date.
        # if asset 1 started on 2015-01-03 but its first trade is 2015-01-06
        # 10:31 AM US/Eastern, this dict would store {1: 23675971},
        # which is the minute epoch of that date.
        self._known_zero_volume_dict = {}

    def _get_metadata(self):
        return BcolzMinuteBarMetadata.read(self._rootdir)

    @property
    def trading_calendar(self):
        return self.calendar

    @lazyval
    def last_available_dt(self):
        _, close = self.calendar.open_and_close_for_session(self._end_session)
        return close

    @property
    def first_trading_day(self):
        return self._start_session

    def _ohlc_ratio_inverse_for_sid(self, sid):
        if self._ohlc_inverses_per_sid is not None:
            try:
                return self._ohlc_inverses_per_sid[sid]
            except KeyError:
                pass

        # If we can not get a sid-specific OHLC inverse for this sid,
        # fallback to the default.
        return self._default_ohlc_inverse

    def _minutes_to_exclude(self):
        """
        Calculate the minutes which should be excluded when a window
        occurs on days which had an early close, i.e. days where the close
        based on the regular period of minutes per day and the market close
        do not match.

        Returns
        -------
        List of DatetimeIndex representing the minutes to exclude because
        of early closes.
        """
        market_opens = self._market_opens.values.astype('datetime64[m]')
        market_closes = self._market_closes.values.astype('datetime64[m]')
        minutes_per_day = (market_closes - market_opens).astype(np.int64)
        early_indices = np.where(
            minutes_per_day != self._minutes_per_day - 1)[0]
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
                self._minutes_per_day
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
            try:
                carray = self._carrays[field][sid] = bcolz.carray(
                    rootdir=self._get_carray_path(sid, field),
                    mode='r',
                )
            except IOError:
                raise NoDataForSid('No minute data for sid {}.'.format(sid))

        return carray

    def table_len(self, sid):
        """Returns the length of the underlying table for this sid."""
        return len(self._open_minute_file('close', sid))

    def get_sid_attr(self, sid, name):
        sid_subdir = _sid_subdir_path(sid)
        sid_path = os.path.join(self._rootdir, sid_subdir)
        attrs = bcolz.attrs.attrs(sid_path, 'r')
        try:
            return attrs[name]
        except KeyError:
            return None

    def get_value(self, sid, dt, field):
        """
        Retrieve the pricing info for the given sid, dt, and field.

        Parameters
        ----------
        sid : int
            Asset identifier.
        dt : datetime-like
            The datetime at which the trade occurred.
        field : string
            The type of pricing data to retrieve.
            ('open', 'high', 'low', 'close', 'volume')

        Returns
        -------
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
            try:
                minute_pos = self._find_position_of_minute(dt)
            except ValueError:
                raise NoDataOnDate()

            self._last_get_value_dt_value = dt.value
            self._last_get_value_dt_position = minute_pos

        try:
            value = self._open_minute_file(field, sid)[minute_pos]
        except IndexError:
            value = 0
        if value == 0:
            if field == 'volume':
                return 0
            else:
                return np.nan

        if field != 'volume':
            value *= self._ohlc_ratio_inverse_for_sid(sid)
        return value

    def get_last_traded_dt(self, asset, dt):
        minute_pos = self._find_last_traded_position(asset, dt)
        if minute_pos == -1:
            return pd.NaT
        return self._pos_to_minute(minute_pos)

    def _find_last_traded_position(self, asset, dt):
        volumes = self._open_minute_file('volume', asset)
        start_date_minute = asset.start_date.value / NANOS_IN_MINUTE
        dt_minute = dt.value / NANOS_IN_MINUTE

        try:
            # if we know of a dt before which this asset has no volume,
            # don't look before that dt
            earliest_dt_to_search = self._known_zero_volume_dict[asset.sid]
        except KeyError:
            earliest_dt_to_search = start_date_minute

        if dt_minute < earliest_dt_to_search:
            return -1

        pos = find_last_traded_position_internal(
            self._market_open_values,
            self._market_close_values,
            dt_minute,
            earliest_dt_to_search,
            volumes,
            self._minutes_per_day,
        )

        if pos == -1:
            # if we didn't find any volume before this dt, save it to avoid
            # work in the future.
            try:
                self._known_zero_volume_dict[asset.sid] = max(
                    dt_minute,
                    self._known_zero_volume_dict[asset.sid]
                )
            except KeyError:
                self._known_zero_volume_dict[asset.sid] = dt_minute

        return pos

    def _pos_to_minute(self, pos):
        minute_epoch = minute_value(
            self._market_open_values,
            pos,
            self._minutes_per_day
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
            self._minutes_per_day,
            False,
        )

    def load_raw_arrays(self, fields, start_dt, end_dt, sids):
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
            (minutes in range, sids) with a dtype of float64, containing the
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

        shape = num_minutes, len(sids)

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
                # first slice down to len(where) because we might not have
                # written data for all the minutes requested
                if field != 'volume':
                    out[:len(where), i][where] = (
                        values[where] * self._ohlc_ratio_inverse_for_sid(sid))
                else:
                    out[:len(where), i][where] = values[where]

            results.append(out)
        return results


class MinuteBarUpdateReader(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for minute update readers.
    """

    @abstractmethod
    def read(self, dts, sids):
        """
        Read and return pricing update data.

        Parameters
        ----------
        dts : DatetimeIndex
            The minutes for which to read the pricing updates.
        sids : iter[int]
            The sids for which to read the pricing updates.

        Returns
        -------
        data : iter[(int, DataFrame)]
            Returns an iterable of ``sid`` to the corresponding OHLCV data.
        """
        raise NotImplementedError()


class H5MinuteBarUpdateWriter(object):
    """
    Writer for files containing minute bar updates for consumption by a writer
    for a ``MinuteBarReader`` format.

    Parameters
    ----------
    path : str
        The destination path.
    complevel : int, optional
        The HDF5 complevel, defaults to ``5``.
    complib : str, optional
        The HDF5 complib, defaults to ``zlib``.
    """

    FORMAT_VERSION = 0

    _COMPLEVEL = 5
    _COMPLIB = 'zlib'

    def __init__(self, path, complevel=None, complib=None):
        self._complevel = complevel if complevel \
            is not None else self._COMPLEVEL
        self._complib = complib if complib \
            is not None else self._COMPLIB
        self._path = path

    def write(self, frames):
        """
        Write the frames to the target HDF5 file, using the format used by
        ``pd.Panel.to_hdf``

        Parameters
        ----------
        frames : iter[(int, DataFrame)] or dict[int -> DataFrame]
            An iterable or other mapping of sid to the corresponding OHLCV
            pricing data.
        """
        with HDFStore(self._path, 'w',
                      complevel=self._complevel, complib=self._complib) \
                as store:
            panel = pd.Panel.from_dict(dict(frames))
            panel.to_hdf(store, 'updates')
        with tables.open_file(self._path, mode='r+') as h5file:
            h5file.set_node_attr('/', 'version', 0)


class H5MinuteBarUpdateReader(MinuteBarUpdateReader):
    """
    Reader for minute bar updates stored in HDF5 files.

    Parameters
    ----------
    path : str
        The path of the HDF5 file from which to source data.
    """
    def __init__(self, path):
        self._panel = pd.read_hdf(path)

    def read(self, dts, sids):
        panel = self._panel[sids, dts, :]
        return panel.iteritems()
