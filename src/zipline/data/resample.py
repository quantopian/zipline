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
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from zipline.data._resample import (
    _minute_to_session_open,
    _minute_to_session_high,
    _minute_to_session_low,
    _minute_to_session_close,
    _minute_to_session_volume,
)
from zipline.data.bar_reader import NoDataOnDate
from zipline.data.bcolz_minute_bars import MinuteBarReader
from zipline.data.session_bars import SessionBarReader
from zipline.utils.memoize import lazyval
from zipline.utils.math_utils import nanmax, nanmin

_MINUTE_TO_SESSION_OHCLV_HOW = OrderedDict(
    (
        ("open", "first"),
        ("high", "max"),
        ("low", "min"),
        ("close", "last"),
        ("volume", "sum"),
    )
)


def minute_frame_to_session_frame(minute_frame, calendar):
    """Resample a DataFrame with minute data into the frame expected by a
    BcolzDailyBarWriter.

    Parameters
    ----------
    minute_frame : pd.DataFrame
        A DataFrame with the columns `open`, `high`, `low`, `close`, `volume`,
        and `dt` (minute dts)
    calendar : trading_calendars.trading_calendar.TradingCalendar
        A TradingCalendar on which session labels to resample from minute
        to session.

    Return
    ------
    session_frame : pd.DataFrame
        A DataFrame with the columns `open`, `high`, `low`, `close`, `volume`,
        and `day` (datetime-like).
    """
    how = OrderedDict(
        (c, _MINUTE_TO_SESSION_OHCLV_HOW[c]) for c in minute_frame.columns
    )
    labels = calendar.minutes_to_sessions(minute_frame.index)
    return minute_frame.groupby(labels).agg(how)


def minute_to_session(column, close_locs, data, out):
    """Resample an array with minute data into an array with session data.

    This function assumes that the minute data is the exact length of all
    minutes in the sessions in the output.

    Parameters
    ----------
    column : str
        The `open`, `high`, `low`, `close`, or `volume` column.
    close_locs : array[intp]
        The locations in `data` which are the market close minutes.
    data : array[float64|uint32]
        The minute data to be sampled into session data.
        The first value should align with the market open of the first session,
        containing values for all minutes for all sessions. With the last value
        being the market close of the last session.
    out : array[float64|uint32]
        The output array into which to write the sampled sessions.
    """
    if column == "open":
        _minute_to_session_open(close_locs, data, out)
    elif column == "high":
        _minute_to_session_high(close_locs, data, out)
    elif column == "low":
        _minute_to_session_low(close_locs, data, out)
    elif column == "close":
        _minute_to_session_close(close_locs, data, out)
    elif column == "volume":
        _minute_to_session_volume(close_locs, data, out)
    return out


class DailyHistoryAggregator:
    """Converts minute pricing data into a daily summary, to be used for the
    last slot in a call to history with a frequency of `1d`.

    This summary is the same as a daily bar rollup of minute data, with the
    distinction that the summary is truncated to the `dt` requested.
    i.e. the aggregation slides forward during a the course of simulation day.

    Provides aggregation for `open`, `high`, `low`, `close`, and `volume`.
    The aggregation rules for each price type is documented in their respective

    """

    def __init__(self, market_opens, minute_reader, trading_calendar):
        self._market_opens = market_opens
        self._minute_reader = minute_reader
        self._trading_calendar = trading_calendar

        # The caches are structured as (date, market_open, entries), where
        # entries is a dict of asset -> (last_visited_dt, value)
        #
        # Whenever an aggregation method determines the current value,
        # the entry for the respective asset should be overwritten with a new
        # entry for the current dt.value (int) and aggregation value.
        #
        # When the requested dt's date is different from date the cache is
        # flushed, so that the cache entries do not grow unbounded.
        #
        # Example cache:
        # cache = (date(2016, 3, 17),
        #          pd.Timestamp('2016-03-17 13:31', tz='UTC'),
        #          {
        #              1: (1458221460000000000, np.nan),
        #              2: (1458221460000000000, 42.0),
        #         })
        self._caches = {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
        }

        # The int value is used for deltas to avoid extra computation from
        # creating new Timestamps.
        self._one_min = pd.Timedelta("1 min").value

    def _prelude(self, dt, field):
        session = self._trading_calendar.minute_to_session(dt)
        dt_value = dt.value
        cache = self._caches[field]
        if cache is None or cache[0] != session:
            market_open = self._market_opens.loc[session]
            cache = self._caches[field] = (session, market_open, {})

        _, market_open, entries = cache
        if dt != market_open:
            prev_dt = dt_value - self._one_min
        else:
            prev_dt = None
        return market_open, prev_dt, dt_value, entries

    def opens(self, assets, dt):
        """The open field's aggregation returns the first value that occurs
        for the day, if there has been no data on or before the `dt` the open
        is `nan`.

        Once the first non-nan open is seen, that value remains constant per
        asset for the remainder of the day.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, "open")

        opens = []
        session_label = self._trading_calendar.minute_to_session(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                opens.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, "open")
                entries[asset] = (dt_value, val)
                opens.append(val)
                continue
            else:
                try:
                    last_visited_dt, first_open = entries[asset]
                    if last_visited_dt == dt_value:
                        opens.append(first_open)
                        continue
                    elif not pd.isnull(first_open):
                        opens.append(first_open)
                        entries[asset] = (dt_value, first_open)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz="UTC"
                        )
                        window = self._minute_reader.load_raw_arrays(
                            ["open"],
                            after_last,
                            dt,
                            [asset],
                        )[0]
                        nonnan = window[~pd.isnull(window)]
                        if len(nonnan):
                            val = nonnan[0]
                        else:
                            val = np.nan
                        entries[asset] = (dt_value, val)
                        opens.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ["open"],
                        market_open,
                        dt,
                        [asset],
                    )[0]
                    nonnan = window[~pd.isnull(window)]
                    if len(nonnan):
                        val = nonnan[0]
                    else:
                        val = np.nan
                    entries[asset] = (dt_value, val)
                    opens.append(val)
                    continue
        return np.array(opens)

    def highs(self, assets, dt):
        """The high field's aggregation returns the largest high seen between
        the market open and the current dt.
        If there has been no data on or before the `dt` the high is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, "high")

        highs = []
        session_label = self._trading_calendar.minute_to_session(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                highs.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, "high")
                entries[asset] = (dt_value, val)
                highs.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_max = entries[asset]
                    if last_visited_dt == dt_value:
                        highs.append(last_max)
                        continue
                    elif last_visited_dt == prev_dt:
                        curr_val = self._minute_reader.get_value(asset, dt, "high")
                        if pd.isnull(curr_val):
                            val = last_max
                        elif pd.isnull(last_max):
                            val = curr_val
                        else:
                            val = max(last_max, curr_val)
                        entries[asset] = (dt_value, val)
                        highs.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz="UTC"
                        )
                        window = self._minute_reader.load_raw_arrays(
                            ["high"],
                            after_last,
                            dt,
                            [asset],
                        )[0].T
                        val = nanmax(np.append(window, last_max))
                        entries[asset] = (dt_value, val)
                        highs.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ["high"],
                        market_open,
                        dt,
                        [asset],
                    )[0].T
                    val = nanmax(window)
                    entries[asset] = (dt_value, val)
                    highs.append(val)
                    continue
        return np.array(highs)

    def lows(self, assets, dt):
        """The low field's aggregation returns the smallest low seen between
        the market open and the current dt.
        If there has been no data on or before the `dt` the low is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, "low")

        lows = []
        session_label = self._trading_calendar.minute_to_session(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                lows.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, "low")
                entries[asset] = (dt_value, val)
                lows.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_min = entries[asset]
                    if last_visited_dt == dt_value:
                        lows.append(last_min)
                        continue
                    elif last_visited_dt == prev_dt:
                        curr_val = self._minute_reader.get_value(asset, dt, "low")
                        val = nanmin([last_min, curr_val])
                        entries[asset] = (dt_value, val)
                        lows.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz="UTC"
                        )
                        window = self._minute_reader.load_raw_arrays(
                            ["low"],
                            after_last,
                            dt,
                            [asset],
                        )[0].T
                        val = nanmin(np.append(window, last_min))
                        entries[asset] = (dt_value, val)
                        lows.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ["low"],
                        market_open,
                        dt,
                        [asset],
                    )[0].T
                    val = nanmin(window)
                    entries[asset] = (dt_value, val)
                    lows.append(val)
                    continue
        return np.array(lows)

    def closes(self, assets, dt):
        """The close field's aggregation returns the latest close at the given
        dt.
        If the close for the given dt is `nan`, the most recent non-nan
        `close` is used.
        If there has been no data on or before the `dt` the close is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, "close")

        closes = []
        session_label = self._trading_calendar.minute_to_session(dt)

        def _get_filled_close(asset):
            """
            Returns the most recent non-nan close for the asset in this
            session. If there has been no data in this session on or before the
            `dt`, returns `nan`
            """
            window = self._minute_reader.load_raw_arrays(
                ["close"],
                market_open,
                dt,
                [asset],
            )[0]
            try:
                return window[~np.isnan(window)][-1]
            except IndexError:
                return np.NaN

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                closes.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, "close")
                entries[asset] = (dt_value, val)
                closes.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_close = entries[asset]
                    if last_visited_dt == dt_value:
                        closes.append(last_close)
                        continue
                    elif last_visited_dt == prev_dt:
                        val = self._minute_reader.get_value(asset, dt, "close")
                        if pd.isnull(val):
                            val = last_close
                        entries[asset] = (dt_value, val)
                        closes.append(val)
                        continue
                    else:
                        val = self._minute_reader.get_value(asset, dt, "close")
                        if pd.isnull(val):
                            val = _get_filled_close(asset)
                        entries[asset] = (dt_value, val)
                        closes.append(val)
                        continue
                except KeyError:
                    val = self._minute_reader.get_value(asset, dt, "close")
                    if pd.isnull(val):
                        val = _get_filled_close(asset)
                    entries[asset] = (dt_value, val)
                    closes.append(val)
                    continue
        return np.array(closes)

    def volumes(self, assets, dt):
        """The volume field's aggregation returns the sum of all volumes
        between the market open and the `dt`
        If there has been no data on or before the `dt` the volume is 0.

        Returns
        -------
        np.array with dtype=int64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, "volume")

        volumes = []
        session_label = self._trading_calendar.minute_to_session(dt)

        for asset in assets:
            if not asset.is_alive_for_session(session_label):
                volumes.append(0)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, "volume")
                entries[asset] = (dt_value, val)
                volumes.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_total = entries[asset]
                    if last_visited_dt == dt_value:
                        volumes.append(last_total)
                        continue
                    elif last_visited_dt == prev_dt:
                        val = self._minute_reader.get_value(asset, dt, "volume")
                        val += last_total
                        entries[asset] = (dt_value, val)
                        volumes.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz="UTC"
                        )
                        window = self._minute_reader.load_raw_arrays(
                            ["volume"],
                            after_last,
                            dt,
                            [asset],
                        )[0]
                        val = np.nansum(window) + last_total
                        entries[asset] = (dt_value, val)
                        volumes.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ["volume"],
                        market_open,
                        dt,
                        [asset],
                    )[0]
                    val = np.nansum(window)
                    entries[asset] = (dt_value, val)
                    volumes.append(val)
                    continue
        return np.array(volumes)


class MinuteResampleSessionBarReader(SessionBarReader):
    def __init__(self, calendar, minute_bar_reader):
        self._calendar = calendar
        self._minute_bar_reader = minute_bar_reader

    def _get_resampled(self, columns, start_session, end_session, assets):
        range_open = self._calendar.session_first_minute(start_session)
        range_close = self._calendar.session_close(end_session)

        minute_data = self._minute_bar_reader.load_raw_arrays(
            columns,
            range_open,
            range_close,
            assets,
        )

        # Get the index of the close minute for each session in the range.
        # If the range contains only one session, the only close in the range
        # is the last minute in the data. Otherwise, we need to get all the
        # session closes and find their indices in the range of minutes.
        if start_session == end_session:
            close_ilocs = np.array([len(minute_data[0]) - 1], dtype=np.int64)
        else:
            minutes = self._calendar.minutes_in_range(
                range_open,
                range_close,
            )
            session_closes = self._calendar.closes[start_session:end_session]
            close_ilocs = minutes.searchsorted(pd.DatetimeIndex(session_closes))

        results = []
        shape = (len(close_ilocs), len(assets))

        for col in columns:
            if col != "volume":
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.uint32)
            results.append(out)

        for i in range(len(assets)):
            for j, column in enumerate(columns):
                data = minute_data[j][:, i]
                minute_to_session(column, close_ilocs, data, results[j][:, i])

        return results

    @property
    def trading_calendar(self):
        return self._calendar

    def load_raw_arrays(self, columns, start_dt, end_dt, sids):
        return self._get_resampled(columns, start_dt, end_dt, sids)

    def get_value(self, sid, session, colname):
        # WARNING: This will need caching or other optimization if used in a
        # tight loop.
        # This was developed to complete interface, but has not been tuned
        # for real world use.
        return self._get_resampled([colname], session, session, [sid])[0][0][0]

    @lazyval
    def sessions(self):
        cal = self._calendar
        first = self._minute_bar_reader.first_trading_day
        last = cal.minute_to_session(self._minute_bar_reader.last_available_dt)
        return cal.sessions_in_range(first, last)

    @lazyval
    def last_available_dt(self):
        return self.trading_calendar.minute_to_session(
            self._minute_bar_reader.last_available_dt
        )

    @property
    def first_trading_day(self):
        return self._minute_bar_reader.first_trading_day

    def get_last_traded_dt(self, asset, dt):
        last_dt = self._minute_bar_reader.get_last_traded_dt(asset, dt)
        if pd.isnull(last_dt):
            # todo: this doesn't seem right
            return self.trading_calendar.first_session
        return self.trading_calendar.minute_to_session(last_dt)


class ReindexBarReader(ABC):
    """A base class for readers which reindexes results, filling in the additional
    indices with empty data.

    Used to align the reading assets which trade on different calendars.

    Currently only supports a ``trading_calendar`` which is a superset of the
    ``reader``'s calendar.

    Parameters
    ----------

    - trading_calendar : zipline.utils.trading_calendar.TradingCalendar
       The calendar to use when indexing results from the reader.
    - reader : MinuteBarReader|SessionBarReader
       The reader which has a calendar that is a subset of the desired
       ``trading_calendar``.
    - first_trading_session : pd.Timestamp
       The first trading session the reader should provide. Must be specified,
       since the ``reader``'s first session may not exactly align with the
       desired calendar. Specifically, in the case where the first session
       on the target calendar is a holiday on the ``reader``'s calendar.
    - last_trading_session : pd.Timestamp
       The last trading session the reader should provide. Must be specified,
       since the ``reader``'s last session may not exactly align with the
       desired calendar. Specifically, in the case where the last session
       on the target calendar is a holiday on the ``reader``'s calendar.
    """

    def __init__(
        self,
        trading_calendar,
        reader,
        first_trading_session,
        last_trading_session,
    ):
        self._trading_calendar = trading_calendar
        self._reader = reader
        self._first_trading_session = first_trading_session
        self._last_trading_session = last_trading_session

    @property
    def last_available_dt(self):
        return self._reader.last_available_dt

    def get_last_traded_dt(self, sid, dt):
        return self._reader.get_last_traded_dt(sid, dt)

    @property
    def first_trading_day(self):
        return self._reader.first_trading_day

    def get_value(self, sid, dt, field):
        # Give an empty result if no data is present.
        try:
            return self._reader.get_value(sid, dt, field)
        except NoDataOnDate:
            if field == "volume":
                return 0
            else:
                return np.nan

    @abstractmethod
    def _outer_dts(self, start_dt, end_dt):
        raise NotImplementedError

    @abstractmethod
    def _inner_dts(self, start_dt, end_dt):
        raise NotImplementedError

    @property
    def trading_calendar(self):
        return self._trading_calendar

    @lazyval
    def sessions(self):
        return self.trading_calendar.sessions_in_range(
            self._first_trading_session, self._last_trading_session
        )

    def load_raw_arrays(self, fields, start_dt, end_dt, sids):
        outer_dts = self._outer_dts(start_dt, end_dt)
        inner_dts = self._inner_dts(start_dt, end_dt)

        indices = outer_dts.searchsorted(inner_dts)

        shape = len(outer_dts), len(sids)

        outer_results = []

        if len(inner_dts) > 0:
            inner_results = self._reader.load_raw_arrays(
                fields, inner_dts[0], inner_dts[-1], sids
            )
        else:
            inner_results = None

        for i, field in enumerate(fields):
            if field != "volume":
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.uint32)

            if inner_results is not None:
                out[indices] = inner_results[i]

            outer_results.append(out)

        return outer_results


class ReindexMinuteBarReader(ReindexBarReader, MinuteBarReader):
    """See: ``ReindexBarReader``"""

    def _outer_dts(self, start_dt, end_dt):
        return self._trading_calendar.minutes_in_range(start_dt, end_dt)

    def _inner_dts(self, start_dt, end_dt):
        return self._reader.calendar.minutes_in_range(start_dt, end_dt)


class ReindexSessionBarReader(ReindexBarReader, SessionBarReader):
    """See: ``ReindexBarReader``"""

    def _outer_dts(self, start_dt, end_dt):
        return self.trading_calendar.sessions_in_range(start_dt, end_dt)

    def _inner_dts(self, start_dt, end_dt):
        return self._reader.trading_calendar.sessions_in_range(start_dt, end_dt)
