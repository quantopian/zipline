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
from abc import (
    ABCMeta,
    abstractproperty,
    abstractmethod,
)
from six import with_metaclass

import pandas as pd
from pandas import (
    DataFrame,
    date_range,
    DateOffset,
    DatetimeIndex,
    Timedelta,
)
from pandas.tseries.offsets import CustomBusinessDay

start_default = pd.Timestamp('1990-01-01', tz='UTC')
end_base = pd.Timestamp('today', tz='UTC')
# Give an aggressive buffer for logic that needs to use the next trading
# day or minute.
end_default = end_base + pd.Timedelta(days=365)


def delta_from_time(t):
    """
    Convert a datetime.time into a timedelta.
    """
    return Timedelta(
        hours=t.hour,
        minutes=t.minute,
        seconds=t.second,
    )


def days_at_time(days, t, tz, day_offset=0):
    """
    Shift an index of days to time t, interpreted in tz.

    Overwrites any existing tz info on the input.

    Parameters
    ----------
    days : DatetimeIndex
        The "base" time which we want to change.
    t : datetime.time
        The time we want to offset @days by
    tz : pytz.timezone
        The timezone which these times represent
    day_offset : int
        The number of days we want to offset @days by
    """
    days = DatetimeIndex(days).tz_localize(None).tz_localize(tz)
    days_offset = days + DateOffset(day_offset)
    return days_offset.shift(
        1, freq=DateOffset(hour=t.hour, minute=t.minute, second=t.second)
    ).tz_convert('UTC')


def holidays_at_time(calendar, start, end, time, tz):
    return days_at_time(
        calendar.holidays(
            # Workaround for https://github.com/pydata/pandas/issues/9825.
            start.tz_localize(None),
            end.tz_localize(None),
        ),
        time,
        tz=tz,
    )


def _overwrite_special_dates(midnight_utcs,
                             opens_or_closes,
                             special_opens_or_closes):
    """
    Overwrite dates in open_or_closes with corresponding dates in
    special_opens_or_closes, using midnight_utcs for alignment.
    """
    # Short circuit when nothing to apply.
    if not len(special_opens_or_closes):
        return

    len_m, len_oc = len(midnight_utcs), len(opens_or_closes)
    if len_m != len_oc:
        raise ValueError(
            "Found misaligned dates while building calendar.\n"
            "Expected midnight_utcs to be the same length as open_or_closes,\n"
            "but len(midnight_utcs)=%d, len(open_or_closes)=%d" % len_m, len_oc
        )

    # Find the array indices corresponding to each special date.
    indexer = midnight_utcs.get_indexer(special_opens_or_closes.normalize())

    # -1 indicates that no corresponding entry was found.  If any -1s are
    # present, then we have special dates that doesn't correspond to any
    # trading day.
    if -1 in indexer:
        bad_dates = list(special_opens_or_closes[indexer == -1])
        raise ValueError("Special dates %s are not trading days." % bad_dates)

    # NOTE: This is a slightly dirty hack.  We're in-place overwriting the
    # internal data of an Index, which is conceptually immutable.  Since we're
    # maintaining sorting, this should be ok, but this is a good place to
    # sanity check if things start going haywire with calendar computations.
    opens_or_closes.values[indexer] = special_opens_or_closes.values


class ExchangeCalendar(with_metaclass(ABCMeta)):
    """
    An ExchangeCalendar represents the timing information of a single market
    exchange.

    Properties
    ----------
    name : str
        The name of this exchange calendar.
        e.g.: 'NYSE', 'LSE', 'CME Energy'
    tz : timezone
        The native timezone of the exchange.
    """

    def __init__(self, start=start_default, end=end_default):
        tz = self.tz
        open_offset = self.open_offset
        close_offset = self.close_offset

        # Define those days on which the exchange is usually open.
        self.day = CustomBusinessDay(
            holidays=list(self.holidays_adhoc),
            calendar=self.holidays_calendar,
        )

        # Midnight in UTC for each trading day.
        _all_days = date_range(start, end, freq=self.day, tz='UTC')

        # `DatetimeIndex`s of standard opens/closes for each day.
        _opens = days_at_time(_all_days, self.open_time, tz, open_offset)
        _closes = days_at_time(_all_days, self.close_time, tz, close_offset)

        # `DatetimeIndex`s of nonstandard opens/closes
        _special_opens = self._special_opens(start, end)
        _special_closes = self._special_closes(start, end)

        # Overwrite the special opens and closes on top of the standard ones.
        _overwrite_special_dates(_all_days, _opens, _special_opens)
        _overwrite_special_dates(_all_days, _closes, _special_closes)

        # In pandas 0.16.1 _opens and _closes will lose their timezone
        # information. This looks like it has been resolved in 0.17.1.
        # http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#datetime-with-tz  # noqa
        self.schedule = DataFrame(
            index=_all_days,
            columns=['market_open', 'market_close'],
            data={
                'market_open': _opens,
                'market_close': _closes,
            },
            dtype='datetime64[ns]',
        )

        self.first_trading_day = _all_days[0]
        self.last_trading_day = _all_days[-1]

    def _special_dates(self, calendars, ad_hoc_dates, start_date, end_date):
        """
        Union an iterable of pairs of the form

        (time, calendar)

        and an iterable of pairs of the form

        (time, [dates])

        (This is shared logic for computing special opens and special closes.)
        """
        tz = self.native_timezone
        _dates = DatetimeIndex([], tz='UTC').union_many(
            [
                holidays_at_time(calendar, start_date, end_date, time_, tz)
                for time_, calendar in calendars
            ] + [
                days_at_time(datetimes, time_, tz)
                for time_, datetimes in ad_hoc_dates
            ]
        )
        return _dates[(_dates >= start_date) & (_dates <= end_date)]

    def _special_opens(self, start, end):
        return self._special_dates(
            self.special_opens_calendars,
            self.special_opens_adhoc,
            start,
            end,
        )

    def _special_closes(self, start, end):
        return self._special_dates(
            self.special_closes_calendars,
            self.special_closes_adhoc,
            start,
            end,
        )

    @abstractproperty
    def name(self):
        """
        The name of this exchange calendar.
        E.g.: 'NYSE', 'LSE', 'CME Energy'
        """
        raise NotImplementedError()

    @abstractproperty
    def tz(self):
        """
        The native timezone of the exchange.

        SD: Not clear that this needs to be exposed.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_open_on_minute(self, dt):
        """
        Is the exchange open at minute @dt.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        bool
            True if  exchange is open at the given dt, otherwise False.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_open_on_date(self, dt):
        """
        Is the exchange open anytime during @dt.

        SD: Need to decide whether this method answers the question:
        - Is exchange open at any time during the calendar day containing dt
        or
        - Is exchange open at any time during the trading session containg dt.
        Semantically it seems that the first makes more sense.

        Parameters
        ----------
        dt : Timestamp
            The UTC-canonicalized date.

        Returns
        -------
        bool
            True if exchange is open at any time during @dt.
        """
        raise NotImplementedError()

    @abstractmethod
    def trading_days(self, start, end):
        """
        Calculates all of the exchange sessions between the given
        start and end.

        SD: Presumably @start and @end are UTC-canonicalized, as our exchange
        sessions are. If not, then it's not clear how this method should behave
        if @start and @end are both in the middle of the day.

        Parameters
        ----------
        start : Timestamp
        end : Timestamp

        Returns
        -------
        DatetimeIndex
            A DatetimeIndex populated with all of the trading days between
            the given start and end.
        """
        raise NotImplementedError()

    @abstractmethod
    def opens_and_closes(self, date):
        """
        Given a UTC-canonicalized date, returns a tuple of timestamps of the
        open and close of the exchange session on that date.

        SD: Can @date be an arbitrary datetime, or should we first map it to
        and exchange session using session_date. Need to check what the
        consumers expect.

        Parameters
        ----------
        date : Timestamp
            The UTC-canonicalized date whose open and close are needed.

        Returns
        -------
        (Timestamp, Timestamp)
            The open and close for the given date.
        """
        raise NotImplementedError()

    @abstractmethod
    def session_date(self, dt):
        """
        Given a time, returns the UTC-canonicalized date of the exchange
        session in which the time belongs. If the time is not in an exchange
        session (while the market is closed), returns the date of the next
        exchange session after the time.

        Parameters
        ----------
        dt : Timestamp

        Returns
        -------
        Timestamp
            The date of the exchange session in which dt belongs.
        """
        raise NotImplementedError()

    @abstractmethod
    def minutes_for_date(self, date):
        """
        Given a UTC-canonicalized date, returns a DatetimeIndex of all trading
        minutes in the exchange session for that date.

        SD: Sounds like @date can be an arbitrary datetime, and that we should
        first map to an exchange session by calling self.session_date. Need to
        check what the consumers expect.

        Parameters
        ----------
        date : Timestamp
            The UTC-canonicalized date whose minutes are needed.

        Returns
        -------
        DatetimeIndex
            A DatetimeIndex populated with all of the minutes in the
            given date.
        """
        raise NotImplementedError()

    @abstractmethod
    def minute_window(self, start, count, step=1):
        """
        Return a DatetimeIndex containing `count` market minutes, starting with
        `start` and continuing `step` minutes at a time.

        Parameters
        ----------
        start : Timestamp
            The start of the window.
        count : int
            The number of minutes needed.
        step : int
            The step size by which to increment.

        Returns
        -------
        DatetimeIndex
            A window with @count minutes, starting with @start a returning
            every @step minute.
        """
        raise NotImplementedError()

    def normalize_date(self, date):
        date = pd.Timestamp(date, tz='UTC')
        return pd.tseries.tools.normalize_date(date)
