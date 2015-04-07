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
from abc import (
    ABCMeta,
    abstractproperty,
)
from six import with_metaclass

from pandas import (
    DataFrame,
    date_range,
    DateOffset,
    DatetimeIndex,
    Timedelta,
)
from pandas.tseries.offsets import CustomBusinessDay


def delta_from_time(t):
    """
    Convert a datetime.time into a timedelta.
    """
    return Timedelta(
        hours=t.hour,
        minutes=t.minute,
        seconds=t.second,
    )


def days_at_time(days, t, tz):
    """
    Shift an index of days to time t, interpreted in tz.

    Overwrites any existing tz info on the input.
    """
    days = DatetimeIndex(days).tz_localize(None).tz_localize(tz)
    return days.shift(
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
    Abstract Base Class for Exchange Calendars

    Provides the following public attributes:
    """
    def __init__(self, start, end):
        tz = self.native_timezone

        self.day = CustomBusinessDay(
            holidays=list(self.holidays_adhoc),
            calendar=self.holidays_calendar,
        )

        # Midnight in UTC for each trading day.
        _all_days = date_range(start, end, freq=self.day, tz='UTC')

        # `DatetimeIndex`s of standard opens/closes for each day.
        _opens = days_at_time(_all_days, self.open_time, tz)
        _closes = days_at_time(_all_days, self.close_time, tz)

        # `DatetimeIndex`s of nonstandard opens/closes
        _special_opens = self._special_opens(start, end)
        _special_closes = self._special_closes(start, end)

        # Overwrite the special opens and closes on top of the standard ones.
        _overwrite_special_dates(_all_days, _opens, _special_opens)
        _overwrite_special_dates(_all_days, _closes, _special_closes)

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
    def native_timezone(self):
        """
        Native timezone for the exchange.
        """
        raise NotImplementedError()

    @abstractproperty
    def open_time(self):
        """
        datetime.time at which the exchange opens on normal days
        """
        raise NotImplementedError()

    @abstractproperty
    def close_time(self):
        """
        Hour, in UTC, at which the exchange opens on no.
        """
        raise NotImplementedError()

    @abstractproperty
    def holidays_calendar(self):
        """
        An instance of pd.AbstractHolidayCalendar representing
        regularly-occurring holidays.
        """
        raise NotImplementedError()

    @abstractproperty
    def special_opens_calendars(self):
        """
        Iterable of pairs of the form (datetime.time, AbstractHolidayCalendar).

        Defines dates and times on which the calendar regularly opens at a
        nonstandard time.
        """
        return ()

    @abstractproperty
    def special_closes_calendars(self):
        """
        Iterable of pairs of the form (datetime.time, AbstractHolidayCalendar).

        Defines dates and times on which the calendar regularly closes at a
        nonstandard time.
        """
        return ()

    @abstractproperty
    def holidays_adhoc(self):
        """
        An iterable of datetimes on which the market was closed.

        Intended for use in cases where the market was closed as a result of a
        non-recurring historical event.
        """
        return ()

    @abstractproperty
    def special_opens_adhoc(self):
        """
        Iterable of datetimes on which the exchange opened irregularly.

        Intended for use in cases where the market opened irregularly as a
        result of a non-recurring historical event.
        """
        return ()

    @abstractproperty
    def special_closes_adhoc(self):
        """
        Iterable of datetimes on which the exchange closed irregularly.

        Intended for use in cases where the market closed irregularly as a
        result of a non-recurring historical event.
        """
        return ()
