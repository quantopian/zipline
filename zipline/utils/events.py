#
# Copyright 2014 Quantopian, Inc.
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
from collections import namedtuple
import six

import datetime
import pandas as pd
import pytz

from .context_tricks import nop_context
from zipline.errors import NoFurtherDataError

from zipline.utils.calendars import normalize_date

__all__ = [
    'EventManager',
    'Event',
    'EventRule',
    'StatelessRule',
    'ComposedRule',
    'Always',
    'Never',
    'AfterOpen',
    'BeforeClose',
    'NotHalfDay',
    'NthTradingDayOfWeek',
    'NDaysBeforeLastTradingDayOfWeek',
    'NthTradingDayOfMonth',
    'NDaysBeforeLastTradingDayOfMonth',
    'StatefulRule',
    'OncePerDay',

    # Factory API
    'date_rules',
    'time_rules',
    'make_eventrule',
]


MAX_MONTH_RANGE = 26
MAX_WEEK_RANGE = 5


def naive_to_utc(ts):
    """
    Converts a UTC tz-naive timestamp to a tz-aware timestamp.
    """
    # Drop the nanoseconds field. warn=False suppresses the warning
    # that we are losing the nanoseconds; however, this is intended.
    return pd.Timestamp(ts.to_pydatetime(warn=False), tz='UTC')


def ensure_utc(time, tz='UTC'):
    """
    Normalize a time. If the time is tz-naive, assume it is UTC.
    """
    if not time.tzinfo:
        time = time.replace(tzinfo=pytz.timezone(tz))
    return time.replace(tzinfo=pytz.utc)


def _out_of_range_error(a, b=None, var='offset'):
    start = 0
    if b is None:
        end = a - 1
    else:
        start = a
        end = b - 1
    return ValueError(
        '{var} must be in between {start} and {end} inclusive'.format(
            var=var,
            start=start,
            end=end,
        )
    )


def _td_check(td):
    seconds = td.total_seconds()

    # 23400 seconds is 6 hours and 30 minutes.
    if 60 <= seconds <= 23400:
        return td
    else:
        raise ValueError('offset must be in between 1 minute and 6 hours and'
                         ' 30 minutes inclusive')


def _build_offset(offset, kwargs, default):
    """
    Builds the offset argument for event rules.
    """
    if offset is None:
        if not kwargs:
            return default  # use the default.
        else:
            return _td_check(datetime.timedelta(**kwargs))
    elif kwargs:
        raise ValueError('Cannot pass kwargs and an offset')
    elif isinstance(offset, datetime.timedelta):
        return _td_check(offset)
    else:
        raise TypeError("Must pass 'hours' and/or 'minutes' as keywords")


def _build_date(date, kwargs):
    """
    Builds the date argument for event rules.
    """
    if date is None:
        if not kwargs:
            raise ValueError('Must pass a date or kwargs')
        else:
            return datetime.date(**kwargs)

    elif kwargs:
        raise ValueError('Cannot pass kwargs and a date')
    else:
        return date


def _build_time(time, kwargs):
    """
    Builds the time argument for event rules.
    """
    tz = kwargs.pop('tz', 'UTC')
    if time:
        if kwargs:
            raise ValueError('Cannot pass kwargs and a time')
        else:
            return ensure_utc(time, tz)
    elif not kwargs:
        raise ValueError('Must pass a time or kwargs')
    else:
        return datetime.time(**kwargs)


class EventManager(object):
    """Manages a list of Event objects.
    This manages the logic for checking the rules and dispatching to the
    handle_data function of the Events.

    Parameters
    ----------
    create_context : (BarData) -> context manager, optional
        An optional callback to produce a context manager to wrap the calls
        to handle_data. This will be passed the current BarData.
    """
    def __init__(self, create_context=None):
        self._events = []
        self._create_context = (
            create_context
            if create_context is not None else
            lambda *_: nop_context
        )

    def add_event(self, event, prepend=False):
        """
        Adds an event to the manager.
        """
        if prepend:
            self._events.insert(0, event)
        else:
            self._events.append(event)

    def handle_data(self, context, data, dt):
        with self._create_context(data):
            for event in self._events:
                event.handle_data(
                    context,
                    data,
                    dt,
                )


class Event(namedtuple('Event', ['rule', 'callback'])):
    """
    An event is a pairing of an EventRule and a callable that will be invoked
    with the current algorithm context, data, and datetime only when the rule
    is triggered.
    """
    def __new__(cls, rule=None, callback=None):
        callback = callback or (lambda *args, **kwargs: None)
        return super(cls, cls).__new__(cls, rule=rule, callback=callback)

    def handle_data(self, context, data, dt):
        """
        Calls the callable only when the rule is triggered.
        """
        if self.rule.should_trigger(dt):
            self.callback(context, data)


class EventRule(six.with_metaclass(ABCMeta)):
    @abstractmethod
    def should_trigger(self, dt):
        """
        Checks if the rule should trigger with its current state.
        This method should be pure and NOT mutate any state on the object.
        """
        raise NotImplementedError('should_trigger')


class StatelessRule(EventRule):
    """
    A stateless rule has no observable side effects.
    This is reentrant and will always give the same result for the
    same datetime.
    Because these are pure, they can be composed to create new rules.
    """
    def and_(self, rule):
        """
        Logical and of two rules, triggers only when both rules trigger.
        This follows the short circuiting rules for normal and.
        """
        return ComposedRule(self, rule, ComposedRule.lazy_and)
    __and__ = and_


class ComposedRule(StatelessRule):
    """
    A rule that composes the results of two rules with some composing function.
    The composing function should be a binary function that accepts the results
    first(dt) and second(dt) as positional arguments.
    For example, operator.and_.
    If lazy=True, then the lazy composer is used instead. The lazy composer
    expects a function that takes the two should_trigger functions and the
    datetime. This is useful of you don't always want to call should_trigger
    for one of the rules. For example, this is used to implement the & and |
    operators so that they will have the same short circuit logic that is
    expected.
    """
    def __init__(self, first, second, composer):
        if not (isinstance(first, StatelessRule) and
                isinstance(second, StatelessRule)):
            raise ValueError('Only two StatelessRules can be composed')

        self.first = first
        self.second = second
        self.composer = composer

    def should_trigger(self, dt):
        """
        Composes the two rules with a lazy composer.
        """
        return self.composer(
            self.first.should_trigger,
            self.second.should_trigger,
            dt
        )

    @staticmethod
    def lazy_and(first_should_trigger, second_should_trigger, dt):
        """
        Lazily ands the two rules. This will NOT call the should_trigger of the
        second rule if the first one returns False.
        """
        return first_should_trigger(dt) and second_should_trigger(dt)


class Always(StatelessRule):
    """
    A rule that always triggers.
    """
    @staticmethod
    def always_trigger(dt):
        """
        A should_trigger implementation that will always trigger.
        """
        return True
    should_trigger = always_trigger


class Never(StatelessRule):
    """
    A rule that never triggers.
    """
    @staticmethod
    def never_trigger(dt):
        """
        A should_trigger implementation that will never trigger.
        """
        return False
    should_trigger = never_trigger


class AfterOpen(StatelessRule):
    """
    A rule that triggers for some offset after the market opens.
    Example that triggers after 30 minutes of the market opening:

    >>> AfterOpen(minutes=30)
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(
            offset,
            kwargs,
            datetime.timedelta(minutes=1),  # Defaults to the first minute.
        )

        self._period_start = None
        self._period_end = None
        self._period_close = None

        self._one_minute = datetime.timedelta(minutes=1)

    def calculate_dates(self, dt):
        # given a dt, find that day's open and period end (open + offset)
        self._period_start, self._period_close = self.cal.open_and_close(dt)
        self._period_end = self._period_start + self.offset - self._one_minute

    def should_trigger(self, dt):
        # There are two reasons why we might want to recalculate the dates.
        # One is the first time we ever call should_trigger, when
        # self._period_start is none. The second is when we're on a new day,
        # and need to recalculate the dates. For performance reasons, we rely
        # on the fact that our clock only ever ticks forward, since it's
        # cheaper to do dt1 <= dt2 than dt1.date() != dt2.date(). This means
        # that we will NOT correctly recognize a new date if we go backwards
        # in time(which should never happen in a simulation, or in live
        # trading)
        if (
            self._period_start is None or
            self._period_close <= dt
        ):
            self.calculate_dates(dt)

        return dt == self._period_end


class BeforeClose(StatelessRule):
    """
    A rule that triggers for some offset time before the market closes.
    Example that triggers for the last 30 minutes every day:

    >>> BeforeClose(minutes=30)
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(
            offset,
            kwargs,
            datetime.timedelta(minutes=1),  # Defaults to the last minute.
        )

        self._period_start = None
        self._period_end = None

        self._one_minute = datetime.timedelta(minutes=1)

    def calculate_dates(self, dt):
        # given a dt, find that day's close and period start (close - offset)
        self._period_end = self.cal.open_and_close(dt)[1]
        self._period_start = self._period_end - self.offset
        self._period_close = self._period_end

    def should_trigger(self, dt):
        # There are two reasons why we might want to recalculate the dates.
        # One is the first time we ever call should_trigger, when
        # self._period_start is none. The second is when we're on a new day,
        # and need to recalculate the dates. For performance reasons, we rely
        # on the fact that our clock only ever ticks forward, since it's
        # cheaper to do dt1 <= dt2 than dt1.date() != dt2.date(). This means
        # that we will NOT correctly recognize a new date if we go backwards
        # in time(which should never happen in a simulation, or in live
        # trading)
        if (
            self._period_start is None or
            self._period_close <= dt
        ):
            self.calculate_dates(dt)

        return self._period_start == dt


class NotHalfDay(StatelessRule):
    """
    A rule that only triggers when it is not a half day.
    """
    def should_trigger(self, dt):
        return normalize_date(dt) not in self.cal.early_closes


class TradingDayOfWeekRule(six.with_metaclass(ABCMeta, StatelessRule)):
    def __init__(self, n, invert):
        if not 0 <= n < MAX_WEEK_RANGE:
            raise _out_of_range_error(MAX_WEEK_RANGE)
        self.next_date_start = None
        self.next_date_end = None
        self.next_midnight_timestamp = None
        self.td_delta = -n if invert else n

    @abstractmethod
    def date_func(self, dt, cal):
        raise NotImplementedError

    def calculate_start_and_end(self, dt):
        while True:
            next_trading_day = self.cal.add_trading_days(
                self.td_delta,
                self.date_func(dt, self.cal),
            )

            # If after applying the offset to the start/end day of the week, we
            # get day in a different week, skip this week and go on to the next
            if next_trading_day.isocalendar()[1] == dt.isocalendar()[1]:
                break
            else:
                dt += datetime.timedelta(days=7)

        next_open, next_close = self.cal.open_and_close(next_trading_day)
        self.next_date_start = next_open
        self.next_date_end = next_close
        self.next_midnight_timestamp = next_trading_day

    def should_trigger(self, dt):
        if self.next_date_start is None:
            # First time this method has been called. Calculate the midnight,
            # open, and close for the first trigger, which occurs on the week
            # of the simulation start
            self.calculate_start_and_end(dt)

        # If we've passed the trigger, calculate the next one
        if dt > self.next_date_end:
            self.calculate_start_and_end(self.next_date_end +
                                         datetime.timedelta(days=7))

        # if the given dt is within the next matching day, return true.
        if self.next_date_start <= dt <= self.next_date_end or \
                dt == self.next_midnight_timestamp:
            return True

        return False


class NthTradingDayOfWeek(TradingDayOfWeekRule):
    """
    A rule that triggers on the nth trading day of the week.
    This is zero-indexed, n=0 is the first trading day of the week.
    """
    def __init__(self, n):
        super(NthTradingDayOfWeek, self).__init__(n, invert=False)

    @staticmethod
    def get_first_trading_day_of_week(dt, cal):
        prev = None
        # Traverse backward until we hit a week border, then jump back to the
        # previous trading day.
        try:
            while not prev or dt.weekday() < prev.weekday():
                prev = dt
                dt = cal.previous_trading_day(dt)
        except NoFurtherDataError:
            prev = dt

        if cal.is_open_on_day(prev):
            return prev
        else:
            return cal.next_trading_day(prev)

    date_func = get_first_trading_day_of_week


class NDaysBeforeLastTradingDayOfWeek(TradingDayOfWeekRule):
    """
    A rule that triggers n days before the last trading day of the week.
    """
    def __init__(self, n):
        super(NDaysBeforeLastTradingDayOfWeek, self).__init__(n, invert=True)

    @staticmethod
    def get_last_trading_day_of_week(dt, cal):
        prev = None
        # Traverse forward until we hit a week border, then jump back to the
        # previous trading day.
        try:
            while not prev or dt.weekday() > prev.weekday():
                prev = dt
                dt = cal.next_trading_day(dt)
        except NoFurtherDataError:
            prev = dt

        if cal.is_open_on_day(prev):
            return prev
        else:
            return cal.previous_trading_day(prev)

    date_func = get_last_trading_day_of_week


class TradingDayOfMonthRule(six.with_metaclass(ABCMeta, StatelessRule)):
    def __init__(self, n, invert):
        if not 0 <= n < MAX_MONTH_RANGE:
            raise _out_of_range_error(MAX_MONTH_RANGE)
        self.month = None
        self.date = None
        self.td_delta = -n if invert else n

    def should_trigger(self, dt):
        return self.get_trigger_day_of_month(dt) == normalize_date(dt)

    @abstractmethod
    def date_func(self, dt):
        raise NotImplementedError

    def get_trigger_day_of_month(self, dt):
        if self.month == dt.month:
            # We already computed the day for this month.
            return self.date

        self.date = self.date_func(dt)
        if self.td_delta:
            self.date = self.cal.add_trading_days(self.td_delta, self.date)

        return self.date


class NthTradingDayOfMonth(TradingDayOfMonthRule):
    """
    A rule that triggers on the nth trading day of the month.
    This is zero-indexed, n=0 is the first trading day of the month.
    """
    def __init__(self, n):
        super(NthTradingDayOfMonth, self).__init__(n, invert=False)

    def get_first_trading_day_of_month(self, dt):
        self.month = dt.month

        dt = dt.replace(day=1)
        first_day = (normalize_date(dt) if self.cal.is_open_on_day(dt)
                     else self.cal.next_trading_day(dt))
        return first_day

    date_func = get_first_trading_day_of_month


class NDaysBeforeLastTradingDayOfMonth(TradingDayOfMonthRule):
    """
    A rule that triggers n days before the last trading day of the month.
    """
    def __init__(self, n):
        super(NDaysBeforeLastTradingDayOfMonth, self).__init__(n, invert=True)

    def get_last_trading_day_of_month(self, dt):
        self.month = dt.month

        if dt.month == 12:
            # Roll the year forward and start in January.
            year = dt.year + 1
            month = 1
        else:
            # Increment the month in the same year.
            year = dt.year
            month = dt.month + 1

        last_day = self.cal.previous_trading_day(
            dt.replace(year=year, month=month, day=1)
        )
        return last_day

    date_func = get_last_trading_day_of_month


# Stateful rules


class StatefulRule(EventRule):
    """
    A stateful rule has state.
    This rule will give different results for the same datetimes depending
    on the internal state that this holds.
    StatefulRules wrap other rules as state transformers.
    """
    def __init__(self, rule=None):
        self.rule = rule or Always()

    def new_should_trigger(self, callable_):
        """
        Replace the should trigger implementation for the current rule.
        """
        self.should_trigger = callable_


class OncePerDay(StatefulRule):
    def __init__(self, rule=None):
        self.triggered = False

        self.date = None
        self.next_date = None

        super(OncePerDay, self).__init__(rule)

    def should_trigger(self, dt):
        if self.date is None or dt >= self.next_date:
            # initialize or reset for new date
            self.triggered = False
            self.date = dt

            # record the timestamp for the next day, so that we can use it
            # to know if we've moved to the next day
            self.next_date = dt + pd.Timedelta(1, unit="d")

        if not self.triggered and self.rule.should_trigger(dt):
            self.triggered = True
            return True


# Factory API

class date_rules(object):
    every_day = Always

    @staticmethod
    def month_start(days_offset=0):
        return NthTradingDayOfMonth(n=days_offset)

    @staticmethod
    def month_end(days_offset=0):
        return NDaysBeforeLastTradingDayOfMonth(n=days_offset)

    @staticmethod
    def week_start(days_offset=0):
        return NthTradingDayOfWeek(n=days_offset)

    @staticmethod
    def week_end(days_offset=0):
        return NDaysBeforeLastTradingDayOfWeek(n=days_offset)


class time_rules(object):
    market_open = AfterOpen
    market_close = BeforeClose
    every_minute = Always


def make_eventrule(date_rule, time_rule, cal, half_days=True):
    """
    Constructs an event rule from the factory api.
    """

    # Insert the calendar in to the individual rules
    date_rule.cal = cal
    time_rule.cal = cal

    if half_days:
        inner_rule = date_rule & time_rule
    else:
        nhd_rule = NotHalfDay()
        nhd_rule.cal = cal
        inner_rule = date_rule & time_rule & nhd_rule

    return OncePerDay(rule=inner_rule)
