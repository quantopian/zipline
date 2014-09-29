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
from functools import partial
import operator
import six

import datetime
import pandas as pd
import pytz

from zipline.finance.trading import TradingEnvironment


__all__ = [
    'EventManager',
    'Event',
    'EventRule',
    'StatelessRule',
    'InvertedRule',
    'ComposedRule',
    'Always',
    'Never',
    'AfterOpen',
    'BeforeClose',
    'OnDate',
    'AfterDate',
    'BeforeDate',
    'BetweenDates',
    'AtTime',
    'AfterTime',
    'BeforeTime',
    'BetweenTimes',
    'HalfDay',
    'NotHalfDay',
    'NthTradingDayOfWeek',
    'FirstTradingDayOfWeek',
    'NDaysBeforeLastTradingDayOfWeek',
    'LastTradingDayOfWeek',
    'NthTradingDayOfMonth',
    'FirstTradingDayOfMonth',
    'NDaysBeforeLastTradingDayOfMonth',
    'LastTradingDayOfMonth',
    'StatefulRule',
    'RuleFromCallable',
    'Once',
    'DoNTimes',
    'SkipNTimes',
    'NTimesPerPeriod',
    'OncePerPeriod',
    'OncePerDay',
    'OncePerWeek',
    'OncePerMonth',
    'OncePerQuarter',

    # Factory API
    'DateRuleFactory',
    'TimeRuleFactory',
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


def _coerce_datetime(maybe_dt):
    if isinstance(maybe_dt, datetime.datetime):
        return maybe_dt
    elif isinstance(maybe_dt, datetime.date):
        return datetime.datetime(
            year=maybe_dt.year,
            month=maybe_dt.month,
            day=maybe_dt.day,
            tzinfo=pytz.utc,
        )
    elif isinstance(maybe_dt, (tuple, list)) and len(maybe_dt) == 3:
        year, month, day = maybe_dt
        return datetime.datetime(
            year=year,
            month=month,
            day=day,
            tzinfo=pytz.utc,
        )
    else:
        raise TypeError('Cannot coerce %s into a datetime.datetime'
                        % type(maybe_dt).__name__)


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
    """
    Manages a list of Event objects.
    This manages the logic for checking the rules and dispatching to the
    handle_data function of the Events.
    """
    def __init__(self):
        self._events = []

    def add_event(self, event, prepend=False):
        """
        Adds an event to the manager.
        """
        if prepend:
            self._events.insert(0, event)
        else:
            self._events.append(event)

    def handle_data(self, context, data, dt):
        for event in self._events:
            event.handle_data(context, data, dt)


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
    """
    An event rule checks a datetime and sees if it should trigger.
    """
    @property
    def env(self):
        return TradingEnvironment.instance()

    @abstractmethod
    def should_trigger(self, dt):
        """
        Checks if the rule should trigger with it's current state.
        This method should be pure and NOT mutate any state on the object.
        """
        raise NotImplementedError('should_trigger')


class StatelessRule(EventRule):
    """
    A stateless rule has no state.
    This is reentrant and will always give the same result for the
    same datetime.
    Because these are pure, they can be composed to create new rules.
    """
    def and_(self, rule):
        """
        Logical and of two rules, triggers only when both rules trigger.
        This follows the short circuiting rules for normal and.
        """
        return ComposedRule(self, rule, ComposedRule.lazy_and, lazy=True)
    __and__ = and_

    def or_(self, rule):
        """
        Logical or of two rules, triggers when either rule triggers.
        This follows the short circuiting rules for normal or.
        """
        return ComposedRule(self, rule, ComposedRule.lazy_or, lazy=True)
    __or__ = or_

    def xor(self, rule):
        """
        Logical xor of two rules, triggers if exactly one rule is triggered.
        """
        return ComposedRule(self, rule, operator.xor)
    __xor__ = xor

    def invert(self):
        """
        Logical inversion of a rule, triggers only when the rule is not
        triggered.
        """
        return InvertedRule(self)
    __invert__ = invert


# Stateless Rules

class InvertedRule(StatelessRule):
    """
    A rule that inverts the results of another rule.
    """
    def __init__(self, rule):
        if not isinstance(rule, StatelessRule):
            raise ValueError('Only StatelessRules can be inverted')
        self.rule = rule

    def should_trigger(self, dt):
        """
        Triggers only when self.rule.should_trigger(dt) does not trigger.
        """
        return not self.rule.should_trigger(dt)


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
    def __init__(self, first, second, composer, lazy=False):
        if not (isinstance(first, StatelessRule)
                and isinstance(second, StatelessRule)):
            raise ValueError('Only two StatelessRules can be composed')

        self.first = first
        self.second = second
        self.composer = composer
        if lazy:
            # Switch the the lazy should trigger instead.
            self.should_trigger = self._lazy_should_trigger

    def should_trigger(self, dt):
        """
        Composes the results of two rule's should_trigger methods to get the
        result for this rule.
        """
        return self.composer(
            self.first.should_trigger(dt),
            self.second.should_trigger(dt),
        )

    def _lazy_should_trigger(self, dt):
        """
        Composes the two rules with a lazy composer. This is used when
        lazy=True in __init__.
        """
        return self.composer(
            self.first.should_trigger,
            self.second.should_trigger,
            dt,
        )

    @staticmethod
    def lazy_and(first_should_trigger, second_should_trigger, dt):
        """
        Lazily ands the two rules. This will NOT call the should_trigger of the
        second rule if the first one returns False.
        """
        return first_should_trigger(dt) and second_should_trigger(dt)

    @staticmethod
    def lazy_or(first_should_trigger, second_should_trigger, dt):
        """
        Lazily ors the two rules. This will NOT call the should_trigger of the
        second rule the first one returns True.
        """
        return first_should_trigger(dt) or second_should_trigger(dt)


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
    Example that triggers triggers after 30 minutes of the market opening:

    >>> AfterOpen(minutes=30)
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(
            offset,
            kwargs,
            datetime.timedelta(minutes=1),  # Defaults to the first minute.
        )

        self._dt = None

    def should_trigger(self, dt):
        return self._get_open(dt) + self.offset <= dt

    def _get_open(self, dt):
        """
        Cache the open for each day.
        """
        if self._dt is None or (self._dt.date() != dt.date()):
            self._dt = self.env.get_open_and_close(dt)[0] \
                - datetime.timedelta(minutes=1)

        return self._dt


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

        self._dt = None

    def should_trigger(self, dt):
        return self._get_close(dt) - self.offset <= dt

    def _get_close(self, dt):
        """
        Cache the close for each day.
        """
        if self._dt is None or (self._dt.date() != dt.date()):
            self._dt = self.env.get_open_and_close(dt)[1]

        return self._dt


class OnDate(StatelessRule):
    """
    A rule that triggers on a certain date.
    """
    def __init__(self, date=None, **kwargs):
        self.date = _build_date(date, kwargs)

    def should_trigger(self, dt):
        return dt.date() == self.date


class AfterDate(StatelessRule):
    """
    A rule that triggers after a certain date.
    """
    def __init__(self, date, **kwargs):
        self.date = _build_date(date, kwargs)

    def should_trigger(self, dt):
        return dt.date() > self.date


class BeforeDate(StatelessRule):
    """
    A rule that triggers before a certain date.
    """
    def __init__(self, date, **kwargs):
        self.date = _build_date(date, kwargs)

    def should_trigger(self, dt):
        return dt.date() < self.date


def BetweenDates(date1, date2):  # pragma: no cover
    """
    A rule that triggers between in the range [date1, date2).
    """
    return (OnDate(date1) | AfterDate(date1)) & BeforeDate(date2)


class AtTime(StatelessRule):
    """
    A rule that triggers at an exact time.
    """
    def __init__(self, time=None, **kwargs):
        self.time = _build_time(time, kwargs)

    def should_trigger(self, dt):
        return dt.timetz() == self.time


class AfterTime(StatelessRule):
    """
    A rule that triggers after a given time.
    """
    def __init__(self, time=None, **kwargs):
        self.time = _build_time(time, kwargs)

    def should_trigger(self, dt):
        return dt.timetz() > self.time


class BeforeTime(StatelessRule):
    """
    A rule that triggers before a given time.
    """
    def __init__(self, time=None, **kwargs):
        self.time = _build_time(time, kwargs)

    def should_trigger(self, dt):
        return dt.timetz() < self.time


def BetweenTimes(time1=None, time2=None, tz='UTC'):  # pragma: no cover
    """
    A rule that triggers when the datetime is in the range [time1, time2).
    """
    return (AtTime(time1, tz=tz)
            | AfterTime(time1, tz=tz)) & BeforeTime(time2, tz=tz)


class HalfDay(StatelessRule):
    """
    A rule that only triggers on half days.
    """
    def should_trigger(self, dt):
        return dt in self.env.early_closes


class NotHalfDay(StatelessRule):
    """
    A rule that only triggers when it is not a half day.
    """
    def should_trigger(self, dt):
        return dt.date() not in self.env.early_closes


class NthTradingDayOfWeek(StatelessRule):
    """
    A rule that triggers on the nth trading day of the week.
    This is zero-indexed, n=0 is the first trading day of the week.
    """
    def __init__(self, n=0):
        if not 0 <= n < MAX_WEEK_RANGE:
            raise _out_of_range_error(MAX_WEEK_RANGE)
        self.td_delta = n

    def should_trigger(self, dt):
        return _coerce_datetime(self.env.add_trading_days(
            self.td_delta,
            self.get_first_trading_day_of_week(dt),
        )).date() == dt.date()

    def get_first_trading_day_of_week(self, dt):
        prev = dt
        dt = self.env.previous_trading_day(dt)
        while dt.date().weekday() < prev.date().weekday():
            prev = dt
            dt = self.env.previous_trading_day(dt)

        return prev.date()


FirstTradingDayOfWeek = partial(NthTradingDayOfWeek, n=0)


class NDaysBeforeLastTradingDayOfWeek(StatelessRule):
    """
    A rule that triggers n days before the last trading day of the week.
    """
    def __init__(self, n):
        if not 0 <= n < MAX_WEEK_RANGE:
            raise _out_of_range_error(MAX_WEEK_RANGE)
        self.td_delta = -n
        self.date = None

    def should_trigger(self, dt):
        return _coerce_datetime(self.env.add_trading_days(
            self.td_delta,
            self.get_last_trading_day_of_week(dt),
        )).date() == dt.date()

    def get_last_trading_day_of_week(self, dt):
        prev = dt
        dt = self.env.next_trading_day(dt)
        # Traverse forward until we hit a week border, then jump back to the
        # previous trading day.
        while dt.date().weekday() > prev.date().weekday():
            prev = dt
            dt = self.env.next_trading_day(dt)

        return prev.date()


LastTradingDayOfWeek = partial(NDaysBeforeLastTradingDayOfWeek, n=0)  # pragma: no cover  # NOQA


class NthTradingDayOfMonth(StatelessRule):
    """
    A rule that triggers on the nth trading day of the month.
    This is zero-indexed, n=0 is the first trading day of the month.
    """
    def __init__(self, n=0):
        if not 0 <= n < MAX_MONTH_RANGE:
            raise _out_of_range_error(MAX_MONTH_RANGE)
        self.td_delta = n
        self.month = None
        self.day = None

    def should_trigger(self, dt):
        return self.get_nth_trading_day_of_month(dt) == dt.date()

    def get_nth_trading_day_of_month(self, dt):
        if self.month == dt.month:
            # We already computed the day for this month.
            return self.day

        if not self.td_delta:
            self.day = self.get_first_trading_day_of_month(dt)
        else:
            self.day = self.env.add_trading_days(
                self.td_delta,
                self.get_first_trading_day_of_month(dt),
            ).date()

        return self.day

    def get_first_trading_day_of_month(self, dt):
        self.month = dt.month

        dt = dt.replace(day=1)
        self.first_day = (dt if self.env.is_trading_day(dt)
                          else self.env.next_trading_day(dt)).date()
        return self.first_day

FirstTradingDayOfMonth = partial(NthTradingDayOfMonth, n=0)  # pragma: no cover


class NDaysBeforeLastTradingDayOfMonth(StatelessRule):
    """
    A rule that triggers n days before the last trading day of the month.
    """
    def __init__(self, n=0):
        if not 0 <= n < MAX_MONTH_RANGE:
            raise _out_of_range_error(MAX_MONTH_RANGE)
        self.td_delta = -n
        self.month = None
        self.day = None

    def should_trigger(self, dt):
        return self.get_nth_to_last_trading_day_of_month(dt) == dt.date()

    def get_nth_to_last_trading_day_of_month(self, dt):
        if self.month == dt.month:
            # We already computed the last day for this month.
            return self.day

        if not self.td_delta:
            self.day = self.get_last_trading_day_of_month(dt)
        else:
            self.day = self.env.add_trading_days(
                self.td_delta,
                self.get_last_trading_day_of_month(dt),
            ).date()

        return self.day

    def get_last_trading_day_of_month(self, dt):
        self.month = dt.month

        self.last_day = self.env.previous_trading_day(
            dt.replace(month=(dt.month % 12) + 1, day=1)
        ).date()
        return self.last_day


LastTradingDayOfMonth = partial(NDaysBeforeLastTradingDayOfMonth, n=0)  # pragma: no cover  # NOQA


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


class RuleFromCallable(StatefulRule):
    """
    Constructs an EventRule from an arbitrary callable.
    """
    def __init__(self, callback, rule=None):
        """
        Constructs an EventRule from a callable.
        If the provided paramater is not callable, or cannot be called with a
        single paramater, then a ValueError will be raised.
        """
        self.callback = callback
        super(RuleFromCallable, self).__init__(rule)

    def should_trigger(self, dt):
        """
        Only check the wrapped should trigger when callback is true.
        """
        return self.callback(dt) and self.rule.should_trigger(dt)


class DoNTimes(StatefulRule):
    """
    A rule that triggers n times.
    """
    def __init__(self, n, rule=None):
        self.n = n
        if self.n <= 0:
            self.new_should_trigger(Never.never_trigger)
            return
        super(DoNTimes, self).__init__(rule)

    def should_trigger(self, dt):
        """
        Only triggers n times before switching to a never_trigger mode.
        """
        triggered = self.rule.should_trigger(dt)
        if triggered:
            self.n -= 1
            if self.n == 0:
                self.new_should_trigger(Never.never_trigger)
            return True
        return False


Once = partial(DoNTimes, n=1)  # pragma: no cover


class SkipNTimes(StatefulRule):
    """
    A rule that skips the first n times.
    """
    def __init__(self, n, rule=None):
        self.n = n
        if self.n <= 0:
            self.new_should_trigger(Always.always_trigger)
            return
        super(SkipNTimes, self).__init__(rule)

    def should_trigger(self, dt):
        """
        Skips the first n times before switching to the inner rule's
        should_trigger.
        """
        triggered = self.rule.should_trigger(dt)
        if triggered:
            self.n -= 1
            if self.n == 0:
                self.new_should_trigger(self.rule.should_trigger)
        return False


class NTimesPerPeriod(StatefulRule):
    """
    A rule that triggers n times in a given period.
    """
    def __init__(self, n=1, freq='B', rule=None):
        self.n = n
        self.freq = freq
        if self.n <= 0:
            self.new_should_trigger(Never.never_trigger)
        self.period = pd.Period(self.env.first_trading_day, freq=freq)
        super(NTimesPerPeriod, self).__init__(rule)

    def should_trigger(self, dt):
        if dt < self.end_time:
            if self.rule.should_trigger(dt):
                self.hit_this_period += 1
                return self.hit_this_period <= self.n
            else:
                return False
        else:
            # We are now in the next period, compute the next period end and
            # reset the counter.
            self.advance_period(dt)
            self.hit_this_period = 1
            return self.hit_this_period <= self.n

    def advance_period(self, dt):
        """
        Advance the internal period by jumping in steps of size freq from the
        last known end_time.
        """
        while self.end_time < dt:
            self.period += 1

    @property
    def end_time(self):
        return naive_to_utc(self.period.end_time)


class OncePerDay(StatefulRule):
    """
    A special case implementation of `NTimesPerPeriod`.
    """
    def __init__(self, rule=None):
        self.date = None
        self.triggered = False
        super(OncePerDay, self).__init__(rule)

    def should_trigger(self, dt):
        dt_date = dt.date()
        if self.date is None or self.date != dt_date:
            # initialize or reset for new date
            self.triggered = False
            self.date = dt_date

        if not self.triggered and self.rule.should_trigger(dt):
            self.triggered = True
            return True


# Convenience aliases for common use cases on NTimesPerPeriod.
OncePerPeriod = partial(NTimesPerPeriod, n=1)  # pragma: no cover
OncePerWeek = partial(NTimesPerPeriod, n=1, freq='W')  # pragma: no cover
OncePerMonth = partial(NTimesPerPeriod, n=1, freq='M')  # pragma: no cover
OncePerQuarter = partial(NTimesPerPeriod, n=1, freq='Q')  # pragma: no cover


# Factory API

class DateRuleFactory(object):
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


class TimeRuleFactory(object):
    market_open = AfterOpen
    market_close = BeforeClose


# Convenience aliases.
date_rules = DateRuleFactory
time_rules = TimeRuleFactory


def make_eventrule(date_rule, time_rule, half_days=True):
    """
    Constructs an event rule from the factory api.
    """
    if half_days:
        inner_rule = date_rule & time_rule
    else:
        inner_rule = date_rule & time_rule & NotHalfDay()

    return OncePerDay(rule=inner_rule)
