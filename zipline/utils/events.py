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

from zipline.finance.trading import TradingEnvironment
from zipline.utils.argcheck import verify_callable_argspec, Argument


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
    'DateRuleFactory',
    'TimeRuleFactory',
    'make_eventrule',
]


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


def _build_offset(offset, kwargs):
    """
    Builds the offset argument for event rules.
    """
    if offset is None:
        if not kwargs:
            return datetime.timedelta()  # An empty offset (+0).
        else:
            return datetime.timedelta(**kwargs)
    elif kwargs:
        raise ValueError('Cannot pass kwargs and an offset')
    else:
        return offset


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
    def __new__(cls, rule=None, callback=None, check_args=True):
        callback = callback or (lambda *args, **kwargs: None)
        if check_args:
            # Check the callback provided.
            verify_callable_argspec(
                callback,
                [Argument('context' if check_args else Argument.ignore),
                 Argument('data' if check_args else Argument.ignore)]
            )

            # Make sure that the rule's should_trigger is valid. This will
            # catch potential errors much more quickly and give a more helpful
            # error.
            verify_callable_argspec(
                getattr(rule, 'should_trigger'),
                [Argument('dt')]
            )

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
    env = TradingEnvironment.instance()

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
        if not (isinstance(first, StatelessRule)
                and isinstance(second, StatelessRule)):
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
            dt,
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
    Example that triggers triggers after 30 minutes of the market opening:

    >>> AfterOpen(minutes=30)
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(offset, kwargs)

    def should_trigger(self, dt):
        return self.env.get_open_and_close(dt)[0] + self.offset <= dt


class BeforeClose(StatelessRule):
    """
    A rule that triggers for some offset time before the market closes.
    Example that triggers for the last 30 minutes every day:

    >>> BeforeClose(minutes=30)
    """
    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(offset, kwargs)

    def should_trigger(self, dt):
        return self.env.get_open_and_close(dt)[1] - self.offset < dt


class NotHalfDay(StatelessRule):
    """
    A rule that only triggers when it is not a half day.
    """
    def should_trigger(self, dt):
        return dt not in self.env.early_closes


class NthTradingDayOfWeek(StatelessRule):
    """
    A rule that triggers on the nth trading day of the week.
    This is zero-indexed, n=0 is the first trading day of the week.
    """
    def __init__(self, n=0):
        if n not in range(5):
            raise ValueError('n must be in [0,5)')
        self.td_delta = n

    def should_trigger(self, dt):
        return self.env.add_trading_days(
            self.td_delta,
            self.get_first_trading_day_of_week(dt),
        ) == dt.date()

    def get_first_trading_day_of_week(self, dt):
        prev = dt
        dt = self.env.previous_trading_day(dt)
        # Backtrack until we hit a week border, then jump to the next trading
        # day.
        while dt.day < prev.day:
            prev = dt
            dt = self.env.previous_trading_day(dt)
        return prev.date()


class NDaysBeforeLastTradingDayOfWeek(StatelessRule):
    """
    A rule that triggers n days before the last trading day of the week.
    """
    def __init__(self, n):
        if n not in range(5):
            raise ValueError('n must be in [0,5)')
        self.td_delta = -n
        self.date = None

    def should_trigger(self, dt):
        return self.env.add_trading_days(
            self.td_delta,
            self.get_last_trading_day_of_week(dt),
        ) == dt.date()

    def get_last_trading_day_of_week(self, dt):
        prev = dt
        dt = self.env.next_trading_day(dt)
        # Traverse forward until we hit a week border, then jump back to the
        # previous trading day.
        while dt.day > prev.day:
            prev = dt
            dt = self.env.next_trading_day(dt)
        return prev.date()


class NthTradingDayOfMonth(StatelessRule):
    """
    A rule that triggers on the nth trading day of the month.
    This is zero-indexed, n=0 is the first trading day of the month.
    """
    def __init__(self, n=0):
        if n not in range(31):
            raise ValueError('n must be in [0,31)')
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


class NDaysBeforeLastTradingDayOfMonth(StatelessRule):
    """
    A rule that triggers n days before the last trading day of the month.
    """
    def __init__(self, n=0):
        if n not in range(31):
            raise ValueError('n must be in [0,31)')
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


# Factory API

class DateRuleFactory(object):
    every_day = Always

    @staticmethod
    def month_start(offset=0):
        return NthTradingDayOfMonth(n=offset)

    @staticmethod
    def month_end(offset=0):
        return NDaysBeforeLastTradingDayOfMonth(n=offset)

    @staticmethod
    def week_start(offset=0):
        return NthTradingDayOfWeek(n=offset)

    @staticmethod
    def week_end(offset=0):
        return NDaysBeforeLastTradingDayOfWeek(n=offset)


class TimeRuleFactory(object):
    market_open = AfterOpen
    market_close = BeforeClose


def make_eventrule(date_rule, time_rule, half_days=True):
    """
    Constructs an event rule from the factory api.
    """
    if half_days:
        inner_rule = date_rule & time_rule
    else:
        inner_rule = date_rule & time_rule & NotHalfDay()

    return OncePerDay(rule=inner_rule)
