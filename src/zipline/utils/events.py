#
# Copyright 2019 Quantopian, Inc.
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
from abc import ABCMeta, ABC, abstractmethod
from collections import namedtuple
import inspect
import warnings

import datetime
import numpy as np
import pandas as pd
import pytz
from toolz import curry

from zipline.utils.input_validation import preprocess
from zipline.utils.memoize import lazyval
from zipline.utils.sentinel import sentinel

from .context_tricks import nop_context

__all__ = [
    "EventManager",
    "Event",
    "EventRule",
    "StatelessRule",
    "ComposedRule",
    "Always",
    "Never",
    "AfterOpen",
    "BeforeClose",
    "NotHalfDay",
    "NthTradingDayOfWeek",
    "NDaysBeforeLastTradingDayOfWeek",
    "NthTradingDayOfMonth",
    "NDaysBeforeLastTradingDayOfMonth",
    "StatefulRule",
    "OncePerDay",
    # Factory API
    "date_rules",
    "time_rules",
    "calendars",
    "make_eventrule",
]

MAX_MONTH_RANGE = 23
MAX_WEEK_RANGE = 5


def ensure_utc(time, tz="UTC"):
    """Normalize a time. If the time is tz-naive, assume it is UTC."""
    if not time.tzinfo:
        time = time.replace(tzinfo=pytz.timezone(tz))
    return time.replace(tzinfo=pytz.utc)


def _out_of_range_error(a, b=None, var="offset"):
    start = 0
    if b is None:
        end = a - 1
    else:
        start = a
        end = b - 1
    return ValueError(
        "{var} must be in between {start} and {end} inclusive".format(
            var=var,
            start=start,
            end=end,
        )
    )


def _td_check(td):
    seconds = td.total_seconds()

    # 43200 seconds = 12 hours
    if 60 <= seconds <= 43200:
        return td
    else:
        raise ValueError(
            "offset must be in between 1 minute and 12 hours, " "inclusive."
        )


def _build_offset(offset, kwargs, default):
    """
    Builds the offset argument for event rules.
    """
    # Filter down to just kwargs that were actually passed.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if offset is None:
        if not kwargs:
            return default  # use the default.
        else:
            return _td_check(datetime.timedelta(**kwargs))
    elif kwargs:
        raise ValueError("Cannot pass kwargs and an offset")
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
            raise ValueError("Must pass a date or kwargs")
        else:
            return datetime.date(**kwargs)

    elif kwargs:
        raise ValueError("Cannot pass kwargs and a date")
    else:
        return date


# TODO: only used in tests
# TODO FIX TZ
def _build_time(time, kwargs):
    """Builds the time argument for event rules."""
    tz = kwargs.pop("tz", "UTC")
    if time:
        if kwargs:
            raise ValueError("Cannot pass kwargs and a time")
        else:
            return ensure_utc(time, tz)
    elif not kwargs:
        raise ValueError("Must pass a time or kwargs")
    else:
        return datetime.time(**kwargs)


@curry
def lossless_float_to_int(funcname, func, argname, arg):
    """
    A preprocessor that coerces integral floats to ints.

    Receipt of non-integral floats raises a TypeError.
    """
    if not isinstance(arg, float):
        return arg

    arg_as_int = int(arg)
    if arg == arg_as_int:
        warnings.warn(
            "{f} expected an int for argument {name!r}, but got float {arg}."
            " Coercing to int.".format(
                f=funcname,
                name=argname,
                arg=arg,
            ),
        )
        return arg_as_int

    raise TypeError(arg)


class EventManager:
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
            create_context if create_context is not None else lambda *_: nop_context
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


class Event(namedtuple("Event", ["rule", "callback"])):
    """
    An event is a pairing of an EventRule and a callable that will be invoked
    with the current algorithm context, data, and datetime only when the rule
    is triggered.
    """

    def __new__(cls, rule, callback=None):
        callback = callback or (lambda *args, **kwargs: None)
        return super(cls, cls).__new__(cls, rule=rule, callback=callback)

    def handle_data(self, context, data, dt):
        """
        Calls the callable only when the rule is triggered.
        """
        if self.rule.should_trigger(dt):
            self.callback(context, data)


class EventRule(ABC):
    """A rule defining when a scheduled function should execute."""

    # Instances of EventRule are assigned a calendar instance when scheduling
    # a function.
    _cal = None

    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self, value):
        self._cal = value

    @abstractmethod
    def should_trigger(self, dt):
        """
        Checks if the rule should trigger with its current state.
        This method should be pure and NOT mutate any state on the object.
        """
        raise NotImplementedError("should_trigger")


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
        if not (isinstance(first, StatelessRule) and isinstance(second, StatelessRule)):
            raise ValueError("Only two StatelessRules can be composed")

        self.first = first
        self.second = second
        self.composer = composer

    def should_trigger(self, dt):
        """
        Composes the two rules with a lazy composer.
        """
        return self.composer(self.first.should_trigger, self.second.should_trigger, dt)

    @staticmethod
    def lazy_and(first_should_trigger, second_should_trigger, dt):
        """
        Lazily ands the two rules. This will NOT call the should_trigger of the
        second rule if the first one returns False.
        """
        return first_should_trigger(dt) and second_should_trigger(dt)

    @property
    def cal(self):
        return self.first.cal

    @cal.setter
    def cal(self, value):
        # Thread the calendar through to the underlying rules.
        self.first.cal = self.second.cal = value


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

    >>> AfterOpen(minutes=30)  # doctest: +ELLIPSIS
    <zipline.utils.events.AfterOpen object at ...>
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
        """Given a date, find that day's open and period end (open + offset)."""

        period_start = self.cal.session_first_minute(self.cal.minute_to_session(dt))
        period_close = self.cal.session_close(self.cal.minute_to_session(dt))

        # Align the market open and close times here with the execution times
        # used by the simulation clock. This ensures that scheduled functions
        # trigger at the correct times.
        if self.cal.name == "us_futures":
            self._period_start = self.cal.execution_time_from_open(period_start)
            self._period_close = self.cal.execution_time_from_close(period_close)
        else:
            self._period_start = period_start
            self._period_close = period_close

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
        if self._period_start is None or self._period_close <= dt:
            self.calculate_dates(dt)

        return dt == self._period_end


class BeforeClose(StatelessRule):
    """A rule that triggers for some offset time before the market closes.
    Example that triggers for the last 30 minutes every day:

    >>> BeforeClose(minutes=30)  # doctest: +ELLIPSIS
    <zipline.utils.events.BeforeClose object at ...>
    """

    def __init__(self, offset=None, **kwargs):
        self.offset = _build_offset(
            offset,
            kwargs,
            datetime.timedelta(minutes=1),  # Defaults to the last minute.
        )

        self._period_start = None
        self._period_close = None
        self._period_end = None

        self._one_minute = datetime.timedelta(minutes=1)

    def calculate_dates(self, dt):
        """
        Given a dt, find that day's close and period start (close - offset).
        """
        period_end = self.cal.session_close(self.cal.minute_to_session(dt))

        # Align the market close time here with the execution time used by the
        # simulation clock. This ensures that scheduled functions trigger at
        # the correct times.
        if self.cal == "us_futures":
            self._period_end = self.cal.execution_time_from_close(period_end)
        else:
            self._period_end = period_end

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
        if self._period_start is None or self._period_close <= dt:
            self.calculate_dates(dt)

        return self._period_start == dt


class NotHalfDay(StatelessRule):
    """
    A rule that only triggers when it is not a half day.
    """

    def should_trigger(self, dt):
        return self.cal.minute_to_session(dt) not in self.cal.early_closes


class TradingDayOfWeekRule(StatelessRule, metaclass=ABCMeta):
    @preprocess(n=lossless_float_to_int("TradingDayOfWeekRule"))
    def __init__(self, n, invert):
        if not 0 <= n < MAX_WEEK_RANGE:
            raise _out_of_range_error(MAX_WEEK_RANGE)

        self.td_delta = (-n - 1) if invert else n

    def should_trigger(self, dt):
        # is this market minute's period in the list of execution periods?
        val = self.cal.minute_to_session(dt, direction="none").value
        return val in self.execution_period_values

    @lazyval
    def execution_period_values(self):
        # calculate the list of periods that match the given criteria
        sessions = self.cal.sessions
        return set(
            pd.Series(data=sessions)
            # Group by ISO year (0) and week (1)
            .groupby(sessions.map(lambda x: x.isocalendar()[0:2]))
            .nth(self.td_delta)
            .view(np.int64)
        )


class NthTradingDayOfWeek(TradingDayOfWeekRule):
    """
    A rule that triggers on the nth trading day of the week.
    This is zero-indexed, n=0 is the first trading day of the week.
    """

    def __init__(self, n):
        super(NthTradingDayOfWeek, self).__init__(n, invert=False)


class NDaysBeforeLastTradingDayOfWeek(TradingDayOfWeekRule):
    """
    A rule that triggers n days before the last trading day of the week.
    """

    def __init__(self, n):
        super(NDaysBeforeLastTradingDayOfWeek, self).__init__(n, invert=True)


class TradingDayOfMonthRule(StatelessRule, metaclass=ABCMeta):
    @preprocess(n=lossless_float_to_int("TradingDayOfMonthRule"))
    def __init__(self, n, invert):
        if not 0 <= n < MAX_MONTH_RANGE:
            raise _out_of_range_error(MAX_MONTH_RANGE)
        if invert:
            self.td_delta = -n - 1
        else:
            self.td_delta = n

    def should_trigger(self, dt):
        # is this market minute's period in the list of execution periods?
        value = self.cal.minute_to_session(dt, direction="none").value
        return value in self.execution_period_values

    @lazyval
    def execution_period_values(self):
        # calculate the list of periods that match the given criteria
        sessions = self.cal.sessions
        return set(
            pd.Series(data=sessions)
            .groupby([sessions.year, sessions.month])
            .nth(self.td_delta)
            .astype(np.int64)
        )


class NthTradingDayOfMonth(TradingDayOfMonthRule):
    """
    A rule that triggers on the nth trading day of the month.
    This is zero-indexed, n=0 is the first trading day of the month.
    """

    def __init__(self, n):
        super(NthTradingDayOfMonth, self).__init__(n, invert=False)


class NDaysBeforeLastTradingDayOfMonth(TradingDayOfMonthRule):
    """
    A rule that triggers n days before the last trading day of the month.
    """

    def __init__(self, n):
        super(NDaysBeforeLastTradingDayOfMonth, self).__init__(n, invert=True)


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

    @property
    def cal(self):
        return self.rule.cal

    @cal.setter
    def cal(self, value):
        # Thread the calendar through to the underlying rule.
        self.rule.cal = value


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


class date_rules:
    """Factories for date-based :func:`~zipline.api.schedule_function` rules.

    See Also
    --------
    :func:`~zipline.api.schedule_function`
    """

    @staticmethod
    def every_day():
        """Create a rule that triggers every day.

        Returns
        -------
        rule : zipline.utils.events.EventRule
        """
        return Always()

    @staticmethod
    def month_start(days_offset=0):
        """
        Create a rule that triggers a fixed number of trading days after the
        start of each month.

        Parameters
        ----------
        days_offset : int, optional
            Number of trading days to wait before triggering each
            month. Default is 0, i.e., trigger on the first trading day of the
            month.

        Returns
        -------
        rule : zipline.utils.events.EventRule
        """
        return NthTradingDayOfMonth(n=days_offset)

    @staticmethod
    def month_end(days_offset=0):
        """
        Create a rule that triggers a fixed number of trading days before the
        end of each month.

        Parameters
        ----------
        days_offset : int, optional
            Number of trading days prior to month end to trigger. Default is 0,
            i.e., trigger on the last day of the month.

        Returns
        -------
        rule : zipline.utils.events.EventRule
        """
        return NDaysBeforeLastTradingDayOfMonth(n=days_offset)

    @staticmethod
    def week_start(days_offset=0):
        """
        Create a rule that triggers a fixed number of trading days after the
        start of each week.

        Parameters
        ----------
        days_offset : int, optional
            Number of trading days to wait before triggering each week. Default
            is 0, i.e., trigger on the first trading day of the week.
        """
        return NthTradingDayOfWeek(n=days_offset)

    @staticmethod
    def week_end(days_offset=0):
        """
        Create a rule that triggers a fixed number of trading days before the
        end of each week.

        Parameters
        ----------
        days_offset : int, optional
            Number of trading days prior to week end to trigger. Default is 0,
            i.e., trigger on the last trading day of the week.
        """
        return NDaysBeforeLastTradingDayOfWeek(n=days_offset)


class time_rules:
    """Factories for time-based :func:`~zipline.api.schedule_function` rules.

    See Also
    --------
    :func:`~zipline.api.schedule_function`
    """

    @staticmethod
    def market_open(offset=None, hours=None, minutes=None):
        """
        Create a rule that triggers at a fixed offset from market open.

        The offset can be specified either as a :class:`datetime.timedelta`, or
        as a number of hours and minutes.

        Parameters
        ----------
        offset : datetime.timedelta, optional
            If passed, the offset from market open at which to trigger. Must be
            at least 1 minute.
        hours : int, optional
            If passed, number of hours to wait after market open.
        minutes : int, optional
            If passed, number of minutes to wait after market open.

        Returns
        -------
        rule : zipline.utils.events.EventRule

        Notes
        -----
        If no arguments are passed, the default offset is one minute after
        market open.

        If ``offset`` is passed, ``hours`` and ``minutes`` must not be
        passed. Conversely, if either ``hours`` or ``minutes`` are passed,
        ``offset`` must not be passed.
        """
        return AfterOpen(offset=offset, hours=hours, minutes=minutes)

    @staticmethod
    def market_close(offset=None, hours=None, minutes=None):
        """
        Create a rule that triggers at a fixed offset from market close.

        The offset can be specified either as a :class:`datetime.timedelta`, or
        as a number of hours and minutes.

        Parameters
        ----------
        offset : datetime.timedelta, optional
            If passed, the offset from market close at which to trigger. Must
            be at least 1 minute.
        hours : int, optional
            If passed, number of hours to wait before market close.
        minutes : int, optional
            If passed, number of minutes to wait before market close.

        Returns
        -------
        rule : zipline.utils.events.EventRule

        Notes
        -----
        If no arguments are passed, the default offset is one minute before
        market close.

        If ``offset`` is passed, ``hours`` and ``minutes`` must not be
        passed. Conversely, if either ``hours`` or ``minutes`` are passed,
        ``offset`` must not be passed.
        """
        return BeforeClose(offset=offset, hours=hours, minutes=minutes)

    every_minute = Always


class calendars:
    US_EQUITIES = sentinel("US_EQUITIES")
    US_FUTURES = sentinel("US_FUTURES")


def _invert(d):
    return dict(zip(d.values(), d.keys()))


_uncalled_rules = _invert(vars(date_rules))
_uncalled_rules.update(_invert(vars(time_rules)))


def _check_if_not_called(v):
    try:
        name = _uncalled_rules[v]
    except KeyError:
        if not (inspect.isclass(v) and issubclass(v, EventRule)):
            return

        name = getattr(v, "__name__", None)

    msg = "invalid rule: %r" % (v,)
    if name is not None:
        msg += " (hint: did you mean %s())" % name

    raise TypeError(msg)


def make_eventrule(date_rule, time_rule, cal, half_days=True):
    """
    Constructs an event rule from the factory api.
    """
    _check_if_not_called(date_rule)
    _check_if_not_called(time_rule)

    if half_days:
        inner_rule = date_rule & time_rule
    else:
        inner_rule = date_rule & time_rule & NotHalfDay()

    opd = OncePerDay(rule=inner_rule)
    # This is where a scheduled function's rule is associated with a calendar.
    opd.cal = cal
    return opd
