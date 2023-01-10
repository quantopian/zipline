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
import datetime
from inspect import isabstract
import random
import warnings

from parameterized import parameterized
import pandas as pd
from zipline.utils.calendar_utils import get_calendar

import zipline.utils.events
from zipline.utils.events import (
    EventRule,
    StatelessRule,
    Always,
    Never,
    AfterOpen,
    ComposedRule,
    BeforeClose,
    NotHalfDay,
    NthTradingDayOfWeek,
    NDaysBeforeLastTradingDayOfWeek,
    NthTradingDayOfMonth,
    NDaysBeforeLastTradingDayOfMonth,
    StatefulRule,
    OncePerDay,
    _build_offset,
    _build_date,
    _build_time,
    EventManager,
    Event,
    MAX_MONTH_RANGE,
    MAX_WEEK_RANGE,
    TradingDayOfMonthRule,
    TradingDayOfWeekRule,
)

import pytest


def param_range(*args):
    return ([n] for n in range(*args))


class TestUtils:
    @pytest.mark.parametrize(
        "name, f",
        [
            ("_build_date", _build_date),
            ("_build_time", _build_time),
        ],
    )
    def test_build_none(self, name, f):
        with pytest.raises(ValueError):
            f(None, {})

    def test_build_offset_default(self):
        default = object()
        assert default is _build_offset(None, {}, default)

    def test_build_offset_both(self):
        with pytest.raises(ValueError):
            _build_offset(datetime.timedelta(minutes=1), {"minutes": 1}, None)

    def test_build_offset_exc(self):
        with pytest.raises(TypeError):
            # object() is not an instance of a timedelta.
            _build_offset(object(), {}, None)

    def test_build_offset_kwargs(self):
        kwargs = {"minutes": 1}
        assert _build_offset(None, kwargs, None) == datetime.timedelta(**kwargs)

    def test_build_offset_td(self):
        td = datetime.timedelta(minutes=1)
        assert _build_offset(td, {}, None) == td

    def test_build_date_both(self):
        with pytest.raises(ValueError):
            _build_date(
                datetime.date(year=2014, month=9, day=25),
                {
                    "year": 2014,
                    "month": 9,
                    "day": 25,
                },
            )

    def test_build_date_kwargs(self):
        kwargs = {"year": 2014, "month": 9, "day": 25}
        assert _build_date(None, kwargs) == datetime.date(**kwargs)

    def test_build_date_date(self):
        date = datetime.date(year=2014, month=9, day=25)
        assert _build_date(date, {}) == date

    def test_build_time_both(self):
        with pytest.raises(ValueError):
            _build_time(
                datetime.time(hour=1, minute=5),
                {
                    "hour": 1,
                    "minute": 5,
                },
            )

    def test_build_time_kwargs(self):
        kwargs = {"hour": 1, "minute": 5}
        assert _build_time(None, kwargs) == datetime.time(**kwargs)


@pytest.fixture(scope="function")
def set_event_manager(request):
    request.cls.em = EventManager()
    request.cls.event1 = Event(Always())
    request.cls.event2 = Event(Always())


@pytest.mark.usefixtures("set_event_manager")
class TestEventManager:
    def test_add_event(self):
        self.em.add_event(self.event1)
        assert len(self.em._events) == 1

    def test_add_event_prepend(self):
        self.em.add_event(self.event1)
        self.em.add_event(self.event2, prepend=True)
        assert [self.event2, self.event1] == self.em._events

    def test_add_event_append(self):
        self.em.add_event(self.event1)
        self.em.add_event(self.event2)
        assert [self.event1, self.event2] == self.em._events

    def test_checks_should_trigger(self):
        class CountingRule(Always):
            count = 0

            def should_trigger(self, dt):
                CountingRule.count += 1
                return True

        for r in [CountingRule] * 5:
            self.em.add_event(Event(r()))

        self.em.handle_data(None, None, datetime.datetime.now())
        assert CountingRule.count == 5


class TestEventRule:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            EventRule()

    def test_not_implemented(self):
        with pytest.raises(NotImplementedError):
            super(Always, Always()).should_trigger("a")


def minutes_for_days(cal, ordered_days=False):
    """
    500 randomly selected days.
    This is used to make sure our test coverage is unbiased towards any rules.
    We use a random sample because testing on all the trading days took
    around 180 seconds on my laptop, which is far too much for normal unit
    testing.

    We manually set the seed so that this will be deterministic.
    Results of multiple runs were compared to make sure that this is actually
    true.

    This returns a generator of tuples each wrapping a single generator.
    Iterating over this yields a single day, iterating over the day yields
    the minutes for that day.
    """
    random.seed("deterministic")
    if ordered_days:
        # Get a list of 500 trading days, in order. As a performance
        # optimization in AfterOpen and BeforeClose, we rely on the fact that
        # the clock only ever moves forward in a simulation. For those cases,
        # we guarantee that the list of trading days we test is ordered.
        ordered_session_list = random.sample(list(cal.sessions), 500)
        ordered_session_list.sort()

        def session_picker(day):
            return ordered_session_list[day]

    else:
        # Other than AfterOpen and BeforeClose, we don't rely on the nature
        # of the clock, so we don't care.
        def session_picker(day):
            return random.choice(cal.sessions[:-1].tolist())

    return [cal.session_minutes(session_picker(cnt)) for cnt in range(500)]


# THE CLASS BELOW ARE GOING TO BE IMPORTED BY test_events_cme and nyse
class RuleTestCase:
    CALENDAR_STRING = "foo"

    @classmethod
    def setup_class(cls):
        # On the AfterOpen and BeforeClose tests, we want ensure that the
        # functions are pure, and that running them with the same input will
        # provide the same output, regardless of whether the function is run 1
        # or N times. (For performance reasons, we cache some internal state
        # in AfterOpen and BeforeClose, but we don't want it to affect
        # purity). Hence, we use the same before_close and after_open across
        # subtests.
        cls.before_close = BeforeClose(hours=1, minutes=5)
        cls.after_open = AfterOpen(hours=1, minutes=5)
        cls.class_ = None  # Mark that this is the base class.

        cal = get_calendar(cls.CALENDAR_STRING)
        cls.before_close.cal = cal
        cls.after_open.cal = cal

    def test_completeness(self):
        """
        Tests that all rules are being tested.
        """
        if not self.class_:
            return  # This is the base class testing, it is always complete.

        classes_to_ignore = [TradingDayOfWeekRule, TradingDayOfMonthRule]

        dem = {
            k
            for k, v in vars(zipline.utils.events).items()
            if isinstance(v, type)
            and issubclass(v, self.class_)
            and v is not self.class_
            and v not in classes_to_ignore
            and not isabstract(v)
        }
        ds = {k[5:] for k in dir(self) if k.startswith("test") and k[5:] in dem}
        assert (
            dem <= ds
        ), "This suite is missing tests for the following classes:\n" + "\n".join(
            map(repr, dem - ds)
        )


class StatelessRulesTests(RuleTestCase):
    @classmethod
    def setup_class(cls):
        super(StatelessRulesTests, cls).setup_class()

        cls.class_ = StatelessRule
        cls.cal = get_calendar(cls.CALENDAR_STRING)

        # First day of 09/2014 is closed whereas that for 10/2014 is open
        cls.sept_sessions = cls.cal.sessions_in_range(
            pd.Timestamp("2014-09-01"),
            pd.Timestamp("2014-09-30"),
        )
        cls.oct_sessions = cls.cal.sessions_in_range(
            pd.Timestamp("2014-10-01"),
            pd.Timestamp("2014-10-31"),
        )

        cls.sept_week = cls.cal.sessions_minutes(
            pd.Timestamp("2014-09-22"),
            pd.Timestamp("2014-09-26"),
        )

        cls.HALF_SESSION = None
        cls.FULL_SESSION = None

    def test_Always(self):
        should_trigger = Always().should_trigger
        for session_minutes in minutes_for_days(self.cal):
            assert all(map(should_trigger, session_minutes))

    def test_Never(self):
        should_trigger = Never().should_trigger
        for session_minutes in minutes_for_days(self.cal):
            assert not any(map(should_trigger, session_minutes))

    def test_AfterOpen(self):
        minute_groups = minutes_for_days(self.cal, ordered_days=True)
        should_trigger = self.after_open.should_trigger
        for session_minutes in minute_groups:
            for i, minute in enumerate(session_minutes):
                # Should only trigger at the 64th minute
                if i != 64:
                    assert not should_trigger(minute)
                else:
                    assert should_trigger(minute)

    def test_invalid_offset(self):
        with pytest.raises(ValueError):
            AfterOpen(hours=12, minutes=1)

        with pytest.raises(ValueError):
            AfterOpen(hours=0, minutes=0)

        with pytest.raises(ValueError):
            BeforeClose(hours=12, minutes=1)

        with pytest.raises(ValueError):
            BeforeClose(hours=0, minutes=0)

    def test_BeforeClose(self):
        minute_groups = minutes_for_days(self.cal, ordered_days=True)
        should_trigger = self.before_close.should_trigger
        for minute_group in minute_groups:
            for minute in minute_group:
                # Should only trigger at the 65th-to-last minute
                if minute != minute_group[-66]:
                    assert not should_trigger(minute)
                else:
                    assert should_trigger(minute)

    def test_NotHalfDay(self):
        rule = NotHalfDay()
        rule.cal = self.cal

        if self.HALF_SESSION:
            for minute in self.cal.session_minutes(self.HALF_SESSION):
                assert not rule.should_trigger(minute)

        if self.FULL_SESSION:
            for minute in self.cal.session_minutes(self.FULL_SESSION):
                assert rule.should_trigger(minute)

    def test_NthTradingDayOfWeek_day_zero(self):
        """Test that we don't blow up when trying to call week_start's
        should_trigger on the first day of a trading environment.
        """
        rule = NthTradingDayOfWeek(0)
        rule.cal = self.cal
        first_open = self.cal.session_open_close(self.cal.sessions[0])
        assert first_open

    def test_NthTradingDayOfWeek(self):
        for n in range(MAX_WEEK_RANGE):
            rule = NthTradingDayOfWeek(n)
            rule.cal = self.cal
            should_trigger = rule.should_trigger
            prev_period = self.cal.minute_to_session(self.sept_week[0])
            n_tdays = 0
            for minute in self.sept_week:
                period = self.cal.minute_to_session(minute)

                if prev_period < period:
                    n_tdays += 1
                    prev_period = period

                if should_trigger(minute):
                    assert n_tdays == n
                else:
                    assert n_tdays != n

    def test_NDaysBeforeLastTradingDayOfWeek(self):
        for n in range(MAX_WEEK_RANGE):
            rule = NDaysBeforeLastTradingDayOfWeek(n)
            rule.cal = self.cal
            should_trigger = rule.should_trigger
            for minute in self.sept_week:
                if should_trigger(minute):
                    n_tdays = 0
                    session = self.cal.minute_to_session(minute, direction="none")
                    next_session = self.cal.next_session(session)
                    while next_session.dayofweek > session.dayofweek:
                        session = next_session
                        next_session = self.cal.next_session(session)
                        n_tdays += 1

                    assert n_tdays == n

    def test_NthTradingDayOfMonth(self):
        for n in range(MAX_MONTH_RANGE):
            rule = NthTradingDayOfMonth(n)
            rule.cal = self.cal
            should_trigger = rule.should_trigger
            for sessions_list in (self.sept_sessions, self.oct_sessions):
                for n_tdays, session in enumerate(sessions_list):
                    # just check the first 10 minutes of each session
                    for m in self.cal.session_minutes(session)[0:10]:
                        if should_trigger(m):
                            assert n_tdays == n
                        else:
                            assert n_tdays != n

    def test_NDaysBeforeLastTradingDayOfMonth(self):
        for n in range(MAX_MONTH_RANGE):
            rule = NDaysBeforeLastTradingDayOfMonth(n)
            rule.cal = self.cal
            should_trigger = rule.should_trigger
            sessions = reversed(self.oct_sessions)
            for n_days_before, session in enumerate(sessions):
                for m in self.cal.session_minutes(session)[0:10]:
                    if should_trigger(m):
                        assert n_days_before == n
                    else:
                        assert n_days_before != n

    def test_ComposedRule(self):
        minute_groups = minutes_for_days(self.cal)
        rule1 = Always()
        rule2 = Never()

        for minute in minute_groups:
            composed = rule1 & rule2
            should_trigger = composed.should_trigger
            assert isinstance(composed, ComposedRule)
            assert composed.first is rule1
            assert composed.second is rule2
            assert not any(map(should_trigger, minute))

    @parameterized.expand(
        [
            ("month_start", NthTradingDayOfMonth),
            ("month_end", NDaysBeforeLastTradingDayOfMonth),
            ("week_start", NthTradingDayOfWeek),
            ("week_end", NthTradingDayOfWeek),
        ],
    )
    def test_pass_float_to_day_of_period_rule(self, name, rule_type):
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter("always")
            rule_type(n=3)  # Shouldn't trigger a warning.
            rule_type(n=3.0)  # Should trigger a warning about float coercion.

        assert len(raised_warnings) == 1

        # We only implicitly convert from float to int when there's no loss of
        # precision.
        with pytest.raises(TypeError):
            rule_type(3.1)

    def test_invalid_offsets(self):
        with pytest.raises(ValueError):
            NthTradingDayOfWeek(5)

        with pytest.raises(ValueError):
            NthTradingDayOfWeek(-1)

        with pytest.raises(ValueError):
            NthTradingDayOfMonth(-1)

        with pytest.raises(ValueError):
            NthTradingDayOfMonth(24)


class StatefulRulesTests(RuleTestCase):
    CALENDAR_STRING = "NYSE"

    @classmethod
    def setup_class(cls):
        super(StatefulRulesTests, cls).setup_class()

        cls.class_ = StatefulRule
        cls.cal = get_calendar(cls.CALENDAR_STRING)

    def test_OncePerDay(self):
        class RuleCounter(StatefulRule):
            """
            A rule that counts the number of times another rule triggers
            but forwards the results out.
            """

            count = 0

            def should_trigger(self, dt):
                st = self.rule.should_trigger(dt)
                if st:
                    self.count += 1
                return st

        for minute_group in minutes_for_days(self.cal):
            rule = RuleCounter(OncePerDay())

            for minute in minute_group:
                rule.should_trigger(minute)

            assert rule.count == 1
