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
from unittest import TestCase
from datetime import timedelta

from nose_parameterized import parameterized
import pandas as pd
from six import iteritems
from six.moves import range, map

from zipline.testing import subtest, parameter_space
import zipline.utils.events
from zipline.utils.calendars import get_calendar
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
    TradingDayOfWeekRule
)


def param_range(*args):
    return ([n] for n in range(*args))


class TestUtils(TestCase):
    @parameterized.expand([
        ('_build_date', _build_date),
        ('_build_time', _build_time),
    ])
    def test_build_none(self, name, f):
        with self.assertRaises(ValueError):
            f(None, {})

    def test_build_offset_default(self):
        default = object()
        self.assertIs(default, _build_offset(None, {}, default))

    def test_build_offset_both(self):
        with self.assertRaises(ValueError):
            _build_offset(datetime.timedelta(minutes=1), {'minutes': 1}, None)

    def test_build_offset_exc(self):
        with self.assertRaises(TypeError):
            # object() is not an instance of a timedelta.
            _build_offset(object(), {}, None)

    def test_build_offset_kwargs(self):
        kwargs = {'minutes': 1}
        self.assertEqual(
            _build_offset(None, kwargs, None),
            datetime.timedelta(**kwargs),
        )

    def test_build_offset_td(self):
        td = datetime.timedelta(minutes=1)
        self.assertEqual(
            _build_offset(td, {}, None),
            td,
        )

    def test_build_date_both(self):
        with self.assertRaises(ValueError):
            _build_date(
                datetime.date(year=2014, month=9, day=25), {
                    'year': 2014,
                    'month': 9,
                    'day': 25,
                },
            )

    def test_build_date_kwargs(self):
        kwargs = {'year': 2014, 'month': 9, 'day': 25}
        self.assertEqual(
            _build_date(None, kwargs),
            datetime.date(**kwargs),
        )

    def test_build_date_date(self):
        date = datetime.date(year=2014, month=9, day=25)
        self.assertEqual(
            _build_date(date, {}),
            date,
        )

    def test_build_time_both(self):
        with self.assertRaises(ValueError):
            _build_time(
                datetime.time(hour=1, minute=5), {
                    'hour': 1,
                    'minute': 5,
                },
            )

    def test_build_time_kwargs(self):
        kwargs = {'hour': 1, 'minute': 5}
        self.assertEqual(
            _build_time(None, kwargs),
            datetime.time(**kwargs),
        )


class TestEventManager(TestCase):
    def setUp(self):
        self.em = EventManager()
        self.event1 = Event(Always(), lambda context, data: None)
        self.event2 = Event(Always(), lambda context, data: None)

    def test_add_event(self):
        self.em.add_event(self.event1)
        self.assertEqual(len(self.em._events), 1)

    def test_add_event_prepend(self):
        self.em.add_event(self.event1)
        self.em.add_event(self.event2, prepend=True)
        self.assertEqual([self.event2, self.event1], self.em._events)

    def test_add_event_append(self):
        self.em.add_event(self.event1)
        self.em.add_event(self.event2)
        self.assertEqual([self.event1, self.event2], self.em._events)

    def test_checks_should_trigger(self):
        class CountingRule(Always):
            count = 0

            def should_trigger(self, dt):
                CountingRule.count += 1
                return True

        for r in [CountingRule] * 5:
                self.em.add_event(
                    Event(r(), lambda context, data: None)
                )

        self.em.handle_data(None, None, datetime.datetime.now())

        self.assertEqual(CountingRule.count, 5)


class TestEventRule(TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            EventRule()

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            super(Always, Always()).should_trigger('a')


def minutes_for_days(ordered_days=False):
    """
    500 randomly selected days.
    This is used to make sure our test coverage is unbaised towards any rules.
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
    cal = get_calendar('NYSE')
    random.seed('deterministic')
    if ordered_days:
        # Get a list of 500 trading days, in order. As a performance
        # optimization in AfterOpen and BeforeClose, we rely on the fact that
        # the clock only ever moves forward in a simulation. For those cases,
        # we guarantee that the list of trading days we test is ordered.
        ordered_session_list = random.sample(list(cal.all_sessions), 500)
        ordered_session_list.sort()

        def session_picker(day):
            return ordered_session_list[day]
    else:
        # Other than AfterOpen and BeforeClose, we don't rely on the the nature
        # of the clock, so we don't care.
        def session_picker(day):
            return random.choice(cal.all_sessions[:-1])

    return ((cal.minutes_for_session(session_picker(cnt)),)
            for cnt in range(500))


class RuleTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
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

        cal = get_calendar('NYSE')
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
            k for k, v in iteritems(vars(zipline.utils.events))
            if isinstance(v, type) and
            issubclass(v, self.class_) and
            v is not self.class_ and
            v not in classes_to_ignore and
            not isabstract(v)
        }
        ds = {
            k[5:] for k in dir(self)
            if k.startswith('test') and k[5:] in dem
        }
        self.assertTrue(
            dem <= ds,
            msg='This suite is missing tests for the following classes:\n' +
            '\n'.join(map(repr, dem - ds)),
        )


class TestStatelessRules(RuleTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestStatelessRules, cls).setUpClass()

        cls.class_ = StatelessRule

        cls.nyse_cal = get_calendar('NYSE')

        # First day of 09/2014 is closed whereas that for 10/2014 is open
        cls.sept_sessions = cls.nyse_cal.sessions_in_range(
            pd.Timestamp('2014-09-01', tz='UTC'),
            pd.Timestamp('2014-09-30', tz='UTC'),
        )
        cls.oct_sessions = cls.nyse_cal.sessions_in_range(
            pd.Timestamp('2014-10-01', tz='UTC'),
            pd.Timestamp('2014-10-31', tz='UTC'),
        )

        cls.sept_week = cls.nyse_cal.minutes_for_sessions_in_range(
            pd.Timestamp("2014-09-22", tz='UTC'),
            pd.Timestamp("2014-09-26", tz='UTC')
        )

    @subtest(minutes_for_days(), 'ms')
    def test_Always(self, ms):
        should_trigger = Always().should_trigger
        self.assertTrue(all(map(should_trigger, ms)))

    @subtest(minutes_for_days(), 'ms')
    def test_Never(self, ms):
        should_trigger = Never().should_trigger
        self.assertFalse(any(map(should_trigger, ms)))

    @subtest(minutes_for_days(ordered_days=True), 'ms')
    def test_AfterOpen(self, ms):
        should_trigger = self.after_open.should_trigger
        for i, m in enumerate(ms):
            # Should only trigger at the 64th minute
            if i != 64:
                self.assertFalse(should_trigger(m))
            else:
                self.assertTrue(should_trigger(m))

    @subtest(minutes_for_days(ordered_days=True), 'ms')
    def test_BeforeClose(self, ms):
        ms = list(ms)
        should_trigger = self.before_close.should_trigger
        for m in ms:
            # Should only trigger at the 65th-to-last minute
            if m != ms[-66]:
                self.assertFalse(should_trigger(m))
            else:
                self.assertTrue(should_trigger(m))

    def test_NotHalfDay(self):
        rule = NotHalfDay()
        rule.cal = self.nyse_cal

        half_day_period = pd.Timestamp("2014-07-03", tz='UTC')
        full_day_period = pd.Timestamp("2014-09-24", tz='UTC')

        for minute in self.nyse_cal.minutes_for_session(half_day_period):
            self.assertFalse(rule.should_trigger(minute))

        for minute in self.nyse_cal.minutes_for_session(full_day_period):
            self.assertTrue(rule.should_trigger(minute))

    def test_NthTradingDayOfWeek_day_zero(self):
        """
        Test that we don't blow up when trying to call week_start's
        should_trigger on the first day of a trading environment.
        """
        cal = get_calendar('NYSE')
        rule = NthTradingDayOfWeek(0)
        rule.cal = cal
        first_open = self.nyse_cal.open_and_close_for_session(
            self.nyse_cal.all_sessions[0]
        )
        self.assertTrue(first_open)

    @subtest(param_range(MAX_WEEK_RANGE), 'n')
    def test_NthTradingDayOfWeek(self, n):
        cal = get_calendar('NYSE')
        rule = NthTradingDayOfWeek(n)
        rule.cal = cal
        should_trigger = rule.should_trigger
        prev_period = self.nyse_cal.minute_to_session_label(self.sept_week[0])
        n_tdays = 0
        for minute in self.sept_week:
            period = self.nyse_cal.minute_to_session_label(
                minute, direction="none"
            )

            if prev_period < period:
                n_tdays += 1
                prev_period = period

            if should_trigger(minute):
                self.assertEqual(n_tdays, n)
            else:
                self.assertNotEqual(n_tdays, n)

    @subtest(param_range(MAX_WEEK_RANGE), 'n')
    def test_NDaysBeforeLastTradingDayOfWeek(self, n):
        cal = get_calendar('NYSE')
        rule = NDaysBeforeLastTradingDayOfWeek(n)
        rule.cal = cal
        should_trigger = rule.should_trigger
        for minute in self.sept_week:
            if should_trigger(minute):
                n_tdays = 0
                session = self.nyse_cal.minute_to_session_label(
                    minute,
                    direction="none"
                )
                next_session = self.nyse_cal.next_session_label(session)
                while next_session.dayofweek > session.dayofweek:
                    session = next_session
                    next_session = self.nyse_cal.next_session_label(session)
                    n_tdays += 1

                self.assertEqual(n_tdays, n)

    @parameter_space(
        rule_offset=(0, 1, 2, 3, 4),
        start_offset=(0, 1, 2, 3, 4),
        type=('week_start', 'week_end')
    )
    def test_edge_cases_for_TradingDayOfWeek(self,
                                             rule_offset,
                                             start_offset,
                                             type):
        """
        Test that we account for midweek holidays. Monday 01/20 is a holiday.
        Ensure that the trigger date for that week is adjusted
        appropriately, or thrown out if not enough trading days. Also, test
        that if we start the simulation on a day where we miss the trigger
        for that week, that the trigger is recalculated for next week.
        """

        sim_start = pd.Timestamp('2014-01-06', tz='UTC') + \
            timedelta(days=start_offset)

        delta = timedelta(days=start_offset)

        jan_minutes = self.nyse_cal.minutes_for_sessions_in_range(
            pd.Timestamp("2014-01-06", tz='UTC') + delta,
            pd.Timestamp("2014-01-31", tz='UTC')
        )

        if type == 'week_start':
            rule = NthTradingDayOfWeek
            # Expect to trigger on the first trading day of the week, plus the
            # offset
            trigger_periods = [
                pd.Timestamp('2014-01-06', tz='UTC'),
                pd.Timestamp('2014-01-13', tz='UTC'),
                pd.Timestamp('2014-01-21', tz='UTC'),
                pd.Timestamp('2014-01-27', tz='UTC'),
            ]
            trigger_periods = \
                [x + timedelta(days=rule_offset) for x in trigger_periods]
        else:
            rule = NDaysBeforeLastTradingDayOfWeek
            # Expect to trigger on the last trading day of the week, minus the
            # offset
            trigger_periods = [
                pd.Timestamp('2014-01-10', tz='UTC'),
                pd.Timestamp('2014-01-17', tz='UTC'),
                pd.Timestamp('2014-01-24', tz='UTC'),
                pd.Timestamp('2014-01-31', tz='UTC'),
            ]
            trigger_periods = \
                [x - timedelta(days=rule_offset) for x in trigger_periods]

        rule.cal = self.nyse_cal
        should_trigger = rule(rule_offset).should_trigger

        # If offset is 4, there is not enough trading days in the short week,
        # and so it should not trigger
        if rule_offset == 4:
            del trigger_periods[2]

        # Filter out trigger dates that happen before the simulation starts
        trigger_periods = [x for x in trigger_periods if x >= sim_start]

        # Get all the minutes on the trigger dates
        trigger_minutes = self.nyse_cal.minutes_for_session(trigger_periods[0])
        for period in trigger_periods[1:]:
            trigger_minutes += self.nyse_cal.minutes_for_session(period)

        expected_n_triggered = len(trigger_minutes)
        trigger_minutes_iter = iter(trigger_minutes)

        n_triggered = 0
        for m in jan_minutes:
            if should_trigger(m):
                self.assertEqual(m, next(trigger_minutes_iter))
                n_triggered += 1

        self.assertEqual(n_triggered, expected_n_triggered)

    @parameterized.expand([('week_start',), ('week_end',)])
    def test_week_and_time_composed_rule(self, type):
        week_rule = NthTradingDayOfWeek(0) if type == 'week_start' else \
            NDaysBeforeLastTradingDayOfWeek(4)
        time_rule = AfterOpen(minutes=60)

        week_rule.cal = self.nyse_cal
        time_rule.cal = self.nyse_cal

        composed_rule = week_rule & time_rule

        should_trigger = composed_rule.should_trigger

        week_minutes = self.nyse_cal.minutes_for_sessions_in_range(
            pd.Timestamp("2014-01-06", tz='UTC'),
            pd.Timestamp("2014-01-10", tz='UTC')
        )

        dt = pd.Timestamp('2014-01-06 14:30:00', tz='UTC')
        trigger_day_offset = 0
        trigger_minute_offset = 60
        n_triggered = 0

        for m in week_minutes:
            if should_trigger(m):
                self.assertEqual(m, dt + timedelta(days=trigger_day_offset) +
                                 timedelta(minutes=trigger_minute_offset))
                n_triggered += 1

        self.assertEqual(n_triggered, 1)

    @subtest(param_range(MAX_MONTH_RANGE), 'n')
    def test_NthTradingDayOfMonth(self, n):
        cal = get_calendar('NYSE')
        rule = NthTradingDayOfMonth(n)
        rule.cal = cal
        should_trigger = rule.should_trigger
        for sessions_list in (self.sept_sessions, self.oct_sessions):
            for n_tdays, session in enumerate(sessions_list):
                for m in self.nyse_cal.minutes_for_session(session):
                    if should_trigger(m):
                        self.assertEqual(n_tdays, n)
                    else:
                        self.assertNotEqual(n_tdays, n)

    @subtest(param_range(MAX_MONTH_RANGE), 'n')
    def test_NDaysBeforeLastTradingDayOfMonth(self, n):
        cal = get_calendar('NYSE')
        rule = NDaysBeforeLastTradingDayOfMonth(n)
        rule.cal = cal
        should_trigger = rule.should_trigger
        for n_days_before, session in enumerate(reversed(self.oct_sessions)):
            for m in self.nyse_cal.minutes_for_session(session):
                if should_trigger(m):
                    self.assertEqual(n_days_before, n)
                else:
                    self.assertNotEqual(n_days_before, n)

    @subtest(minutes_for_days(), 'ms')
    def test_ComposedRule(self, ms):
        rule1 = Always()
        rule2 = Never()

        composed = rule1 & rule2
        should_trigger = composed.should_trigger
        self.assertIsInstance(composed, ComposedRule)
        self.assertIs(composed.first, rule1)
        self.assertIs(composed.second, rule2)
        self.assertFalse(any(map(should_trigger, ms)))


class TestStatefulRules(RuleTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestStatefulRules, cls).setUpClass()

        cls.class_ = StatefulRule

    @subtest(minutes_for_days(), 'ms')
    def test_OncePerDay(self, ms):
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

        rule = RuleCounter(OncePerDay())
        for m in ms:
            rule.should_trigger(m)

        self.assertEqual(rule.count, 1)
