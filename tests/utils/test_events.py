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
from collections import namedtuple
import datetime
from functools import partial
from inspect import isabstract
import random
from unittest import TestCase

from nose_parameterized import parameterized
import pandas as pd
from six import iteritems
from six.moves import range, map

from zipline.finance.trading import TradingEnvironment
from zipline.testing import subtest
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
)


# A day known to be a half day.
HALF_DAY = datetime.datetime(year=2014, month=7, day=3)

# A day known to be a full day.
FULL_DAY = datetime.datetime(year=2014, month=9, day=24)


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

            def should_trigger(self, dt, env):
                CountingRule.count += 1
                return True

        for r in [CountingRule] * 5:
                self.em.add_event(
                    Event(r(), lambda context, data: None)
                )

        mock_algo_class = namedtuple('FakeAlgo', ['trading_environment'])
        mock_algo = mock_algo_class(trading_environment="fake_env")
        self.em.handle_data(mock_algo, None, datetime.datetime.now())

        self.assertEqual(CountingRule.count, 5)


class TestEventRule(TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            EventRule()

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            super(Always, Always()).should_trigger('a', env=None)


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
    env = TradingEnvironment()
    random.seed('deterministic')
    if ordered_days:
        # Get a list of 500 trading days, in order. As a performance
        # optimization in AfterOpen and BeforeClose, we rely on the fact that
        # the clock only ever moves forward in a simulation. For those cases,
        # we guarantee that the list of trading days we test is ordered.
        ordered_day_list = random.sample(list(env.trading_days), 500)
        ordered_day_list.sort()

        def day_picker(day):
            return ordered_day_list[day]
    else:
        # Other than AfterOpen and BeforeClose, we don't rely on the the nature
        # of the clock, so we don't care.
        def day_picker(day):
            return random.choice(env.trading_days[:-1])

    return ((env.market_minutes_for_day(day_picker(cnt)),)
            for cnt in range(500))


class RuleTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
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

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def test_completeness(self):
        """
        Tests that all rules are being tested.
        """
        if not self.class_:
            return  # This is the base class testing, it is always complete.

        dem = {
            k for k, v in iteritems(vars(zipline.utils.events))
            if isinstance(v, type) and
            issubclass(v, self.class_) and
            v is not self.class_ and
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

        cls.sept_days = cls.env.days_in_range(
            pd.Timestamp('2014-09-01'),
            pd.Timestamp('2014-09-30'),
        )

        cls.sept_week = cls.env.minutes_for_days_in_range(
            datetime.date(year=2014, month=9, day=21),
            datetime.date(year=2014, month=9, day=26),
        )

    @subtest(minutes_for_days(), 'ms')
    def test_Always(self, ms):
        should_trigger = partial(Always().should_trigger, env=self.env)
        self.assertTrue(all(map(should_trigger, ms)))

    @subtest(minutes_for_days(), 'ms')
    def test_Never(self, ms):
        should_trigger = partial(Never().should_trigger, env=self.env)
        self.assertFalse(any(map(should_trigger, ms)))

    @subtest(minutes_for_days(ordered_days=True), 'ms')
    def test_AfterOpen(self, ms):
        should_trigger = partial(
            self.after_open.should_trigger,
            env=self.env,
        )
        for i, m in enumerate(ms):
            # Should only trigger at the 64th minute
            if i != 64:
                self.assertFalse(should_trigger(m))
            else:
                self.assertTrue(should_trigger(m))

    @subtest(minutes_for_days(ordered_days=True), 'ms')
    def test_BeforeClose(self, ms):
        ms = list(ms)
        should_trigger = partial(
            self.before_close.should_trigger,
            env=self.env
        )
        for m in ms:
            # Should only trigger at the 65th-to-last minute
            if m != ms[-66]:
                self.assertFalse(should_trigger(m))
            else:
                self.assertTrue(should_trigger(m))

    @subtest(minutes_for_days(), 'ms')
    def test_NotHalfDay(self, ms):
        should_trigger = partial(NotHalfDay().should_trigger, env=self.env)
        self.assertTrue(should_trigger(FULL_DAY))
        self.assertFalse(should_trigger(HALF_DAY))

    def test_NthTradingDayOfWeek_day_zero(self):
        """
        Test that we don't blow up when trying to call week_start's
        should_trigger on the first day of a trading environment.
        """
        self.assertTrue(
            NthTradingDayOfWeek(0).should_trigger(
                self.env.trading_days[0], self.env
            )
        )

    @subtest(param_range(MAX_WEEK_RANGE), 'n')
    def test_NthTradingDayOfWeek(self, n):
        should_trigger = partial(NthTradingDayOfWeek(n).should_trigger,
                                 env=self.env)
        prev_day = self.sept_week[0].date()
        n_tdays = 0
        for m in self.sept_week:
            if prev_day < m.date():
                n_tdays += 1
            prev_day = m.date()

            if should_trigger(m):
                self.assertEqual(n_tdays, n)
            else:
                self.assertNotEqual(n_tdays, n)

    @subtest(param_range(MAX_WEEK_RANGE), 'n')
    def test_NDaysBeforeLastTradingDayOfWeek(self, n):
        should_trigger = partial(
            NDaysBeforeLastTradingDayOfWeek(n).should_trigger, env=self.env
        )
        for m in self.sept_week:
            if should_trigger(m):
                n_tdays = 0
                date = m.to_datetime().date()
                next_date = self.env.next_trading_day(date)
                while next_date.weekday() > date.weekday():
                    date = next_date
                    next_date = self.env.next_trading_day(date)
                    n_tdays += 1

                self.assertEqual(n_tdays, n)

    @subtest(param_range(MAX_MONTH_RANGE), 'n')
    def test_NthTradingDayOfMonth(self, n):
        should_trigger = partial(NthTradingDayOfMonth(n).should_trigger,
                                 env=self.env)
        for n_tdays, d in enumerate(self.sept_days):
            for m in self.env.market_minutes_for_day(d):
                if should_trigger(m):
                    self.assertEqual(n_tdays, n)
                else:
                    self.assertNotEqual(n_tdays, n)

    @subtest(param_range(MAX_MONTH_RANGE), 'n')
    def test_NDaysBeforeLastTradingDayOfMonth(self, n):
        should_trigger = partial(
            NDaysBeforeLastTradingDayOfMonth(n).should_trigger, env=self.env
        )
        for n_days_before, d in enumerate(reversed(self.sept_days)):
            for m in self.env.market_minutes_for_day(d):
                if should_trigger(m):
                    self.assertEqual(n_days_before, n)
                else:
                    self.assertNotEqual(n_days_before, n)

    @subtest(minutes_for_days(), 'ms')
    def test_ComposedRule(self, ms):
        rule1 = Always()
        rule2 = Never()

        composed = rule1 & rule2
        should_trigger = partial(composed.should_trigger, env=self.env)
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

            def should_trigger(self, dt, env):
                st = self.rule.should_trigger(dt, env)
                if st:
                    self.count += 1
                return st

        rule = RuleCounter(OncePerDay())
        for m in ms:
            rule.should_trigger(m, env=self.env)

        self.assertEqual(rule.count, 1)
