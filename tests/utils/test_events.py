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
import datetime
from itertools import islice
from six import iteritems
import random
from six.moves import range, map
from nose_parameterized import parameterized
from unittest import TestCase

import numpy as np

from zipline.finance.trading import TradingEnvironment, with_environment
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


@with_environment()
def minutes_for_days(env=None):
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
    Iterating over this yeilds a single day, iterating over the day yields
    the minutes for that day.
    """
    random.seed('deterministic')
    return ((env.market_minutes_for_day(random.choice(env.trading_days)),)
            for _ in range(500))


class RuleTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment.instance()
        cls.class_ = None  # Mark that this is the base class.

    def test_completeness(self):
        """
        Tests that all rules are being tested.
        """
        if not self.class_:
            return  # This is the base class testing, it is always complete.

        dem = {
            k for k, v in iteritems(vars(zipline.utils.events))
            if isinstance(v, type)
            and issubclass(v, self.class_)
            and v is not self.class_
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
            np.datetime64(datetime.date(year=2014, month=9, day=1)),
            np.datetime64(datetime.date(year=2014, month=9, day=30)),
        )

        cls.sept_week = cls.env.minutes_for_days_in_range(
            datetime.date(year=2014, month=9, day=21),
            datetime.date(year=2014, month=9, day=26),
        )

    @parameterized.expand(minutes_for_days())
    def test_Always(self, ms):
        should_trigger = Always().should_trigger
        self.assertTrue(all(map(should_trigger, ms)))

    @parameterized.expand(minutes_for_days())
    def test_Never(self, ms):
        should_trigger = Never().should_trigger
        self.assertFalse(any(map(should_trigger, ms)))

    @parameterized.expand(minutes_for_days())
    def test_AfterOpen(self, ms):
        should_trigger = AfterOpen(minutes=5, hours=1).should_trigger
        for m in islice(ms, 64):
            # Check the first 64 minutes of data.
            # We use 64 because the offset is from market open
            # at 13:30 UTC, meaning the first minute of data has an
            # offset of 1.
            self.assertFalse(should_trigger(m))
        for m in islice(ms, 64, None):
            # Check the rest of the day.
            self.assertTrue(should_trigger(m))

    @parameterized.expand(minutes_for_days())
    def test_BeforeClose(self, ms):
        ms = list(ms)
        should_trigger = BeforeClose(hours=1, minutes=5).should_trigger
        for m in ms[0:-65]:
            self.assertFalse(should_trigger(m))
        for m in ms[-65:]:
            self.assertTrue(should_trigger(m))

    def test_NotHalfDay(self):
        should_trigger = NotHalfDay().should_trigger
        self.assertTrue(should_trigger(FULL_DAY))
        self.assertFalse(should_trigger(HALF_DAY))

    @parameterized.expand(param_range(MAX_WEEK_RANGE))
    def test_NthTradingDayOfWeek(self, n):
        should_trigger = NthTradingDayOfWeek(n).should_trigger
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

    @parameterized.expand(param_range(MAX_WEEK_RANGE))
    def test_NDaysBeforeLastTradingDayOfWeek(self, n):
        should_trigger = NDaysBeforeLastTradingDayOfWeek(n).should_trigger
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

    @parameterized.expand(param_range(MAX_MONTH_RANGE))
    def test_NthTradingDayOfMonth(self, n):
        should_trigger = NthTradingDayOfMonth(n).should_trigger
        for n_tdays, d in enumerate(self.sept_days):
            for m in self.env.market_minutes_for_day(d):
                if should_trigger(m):
                    self.assertEqual(n_tdays, n)
                else:
                    self.assertNotEqual(n_tdays, n)

    @parameterized.expand(param_range(MAX_MONTH_RANGE))
    def test_NDaysBeforeLastTradingDayOfMonth(self, n):
        should_trigger = NDaysBeforeLastTradingDayOfMonth(n).should_trigger
        for n_days_before, d in enumerate(reversed(self.sept_days)):
            for m in self.env.market_minutes_for_day(d):
                if should_trigger(m):
                    self.assertEqual(n_days_before, n)
                else:
                    self.assertNotEqual(n_days_before, n)

    @parameterized.expand(minutes_for_days())
    def test_ComposedRule(self, ms):
        rule1 = Always()
        rule2 = Never()

        composed = rule1 & rule2
        self.assertIsInstance(composed, ComposedRule)
        self.assertIs(composed.first, rule1)
        self.assertIs(composed.second, rule2)
        self.assertFalse(any(map(composed.should_trigger, ms)))


class TestStatefulRules(RuleTestCase):
    @classmethod
    def setUpClass(cls):
        super(TestStatefulRules, cls).setUpClass()

        cls.class_ = StatefulRule

    @parameterized.expand(minutes_for_days())
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
