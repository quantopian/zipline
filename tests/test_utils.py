#
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

from unittest import TestCase
from zipline.utils.factory import (load_from_yahoo,
                                   load_bars_from_yahoo)
from zipline.utils.event_management import(
    PeriodicEvent,
    AfterOpen,
    BeforeClose,
    AtTime,
    BetweenTimes,
    at_market_open,
    at_market_close
)
from zipline.finance.trading import TradingEnvironment
import pandas as pd
import pytz
import numpy as np
import random


class TestFactory(TestCase):
    def test_load_from_yahoo(self):
        stocks = ['AAPL', 'GE']
        start = pd.datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
        data = load_from_yahoo(stocks=stocks, start=start, end=end)

        assert data.index[0] == pd.Timestamp('1993-01-04 00:00:00+0000')
        assert data.index[-1] == pd.Timestamp('2001-12-31 00:00:00+0000')
        for stock in stocks:
            assert stock in data.columns

        np.testing.assert_raises(
            AssertionError, load_from_yahoo, stocks=stocks,
            start=end, end=start
        )

    def test_load_bars_from_yahoo(self):
        stocks = ['AAPL', 'GE']
        start = pd.datetime(1993, 1, 1, 0, 0, 0, 0, pytz.utc)
        end = pd.datetime(2002, 1, 1, 0, 0, 0, 0, pytz.utc)
        data = load_bars_from_yahoo(stocks=stocks, start=start, end=end)

        assert data.major_axis[0] == pd.Timestamp('1993-01-04 00:00:00+0000')
        assert data.major_axis[-1] == pd.Timestamp('2001-12-31 00:00:00+0000')
        for stock in stocks:
            assert stock in data.items

        for ohlc in ['open', 'high', 'low', 'close', 'volume', 'price']:
            assert ohlc in data.minor_axis

        np.testing.assert_raises(
            AssertionError, load_bars_from_yahoo, stocks=stocks,
            start=end, end=start
        )


class TestEntryRules(TestCase):

    def setUp(self):
        env = TradingEnvironment()
        index = random.choice(range(len(env.trading_days)))
        test_dt = env.trading_days[index]
        self.open_time, self.close_time = env.get_open_and_close(test_dt)
        self.market_mins = env.market_minutes_for_day(test_dt)

    def test_AfterOpen(self):
        """
        Only the first 65 minutes in the trading day
        should evaluate to False.
        """
        after_open = AfterOpen(minutes=5, hours=1)
        results = {dt: after_open(dt) for dt in self.market_mins}
        for dt in self.market_mins[0:65]:
            assert results[dt] is False
        for dt in self.market_mins[65::]:
            assert results[dt] is True

    def test_BeforeClose(self):
        """
        Only the last 65 minutes in the trading day
        should evaluate to True.
        """
        before_close = BeforeClose(minutes=5, hours=1)
        results = {dt: before_close(dt) for dt in self.market_mins}
        for dt in self.market_mins[0:-65]:
            assert results[dt] is False
        for dt in self.market_mins[-65::]:
            assert results[dt] is True

    def test_AtTime(self):
        # 9:40 EST / 13:40 UTC test date
        test_dt = self.market_mins[9]
        at_time = AtTime(hour=9, minute=40, tz='US/Eastern')
        results = {i: at_time(i) for i in self.market_mins}
        for dt in self.market_mins:
            if dt == test_dt:
                assert results[dt] is True
            else:
                assert results[dt] is False

    def test_BetweenTimes(self):
        # criteria: t1 <= dt < t2
        bw_times = BetweenTimes((9, 40), (9, 50), tz='US/Eastern')
        test_dts = self.market_mins[9:19]
        results = {i: bw_times(i) for i in self.market_mins}
        for dt in self.market_mins:
            if dt in test_dts:
                assert results[dt] is True
            else:
                assert results[dt] is False

    def test_at_market_open(self):
        results = {dt: at_market_open(dt) for dt in self.market_mins}
        for dt in self.market_mins:
            if dt == self.open_time:
                assert results[dt] is True
            else:
                assert results[dt] is False

    def test_at_market_close(self):
        results = {dt: at_market_close(dt) for dt in self.market_mins}
        for dt in self.market_mins:
            if dt == self.close_time:
                assert results[dt] is True
            else:
                assert results[dt] is False


class TestPeriodicEvent(TestCase):

    def setUp(self):
        self.env = TradingEnvironment()
        self.early_closes = self.env.early_closes
        index = random.choice(range(len(self.env.trading_days) - 5))
        self.trading_days = self.env.trading_days[index:index + 5]
        self.market_mins = {dt: self.env.market_minutes_for_day(dt)
                            for dt in self.trading_days}
        # Set the rule to always return True so that it does not
        # affect the outcome of the call to .signal()
        self.rule = lambda x: True

    def test_max_daily_hits(self):
        event = PeriodicEvent(rule=self.rule, max_daily_hits=3)
        days_hits = {dt: 0 for dt in self.trading_days}
        for dt in self.trading_days:
            for t in self.market_mins[dt]:
                if event(t):
                    days_hits[dt] += 1
        for d in self.trading_days:
            assert days_hits[d] == 3

    def test_skip_early_close_days(self):
        event = PeriodicEvent(rule=self.rule,
                               period=1,
                               skip_early_close_days=True)
        for dt in self.early_closes:
            market_mins = self.env.market_minutes_for_day(dt)
            for t in market_mins:
                assert event(t) is False

    def test_period(self):
        event = PeriodicEvent(rule=self.rule, period=2)
        # Only even indexed days should pass
        valid_days = set(self.trading_days[::2])
        count = 0
        results = set()
        for dt in self.trading_days:
            for t in self.market_mins[dt]:
                if event(t):
                    count += 1
                    results.add(dt)
        # Check max_daily_hits via count
        assert count == len(valid_days)
        assert results == valid_days
