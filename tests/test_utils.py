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
import pandas as pd
import pytz
import numpy as np

import random
from zipline.finance.trading import TradingEnvironment

from zipline.utils.events import (
    AfterOpen,
    BeforeClose,
    AtTime,
    BetweenTimes,
    market_open,
    market_close,
)


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


class TestMarketTimingRules(TestCase):

    def setUp(self):
        env = TradingEnvironment.instance()
        # Select a random sample of 5 trading days
        index = random.sample(range(len(env.trading_days)), 5)
        test_dts = [env.trading_days[i] for i in index]
        self.open_close_times = [env.get_open_and_close(dt)
                                 for dt in test_dts]
        self.market_minutes = [env.market_minutes_for_day(dt)
                               for dt in test_dts]

    def test_AfterOpen(self):
        """
        Only the first 65 minutes in the trading day
        should evaluate to False.
        """
        after_open = AfterOpen(minutes=5, hours=1)
        for trading_day in self.market_minutes:
            for dt in trading_day[0:65]:
                assert after_open(dt) is False
            for dt in trading_day[65::]:
                assert after_open(dt) is True

    def test_BeforeClose(self):
        """
        Only the last 65 minutes in the trading day
        should evaluate to True.
        """
        before_close = BeforeClose(minutes=5, hours=1)
        for i, trading_day in enumerate(self.market_minutes):
            for dt in trading_day[0:-65]:
                assert before_close(dt) is False
            for dt in trading_day[-65::]:
                assert before_close(dt) is True

    def test_AtTime(self):
        # Only the 9th minute on any trading day should be True.
        at_time = AtTime(hour=9, minute=40, tz='US/Eastern')
        for minutes in self.market_minutes:
            for minute in minutes:
                if at_time(minute):
                    assert minutes.searchsorted(minute) == 9

    def test_BetweenTimes(self):
        # criteria: t1 <= dt < t2
        bw_times = BetweenTimes((9, 40), (9, 50), tz='US/Eastern')
        indexes = range(9, 19)
        for minutes in self.market_minutes:
            for minute in minutes:
                if bw_times(minute):
                    assert minutes.searchsorted(minute) in indexes

    def test_market_open(self):
        for i, minutes in enumerate(self.market_minutes):
            open_time = self.open_close_times[i][0]
            for minute in minutes:
                if market_open(minute):
                    assert minute == open_time

    def test_market_close(self):
        for i, minutes in enumerate(self.market_minutes):
            close_time = self.open_close_times[i][1]
            for minute in minutes:
                if market_close(minute):
                    assert minute == close_time
