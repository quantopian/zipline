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

import datetime
from unittest import TestCase
from zipline.utils import tradingcalendar
from zipline.utils.factory import (load_from_yahoo,
                                   load_bars_from_yahoo)
from zipline.utils.event_management import(
    AfterOpen,
    BeforeClose,
    AtTime,
    BetweenTimes,
    at_market_open,
    at_market_close
)
import pandas as pd
import pytz
import numpy as np


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
        open_close = tradingcalendar.open_and_closes.iloc[0]
        self.open_time = open_close['market_open']
        self.close_time = open_close['market_close']
        self.interval = datetime.timedelta(minutes=1)
        market_mins = []
        dt = self.open_time
        while dt <= self.close_time:
            market_mins.append(dt)
            dt += self.interval
        self.market_mins = market_mins

    def test_AfterOpen(self):
        """
        Only the first 65 minutes in the trading day
        should evaluate to False.
        """
        after_open = AfterOpen(minutes=5, hours=1)
        results = pd.Series(
            {dt: after_open(dt) for dt in self.market_mins}
        )
        invalid_times = results[results == False]
        assert len(invalid_times) == 65
        assert np.all(invalid_times.index == self.market_mins[0:65])

    def test_BeforeClose(self):
        """
        Only the last 65 minutes in the trading day
        should evaluate to True.
        """
        before_close = BeforeClose(minutes=5, hours=1)
        results = pd.Series(
            {dt: before_close(dt) for dt in self.market_mins}
        )
        valid_times = results[results == True]
        assert len(valid_times) == 65
        assert np.all(valid_times.index == self.market_mins[-65::])

    def test_AtTime(self):
        # 10:00 EST / 15:00 UTC test date
        testdt = pd.Timestamp('1990-01-02 15:00:00+0000', tz='UTC')
        at_time = AtTime(hour=10, minute=0, tz='US/Eastern')
        results = pd.Series({i: at_time(i) for i in self.market_mins})
        valid_times = results[results == True]
        assert len(valid_times) == 1
        assert testdt in valid_times

    def test_BetweenTimes(self):
        # 10:00 EST / 15:00 UTC to 5 min later test times
        bw_times = BetweenTimes((10, 0), (10, 5), tz='US/Eastern')
        testdts = pd.Index([
            pd.Timestamp('1990-01-02 15:00:00+0000', tz='UTC'),
            pd.Timestamp('1990-01-02 15:01:00+0000', tz='UTC'),
            pd.Timestamp('1990-01-02 15:02:00+0000', tz='UTC'),
            pd.Timestamp('1990-01-02 15:03:00+0000', tz='UTC'),
            pd.Timestamp('1990-01-02 15:04:00+0000', tz='UTC')
        ])
        results = pd.Series({i: bw_times(i) for i in self.market_mins})
        valid_times = results[results == True]
        assert len(valid_times) == 5
        assert np.all(testdts == valid_times.index)

    def test_at_market_open(self):
        results = pd.Series({dt: at_market_open(dt)
                             for dt in self.market_mins})
        valid_times = results[results == True]
        assert len(valid_times) == 1
        assert self.open_time in valid_times

    def test_at_market_close(self):
        results = pd.Series({dt: at_market_close(dt)
                             for dt in self.market_mins})
        valid_times = results[results == True]
        assert len(valid_times) == 1
        assert self.close_time in valid_times
