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
from collections import OrderedDict

from numpy import array, append, nan, full
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas.tslib import Timedelta

from zipline.assets import Equity
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithTradingSessions,
    WithDataPortal,
    alias,
)


class DataPortalTestBase(WithDataPortal,
                         WithTradingSessions,
                         ZiplineTestCase):

    ASSET_FINDER_EQUITY_SIDS = (1,)
    START_DATE = pd.Timestamp('2016-08-01')
    END_DATE = pd.Timestamp('2016-08-08')

    TRADING_CALENDAR_STRS = ('NYSE', 'CME')

    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True

    @classmethod
    def make_futures_info(cls):
        trading_sessions = cls.trading_sessions['CME']
        return pd.DataFrame({
            'sid': [10000],
            'root_symbol': ['BAR'],
            'symbol': ['BARA'],
            'start_date': [trading_sessions[1]],
            'end_date': [cls.END_DATE],
            # TODO: Make separate from 'end_date'
            'notice_date': [cls.END_DATE],
            'expiration_date': [cls.END_DATE],
            'multiplier': [500],
            'exchange': ['CME'],
        })

    @classmethod
    def make_equity_minute_bar_data(cls):
        trading_calendar = cls.trading_calendars[Equity]
        # No data on first day.
        dts = trading_calendar.minutes_for_session(cls.trading_days[0])
        dfs = []
        dfs.append(pd.DataFrame(
            {
                'open': full(len(dts), nan),
                'high': full(len(dts), nan),
                'low': full(len(dts), nan),
                'close': full(len(dts), nan),
                'volume': full(len(dts), 0),
            },
            index=dts))
        dts = trading_calendar.minutes_for_session(cls.trading_days[1])
        dfs.append(pd.DataFrame(
            {
                'open': append(100.5, full(len(dts) - 1, nan)),
                'high': append(100.9, full(len(dts) - 1, nan)),
                'low': append(100.1, full(len(dts) - 1, nan)),
                'close': append(100.3, full(len(dts) - 1, nan)),
                'volume': append(1000, full(len(dts) - 1, nan)),
            },
            index=dts))
        dts = trading_calendar.minutes_for_session(cls.trading_days[2])
        dfs.append(pd.DataFrame(
            {
                'open': [nan, 103.50, 102.50, 104.50, 101.50, nan],
                'high': [nan, 103.90, 102.90, 104.90, 101.90, nan],
                'low': [nan, 103.10, 102.10, 104.10, 101.10, nan],
                'close': [nan, 103.30, 102.30, 104.30, 101.30, nan],
                'volume': [0, 1003, 1002, 1004, 1001, 0]
            },
            index=dts[:6]
        ))
        dts = trading_calendar.minutes_for_session(cls.trading_days[3])
        dfs.append(pd.DataFrame(
            {
                'open': full(len(dts), nan),
                'high': full(len(dts), nan),
                'low': full(len(dts), nan),
                'close': full(len(dts), nan),
                'volume': full(len(dts), 0),
            },
            index=dts))
        yield 1, pd.concat(dfs)

    @classmethod
    def make_future_minute_bar_data(cls):
        asset = cls.asset_finder.retrieve_asset(10000)
        trading_calendar = cls.trading_calendars[asset.exchange]
        trading_sessions = cls.trading_sessions[asset.exchange]
        # No data on first day, future asset intentionally not on the same
        # dates as equities, so that cross-wiring of results do not create a
        # false positive.
        dts = trading_calendar.minutes_for_session(trading_sessions[1])
        dfs = []
        dfs.append(pd.DataFrame(
            {
                'open': full(len(dts), nan),
                'high': full(len(dts), nan),
                'low': full(len(dts), nan),
                'close': full(len(dts), nan),
                'volume': full(len(dts), 0),
            },
            index=dts))
        dts = trading_calendar.minutes_for_session(trading_sessions[2])
        dfs.append(pd.DataFrame(
            {
                'open': append(200.5, full(len(dts) - 1, nan)),
                'high': append(200.9, full(len(dts) - 1, nan)),
                'low': append(200.1, full(len(dts) - 1, nan)),
                'close': append(200.3, full(len(dts) - 1, nan)),
                'volume': append(2000, full(len(dts) - 1, nan)),
            },
            index=dts))
        dts = trading_calendar.minutes_for_session(trading_sessions[3])
        dfs.append(pd.DataFrame(
            {
                'open': [nan, 203.50, 202.50, 204.50, 201.50, nan],
                'high': [nan, 203.90, 202.90, 204.90, 201.90, nan],
                'low': [nan, 203.10, 202.10, 204.10, 201.10, nan],
                'close': [nan, 203.30, 202.30, 204.30, 201.30, nan],
                'volume': [0, 2003, 2002, 2004, 2001, 0]
            },
            index=dts[:6]
        ))
        dts = trading_calendar.minutes_for_session(trading_sessions[4])
        dfs.append(pd.DataFrame(
            {
                'open': full(len(dts), nan),
                'high': full(len(dts), nan),
                'low': full(len(dts), nan),
                'close': full(len(dts), nan),
                'volume': full(len(dts), 0),
            },
            index=dts))
        yield asset.sid, pd.concat(dfs)

    def test_get_last_traded_equity_minute(self):
        trading_calendar = self.trading_calendars[Equity]
        # Case: Missing data at front of data set, and request dt is before
        # first value.
        dts = trading_calendar.minutes_for_session(self.trading_days[0])
        asset = self.asset_finder.retrieve_asset(1)
        self.assertTrue(pd.isnull(
            self.data_portal.get_last_traded_dt(
                asset, dts[0], 'minute')))

        # Case: Data on requested dt.
        dts = trading_calendar.minutes_for_session(self.trading_days[2])

        self.assertEqual(dts[1],
                         self.data_portal.get_last_traded_dt(
                             asset, dts[1], 'minute'))

        # Case: No data on dt, but data occuring before dt.
        self.assertEqual(dts[4],
                         self.data_portal.get_last_traded_dt(
                             asset, dts[5], 'minute'))

    def test_get_last_traded_future_minute(self):
        asset = self.asset_finder.retrieve_asset(10000)
        trading_calendar = self.trading_calendars[asset.exchange]
        # Case: Missing data at front of data set, and request dt is before
        # first value.
        dts = trading_calendar.minutes_for_session(self.trading_days[0])
        self.assertTrue(pd.isnull(
            self.data_portal.get_last_traded_dt(
                asset, dts[0], 'minute')))

        # Case: Data on requested dt.
        dts = trading_calendar.minutes_for_session(self.trading_days[3])

        self.assertEqual(dts[1],
                         self.data_portal.get_last_traded_dt(
                             asset, dts[1], 'minute'))

        # Case: No data on dt, but data occuring before dt.
        self.assertEqual(dts[4],
                         self.data_portal.get_last_traded_dt(
                             asset, dts[5], 'minute'))

    def test_get_last_traded_dt_equity_daily(self):
        # Case: Missing data at front of data set, and request dt is before
        # first value.
        asset = self.asset_finder.retrieve_asset(1)
        self.assertTrue(pd.isnull(
            self.data_portal.get_last_traded_dt(
                asset, self.trading_days[0], 'daily')))

        # Case: Data on requested dt.
        self.assertEqual(self.trading_days[1],
                         self.data_portal.get_last_traded_dt(
                             asset, self.trading_days[1], 'daily'))

        # Case: No data on dt, but data occuring before dt.
        self.assertEqual(self.trading_days[2],
                         self.data_portal.get_last_traded_dt(
                             asset, self.trading_days[3], 'daily'))

    def test_get_spot_value_equity_minute(self):
        trading_calendar = self.trading_calendars[Equity]
        asset = self.asset_finder.retrieve_asset(1)
        dts = trading_calendar.minutes_for_session(self.trading_days[2])

        # Case: Get data on exact dt.
        dt = dts[1]
        expected = OrderedDict({
            'open': 103.5,
            'high': 103.9,
            'low': 103.1,
            'close': 103.3,
            'volume': 1003,
            'price': 103.3
        })
        result = [self.data_portal.get_spot_value(asset,
                                                  field,
                                                  dt,
                                                  'minute')
                  for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)

        # Case: Get data on empty dt, return nan or most recent data for price.
        dt = dts[100]
        expected = OrderedDict({
            'open': nan,
            'high': nan,
            'low': nan,
            'close': nan,
            'volume': 0,
            'price': 101.3
        })
        result = [self.data_portal.get_spot_value(asset,
                                                  field,
                                                  dt,
                                                  'minute')
                  for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)

    def test_get_spot_value_future_minute(self):
        trading_calendar = self.trading_calendars['CME']
        asset = self.asset_finder.retrieve_asset(10000)
        dts = trading_calendar.minutes_for_session(self.trading_days[3])

        # Case: Get data on exact dt.
        dt = dts[1]
        expected = OrderedDict({
            'open': 203.5,
            'high': 203.9,
            'low': 203.1,
            'close': 203.3,
            'volume': 2003,
            'price': 203.3
        })
        result = [self.data_portal.get_spot_value(asset,
                                                  field,
                                                  dt,
                                                  'minute')
                  for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)

        # Case: Get data on empty dt, return nan or most recent data for price.
        dt = dts[100]
        expected = OrderedDict({
            'open': nan,
            'high': nan,
            'low': nan,
            'close': nan,
            'volume': 0,
            'price': 201.3
        })
        result = [self.data_portal.get_spot_value(asset,
                                                  field,
                                                  dt,
                                                  'minute')
                  for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)

    def test_bar_count_for_simple_transforms(self):
        # July 2015
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31

        # half an hour into july 9, getting a 4-"day" window should get us
        # all the minutes of 7/6, 7/7, 7/8, and 31 minutes of 7/9

        july_9_dt = self.trading_calendar.open_and_close_for_session(
            pd.Timestamp("2015-07-09", tz='UTC')
        )[0] + Timedelta("30 minutes")

        self.assertEqual(
            (3 * 390) + 31,
            self.data_portal._get_minute_count_for_transform(july_9_dt, 4)
        )

        #    November 2015
        # Su Mo Tu We Th Fr Sa
        #  1  2  3  4  5  6  7
        #  8  9 10 11 12 13 14
        # 15 16 17 18 19 20 21
        # 22 23 24 25 26 27 28
        # 29 30

        # nov 26th closed
        # nov 27th was an early close

        # half an hour into nov 30, getting a 4-"day" window should get us
        # all the minutes of 11/24, 11/25, 11/27 (half day!), and 31 minutes
        # of 11/30
        nov_30_dt = self.trading_calendar.open_and_close_for_session(
            pd.Timestamp("2015-11-30", tz='UTC')
        )[0] + Timedelta("30 minutes")

        self.assertEqual(
            390 + 390 + 210 + 31,
            self.data_portal._get_minute_count_for_transform(nov_30_dt, 4)
        )

    def test_get_last_traded_dt_minute(self):
        minutes = self.nyse_calendar.minutes_for_session(
            self.trading_days[2])
        equity = self.asset_finder.retrieve_asset(1)
        result = self.data_portal.get_last_traded_dt(equity,
                                                     minutes[3],
                                                     'minute')
        self.assertEqual(minutes[3], result,
                         "Asset 1 had a trade on third minute, so should "
                         "return that as the last trade on that dt.")

        result = self.data_portal.get_last_traded_dt(equity,
                                                     minutes[5],
                                                     'minute')
        self.assertEqual(minutes[4], result,
                         "Asset 1 had a trade on fourth minute, so should "
                         "return that as the last trade on the fifth.")

        future = self.asset_finder.retrieve_asset(10000)
        calendar = self.trading_calendars[future.exchange]
        minutes = calendar.minutes_for_session(self.trading_days[3])
        result = self.data_portal.get_last_traded_dt(future,
                                                     minutes[3],
                                                     'minute')

        self.assertEqual(minutes[3], result,
                         "Asset 10000 had a trade on the third minute, so "
                         "return that as the last trade on that dt.")

        result = self.data_portal.get_last_traded_dt(future,
                                                     minutes[5],
                                                     'minute')
        self.assertEqual(minutes[4], result,
                         "Asset 10000 had a trade on fourth minute, so should "
                         "return that as the last trade on the fifth.")


class TestDataPortal(DataPortalTestBase):
    DATA_PORTAL_LAST_AVAILABLE_SESSION = None
    DATA_PORTAL_LAST_AVAILABLE_MINUTE = None


class TestDataPortalExplicitLastAvailable(DataPortalTestBase):
    DATA_PORTAL_LAST_AVAILABLE_SESSION = alias('START_DATE')
    DATA_PORTAL_LAST_AVAILABLE_MINUTE = alias('END_DATE')
