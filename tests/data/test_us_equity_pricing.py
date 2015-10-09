#
# Copyright 2015 Quantopian, Inc.
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

from numpy import (
    arange,
    datetime64,
)
from pandas import (
    DataFrame,
    Timestamp,
)
from testfixtures import TempDirectory

from zipline.pipeline.loaders.synthetic import SyntheticDailyBarWriter
from zipline.finance.trading import TradingEnvironment

from zipline.data.us_equity_pricing import (
    BcolzDailyBarSpotReader,
    NoDataOnDate
)


# Test calendar ranges over the month of May and June 2015
#      May 2015
# Su Mo Tu We Th Fr Sa
#                1  2
#  3  4  5  6  7  8  9
# 10 11 12 13 14 15 16
# 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30
# 31


#      June 2015
# Mo Tu We Th Fr Sa Su
#  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14
# 15 16 17 18 19 20 21
# 22 23 24 25 26 27 28
# 29 30
TEST_CALENDAR_START = Timestamp('2015-05-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')


# One asset for each of the cases enumerated in load_raw_arrays_from_bcolz.
EQUITY_INFO = DataFrame(
    [
        # 1) The equity's trades start and end before query.
        {'start_date': '2015-06-01', 'end_date': '2015-06-05'},
        # 2) The equity's trades start and end after query.
        {'start_date': '2015-06-22', 'end_date': '2015-06-30'},
        # 3) The equity's data covers all dates in range.
        {'start_date': '2015-06-02', 'end_date': '2015-06-30'},
        # 4) The equity's trades start before the query start, but stop
        #    before the query end.
        {'start_date': '2015-06-01', 'end_date': '2015-06-15'},
        # 5) The equity's trades start and end during the query.
        {'start_date': '2015-06-12', 'end_date': '2015-06-18'},
        # 6) The equity's trades start during the query, but extend through
        #    the whole query.
        {'start_date': '2015-06-15', 'end_date': '2015-06-25'},
    ],
    index=arange(1, 7),
    columns=['start_date', 'end_date'],
).astype(datetime64)


class BcolzDailyBarSpotReaderTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        all_trading_days = TradingEnvironment().trading_days
        cls.trading_days = all_trading_days[
            all_trading_days.get_loc(TEST_CALENDAR_START):
            all_trading_days.get_loc(TEST_CALENDAR_STOP) + 1
        ]

    def setUp(self):

        self.asset_info = EQUITY_INFO
        self.writer = SyntheticDailyBarWriter(
            self.asset_info,
            self.trading_days,
        )

        self.dir_ = TempDirectory()
        self.dir_.create()
        self.dest = self.dir_.getpath('daily_equity_pricing.bcolz')

        self.writer.write(self.dest, self.trading_days, self.assets)

        self.daily_bar_spot_reader = BcolzDailyBarSpotReader(self.dest)

    def tearDown(self):
        self.dir_.cleanup()

    @property
    def assets(self):
        return self.asset_info.index

    def test_unadjusted_spot_price(self):
        # At beginning
        price = self.daily_bar_spot_reader.unadjusted_spot_price(
            1, Timestamp('2015-06-01', tz='UTC'))
        # Synthetic writes price for date.
        self.assertEqual(135630.0, price)

        # Middle
        price = self.daily_bar_spot_reader.unadjusted_spot_price(
            1, Timestamp('2015-06-02', tz='UTC'))
        self.assertEqual(135631.0, price)

        # End
        price = self.daily_bar_spot_reader.unadjusted_spot_price(
            1, Timestamp('2015-06-05', tz='UTC'))
        self.assertEqual(135634.0, price)

        # Another sid at beginning.
        price = self.daily_bar_spot_reader.unadjusted_spot_price(
            2, Timestamp('2015-06-22', tz='UTC'))
        self.assertEqual(235651.0, price)

    def test_unadjusted_spot_price_no_data(self):

        # before
        with self.assertRaises(NoDataOnDate):
            self.daily_bar_spot_reader.unadjusted_spot_price(
                1, Timestamp('2015-05-29', tz='UTC'))

        # after
        with self.assertRaises(NoDataOnDate):
            self.daily_bar_spot_reader.unadjusted_spot_price(
                1, Timestamp('2015-06-08', tz='UTC'))
