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
from pandas.tslib import Timedelta

from zipline.data.data_portal import DataPortal
from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase
import pandas as pd


# Note: most of dataportal functionality is tested in various other places,
# such as test_history.

class TestDataPortal(WithTradingEnvironment, ZiplineTestCase):
    def init_instance_fixtures(self):
        super(TestDataPortal, self).init_instance_fixtures()

        self.data_portal = DataPortal(self.env.asset_finder,
                                      self.trading_calendar,
                                      first_trading_day=None)

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
