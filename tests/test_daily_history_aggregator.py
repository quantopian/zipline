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
from numbers import Real

from nose_parameterized import parameterized
from numpy.testing import assert_almost_equal
from numpy import nan
import pandas as pd

from zipline.data.data_portal import DailyHistoryAggregator

from zipline.testing.fixtures import (
    WithBcolzEquityMinuteBarReader,
    ZiplineTestCase,
)

OHLC = ['open', 'high', 'low', 'close']
OHLCV = OHLC + ['volume']


class MinuteToDailyAggregationTestCase(WithBcolzEquityMinuteBarReader,
                                       ZiplineTestCase):

    #    March 2016
    # Su Mo Tu We Th Fr Sa
    #        1  2  3  4  5
    #  6  7  8  9 10 11 12
    # 13 14 15 16 17 18 19
    # 20 21 22 23 24 25 26
    # 27 28 29 30 31

    TRADING_ENV_MIN_DATE = START_DATE = pd.Timestamp(
        '2016-03-01', tz='UTC',
    )
    TRADING_ENV_MAX_DATE = END_DATE = pd.Timestamp(
        '2016-03-31', tz='UTC',
    )
    ASSET_FINDER_EQUITY_SIDS = 1, 2

    minutes = pd.date_range('2016-03-15 9:31',
                            '2016-03-15 9:36',
                            freq='min',
                            tz='US/Eastern').tz_convert('UTC')

    @classmethod
    def make_equity_minute_bar_data(cls):
        # sid data is created so that at least one high is lower than a
        # previous high, and the inverse for low
        yield 1, pd.DataFrame(
            {
                'open': [nan, 103.50, 102.50, 104.50, 101.50, nan],
                'high': [nan, 103.90, 102.90, 104.90, 101.90, nan],
                'low': [nan, 103.10, 102.10, 104.10, 101.10, nan],
                'close': [nan, 103.30, 102.30, 104.30, 101.30, nan],
                'volume': [0, 1003, 1002, 1004, 1001, 0]
            },
            index=cls.minutes,
        )
        # sid 2 is included to provide data on different bars than sid 1,
        # as will as illiquidty mid-day
        yield 2, pd.DataFrame(
            {
                'open': [201.50, nan, 204.50, nan, 200.50, 202.50],
                'high': [201.90, nan, 204.90, nan, 200.90, 202.90],
                'low': [201.10, nan, 204.10, nan, 200.10, 202.10],
                'close': [201.30, nan, 203.50, nan, 200.30, 202.30],
                'volume': [2001, 0, 2004, 0, 2000, 2002],
            },
            index=cls.minutes,
        )

    expected_values = {
        1: pd.DataFrame(
            {
                'open': [nan, 103.50, 103.50, 103.50, 103.50, 103.50],
                'high': [nan, 103.90, 103.90, 104.90, 104.90, 104.90],
                'low': [nan, 103.10, 102.10, 102.10, 101.10, 101.10],
                'close': [nan, 103.30, 102.30, 104.30, 101.30, 101.30],
                'volume': [0, 1003, 2005, 3009, 4010, 4010]
            },
            index=minutes,
        ),
        2: pd.DataFrame(
            {
                'open': [201.50, 201.50, 201.50, 201.50, 201.50, 201.50],
                'high': [201.90, 201.90, 204.90, 204.90, 204.90, 204.90],
                'low': [201.10, 201.10, 201.10, 201.10, 200.10, 200.10],
                'close': [201.30, 201.30, 203.50, 203.50, 200.30, 202.30],
                'volume': [2001, 2001, 4005, 4005, 6005, 8007],
            },
            index=minutes,
        )
    }

    @classmethod
    def init_class_fixtures(cls):
        super(MinuteToDailyAggregationTestCase, cls).init_class_fixtures()

        cls.EQUITIES = {
            1: cls.env.asset_finder.retrieve_asset(1),
            2: cls.env.asset_finder.retrieve_asset(2)
        }

    def init_instance_fixtures(self):
        super(MinuteToDailyAggregationTestCase, self).init_instance_fixtures()
        # Set up a fresh data portal for each test, since order of calling
        # needs to be tested.
        self.equity_daily_aggregator = DailyHistoryAggregator(
            self.trading_calendar.schedule.market_open,
            self.bcolz_equity_minute_bar_reader,
            self.trading_calendar
        )

    @parameterized.expand([
        ('open_sid_1', 'open', 1),
        ('high_1', 'high', 1),
        ('low_1', 'low', 1),
        ('close_1', 'close', 1),
        ('volume_1', 'volume', 1),
        ('open_2', 'open', 2),
        ('high_2', 'high', 2),
        ('low_2', 'low', 2),
        ('close_2', 'close', 2),
        ('volume_2', 'volume', 2),

    ])
    def test_contiguous_minutes_individual(self, name, field, sid):
        # First test each minute in order.
        method_name = field + 's'
        results = []
        repeat_results = []
        asset = self.EQUITIES[sid]
        for minute in self.minutes:
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            results.append(value)

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            repeat_results.append(value)

        assert_almost_equal(results, self.expected_values[asset][field],
                            err_msg='sid={0} field={1}'.format(asset, field))
        assert_almost_equal(repeat_results, self.expected_values[asset][field],
                            err_msg='sid={0} field={1}'.format(asset, field))

    @parameterized.expand([
        ('open_sid_1', 'open', 1),
        ('high_1', 'high', 1),
        ('low_1', 'low', 1),
        ('close_1', 'close', 1),
        ('volume_1', 'volume', 1),
        ('open_2', 'open', 2),
        ('high_2', 'high', 2),
        ('low_2', 'low', 2),
        ('close_2', 'close', 2),
        ('volume_2', 'volume', 2),

    ])
    def test_skip_minutes_individual(self, name, field, sid):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        for i in [1, 5]:
            minute = self.minutes[i]
            asset = self.EQUITIES[sid]
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            assert_almost_equal(value,
                                self.expected_values[sid][field][i],
                                err_msg='sid={0} field={1} dt={2}'.format(
                                    sid, field, minute))

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            value = getattr(self.equity_daily_aggregator, method_name)(
                [asset], minute)[0]
            # Prevent regression on building an array when scalar is intended.
            self.assertIsInstance(value, Real)
            assert_almost_equal(value,
                                self.expected_values[sid][field][i],
                                err_msg='sid={0} field={1} dt={2}'.format(
                                    sid, field, minute))

    @parameterized.expand(OHLCV)
    def test_contiguous_minutes_multiple(self, field):
        # First test each minute in order.
        method_name = field + 's'
        assets = sorted(self.EQUITIES.values())
        results = {asset: [] for asset in assets}
        repeat_results = {asset: [] for asset in assets}
        for minute in self.minutes:
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                results[asset].append(value)

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                repeat_results[asset].append(value)
        for asset in assets:
            assert_almost_equal(results[asset],
                                self.expected_values[asset][field],
                                err_msg='sid={0} field={1}'.format(
                                    asset, field))
            assert_almost_equal(repeat_results[asset],
                                self.expected_values[asset][field],
                                err_msg='sid={0} field={1}'.format(
                                    asset, field))

    @parameterized.expand(OHLCV)
    def test_skip_minutes_multiple(self, field):
        # Test skipping minutes, to exercise backfills.
        # Tests initial backfill and mid day backfill.
        method_name = field + 's'
        assets = sorted(self.EQUITIES.values())
        for i in [1, 5]:
            minute = self.minutes[i]
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                assert_almost_equal(
                    value,
                    self.expected_values[asset][field][i],
                    err_msg='sid={0} field={1} dt={2}'.format(
                        asset, field, minute))

            # Call a second time with the same dt, to prevent regression
            # against case where crossed start and end dts caused a crash
            # instead of the last value.
            values = getattr(self.equity_daily_aggregator, method_name)(
                assets, minute)
            for j, asset in enumerate(assets):
                value = values[j]
                # Prevent regression on building an array when scalar is
                # intended.
                self.assertIsInstance(value, Real)
                assert_almost_equal(
                    value,
                    self.expected_values[asset][field][i],
                    err_msg='sid={0} field={1} dt={2}'.format(
                        asset, field, minute))
