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
"""
Tests for zipline.data.ffc.loaders.us_equity_pricing
"""
from unittest import TestCase

from nose_parameterized import parameterized
from numpy import (
    arange,
    datetime64,
    float64,
    full,
    uint32,
)
from numpy.testing import assert_array_equal
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Timedelta,
    Timestamp,
)
from pandas.util.testing import assert_index_equal
from testfixtures import TempDirectory

from zipline.data.adjustment import Float64Multiply
from zipline.data.equities import USEquityPricing
from zipline.data.ffc.synthetic import (
    NullAdjustmentReader,
    SyntheticDailyBarWriter,
)
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzDailyBarReader,
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
)
from zipline.finance.trading import TradingEnvironment

# Test calendar ranges over the month of June 2015
#      June 2015
# Mo Tu We Th Fr Sa Su
#  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14
# 15 16 17 18 19 20 21
# 22 23 24 25 26 27 28
# 29 30
TEST_CALENDAR_START = Timestamp('2015-06-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')

TEST_QUERY_START = Timestamp('2015-06-10', tz='UTC')
TEST_QUERY_STOP = Timestamp('2015-06-19', tz='UTC')
TEST_QUERY_COLUMNS = [USEquityPricing.close, USEquityPricing.volume]

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

TEST_QUERY_ASSETS = EQUITY_INFO.index
EPOCH = pd.Timestamp(0, tz='UTC')


def str_to_seconds(s):
    return (Timestamp(s, tz='UTC') - EPOCH).total_seconds()


def seconds_to_timestamp(seconds):
    return EPOCH + Timedelta(seconds=seconds)


class DailyBarReaderWriterTestCase(TestCase):

    def setUp(self):
        all_trading_days = TradingEnvironment.instance().trading_days
        self.trading_days = all_trading_days[
            all_trading_days.get_loc(TEST_CALENDAR_START):
            all_trading_days.get_loc(TEST_CALENDAR_STOP) + 1
        ]

        self.asset_info = EQUITY_INFO
        self.writer = SyntheticDailyBarWriter(
            self.asset_info,
            self.trading_days,
        )
        self.dir_ = TempDirectory()
        self.dir_.create()
        self.dest = self.dir_.getpath('daily_equity_pricing.bcolz')

    def tearDown(self):
        self.dir_.cleanup()

    @property
    def assets(self):
        return self.asset_info.index

    def trading_days_between(self, start, end):
        return self.trading_days[self.trading_days.slice_indexer(start, end)]

    def asset_start(self, asset_id):
        return self.asset_info.loc[asset_id]['start_date'].tz_localize('UTC')

    def asset_end(self, asset_id):
        return self.asset_info.loc[asset_id]['end_date'].tz_localize('UTC')

    def dates_for_asset(self, asset_id):
        start, end = self.asset_start(asset_id), self.asset_end(asset_id)
        return self.trading_days_between(start, end)

    def test_write_ohlcv_content(self):
        result = self.writer.write(self.dest, self.trading_days, self.assets)
        for column in SyntheticDailyBarWriter.OHLCV:
            idx = 0
            data = result[column][:]
            multiplier = 1 if column == 'volume' else 1000
            for asset_id in self.assets:
                for date in self.dates_for_asset(asset_id):
                    self.assertEqual(
                        SyntheticDailyBarWriter.expected_value(
                            asset_id,
                            date,
                            column
                        ) * multiplier,
                        data[idx],
                    )
                    idx += 1
            self.assertEqual(idx, len(data))

    def test_write_day_and_id(self):
        result = self.writer.write(self.dest, self.trading_days, self.assets)
        idx = 0
        ids = result['id']
        days = result['day']
        for asset_id in self.assets:
            for date in self.dates_for_asset(asset_id):
                self.assertEqual(ids[idx], asset_id)
                self.assertEqual(date, seconds_to_timestamp(days[idx]))
                idx += 1

    def test_write_attrs(self):
        result = self.writer.write(self.dest, self.trading_days, self.assets)
        expected_first_row = {
            '1': 0,
            '2': 5,   # Asset 1 has 5 trading days.
            '3': 12,  # Asset 2 has 7 trading days.
            '4': 33,  # Asset 3 has 21 trading days.
            '5': 44,  # Asset 4 has 11 trading days.
            '6': 49,  # Asset 5 has 5 trading days.
        }
        expected_last_row = {
            '1': 4,
            '2': 11,
            '3': 32,
            '4': 43,
            '5': 48,
            '6': 57,    # Asset 6 has 9 trading days.
        }
        expected_calendar_offset = {
            '1': 0,   # Starts on 6-01, 1st trading day of month.
            '2': 15,  # Starts on 6-22, 16th trading day of month.
            '3': 1,   # Starts on 6-02, 2nd trading day of month.
            '4': 0,   # Starts on 6-01, 1st trading day of month.
            '5': 9,   # Starts on 6-12, 10th trading day of month.
            '6': 10,  # Starts on 6-15, 11th trading day of month.
        }
        self.assertEqual(result.attrs['first_row'], expected_first_row)
        self.assertEqual(result.attrs['last_row'], expected_last_row)
        self.assertEqual(
            result.attrs['calendar_offset'],
            expected_calendar_offset,
        )
        assert_index_equal(
            self.trading_days,
            DatetimeIndex(result.attrs['calendar'], tz='UTC'),
        )

    def expected_read_values(self, dates, assets, column):
        if column == 'volume':
            dtype = uint32
            missing = 0
        else:
            dtype = float64
            missing = float('nan')

        data = full((len(dates), len(assets)), missing, dtype=dtype)
        for j, asset in enumerate(assets):
            start = self.asset_start(asset)
            stop = self.asset_end(asset)
            for i, date in enumerate(dates):
                # No value expected for dates outside the asset's start/end
                # date.
                if not (start <= date <= stop):
                    continue
                data[i, j] = SyntheticDailyBarWriter.expected_value(
                    asset,
                    date,
                    column,
                )
        return data

    def _check_read_results(self, columns, assets, start_date, end_date):
        table = self.writer.write(self.dest, self.trading_days, self.assets)
        reader = BcolzDailyBarReader(table)
        dates = self.trading_days_between(start_date, end_date)
        results = reader.load_raw_arrays(columns, dates, assets)
        for column, result in zip(columns, results):
            assert_array_equal(
                result,
                self.expected_read_values(dates, assets, column.name),
            )

    @parameterized.expand([
        ([USEquityPricing.open],),
        ([USEquityPricing.close, USEquityPricing.volume],),
        ([USEquityPricing.volume, USEquityPricing.high, USEquityPricing.low],),
        (USEquityPricing.columns,),
    ])
    def test_read(self, columns):
        self._check_read_results(
            columns,
            self.assets,
            TEST_QUERY_START,
            TEST_QUERY_STOP,
        )

    def test_start_on_asset_start(self):
        """
        Test loading with queries that starts on the first day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.high, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_start(asset),
                end_date=self.trading_days[-1],
            )

    def test_start_on_asset_end(self):
        """
        Test loading with queries that start on the last day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.close, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_end(asset),
                end_date=self.trading_days[-1],
            )

    def test_end_on_asset_start(self):
        """
        Test loading with queries that end on the first day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.close, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.trading_days[0],
                end_date=self.asset_start(asset),
            )

    def test_end_on_asset_end(self):
        """
        Test loading with queries that end on the last day of each asset's
        lifetime.
        """
        columns = [USEquityPricing.close, USEquityPricing.volume]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.trading_days[0],
                end_date=self.asset_end(asset),
            )


# ADJUSTMENTS use the following scheme to indicate information about the value
# upon inspection.
#
# 1s place is the equity
#
# 0.1s place is the action type, with:
#
# splits, 1
# mergers, 2
# dividends, 3
#
# 0.001s is the date
SPLITS = DataFrame(
    [
        # Before query range, should be excluded.
        {'effective_date': str_to_seconds('2015-06-03'),
         'ratio': 1.103,
         'sid': 1},
        # First day of query range, should have last_row of 0
        {'effective_date': str_to_seconds('2015-06-10'),
         'ratio': 1.110,
         'sid': 1},
        # Third day of query range, should have last_row of 2
        {'effective_date': str_to_seconds('2015-06-12'),
         'ratio': 1.112,
         'sid': 1},
        # After query range, should be excluded.
        {'effective_date': str_to_seconds('2015-06-21'),
         'ratio': 1.121,
         'sid': 1},
        # Another action in query range, should have last_row of 1
        {'effective_date': str_to_seconds('2015-06-11'),
         'ratio': 2.111,
         'sid': 2},
        # Last day of range.  Should have last_row of 7
        {'effective_date': str_to_seconds('2015-06-19'),
         'ratio': 3.119,
         'sid': 3},
    ]
)


MERGERS = DataFrame(
    [
        # Before query range, should be excluded.
        {'effective_date': str_to_seconds('2015-06-03'),
         'ratio': 1.203,
         'sid': 1},
        # First day of query range, should have last_row of 0
        {'effective_date': str_to_seconds('2015-06-10'),
         'ratio': 1.210,
         'sid': 1},
        # Third day of query range, should have last_row of 2
        {'effective_date': str_to_seconds('2015-06-12'),
         'ratio': 1.212,
         'sid': 1},
        # After query range, should be excluded.
        {'effective_date': str_to_seconds('2015-06-26'),
         'ratio': 1.226,
         'sid': 1},
        # Another action in query range, should have last_row of 2
        {'effective_date': str_to_seconds('2015-06-12'),
         'ratio': 2.212,
         'sid': 2},
        # Last day of range.  Should have last_row of 7
        {'effective_date': str_to_seconds('2015-06-19'),
         'ratio': 3.219,
         'sid': 3},
    ],
)

DIVIDENDS = DataFrame(
    [
        # Before query range, should be excluded.
        {'effective_date': str_to_seconds('2015-06-01'),
         'ratio': 1.301,
         'sid': 1},
        # First day of query range, should have last_row of 0
        {'effective_date': str_to_seconds('2015-06-10'),
         'ratio': 1.310,
         'sid': 1},
        # Third day of query range, should have last_row of 2
        {'effective_date': str_to_seconds('2015-06-12'),
         'ratio': 1.312,
         'sid': 1},
        # After query range, should be excluded.
        {'effective_date': str_to_seconds('2015-06-25'),
         'ratio': 1.325,
         'sid': 1},
        # Another action in query range, should have last_row of 3
        {'effective_date': str_to_seconds('2015-06-15'),
         'ratio': 2.315,
         'sid': 2},
        # Last day of range.  Should have last_row of 7
        {'effective_date': str_to_seconds('2015-06-19'),
         'ratio': 3.319,
         'sid': 3},
    ],
)


class UsEquityPricingLoaderTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = TempDirectory()
        cls.db_path = cls.test_data_dir.getpath('adjustments.db')
        writer = SQLiteAdjustmentWriter(cls.db_path)
        writer.write(SPLITS, MERGERS, DIVIDENDS)

        cls.assets = TEST_QUERY_ASSETS
        all_trading_days = TradingEnvironment.instance().trading_days
        cls.trading_days = all_trading_days[
            all_trading_days.slice_indexer(TEST_QUERY_START, TEST_QUERY_STOP)
        ]

    @classmethod
    def tearDownClass(cls):
        cls.test_data_dir.cleanup()

    def test_load_adjustments_from_sqlite(self):
        reader = SQLiteAdjustmentReader(self.db_path)
        adjustments = reader.load_adjustments(
            TEST_QUERY_COLUMNS,
            self.trading_days,
            self.assets,
        )

        close_adjustments = adjustments[0]
        volume_adjustments = adjustments[1]

        # See SPLITS, MERGERS, DIVIDENDS module variables for details of
        # expected values.
        EXPECTED_CLOSES = {
            # 2015-06-10
            0: [
                Float64Multiply(first_row=0, last_row=0, col=0, value=1.110),
                Float64Multiply(first_row=0, last_row=0, col=0, value=1.210),
                Float64Multiply(first_row=0, last_row=0, col=0, value=1.310),
            ],
            # 2015-06-11
            1: [
                Float64Multiply(first_row=0, last_row=1, col=1, value=2.111),
            ],
            # 2015-06-12
            2: [
                Float64Multiply(first_row=0, last_row=2, col=0, value=1.112),
                Float64Multiply(first_row=0, last_row=2, col=0, value=1.212),
                Float64Multiply(first_row=0, last_row=2, col=1, value=2.212),
                Float64Multiply(first_row=0, last_row=2, col=0, value=1.312),
            ],
            # 2015-06-15
            3: [
                Float64Multiply(first_row=0, last_row=3, col=1, value=2.315),
            ],
            # 2015-06-19
            7: [
                Float64Multiply(first_row=0, last_row=7, col=2, value=3.119),
                Float64Multiply(first_row=0, last_row=7, col=2, value=3.219),
                Float64Multiply(first_row=0, last_row=7, col=2, value=3.319),
            ]
        }

        # Volume adjustments should only have data for splits.
        EXPECTED_VOLUMES = {
            # 2015-06-10
            0: [
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.0 / 1.110
                ),
            ],
            # 2015-06-11
            1: [
                Float64Multiply(
                    first_row=0, last_row=1, col=1, value=1.0 / 2.111,
                )
            ],
            # 2015-06-12
            2: [
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.0 / 1.112,
                )
            ],
            7: [
                Float64Multiply(
                    first_row=0, last_row=7, col=2, value=1.0 / 3.119,
                )
            ],
        }

        self.assertEqual(close_adjustments, EXPECTED_CLOSES)
        self.assertEqual(volume_adjustments, EXPECTED_VOLUMES)

    def test_null_adjustments(self):
        reader = NullAdjustmentReader()
        adjustments = reader.load_adjustments(
            TEST_QUERY_COLUMNS,
            self.trading_days,
            self.assets,
        )
        self.assertEqual(adjustments, [{}, {}])
