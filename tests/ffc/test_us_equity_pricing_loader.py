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
from collections import OrderedDict
import os
import sqlite3
from unittest import TestCase

import bcolz
import numpy as np
import pandas as pd
from testfixtures import TempDirectory

from zipline.data.adjustment import Float64Multiply
from zipline.data.equities import USEquityPricing
from zipline.data.ffc.loaders.us_equity_pricing import (
    BcolzRawPriceLoader,
    SQLiteAdjustmentLoader,
)

# Custom trading calendar for us equity test.
TEST_TRADING_DAYS = pd.date_range('2015-05-31', '2015-06-10', tz='UTC')

# Query is smaller than the entire trading days, so that equities that go
# beyond the range are tested..
TEST_QUERY_RANGE = pd.date_range('2015-06-04', '2015-06-08', tz='UTC')
TEST_QUERY_COLUMNS = [USEquityPricing.close, USEquityPricing.volume]

# The keys are the asset id.
EQUITY_INFO = OrderedDict((
    # 1) This equity's data covers all dates in range.
    (1, {
        'start_date': '2015-06-01',
        'end_date': '2015-06-10',
    }),
    # 2) The equity's trades are all before the start of the query.
    (2, {
        'start_date': '2015-06-01',
        'end_date': '2015-06-03',
    }),
    # 3) The equity's trades start before the query start, but stop
    #    before the query end.
    (3, {
        'start_date': '2015-06-01',
        'end_date': '2015-06-05',
    }),
    # 4) The equity's trades start after query start and ends before
    #    the query end.
    (4, {
        'start_date': '2015-06-05',
        'end_date': '2015-06-06',
    }),
    # 5) The equity's trades start after query start, but trade through or
    #    past the query end
    (5, {
        'start_date': '2015-06-05',
        'end_date': '2015-06-10',
    }),
    # 6) The equity's trades start and end after query end.
    (6, {
        'start_date': '2015-06-09',
        'end_date': '2015-06-10',
    }),
))

TEST_QUERY_ASSETS = pd.Int64Index(EQUITY_INFO.keys())

# price type identifiers
PT_OPEN, PT_HIGH, PT_LOW, PT_CLOSE, PT_VOLUME = range(1000, 6000, 1000)


def create_bcolz_data(test_data_dir):
    sid_col = []
    days = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    start_pos = {}
    start_day_offset = {}
    end_day_offset = {}

    for asset, info in EQUITY_INFO.iteritems():
        asset_day_range = pd.date_range(info['start_date'],
                                        info['end_date'])
        asset_len = len(asset_day_range)
        start_pos[asset] = len(sid_col)
        sid_col.extend([asset] * asset_len)

        start_day_offset[asset] = TEST_TRADING_DAYS.searchsorted(
            asset_day_range[0])
        end_day_offset[asset] = TEST_TRADING_DAYS.searchsorted(
            asset_day_range[-1])

        for day in asset_day_range:
            days.append(int(day.strftime("%s")))
        # Prices are 1000 times the equity float, except for volume which is
        # the integer of the float.
        #
        # Create synthetic prices that code information about the price.
        # The  10000 place is the asset id
        # The   1000 place is the price type, i.e. OHLCV
        # The    100 place is the row position of the assets date range
        #            starting at 1 for the first day
        asset_place = int(asset * 10000)
        # Create the row identifier place
        for i in range(asset_len):
            row_id = i + 1
            opens.append(asset_place + PT_OPEN + row_id)
            highs.append(asset_place + PT_HIGH + row_id)
            lows.append(asset_place + PT_LOW + row_id)
            volumes.append(asset_place + PT_VOLUME + row_id)
            closes.append(asset_place + PT_CLOSE + row_id)

    bcolz_path = os.path.join(test_data_dir, 'equity_test_daily_bars.bcolz')

    table = bcolz.ctable(
        names=[
            'open',
            'high',
            'low',
            'close',
            'volume',
            'day',
            'sid'],
        columns=[
            np.array(opens).astype(np.uint32),
            np.array(highs).astype(np.uint32),
            np.array(lows).astype(np.uint32),
            np.array(closes).astype(np.uint32),
            np.array(volumes).astype(np.uint32),
            np.array(days).astype(np.uint32),
            np.array(sid_col).astype(np.uint32),
        ],
        rootdir=bcolz_path,
    )
    table.attrs['start_pos'] = {str(k): v for k, v
                                in start_pos.iteritems()}
    table.attrs['start_day_offset'] = {str(k): v for k, v
                                       in start_day_offset.iteritems()}
    table.attrs['end_day_offset'] = {str(k): v for k, v
                                     in end_day_offset.iteritems()}
    table.flush()

    return bcolz_path


# Adjustments use the following scheme to indicate information about the value
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

SPLITS = [
    # Before query range, should be excluded.
    {'effective_date': '2015-06-03',
     'ratio': 1.103,
     'sid': 1},
    # First day of query range, should have last_row of 0
    {'effective_date': '2015-06-04',
     'ratio': 1.104,
     'sid': 1},
    # Second day of query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 1.106,
     'sid': 1},
    # After query range, should be excluded.
    {'effective_date': '2015-06-09',
     'ratio': 1.109,
     'sid': 1},
    # Another action in query range, should have last_row of 1
    {'effective_date': '2015-06-05',
     'ratio': 2.105,
     'sid': 2},
]


MERGERS = [
    # Before query range, should be excluded.
    {'effective_date': '2015-06-03',
     'ratio': 1.203,
     'sid': 1},
    # First day of query range, should have last_row of 0
    {'effective_date': '2015-06-04',
     'ratio': 1.204,
     'sid': 1},
    # Second day of query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 1.206,
     'sid': 1},
    # After query range, should be excluded.
    {'effective_date': '2015-06-09',
     'ratio': 1.209,
     'sid': 1},
    # Another action in query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 2.206,
     'sid': 2},
]


DIVIDENDS = [
    # Before query range, should be excluded.
    {'effective_date': '2015-06-03',
     'ratio': 1.303,
     'sid': 1},
    # First day of query range, should have last_row of 0
    {'effective_date': '2015-06-04',
     'ratio': 1.304,
     'sid': 1},
    # Second day of query range, should have last_row of 2
    {'effective_date': '2015-06-06',
     'ratio': 1.306,
     'sid': 1},
    # After query range, should be excluded.
    {'effective_date': '2015-06-09',
     'ratio': 1.309,
     'sid': 1},
    # Another action in query range, should have last_row of 3
    {'effective_date': '2015-06-07',
     'ratio': 2.307,
     'sid': 2},
]

EPOCH = pd.Timestamp(0, tz='UTC')


def date_col_to_seconds(date_column):
    return date_column.apply(
        lambda x: (pd.Timestamp(x, tz='UTC') - EPOCH).total_seconds())


def create_adjustments_data(test_data_dir):
    db_path = os.path.join(test_data_dir, 'adjustments.db')
    conn = sqlite3.connect(db_path)

    splits_df = pd.DataFrame(SPLITS)
    splits_df['effective_date'] = date_col_to_seconds(
        splits_df['effective_date'])

    mergers_df = pd.DataFrame(MERGERS)
    mergers_df['effective_date'] = date_col_to_seconds(
        mergers_df['effective_date'])

    dividends_df = pd.DataFrame(DIVIDENDS)
    dividends_df['effective_date'] = date_col_to_seconds(
        dividends_df['effective_date'])

    splits_df.to_sql('splits', conn)
    mergers_df.to_sql('mergers', conn)
    dividends_df.to_sql('dividends', conn)

    c = conn.cursor()
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='dividends_sid', tn='dividends', cn='sid'))
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='dividends_pay_date', tn='dividends', cn='effective_date'))

    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='mergers_sid', tn='mergers', cn='sid'))
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='mergers_effective_date', tn='mergers', cn='effective_date'))

    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='splits_sid', tn='splits', cn='sid'))
    c.execute('CREATE INDEX {ix} on {tn}({cn})'.format(
        ix='splits_effective_date', tn='mergers', cn='effective_date'))

    conn.close()

    return db_path


class UsEquityPricingLoaderTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = TempDirectory()
        cls.bcolz_test_data_path = create_bcolz_data(
            cls.test_data_dir.path)
        cls.adjustments_test_data_path = create_adjustments_data(
            cls.test_data_dir.path)

    @classmethod
    def tearDownClass(cls):
        cls.test_data_dir.cleanup()

    def test_load_from_bcolz(self):
        # 1) The equity's trades cover all query dates.
        # 2) The equity's trades are all before the start of the query.
        # 3) The equity's trades start before the query start, but stop
        #    before the query end.
        # 4) The equity's trades start after query start but end before
        #    the query end.
        # 5) The equity's trades start after query start, but trade through or
        #    past the query end
        # 6) The equity's trades are start after query end.
        assets = pd.Int64Index(EQUITY_INFO.keys())
        columns = TEST_QUERY_COLUMNS
        query_dates = TEST_QUERY_RANGE
        table = bcolz.ctable(
            rootdir=self.bcolz_test_data_path,
            mode='r')
        trading_days = TEST_TRADING_DAYS
        raw_price_loader = BcolzRawPriceLoader(table, trading_days)
        raw_arrays = raw_price_loader.load_raw_arrays(
            columns,
            assets,
            query_dates)

        close_array = raw_arrays[0]
        # See create_test_bcolz_data for encoding of expected values.
        # Created as the column, so that test is isolated on the individual
        # asset date ranges.
        expected = [
            # Asset 1 should have trade data for all days.
            np.array([14.004, 14.005, 14.006, 14.007, 14.008]),
            # Asset 2 should have no values, all data occurs before query.
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            # Asset 3 should have the first two days of data.
            np.array([34.004, 34.005, np.nan, np.nan, np.nan]),
            # Asset 4 should have data starting on the second day and ending
            # on the third.
            np.array([np.nan, 44.001, 44.002, np.nan, np.nan]),
            # Asset 4 should have data starting on the second day through the
            # end of the range
            np.array([np.nan, 54.001, 54.002, 54.003, 54.004]),
            # Asset 5 should have no data
            np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        ]

        for i, expected_col in enumerate(expected):
            np.testing.assert_allclose(expected_col, close_array[:, i])

    def test_load_adjustments_from_sqlite(self):
        conn = sqlite3.connect(self.adjustments_test_data_path)

        adjustments_loader = SQLiteAdjustmentLoader(conn)

        adjustments = adjustments_loader.load_adjustments(
            TEST_QUERY_COLUMNS,
            TEST_QUERY_ASSETS,
            TEST_QUERY_RANGE,
        )

        close_adjustments = adjustments[0]
        volume_adjustments = adjustments[1]

        # See SPLITS, MERGERS, DIVIDENDS module variables for details of
        # expected values.
        EXPECTED_CLOSES = {
            # 2015-06-04
            0: [
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.104),
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.204),
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.304)
            ],
            1: [
                Float64Multiply(
                    first_row=0, last_row=1, col=1, value=2.105)
            ],
            2: [
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.106),
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.206),
                Float64Multiply(
                    first_row=0, last_row=2, col=1, value=2.206),
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.306)
            ],
            3: [
                Float64Multiply(
                    first_row=0, last_row=3, col=1, value=2.307000)
            ]
        }

        EXPECTED_VOLUMES = {
            0: [
                Float64Multiply(
                    first_row=0, last_row=0, col=0, value=1.0 / 1.104)
            ],
            1: [
                Float64Multiply(
                    first_row=0, last_row=1, col=1, value=1.0 / 2.105)
            ],
            2: [
                Float64Multiply(
                    first_row=0, last_row=2, col=0, value=1.0 / 1.106)
            ]}

        self.assertEqual(close_adjustments, EXPECTED_CLOSES)
        self.assertEqual(volume_adjustments, EXPECTED_VOLUMES)
