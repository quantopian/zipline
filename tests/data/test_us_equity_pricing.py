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
from sys import maxsize

from nose_parameterized import parameterized
from numpy import (
    arange,
    datetime64,
    nan,
)
from numpy.testing import (
    assert_array_equal,
)
from pandas import (
    DataFrame,
    Timestamp,
)
from pandas.util.testing import assert_index_equal

from zipline.data.us_equity_pricing import (
    BcolzDailyBarReader,
    NoDataBeforeDate,
    NoDataAfterDate,
)
from zipline.pipeline.loaders.synthetic import (
    OHLCV,
    asset_start,
    asset_end,
    expected_bar_value,
    expected_bar_values_2d,
    make_bar_data,
)
from zipline.testing import seconds_to_timestamp
from zipline.testing.fixtures import (
    WithBcolzEquityDailyBarReader,
    ZiplineTestCase,
)
from zipline.utils.calendars import get_calendar

TEST_CALENDAR_START = Timestamp('2015-06-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')

TEST_QUERY_START = Timestamp('2015-06-10', tz='UTC')
TEST_QUERY_STOP = Timestamp('2015-06-19', tz='UTC')

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
EQUITY_INFO['symbol'] = [chr(ord('A') + n) for n in range(len(EQUITY_INFO))]

TEST_QUERY_ASSETS = EQUITY_INFO.index


class BcolzDailyBarTestCase(WithBcolzEquityDailyBarReader, ZiplineTestCase):
    EQUITY_DAILY_BAR_START_DATE = TEST_CALENDAR_START
    EQUITY_DAILY_BAR_END_DATE = TEST_CALENDAR_STOP

    @classmethod
    def make_equity_info(cls):
        return EQUITY_INFO

    @classmethod
    def make_equity_daily_bar_data(cls):
        return make_bar_data(
            EQUITY_INFO,
            cls.equity_daily_bar_days,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(BcolzDailyBarTestCase, cls).init_class_fixtures()
        cls.sessions = cls.trading_calendar.sessions_in_range(
            cls.trading_calendar.minute_to_session_label(TEST_CALENDAR_START),
            cls.trading_calendar.minute_to_session_label(TEST_CALENDAR_STOP)
        )

    @property
    def assets(self):
        return EQUITY_INFO.index

    def trading_days_between(self, start, end):
        return self.sessions[self.sessions.slice_indexer(start, end)]

    def asset_start(self, asset_id):
        return asset_start(EQUITY_INFO, asset_id)

    def asset_end(self, asset_id):
        return asset_end(EQUITY_INFO, asset_id)

    def dates_for_asset(self, asset_id):
        start, end = self.asset_start(asset_id), self.asset_end(asset_id)
        return self.trading_days_between(start, end)

    def test_write_ohlcv_content(self):
        result = self.bcolz_daily_bar_ctable
        for column in OHLCV:
            idx = 0
            data = result[column][:]
            multiplier = 1 if column == 'volume' else 1000
            for asset_id in self.assets:
                for date in self.dates_for_asset(asset_id):
                    self.assertEqual(
                        expected_bar_value(
                            asset_id,
                            date,
                            column
                        ) * multiplier,
                        data[idx],
                    )
                    idx += 1
            self.assertEqual(idx, len(data))

    def test_write_day_and_id(self):
        result = self.bcolz_daily_bar_ctable
        idx = 0
        ids = result['id']
        days = result['day']
        for asset_id in self.assets:
            for date in self.dates_for_asset(asset_id):
                self.assertEqual(ids[idx], asset_id)
                self.assertEqual(date, seconds_to_timestamp(days[idx]))
                idx += 1

    def test_write_attrs(self):
        result = self.bcolz_daily_bar_ctable
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
        cal = get_calendar(result.attrs['calendar_name'])
        first_session = Timestamp(result.attrs['start_session_ns'], tz='UTC')
        end_session = Timestamp(result.attrs['end_session_ns'], tz='UTC')
        sessions = cal.sessions_in_range(first_session, end_session)

        assert_index_equal(
            self.sessions,
            sessions
        )

    def test_read_first_trading_day(self):
        self.assertEqual(
            self.bcolz_equity_daily_bar_reader.first_trading_day,
            self.sessions[0],
        )

    def _check_read_results(self, columns, assets, start_date, end_date):
        results = self.bcolz_equity_daily_bar_reader.load_raw_arrays(
            columns,
            start_date,
            end_date,
            assets,
        )
        dates = self.trading_days_between(start_date, end_date)
        for column, result in zip(columns, results):
            assert_array_equal(
                result,
                expected_bar_values_2d(
                    dates,
                    EQUITY_INFO,
                    column,
                )
            )

    @parameterized.expand([
        (['open'],),
        (['close', 'volume'],),
        (['volume', 'high', 'low'],),
        (['open', 'high', 'low', 'close', 'volume'],),
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
        columns = ['high', 'volume']
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_start(asset),
                end_date=self.sessions[-1],
            )

    def test_start_on_asset_end(self):
        """
        Test loading with queries that start on the last day of each asset's
        lifetime.
        """
        columns = ['close', 'volume']
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_end(asset),
                end_date=self.sessions[-1],
            )

    def test_end_on_asset_start(self):
        """
        Test loading with queries that end on the first day of each asset's
        lifetime.
        """
        columns = ['close', 'volume']
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_start(asset),
            )

    def test_end_on_asset_end(self):
        """
        Test loading with queries that end on the last day of each asset's
        lifetime.
        """
        columns = ['close', 'volume']
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_end(asset),
            )

    def test_unadjusted_get_value(self):
        reader = self.bcolz_equity_daily_bar_reader
        # At beginning
        price = reader.get_value(1, Timestamp('2015-06-01', tz='UTC'),
                                 'close')
        # Synthetic writes price for date.
        self.assertEqual(108630.0, price)

        # Middle
        price = reader.get_value(1, Timestamp('2015-06-02', tz='UTC'),
                                 'close')
        self.assertEqual(108631.0, price)
        # End
        price = reader.get_value(1, Timestamp('2015-06-05', tz='UTC'),
                                 'close')
        self.assertEqual(108634.0, price)

        # Another sid at beginning.
        price = reader.get_value(2, Timestamp('2015-06-22', tz='UTC'),
                                 'close')
        self.assertEqual(208651.0, price)

        # Ensure that volume does not have float adjustment applied.
        volume = reader.get_value(1, Timestamp('2015-06-02', tz='UTC'),
                                  'volume')
        self.assertEqual(109631, volume)

    def test_unadjusted_get_value_no_data(self):
        table = self.bcolz_daily_bar_ctable
        reader = BcolzDailyBarReader(table)
        # before
        with self.assertRaises(NoDataBeforeDate):
            reader.get_value(2, Timestamp('2015-06-08', tz='UTC'), 'close')

        # after
        with self.assertRaises(NoDataAfterDate):
            reader.get_value(4, Timestamp('2015-06-16', tz='UTC'), 'close')

    def test_unadjusted_get_value_empty_value(self):
        reader = self.bcolz_equity_daily_bar_reader

        # A sid, day and corresponding index into which to overwrite a zero.
        zero_sid = 1
        zero_day = Timestamp('2015-06-02', tz='UTC')
        zero_ix = reader.sid_day_index(zero_sid, zero_day)

        old = reader._spot_col('close')[zero_ix]
        try:
            # Write a zero into the synthetic pricing data at the day and sid,
            # so that a read should now return -1.
            # This a little hacky, in lieu of changing the synthetic data set.
            reader._spot_col('close')[zero_ix] = 0

            close = reader.get_value(zero_sid, zero_day, 'close')
            assert_array_equal(nan, close)
        finally:
            reader._spot_col('close')[zero_ix] = old


class BcolzDailyBarAlwaysReadAllTestCase(BcolzDailyBarTestCase):
    """
    Force tests defined in BcolzDailyBarTestCase to always read the entire
    column into memory before selecting desired asset data, when invoking
    `load_raw_array`.
    """
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD = 0


class BcolzDailyBarNeverReadAllTestCase(BcolzDailyBarTestCase):
    """
    Force tests defined in BcolzDailyBarTestCase to never read the entire
    column into memory before selecting desired asset data, when invoking
    `load_raw_array`.
    """
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD = maxsize
