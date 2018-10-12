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
import re

from nose_parameterized import parameterized
import numpy as np
from numpy import (
    arange,
    array,
    float64,
    nan,
)
from pandas import (
    concat,
    DataFrame,
    NaT,
    Timestamp,
)
from six import iteritems
from toolz import merge
from trading_calendars import get_calendar

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
    NoDataForSid,
    NoDataOnDate,
)
from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter
from zipline.data.hdf5_daily_bars import (
    CLOSE,
    DEFAULT_SCALING_FACTORS,
    HIGH,
    LOW,
    OPEN,
    VOLUME,
    coerce_to_uint32,
)
from zipline.pipeline.loaders.synthetic import (
    OHLCV,
    asset_start,
    asset_end,
    expected_bar_value_with_holes,
    expected_bar_values_2d,
    make_bar_data,
)
from zipline.testing import seconds_to_timestamp
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithBcolzEquityDailyBarReader,
    WithEquityDailyBarData,
    WithHDF5EquityMultiCountryDailyBarReader,
    WithSeededRandomState,
    WithTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal, assert_sequence_equal
from zipline.utils.classproperty import classproperty

TEST_CALENDAR_START = Timestamp('2015-06-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')

TEST_QUERY_START = Timestamp('2015-06-10', tz='UTC')
TEST_QUERY_STOP = Timestamp('2015-06-19', tz='UTC')

# One asset for each of the cases enumerated in load_raw_arrays_from_bcolz.
us_info = DataFrame(
    [
        # 1) The equity's trades start and end before query.
        {'start_date': '2015-06-01', 'end_date': '2015-06-05'},
        # 2) The equity's trades start and end after query.
        {'start_date': '2015-06-22', 'end_date': '2015-06-30'},
        # 3) The equity's data covers all dates in range (but we define
        #    a hole for it on 2015-06-17).
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
).astype('datetime64[ns]')
us_info['exchange'] = 'NYSE'

ca_info = DataFrame(
    [
        # 7) The equity's trades start and end before query.
        {'start_date': '2015-06-01', 'end_date': '2015-06-05'},
        # 8) The equity's trades start and end after query.
        {'start_date': '2015-06-22', 'end_date': '2015-06-30'},
        # 9) The equity's data covers all dates in range.
        {'start_date': '2015-06-02', 'end_date': '2015-06-30'},
        # 10) The equity's trades start before the query start, but stop
        #    before the query end.
        {'start_date': '2015-06-01', 'end_date': '2015-06-15'},
        # 11) The equity's trades start and end during the query.
        {'start_date': '2015-06-12', 'end_date': '2015-06-18'},
        # 12) The equity's trades start during the query, but extend through
        #    the whole query.
        {'start_date': '2015-06-15', 'end_date': '2015-06-25'},
    ],
    index=arange(7, 13),
    columns=['start_date', 'end_date'],
).astype('datetime64[ns]')
ca_info['exchange'] = 'TSX'

EQUITY_INFO = concat([us_info, ca_info])
EQUITY_INFO['symbol'] = [chr(ord('A') + n) for n in range(len(EQUITY_INFO))]

TEST_QUERY_ASSETS = EQUITY_INFO.index

HOLES = {
    'US': {3: (Timestamp('2015-06-17', tz='UTC'),)},
    'CA': {9: (Timestamp('2015-06-17', tz='UTC'),)},
}


class _DailyBarsTestCase(WithEquityDailyBarData,
                         WithSeededRandomState,
                         ZiplineTestCase):
    EQUITY_DAILY_BAR_START_DATE = TEST_CALENDAR_START
    EQUITY_DAILY_BAR_END_DATE = TEST_CALENDAR_STOP

    # The country under which these tests should be run.
    DAILY_BARS_TEST_QUERY_COUNTRY_CODE = 'US'

    @classmethod
    def init_class_fixtures(cls):
        super(_DailyBarsTestCase, cls).init_class_fixtures()

        cls.sessions = cls.trading_calendar.sessions_in_range(
            cls.trading_calendar.minute_to_session_label(TEST_CALENDAR_START),
            cls.trading_calendar.minute_to_session_label(TEST_CALENDAR_STOP)
        )

    @classmethod
    def make_equity_info(cls):
        return EQUITY_INFO

    @classmethod
    def make_exchanges_info(cls, *args, **kwargs):
        return DataFrame({
            'exchange': ['NYSE', 'TSX'],
            'country_code': ['US', 'CA']
        })

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        # Create the data for all countries.
        return make_bar_data(
            EQUITY_INFO.loc[list(sids)],
            cls.equity_daily_bar_days,
            holes=merge(HOLES.values()),
        )

    @classproperty
    def holes(cls):
        return HOLES[cls.DAILY_BARS_TEST_QUERY_COUNTRY_CODE]

    @property
    def assets(self):
        return list(
            self.asset_finder.equities_sids_for_country_code(
                self.DAILY_BARS_TEST_QUERY_COUNTRY_CODE
            )
        )

    def trading_days_between(self, start, end):
        return self.sessions[self.sessions.slice_indexer(start, end)]

    def asset_start(self, asset_id):
        return asset_start(EQUITY_INFO, asset_id)

    def asset_end(self, asset_id):
        return asset_end(EQUITY_INFO, asset_id)

    def dates_for_asset(self, asset_id):
        start, end = self.asset_start(asset_id), self.asset_end(asset_id)
        return self.trading_days_between(start, end)

    def test_read_first_trading_day(self):
        self.assertEqual(
            self.daily_bar_reader.first_trading_day,
            self.sessions[0],
        )

    def test_sessions(self):
        assert_equal(self.daily_bar_reader.sessions, self.sessions)

    def _check_read_results(self, columns, assets, start_date, end_date):
        results = self.daily_bar_reader.load_raw_arrays(
            columns,
            start_date,
            end_date,
            assets,
        )
        dates = self.trading_days_between(start_date, end_date)
        for column, result in zip(columns, results):
            assert_equal(
                result,
                expected_bar_values_2d(
                    dates,
                    EQUITY_INFO.loc[assets],
                    column,
                    holes=self.holes,
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

        assets_array = np.array(self.assets)
        for _ in range(5):
            assets = assets_array.copy()
            self.rand.shuffle(assets)
            assets = assets[:np.random.randint(1, len(assets))]
            self._check_read_results(
                columns,
                assets,
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
        """Test get_value() on both a price field (CLOSE) and VOLUME."""
        reader = self.daily_bar_reader

        def make_failure_msg(asset, date, field):
            return "Unexpected value for sid={}; date={}; field={}.".format(
                asset,
                date.date(),
                field
            )

        for asset in self.assets:
            # Dates to check.
            asset_start = self.asset_start(asset)

            asset_dates = self.dates_for_asset(asset)
            asset_middle = asset_dates[len(asset_dates) // 2]

            asset_end = self.asset_end(asset)

            # At beginning
            assert_equal(
                reader.get_value(asset, asset_start, CLOSE),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_start,
                    colname=CLOSE,
                    holes=self.holes,
                    missing_value=nan,
                ),
                msg=make_failure_msg(asset, asset_start, CLOSE),
            )

            # Middle
            assert_equal(
                reader.get_value(asset, asset_middle, CLOSE),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_middle,
                    colname=CLOSE,
                    holes=self.holes,
                    missing_value=nan,
                ),
                msg=make_failure_msg(asset, asset_middle, CLOSE),
            )

            # End
            assert_equal(
                reader.get_value(asset, asset_end, CLOSE),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_end,
                    colname=CLOSE,
                    holes=self.holes,
                    missing_value=nan,
                ),
                msg=make_failure_msg(asset, asset_end, CLOSE),
            )

            # Ensure that volume does not have float adjustment applied.
            assert_equal(
                reader.get_value(asset, asset_start, VOLUME),
                expected_bar_value_with_holes(
                    asset_id=asset,
                    date=asset_start,
                    colname=VOLUME,
                    holes=self.holes,
                    missing_value=0,
                ),
                msg=make_failure_msg(asset, asset_start, VOLUME),
            )

    def test_unadjusted_get_value_no_data(self):
        """Test behavior of get_value() around missing data."""
        reader = self.daily_bar_reader

        for asset in self.assets:
            before_start = self.trading_calendar.previous_session_label(
                self.asset_start(asset)
            )
            after_end = self.trading_calendar.next_session_label(
                self.asset_end(asset)
            )

            # Attempting to get data for an asset before its start date
            # should raise NoDataBeforeDate.
            if TEST_CALENDAR_START <= before_start <= TEST_CALENDAR_STOP:
                with self.assertRaises(NoDataBeforeDate):
                    reader.get_value(asset, before_start, CLOSE)

            # Attempting to get data for an asset after its end date
            # should raise NoDataAfterDate.
            if TEST_CALENDAR_START <= after_end <= TEST_CALENDAR_STOP:
                with self.assertRaises(NoDataAfterDate):
                    reader.get_value(asset, after_end, CLOSE)

        # Retrieving data for "holes" (dates with no data, but within
        # an  asset's lifetime) should not raise an exception. nan is
        # returned for OHLC fields, and 0 is returned for volume.
        for asset, dates in iteritems(self.holes):
            for date in dates:
                assert_equal(
                    reader.get_value(asset, date, CLOSE),
                    nan,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-nan value for close."
                    ).format(asset, date.date())
                )
                assert_equal(
                    reader.get_value(asset, date, VOLUME),
                    0.0,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-zero value for volume."
                    ).format(asset, date.date())
                )

    def test_get_last_traded_dt(self):
        for sid in self.assets:
            assert_equal(
                self.daily_bar_reader.get_last_traded_dt(
                    self.asset_finder.retrieve_asset(sid),
                    self.EQUITY_DAILY_BAR_END_DATE,
                ),
                self.asset_end(sid),
            )

            # If an asset is alive by ``mid_date``, its "last trade" dt
            # is either the end date for the asset, or ``mid_date`` if
            # the asset is *still* alive at that point. Otherwise, it
            # is pd.NaT.
            mid_date = Timestamp('2015-06-15', tz='UTC')
            if self.asset_start(sid) <= mid_date:
                expected = min(self.asset_end(sid), mid_date)
            else:
                expected = NaT

            assert_equal(
                self.daily_bar_reader.get_last_traded_dt(
                    self.asset_finder.retrieve_asset(sid),
                    mid_date,
                ),
                expected,
            )

            # If the dt passed comes before any of the assets
            # start trading, the "last traded" dt for each is pd.NaT.
            assert_equal(
                self.daily_bar_reader.get_last_traded_dt(
                    self.asset_finder.retrieve_asset(sid),
                    Timestamp(0, tz='UTC'),
                ),
                NaT,
            )


class BcolzDailyBarTestCase(WithBcolzEquityDailyBarReader, _DailyBarsTestCase):
    EQUITY_DAILY_BAR_COUNTRY_CODES = ['US']

    @classmethod
    def init_class_fixtures(cls):
        super(BcolzDailyBarTestCase, cls).init_class_fixtures()

        cls.daily_bar_reader = cls.bcolz_equity_daily_bar_reader

    def test_write_ohlcv_content(self):
        result = self.bcolz_daily_bar_ctable
        for column in OHLCV:
            idx = 0
            data = result[column][:]
            multiplier = 1 if column == 'volume' else 1000
            for asset_id in self.assets:
                for date in self.dates_for_asset(asset_id):
                    self.assertEqual(
                        data[idx],
                        expected_bar_value_with_holes(
                            asset_id=asset_id,
                            date=date,
                            colname=column,
                            holes=self.holes,
                            missing_value=0,
                        ) * multiplier,
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

        assert_equal(
            self.sessions,
            sessions
        )


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


class BcolzDailyBarWriterMissingDataTestCase(WithAssetFinder,
                                             WithTmpDir,
                                             WithTradingCalendars,
                                             ZiplineTestCase):
    # Sid 3 is active from 2015-06-02 to 2015-06-30.
    MISSING_DATA_SID = 3
    # Leave out data for a day in the middle of the query range.
    MISSING_DATA_DAY = Timestamp('2015-06-15', tz='UTC')

    @classmethod
    def make_equity_info(cls):
        return (
            EQUITY_INFO.loc[EQUITY_INFO.index == cls.MISSING_DATA_SID].copy()
        )

    def test_missing_values_assertion(self):
        sessions = self.trading_calendar.sessions_in_range(
            TEST_CALENDAR_START,
            TEST_CALENDAR_STOP,
        )

        sessions_with_gap = sessions[sessions != self.MISSING_DATA_DAY]
        bar_data = make_bar_data(self.make_equity_info(), sessions_with_gap)

        writer = BcolzDailyBarWriter(
            self.tmpdir.path,
            self.trading_calendar,
            sessions[0],
            sessions[-1],
        )

        # There are 21 sessions between the start and end date for this
        # asset, and we excluded one.
        expected_msg = re.escape(
            "Got 20 rows for daily bars table with first day=2015-06-02, last "
            "day=2015-06-30, expected 21 rows.\n"
            "Missing sessions: "
            "[Timestamp('2015-06-15 00:00:00+0000', tz='UTC')]\n"
            "Extra sessions: []"
        )
        with self.assertRaisesRegexp(AssertionError, expected_msg):
            writer.write(bar_data)


class _HDF5DailyBarTestCase(WithHDF5EquityMultiCountryDailyBarReader,
                            _DailyBarsTestCase):
    @classmethod
    def init_class_fixtures(cls):
        super(_HDF5DailyBarTestCase, cls).init_class_fixtures()

        cls.daily_bar_reader = cls.hdf5_equity_daily_bar_reader

    @property
    def single_country_reader(self):
        return self.single_country_hdf5_equity_daily_bar_readers[
            self.DAILY_BARS_TEST_QUERY_COUNTRY_CODE
        ]

    def test_asset_end_dates(self):
        assert_sequence_equal(self.single_country_reader.sids, self.assets)

        for ix, sid in enumerate(self.single_country_reader.sids):
            assert_equal(
                self.single_country_reader.asset_end_dates[ix],
                self.asset_end(sid).asm8,
                msg=(
                    'asset_end_dates value for sid={} differs from expected'
                ).format(sid)
            )

    def test_asset_start_dates(self):
        assert_sequence_equal(self.single_country_reader.sids, self.assets)

        for ix, sid in enumerate(self.single_country_reader.sids):
            assert_equal(
                self.single_country_reader.asset_start_dates[ix],
                self.asset_start(sid).asm8,
                msg=(
                    'asset_start_dates value for sid={} differs from expected'
                ).format(sid)
            )

    def test_invalid_sid(self):
        INVALID_SID = 100

        with self.assertRaises(NoDataForSid):
            self.daily_bar_reader.load_raw_arrays(
                OHLCV,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
                [INVALID_SID],
            )

        with self.assertRaises(NoDataForSid):
            self.daily_bar_reader.get_value(
                INVALID_SID,
                TEST_QUERY_START,
                'close',
            )

    def test_invalid_sid_single_country(self):
        INVALID_SID = 100

        with self.assertRaises(NoDataForSid):
            self.single_country_reader.load_raw_arrays(
                OHLCV,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
                [INVALID_SID],
            )

        with self.assertRaises(NoDataForSid):
            self.single_country_reader.get_value(
                INVALID_SID,
                TEST_QUERY_START,
                'close',
            )

    def test_invalid_date(self):
        INVALID_DATES = (
            # Before the start of the daily bars.
            self.trading_calendar.previous_session_label(TEST_CALENDAR_START),
            # A Sunday.
            Timestamp('2015-06-07', tz='UTC'),
            # After the end of the daily bars.
            self.trading_calendar.next_session_label(TEST_CALENDAR_STOP),
        )

        for invalid_date in INVALID_DATES:
            with self.assertRaises(NoDataOnDate):
                self.daily_bar_reader.load_raw_arrays(
                    OHLCV,
                    invalid_date,
                    TEST_QUERY_STOP,
                    self.assets,
                )

            with self.assertRaises(NoDataOnDate):
                self.daily_bar_reader.get_value(
                    self.assets[0],
                    invalid_date,
                    'close',
                )


class HDF5DailyBarUSTestCase(_HDF5DailyBarTestCase):
    DAILY_BARS_TEST_QUERY_COUNTRY_CODE = 'US'


class HDF5DailyBarCanadaTestCase(_HDF5DailyBarTestCase):
    TRADING_CALENDAR_PRIMARY_CAL = 'TSX'
    DAILY_BARS_TEST_QUERY_COUNTRY_CODE = 'CA'


class TestCoerceToUint32Price(ZiplineTestCase):
    """Test the coerce_to_uint32() function used by the HDF5DailyBarWriter."""

    @parameterized.expand([
        (OPEN, array([1, 1000, 100000, 100500, 1000005, 130230], dtype='u4')),
        (HIGH, array([1, 1000, 100000, 100500, 1000005, 130230], dtype='u4')),
        (LOW, array([1, 1000, 100000, 100500, 1000005, 130230], dtype='u4')),
        (CLOSE, array([1, 1000, 100000, 100500, 1000005, 130230], dtype='u4')),
        (VOLUME, array([0, 1, 100, 100, 1000, 130], dtype='u4')),
    ])
    def test_coerce_to_uint32_price(self, field, expected):
        # NOTE: 130.23 is not perfectly representable as a double, but we
        # shouldn't truncate and be off by an entire cent
        coerced = coerce_to_uint32(
            array([0.001, 1, 100, 100.5, 1000.005, 130.23], dtype=float64),
            DEFAULT_SCALING_FACTORS[field],
        )

        assert_equal(coerced, expected)
