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
from itertools import cycle, islice
from sys import maxsize
import re

from parameterized import parameterized
import numpy as np
import pandas as pd
from toolz import merge
from zipline.utils.calendar_utils import get_calendar

from zipline.data.bar_reader import (
    NoDataAfterDate,
    NoDataBeforeDate,
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
from zipline.testing import (
    seconds_to_timestamp,
    powerset,
)
from zipline.testing.predicates import (
    assert_equal,
    assert_sequence_equal,
)
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithBcolzEquityDailyBarReader,
    WithHDF5EquityMultiCountryDailyBarReader,
    WithTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
    WithEquityDailyBarData,
    WithSeededRandomState,
)
from zipline.utils.classproperty import classproperty
from zipline.pipeline.loaders.synthetic import (
    OHLCV,
    expected_bar_value_with_holes,
    make_bar_data,
    asset_start,
    asset_end,
    expected_bar_values_2d,
)
import pytest

# NOTE: All sids here are odd, so we can test querying for unknown sids
#       with evens.
us_info = pd.DataFrame(
    [
        # 1) The equity's trades start and end before query.
        {"start_date": "2015-06-01", "end_date": "2015-06-05"},
        # 3) The equity's trades start and end after query.
        {"start_date": "2015-06-22", "end_date": "2015-06-30"},
        # 5) The equity's data covers all dates in range (but we define
        #    a hole for it on 2015-06-17).
        {"start_date": "2015-06-02", "end_date": "2015-06-30"},
        # 7) The equity's trades start before the query start, but stop
        #    before the query end.
        {"start_date": "2015-06-01", "end_date": "2015-06-15"},
        # 9) The equity's trades start and end during the query.
        {"start_date": "2015-06-12", "end_date": "2015-06-18"},
        # 11) The equity's trades start during the query, but extend through
        #    the whole query.
        {"start_date": "2015-06-15", "end_date": "2015-06-25"},
    ],
    index=np.arange(1, 12, step=2),
    columns=["start_date", "end_date"],
).astype("datetime64[ns]")

us_info["exchange"] = "NYSE"

ca_info = pd.DataFrame(
    [
        # 13) The equity's trades start and end before query.
        {"start_date": "2015-06-01", "end_date": "2015-06-05"},
        # 15) The equity's trades start and end after query.
        {"start_date": "2015-06-22", "end_date": "2015-06-30"},
        # 17) The equity's data covers all dates in range.
        {"start_date": "2015-06-02", "end_date": "2015-06-30"},
        # 19) The equity's trades start before the query start, but stop
        #    before the query end.
        {"start_date": "2015-06-01", "end_date": "2015-06-15"},
        # 21) The equity's trades start and end during the query.
        {"start_date": "2015-06-12", "end_date": "2015-06-18"},
        # 23) The equity's trades start during the query, but extend through
        #    the whole query.
        {"start_date": "2015-06-15", "end_date": "2015-06-25"},
    ],
    index=np.arange(13, 24, step=2),
    columns=["start_date", "end_date"],
).astype("datetime64[ns]")

ca_info["exchange"] = "TSX"

EQUITY_INFO = pd.concat([us_info, ca_info])
EQUITY_INFO["symbol"] = [chr(ord("A") + x) for x in range(len(EQUITY_INFO))]

TEST_QUERY_ASSETS = EQUITY_INFO.index

TEST_CALENDAR_START = pd.Timestamp("2015-06-01")
TEST_CALENDAR_STOP = pd.Timestamp("2015-06-30")

TEST_QUERY_START = pd.Timestamp("2015-06-10")
TEST_QUERY_STOP = pd.Timestamp("2015-06-19")


HOLES = {
    "US": {5: (pd.Timestamp("2015-06-17"),)},
    "CA": {17: (pd.Timestamp("2015-06-17"),)},
}


class _DailyBarsTestCase(
    WithEquityDailyBarData,
    WithSeededRandomState,
    ZiplineTestCase,
):
    EQUITY_DAILY_BAR_START_DATE = TEST_CALENDAR_START
    EQUITY_DAILY_BAR_END_DATE = TEST_CALENDAR_STOP

    # The country under which these tests should be run.
    DAILY_BARS_TEST_QUERY_COUNTRY_CODE = "US"

    # Currencies to use for assets in these tests.
    DAILY_BARS_TEST_CURRENCIES = {"US": ["USD"], "CA": ["USD", "CAD"]}

    @classmethod
    def init_class_fixtures(cls):
        super(_DailyBarsTestCase, cls).init_class_fixtures()

        cls.sessions = cls.trading_calendar.sessions_in_range(
            cls.trading_calendar.minute_to_session(TEST_CALENDAR_START),
            cls.trading_calendar.minute_to_session(TEST_CALENDAR_STOP),
        )

    @classmethod
    def make_equity_info(cls):
        return EQUITY_INFO

    @classmethod
    def make_exchanges_info(cls, *args, **kwargs):
        return pd.DataFrame({"exchange": ["NYSE", "TSX"], "country_code": ["US", "CA"]})

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        # Create the data for all countries.
        return make_bar_data(
            EQUITY_INFO.loc[list(sids)],
            cls.equity_daily_bar_days,
            holes=merge(HOLES.values()),
        )

    @classmethod
    def make_equity_daily_bar_currency_codes(cls, country_code, sids):
        # Evenly distribute choices among ``sids``.
        choices = cls.DAILY_BARS_TEST_CURRENCIES[country_code]
        codes = list(islice(cycle(choices), len(sids)))
        return pd.Series(index=sids, data=np.array(codes, dtype=object))

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
                    assets,
                    EQUITY_INFO.loc[self.assets],
                    column,
                    holes=self.holes,
                ),
            )


def test_odd_query_assets():
    assert (TEST_QUERY_ASSETS % 2 == 1).all(), "All sids should be odd."


class BcolzDailyBarTestCase(WithBcolzEquityDailyBarReader, _DailyBarsTestCase):
    EQUITY_DAILY_BAR_COUNTRY_CODES = ["US"]

    @classmethod
    def init_class_fixtures(cls):
        super(BcolzDailyBarTestCase, cls).init_class_fixtures()

        cls.daily_bar_reader = cls.bcolz_equity_daily_bar_reader

    def test_write_ohlcv_content(self):
        result = self.bcolz_daily_bar_ctable
        for column in OHLCV:
            idx = 0
            data = result[column][:]
            multiplier = 1 if column == "volume" else 1000
            for asset_id in self.assets:
                for date in self.dates_for_asset(asset_id):
                    assert (
                        data[idx]
                        == expected_bar_value_with_holes(
                            asset_id=asset_id,
                            date=date,
                            colname=column,
                            holes=self.holes,
                            missing_value=0,
                        )
                        * multiplier
                    )
                    idx += 1
            assert idx == len(data)

    def test_write_day_and_id(self):
        result = self.bcolz_daily_bar_ctable
        idx = 0
        ids = result["id"]
        days = result["day"]
        for asset_id in self.assets:
            for date in self.dates_for_asset(asset_id):
                assert ids[idx] == asset_id
                assert date == seconds_to_timestamp(days[idx])
                idx += 1

    def test_write_attrs(self):
        result = self.bcolz_daily_bar_ctable
        expected_first_row = {
            "1": 0,
            "3": 5,  # Asset 1 has 5 trading days.
            "5": 12,  # Asset 3 has 7 trading days.
            "7": 33,  # Asset 5 has 21 trading days.
            "9": 44,  # Asset 7 has 11 trading days.
            "11": 49,  # Asset 9 has 5 trading days.
        }
        expected_last_row = {
            "1": 4,
            "3": 11,
            "5": 32,
            "7": 43,
            "9": 48,
            "11": 57,  # Asset 11 has 9 trading days.
        }
        expected_calendar_offset = {
            "1": 0,  # Starts on 6-01, 1st trading day of month.
            "3": 15,  # Starts on 6-22, 16th trading day of month.
            "5": 1,  # Starts on 6-02, 2nd trading day of month.
            "7": 0,  # Starts on 6-01, 1st trading day of month.
            "9": 9,  # Starts on 6-12, 10th trading day of month.
            "11": 10,  # Starts on 6-15, 11th trading day of month.
        }
        assert result.attrs["first_row"] == expected_first_row
        assert result.attrs["last_row"] == expected_last_row
        assert result.attrs["calendar_offset"] == expected_calendar_offset
        cal = get_calendar(result.attrs["calendar_name"])
        first_session = pd.Timestamp(result.attrs["start_session_ns"])
        end_session = pd.Timestamp(result.attrs["end_session_ns"])
        sessions = cal.sessions_in_range(first_session, end_session)

        assert_equal(self.sessions, sessions)

    def test_read_first_trading_day(self):
        assert self.daily_bar_reader.first_trading_day == self.sessions[0]

    def test_sessions(self):
        assert_equal(self.daily_bar_reader.sessions, self.sessions)

    @parameterized.expand(
        [
            (["open"],),
            (["close", "volume"],),
            (["volume", "high", "low"],),
            (["open", "high", "low", "close", "volume"],),
        ]
    )
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
            assets = assets[: np.random.randint(1, len(assets))]
            self._check_read_results(
                columns,
                assets,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
            )

    def test_start_on_asset_start(self):
        """Test loading with queries that starts on the first day of each asset's
        lifetime.
        """
        columns = ["high", "volume"]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_start(asset),
                end_date=self.sessions[-1],
            )

    def test_start_on_asset_end(self):
        """Test loading with queries that start on the last day of each asset's
        lifetime.
        """
        columns = ["close", "volume"]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.asset_end(asset),
                end_date=self.sessions[-1],
            )

    def test_end_on_asset_start(self):
        """Test loading with queries that end on the first day of each asset's
        lifetime.
        """
        columns = ["close", "volume"]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_start(asset),
            )

    def test_end_on_asset_end(self):
        """Test loading with queries that end on the last day of each asset's
        lifetime.
        """
        columns = [CLOSE, VOLUME]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_end(asset),
            )

    def test_read_known_and_unknown_sids(self):
        """Test a query with some known sids mixed in with unknown sids."""

        # Construct a list of alternating valid and invalid query sids,
        # bookended by invalid sids.
        #
        # E.g.
        #   INVALID VALID INVALID VALID ... VALID INVALID
        query_assets = (
            [self.assets[-1] + 1]
            + list(range(self.assets[0], self.assets[-1] + 1))
            + [self.assets[-1] + 3]
        )

        columns = [CLOSE, VOLUME]
        self._check_read_results(
            columns,
            query_assets,
            start_date=TEST_QUERY_START,
            end_date=TEST_QUERY_STOP,
        )

    @parameterized.expand(
        [
            # Query for only even sids, only odd ids are valid.
            ([],),
            ([2],),
            ([2, 4, 800],),
        ]
    )
    def test_read_only_unknown_sids(self, query_assets):
        columns = [CLOSE, VOLUME]
        with pytest.raises(ValueError):
            self.daily_bar_reader.load_raw_arrays(
                columns,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
                query_assets,
            )

    def test_unadjusted_get_value(self):
        """Test get_value() on both a price field (CLOSE) and VOLUME."""
        reader = self.daily_bar_reader

        def make_failure_msg(asset, date, field):
            return "Unexpected value for sid={}; date={}; field={}.".format(
                asset, date.date(), field
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
                    missing_value=np.nan,
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
                    missing_value=np.nan,
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
                    missing_value=np.nan,
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
            before_start = self.trading_calendar.previous_session(
                self.asset_start(asset)
            )
            after_end = self.trading_calendar.next_session(self.asset_end(asset))

            # Attempting to get data for an asset before its start date
            # should raise NoDataBeforeDate.
            if TEST_CALENDAR_START <= before_start <= TEST_CALENDAR_STOP:
                with pytest.raises(NoDataBeforeDate):
                    reader.get_value(asset, before_start, CLOSE)

            # Attempting to get data for an asset after its end date
            # should raise NoDataAfterDate.
            if TEST_CALENDAR_START <= after_end <= TEST_CALENDAR_STOP:
                with pytest.raises(NoDataAfterDate):
                    reader.get_value(asset, after_end, CLOSE)

        # Retrieving data for "holes" (dates with no data, but within
        # an  asset's lifetime) should not raise an exception. nan is
        # returned for OHLC fields, and 0 is returned for volume.
        for asset, dates in self.holes.items():
            for date in dates:
                assert_equal(
                    reader.get_value(asset, date, CLOSE),
                    np.nan,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-nan value for close."
                    ).format(asset, date.date()),
                )
                assert_equal(
                    reader.get_value(asset, date, VOLUME),
                    0.0,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-zero value for volume."
                    ).format(asset, date.date()),
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
            mid_date = pd.Timestamp("2015-06-15")
            if self.asset_start(sid) <= mid_date:
                expected = min(self.asset_end(sid), mid_date)
            else:
                expected = pd.NaT

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
                    pd.Timestamp(0),
                ),
                pd.NaT,
            )

    def test_listing_currency(self):
        # Test loading on all assets.
        all_assets = np.array(list(self.assets))
        all_results = self.daily_bar_reader.currency_codes(all_assets)
        all_expected = self.make_equity_daily_bar_currency_codes(
            self.DAILY_BARS_TEST_QUERY_COUNTRY_CODE,
            all_assets,
        ).values
        assert_equal(all_results, all_expected)

        assert all_results.dtype == np.dtype(object)
        for code in all_results:
            assert isinstance(code, str)

        # Check all possible subsets of assets.
        for indices in map(list, powerset(range(len(all_assets)))):
            # Empty queries aren't currently supported.
            if not indices:
                continue
            assets = all_assets[indices]
            results = self.daily_bar_reader.currency_codes(assets)
            expected = all_expected[indices]

            assert_equal(results, expected)

    def test_listing_currency_for_nonexistent_asset(self):
        reader = self.daily_bar_reader

        valid_sid = max(self.assets)
        valid_currency = reader.currency_codes(np.array([valid_sid]))[0]
        invalid_sids = [-1, -2]

        # XXX: We currently require at least one valid sid here, because the
        # MultiCountryDailyBarReader needs one valid sid to be able to dispatch
        # to a child reader. We could probably make that work, but there are no
        # real-world cases where we expect to get all-invalid currency queries,
        # so it's unclear whether we should do work to explicitly support such
        # queries.
        mixed = np.array(invalid_sids + [valid_sid])
        result = self.daily_bar_reader.currency_codes(mixed)
        expected = np.array([None] * 2 + [valid_currency])
        assert_equal(result, expected)


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


class BcolzDailyBarWriterMissingDataTestCase(
    WithAssetFinder, WithTmpDir, WithTradingCalendars, ZiplineTestCase
):
    # Sid 5 is active from 2015-06-02 to 2015-06-30.
    MISSING_DATA_SID = 5
    # Leave out data for a day in the middle of the query range.
    MISSING_DATA_DAY = pd.Timestamp("2015-06-15")

    @classmethod
    def make_equity_info(cls):
        return EQUITY_INFO.loc[EQUITY_INFO.index == cls.MISSING_DATA_SID].copy()

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
            "[Timestamp('2015-06-15 00:00:00')]\n"
            "Extra sessions: []"
        )
        with pytest.raises(AssertionError, match=expected_msg):
            writer.write(bar_data)


class _HDF5DailyBarTestCase(
    WithHDF5EquityMultiCountryDailyBarReader, _DailyBarsTestCase
):
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
                msg=("asset_end_dates value for sid={} differs from expected").format(
                    sid
                ),
            )

    def test_read_first_trading_day(self):
        assert self.daily_bar_reader.first_trading_day == self.sessions[0]

    def test_asset_start_dates(self):
        assert_sequence_equal(self.single_country_reader.sids, self.assets)

        for ix, sid in enumerate(self.single_country_reader.sids):
            assert_equal(
                self.single_country_reader.asset_start_dates[ix],
                self.asset_start(sid).asm8,
                msg=("asset_start_dates value for sid={} differs from expected").format(
                    sid
                ),
            )

    def test_sessions(self):
        assert_equal(self.daily_bar_reader.sessions, self.sessions)

    def test_invalid_date(self):
        INVALID_DATES = (
            # Before the start of the daily bars.
            self.trading_calendar.previous_session(TEST_CALENDAR_START),
            # A Sunday.
            pd.Timestamp("2015-06-07", tz="UTC"),
            # After the end of the daily bars.
            self.trading_calendar.next_session(TEST_CALENDAR_STOP),
        )

        for invalid_date in INVALID_DATES:
            with pytest.raises(NoDataOnDate):
                self.daily_bar_reader.load_raw_arrays(
                    OHLCV,
                    invalid_date,
                    TEST_QUERY_STOP,
                    self.assets,
                )

            with pytest.raises(NoDataOnDate):
                self.daily_bar_reader.get_value(
                    self.assets[0],
                    invalid_date,
                    "close",
                )

    @parameterized.expand(
        [
            (["open"],),
            (["close", "volume"],),
            (["volume", "high", "low"],),
            (["open", "high", "low", "close", "volume"],),
        ]
    )
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
            assets = assets[: np.random.randint(1, len(assets))]
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
        columns = ["high", "volume"]
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
        columns = ["close", "volume"]
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
        columns = ["close", "volume"]
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
        columns = [CLOSE, VOLUME]
        for asset in self.assets:
            self._check_read_results(
                columns,
                self.assets,
                start_date=self.sessions[0],
                end_date=self.asset_end(asset),
            )

    def test_read_known_and_unknown_sids(self):
        """
        Test a query with some known sids mixed in with unknown sids.
        """

        # Construct a list of alternating valid and invalid query sids,
        # bookended by invalid sids.
        #
        # E.g.
        #   INVALID VALID INVALID VALID ... VALID INVALID
        query_assets = (
            [self.assets[-1] + 1]
            + list(range(self.assets[0], self.assets[-1] + 1))
            + [self.assets[-1] + 3]
        )

        columns = [CLOSE, VOLUME]
        self._check_read_results(
            columns,
            query_assets,
            start_date=TEST_QUERY_START,
            end_date=TEST_QUERY_STOP,
        )

    @parameterized.expand(
        [
            # Query for only even sids, only odd ids are valid.
            ([],),
            ([2],),
            ([2, 4, 800],),
        ]
    )
    def test_read_only_unknown_sids(self, query_assets):
        columns = [CLOSE, VOLUME]
        with pytest.raises(ValueError):
            self.daily_bar_reader.load_raw_arrays(
                columns,
                TEST_QUERY_START,
                TEST_QUERY_STOP,
                query_assets,
            )

    def test_unadjusted_get_value(self):
        """Test get_value() on both a price field (CLOSE) and VOLUME."""
        reader = self.daily_bar_reader

        def make_failure_msg(asset, date, field):
            return "Unexpected value for sid={}; date={}; field={}.".format(
                asset, date.date(), field
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
                    missing_value=np.nan,
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
                    missing_value=np.nan,
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
                    missing_value=np.nan,
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
            before_start = self.trading_calendar.previous_session(
                self.asset_start(asset)
            )
            after_end = self.trading_calendar.next_session(self.asset_end(asset))

            # Attempting to get data for an asset before its start date
            # should raise NoDataBeforeDate.
            if TEST_CALENDAR_START <= before_start <= TEST_CALENDAR_STOP:
                with pytest.raises(NoDataBeforeDate):
                    reader.get_value(asset, before_start, CLOSE)

            # Attempting to get data for an asset after its end date
            # should raise NoDataAfterDate.
            if TEST_CALENDAR_START <= after_end <= TEST_CALENDAR_STOP:
                with pytest.raises(NoDataAfterDate):
                    reader.get_value(asset, after_end, CLOSE)

        # Retrieving data for "holes" (dates with no data, but within
        # an  asset's lifetime) should not raise an exception. nan is
        # returned for OHLC fields, and 0 is returned for volume.
        for asset, dates in self.holes.items():
            for date in dates:
                assert_equal(
                    reader.get_value(asset, date, CLOSE),
                    np.nan,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-nan value for close."
                    ).format(asset, date.date()),
                )
                assert_equal(
                    reader.get_value(asset, date, VOLUME),
                    0.0,
                    msg=(
                        "Expected a hole for sid={}; date={}, but got a"
                        " non-zero value for volume."
                    ).format(asset, date.date()),
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
            mid_date = pd.Timestamp("2015-06-15")
            if self.asset_start(sid) <= mid_date:
                expected = min(self.asset_end(sid), mid_date)
            else:
                expected = pd.NaT

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
                    pd.Timestamp(0, tz="UTC"),
                ),
                pd.NaT,
            )

    def test_listing_currency(self):
        # Test loading on all assets.
        all_assets = np.array(list(self.assets))
        all_results = self.daily_bar_reader.currency_codes(all_assets)
        all_expected = self.make_equity_daily_bar_currency_codes(
            self.DAILY_BARS_TEST_QUERY_COUNTRY_CODE,
            all_assets,
        ).values
        assert_equal(all_results, all_expected)

        assert all_results.dtype == np.dtype(object)
        for code in all_results:
            assert isinstance(code, str)

        # Check all possible subsets of assets.
        for indices in map(list, powerset(range(len(all_assets)))):
            # Empty queries aren't currently supported.
            if not indices:
                continue
            assets = all_assets[indices]
            results = self.daily_bar_reader.currency_codes(assets)
            expected = all_expected[indices]

            assert_equal(results, expected)

    def test_listing_currency_for_nonexistent_asset(self):
        reader = self.daily_bar_reader

        valid_sid = max(self.assets)
        valid_currency = reader.currency_codes(np.array([valid_sid]))[0]
        invalid_sids = [-1, -2]

        # XXX: We currently require at least one valid sid here, because the
        # MultiCountryDailyBarReader needs one valid sid to be able to dispatch
        # to a child reader. We could probably make that work, but there are no
        # real-world cases where we expect to get all-invalid currency queries,
        # so it's unclear whether we should do work to explicitly support such
        # queries.
        mixed = np.array(invalid_sids + [valid_sid])
        result = self.daily_bar_reader.currency_codes(mixed)
        expected = np.array([None] * 2 + [valid_currency])
        assert_equal(result, expected)


class HDF5DailyBarUSTestCase(_HDF5DailyBarTestCase):
    DAILY_BARS_TEST_QUERY_COUNTRY_CODE = "US"


class HDF5DailyBarCanadaTestCase(_HDF5DailyBarTestCase):
    TRADING_CALENDAR_PRIMARY_CAL = "TSX"
    DAILY_BARS_TEST_QUERY_COUNTRY_CODE = "CA"


class TestCoerceToUint32Price:
    """Test the coerce_to_uint32() function used by the HDF5DailyBarWriter."""

    @pytest.mark.parametrize(
        "field, expected",
        [
            (OPEN, np.array([1, 1000, 100000, 100500, 1000005, 130230], dtype="u4")),
            (HIGH, np.array([1, 1000, 100000, 100500, 1000005, 130230], dtype="u4")),
            (LOW, np.array([1, 1000, 100000, 100500, 1000005, 130230], dtype="u4")),
            (CLOSE, np.array([1, 1000, 100000, 100500, 1000005, 130230], dtype="u4")),
            (VOLUME, np.array([0, 1, 100, 100, 1000, 130], dtype="u4")),
        ],
    )
    def test_coerce_to_uint32_price(self, field, expected):
        # NOTE: 130.23 is not perfectly representable as a double, but we
        # shouldn't truncate and be off by an entire cent
        coerced = coerce_to_uint32(
            np.array([0.001, 1, 100, 100.5, 1000.005, 130.23], dtype=np.float64),
            DEFAULT_SCALING_FACTORS[field],
        )

        assert_equal(coerced, expected)
