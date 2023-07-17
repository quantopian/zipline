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

"""Tests for USEquityPricingLoader and related classes."""

from parameterized import parameterized
import sys
import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)
import pandas as pd
from pandas.testing import assert_frame_equal
from toolz.curried.operator import getitem

from zipline.lib.adjustment import Float64Multiply
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.loaders.synthetic import (
    NullAdjustmentReader,
    make_bar_data,
    expected_bar_values_2d,
)
from zipline.pipeline.loaders.equity_pricing_loader import (
    USEquityPricingLoader,
)

from zipline.errors import WindowLengthTooLong
from zipline.pipeline.data import USEquityPricing
from zipline.testing import (
    seconds_to_timestamp,
    str_to_seconds,
    MockDailyBarReader,
)
from zipline.testing.fixtures import (
    WithAdjustmentReader,
    ZiplineTestCase,
)
import pytest

# Test calendar ranges over the month of June 2015
#      June 2015
# Mo Tu We Th Fr Sa Su
#  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14
# 15 16 17 18 19 20 21
# 22 23 24 25 26 27 28
# 29 30
TEST_CALENDAR_START = pd.Timestamp("2015-06-01")
TEST_CALENDAR_STOP = pd.Timestamp("2015-06-30")

TEST_QUERY_START = pd.Timestamp("2015-06-10")
TEST_QUERY_STOP = pd.Timestamp("2015-06-19")

# One asset for each of the cases enumerated in load_raw_arrays_from_bcolz.
EQUITY_INFO = pd.DataFrame(
    [
        # 1) The equity's trades start and end before query.
        {"start_date": "2015-06-01", "end_date": "2015-06-05"},
        # 2) The equity's trades start and end after query.
        {"start_date": "2015-06-22", "end_date": "2015-06-30"},
        # 3) The equity's data covers all dates in range.
        {"start_date": "2015-06-02", "end_date": "2015-06-30"},
        # 4) The equity's trades start before the query start, but stop
        #    before the query end.
        {"start_date": "2015-06-01", "end_date": "2015-06-15"},
        # 5) The equity's trades start and end during the query.
        {"start_date": "2015-06-12", "end_date": "2015-06-18"},
        # 6) The equity's trades start during the query, but extend through
        #    the whole query.
        {"start_date": "2015-06-15", "end_date": "2015-06-25"},
    ],
    index=np.arange(1, 7),
    columns=["start_date", "end_date"],
).astype("datetime64[ns]")
EQUITY_INFO["symbol"] = [chr(ord("A") + n) for n in range(len(EQUITY_INFO))]
EQUITY_INFO["exchange"] = "TEST"

TEST_QUERY_SIDS = EQUITY_INFO.index

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
SPLITS = pd.DataFrame(
    [
        # Before query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-03"),
            "ratio": 1.103,
            "sid": 1,
        },
        # First day of query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-10"),
            "ratio": 3.110,
            "sid": 3,
        },
        # Third day of query range, should have last_row of 2
        {
            "effective_date": str_to_seconds("2015-06-12"),
            "ratio": 3.112,
            "sid": 3,
        },
        # After query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-21"),
            "ratio": 6.121,
            "sid": 6,
        },
        # Another action in query range, should have last_row of 1
        {
            "effective_date": str_to_seconds("2015-06-11"),
            "ratio": 3.111,
            "sid": 3,
        },
        # Last day of range.  Should have last_row of 7
        {
            "effective_date": str_to_seconds("2015-06-19"),
            "ratio": 3.119,
            "sid": 3,
        },
    ],
    columns=["effective_date", "ratio", "sid"],
)

MERGERS = pd.DataFrame(
    [
        # Before query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-03"),
            "ratio": 1.203,
            "sid": 1,
        },
        # First day of query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-10"),
            "ratio": 3.210,
            "sid": 3,
        },
        # Third day of query range, should have last_row of 2
        {
            "effective_date": str_to_seconds("2015-06-12"),
            "ratio": 3.212,
            "sid": 3,
        },
        # After query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-25"),
            "ratio": 6.225,
            "sid": 6,
        },
        # Another action in query range, should have last_row of 2
        {
            "effective_date": str_to_seconds("2015-06-12"),
            "ratio": 4.212,
            "sid": 4,
        },
        # Last day of range.  Should have last_row of 7
        {
            "effective_date": str_to_seconds("2015-06-19"),
            "ratio": 3.219,
            "sid": 3,
        },
    ],
    columns=["effective_date", "ratio", "sid"],
)

DIVIDENDS = pd.DataFrame(
    [
        # Before query range, should be excluded.
        {
            "declared_date": pd.Timestamp("2015-05-01", tz="UTC").to_datetime64(),
            "ex_date": pd.Timestamp("2015-06-01", tz="UTC").to_datetime64(),
            "record_date": pd.Timestamp("2015-06-03", tz="UTC").to_datetime64(),
            "pay_date": pd.Timestamp("2015-06-05", tz="UTC").to_datetime64(),
            "amount": 90.0,
            "sid": 1,
        },
        # First day of query range, should be excluded.
        {
            "declared_date": pd.Timestamp("2015-06-01", tz="UTC").to_datetime64(),
            "ex_date": pd.Timestamp("2015-06-10", tz="UTC").to_datetime64(),
            "record_date": pd.Timestamp("2015-06-15", tz="UTC").to_datetime64(),
            "pay_date": pd.Timestamp("2015-06-17", tz="UTC").to_datetime64(),
            "amount": 80.0,
            "sid": 3,
        },
        # Third day of query range, should have last_row of 2
        {
            "declared_date": pd.Timestamp("2015-06-01", tz="UTC").to_datetime64(),
            "ex_date": pd.Timestamp("2015-06-12", tz="UTC").to_datetime64(),
            "record_date": pd.Timestamp("2015-06-15", tz="UTC").to_datetime64(),
            "pay_date": pd.Timestamp("2015-06-17", tz="UTC").to_datetime64(),
            "amount": 70.0,
            "sid": 3,
        },
        # After query range, should be excluded.
        {
            "declared_date": pd.Timestamp("2015-06-01", tz="UTC").to_datetime64(),
            "ex_date": pd.Timestamp("2015-06-25", tz="UTC").to_datetime64(),
            "record_date": pd.Timestamp("2015-06-28", tz="UTC").to_datetime64(),
            "pay_date": pd.Timestamp("2015-06-30", tz="UTC").to_datetime64(),
            "amount": 60.0,
            "sid": 6,
        },
        # Another action in query range, should have last_row of 3
        {
            "declared_date": pd.Timestamp("2015-06-01", tz="UTC").to_datetime64(),
            "ex_date": pd.Timestamp("2015-06-15", tz="UTC").to_datetime64(),
            "record_date": pd.Timestamp("2015-06-18", tz="UTC").to_datetime64(),
            "pay_date": pd.Timestamp("2015-06-20", tz="UTC").to_datetime64(),
            "amount": 50.0,
            "sid": 3,
        },
        # Last day of range.  Should have last_row of 7
        {
            "declared_date": pd.Timestamp("2015-06-01", tz="UTC").to_datetime64(),
            "ex_date": pd.Timestamp("2015-06-19", tz="UTC").to_datetime64(),
            "record_date": pd.Timestamp("2015-06-22", tz="UTC").to_datetime64(),
            "pay_date": pd.Timestamp("2015-06-30", tz="UTC").to_datetime64(),
            "amount": 40.0,
            "sid": 3,
        },
    ],
    columns=[
        "declared_date",
        "ex_date",
        "record_date",
        "pay_date",
        "amount",
        "sid",
    ],
)

DIVIDENDS_EXPECTED = pd.DataFrame(
    [
        # Before query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-01"),
            "ratio": 0.1,
            "sid": 1,
        },
        # First day of query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-10"),
            "ratio": 0.20,
            "sid": 3,
        },
        # Third day of query range, should have last_row of 2
        {
            "effective_date": str_to_seconds("2015-06-12"),
            "ratio": 0.30,
            "sid": 3,
        },
        # After query range, should be excluded.
        {
            "effective_date": str_to_seconds("2015-06-25"),
            "ratio": 0.40,
            "sid": 6,
        },
        # Another action in query range, should have last_row of 3
        {
            "effective_date": str_to_seconds("2015-06-15"),
            "ratio": 0.50,
            "sid": 3,
        },
        # Last day of range.  Should have last_row of 7
        {
            "effective_date": str_to_seconds("2015-06-19"),
            "ratio": 0.60,
            "sid": 3,
        },
    ],
    columns=["effective_date", "ratio", "sid"],
)


class USEquityPricingLoaderTestCase(WithAdjustmentReader, ZiplineTestCase):
    START_DATE = TEST_CALENDAR_START
    END_DATE = TEST_CALENDAR_STOP
    asset_ids = 1, 2, 3

    @classmethod
    def make_equity_info(cls):
        return EQUITY_INFO

    @classmethod
    def make_splits_data(cls):
        return SPLITS

    @classmethod
    def make_mergers_data(cls):
        return MERGERS

    @classmethod
    def make_dividends_data(cls):
        return DIVIDENDS

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return MockDailyBarReader(
            dates=cls.calendar_days_between(cls.START_DATE, cls.END_DATE),
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        return make_bar_data(
            EQUITY_INFO,
            cls.equity_daily_bar_days,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(USEquityPricingLoaderTestCase, cls).init_class_fixtures()
        cls.sids = TEST_QUERY_SIDS
        cls.asset_info = EQUITY_INFO

    def test_input_sanity(self):
        # Ensure that the input data doesn't contain adjustments during periods
        # where the corresponding asset didn't exist.
        for table in SPLITS, MERGERS:
            for eff_date_secs, _, sid in table.itertuples(index=False):
                eff_date = pd.Timestamp(eff_date_secs, unit="s")
                asset_start, asset_end = EQUITY_INFO.loc[
                    sid, ["start_date", "end_date"]
                ]
                assert eff_date >= asset_start
                assert eff_date <= asset_end

    @classmethod
    def calendar_days_between(cls, start_date, end_date, shift=0):
        slice_ = cls.equity_daily_bar_days.slice_indexer(start_date, end_date)
        start = slice_.start + shift
        stop = slice_.stop + shift
        if start < 0:
            raise KeyError(start_date, shift)

        return cls.equity_daily_bar_days[start:stop]

    def expected_adjustments(self, start_date, end_date, tables, adjustment_type):
        price_adjustments = {}
        volume_adjustments = {}

        should_include_price_adjustments = (
            adjustment_type == "all" or adjustment_type == "price"
        )
        should_include_volume_adjustments = (
            adjustment_type == "all" or adjustment_type == "volume"
        )

        query_days = self.calendar_days_between(start_date, end_date)
        start_loc = query_days.get_loc(start_date)

        for table in tables:
            for eff_date_secs, ratio, sid in table.itertuples(index=False):
                eff_date = pd.Timestamp(eff_date_secs, unit="s")

                # Ignore adjustments outside the query bounds.
                if not (start_date <= eff_date <= end_date):
                    continue

                eff_date_loc = query_days.get_loc(eff_date)
                delta = eff_date_loc - start_loc

                # Pricing adjustments should be applied on the date
                # corresponding to the effective date of the input data. They
                # should affect all rows **before** the effective date.
                if should_include_price_adjustments:
                    price_adjustments.setdefault(delta, []).append(
                        Float64Multiply(
                            first_row=0,
                            last_row=delta,
                            first_col=sid - 1,
                            last_col=sid - 1,
                            value=ratio,
                        )
                    )
                # Volume is *inversely* affected by *splits only*.
                if table is SPLITS and should_include_volume_adjustments:
                    volume_adjustments.setdefault(delta, []).append(
                        Float64Multiply(
                            first_row=0,
                            last_row=delta,
                            first_col=sid - 1,
                            last_col=sid - 1,
                            value=1.0 / ratio,
                        )
                    )

        output = {}
        if should_include_price_adjustments:
            output["price_adjustments"] = price_adjustments
        if should_include_volume_adjustments:
            output["volume_adjustments"] = volume_adjustments

        return output

    @parameterized.expand(
        [
            ([SPLITS, MERGERS, DIVIDENDS_EXPECTED], "all"),
            ([SPLITS, MERGERS, DIVIDENDS_EXPECTED], "price"),
            ([SPLITS, MERGERS, DIVIDENDS_EXPECTED], "volume"),
            ([SPLITS, MERGERS, None], "all"),
            ([SPLITS, MERGERS, None], "price"),
        ]
    )
    def test_load_adjustments(self, tables, adjustment_type):
        query_days = self.calendar_days_between(
            TEST_QUERY_START,
            TEST_QUERY_STOP,
        )

        adjustments = self.adjustment_reader.load_adjustments(
            query_days,
            self.sids,
            should_include_splits=tables[0] is not None,
            should_include_mergers=tables[1] is not None,
            should_include_dividends=tables[2] is not None,
            adjustment_type=adjustment_type,
        )
        expected_adjustments = self.expected_adjustments(
            TEST_QUERY_START,
            TEST_QUERY_STOP,
            [table for table in tables if table is not None],
            adjustment_type,
        )

        if adjustment_type == "all" or adjustment_type == "price":
            expected_price_adjustments = expected_adjustments["price_adjustments"]
            for key in expected_price_adjustments:
                price_adjustment = adjustments["price"][key]
                for j, adj in enumerate(price_adjustment):
                    expected = expected_price_adjustments[key][j]
                    assert adj.first_row == expected.first_row
                    assert adj.last_row == expected.last_row
                    assert adj.first_col == expected.first_col
                    assert adj.last_col == expected.last_col
                    assert_allclose(adj.value, expected.value)

        if adjustment_type == "all" or adjustment_type == "volume":
            expected_volume_adjustments = expected_adjustments["volume_adjustments"]
            for key in expected_volume_adjustments:
                volume_adjustment = adjustments["volume"][key]
                for j, adj in enumerate(volume_adjustment):
                    expected = expected_volume_adjustments[key][j]
                    assert adj.first_row == expected.first_row
                    assert adj.last_row == expected.last_row
                    assert adj.first_col == expected.first_col
                    assert adj.last_col == expected.last_col
                    assert_allclose(adj.value, expected.value)

    @parameterized.expand([(True,), (False,)])
    @pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
    def test_load_adjustments_to_df(self, convert_dts):
        reader = self.adjustment_reader
        adjustment_dfs = reader.unpack_db_to_component_dfs(convert_dates=convert_dts)

        name_and_raw = (
            ("splits", SPLITS),
            ("mergers", MERGERS),
            ("dividends", DIVIDENDS_EXPECTED),
        )

        def create_expected_table(df, name):
            expected_df = df.copy()

            if convert_dts:
                for colname in reader._datetime_int_cols[name]:
                    expected_df[colname] = pd.to_datetime(
                        expected_df[colname], unit="s"
                    )

            return expected_df

        def create_expected_div_table(df, name):
            expected_df = df.copy()

            for colname in reader._datetime_int_cols[name]:
                if not convert_dts:
                    expected_df[colname] = (
                        expected_df[colname].astype("datetime64[s]").view(int)
                    )

            return expected_df

        for action_name, raw_tbl in name_and_raw:
            # todo: fix missing dividend value
            if action_name == "dividends":
                continue

            exp = create_expected_table(raw_tbl, action_name)
            assert_frame_equal(adjustment_dfs[action_name], exp)

        # DIVIDENDS is in the opposite form from the rest of the dataframes, so
        # needs to be converted separately.
        div_name = "dividend_payouts"
        exp = create_expected_div_table(DIVIDENDS, div_name)
        assert_frame_equal(adjustment_dfs[div_name].loc[:, exp.columns], exp)

    def test_read_no_adjustments(self):
        adjustment_reader = NullAdjustmentReader()
        columns = [USEquityPricing.close, USEquityPricing.volume]
        query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP)
        # Our expected results for each day are based on values from the
        # previous day.
        shifted_query_days = self.calendar_days_between(
            TEST_QUERY_START,
            TEST_QUERY_STOP,
            shift=-1,
        )

        adjustments = adjustment_reader.load_pricing_adjustments(
            [c.name for c in columns],
            query_days,
            self.sids,
        )
        assert adjustments == [{}, {}]

        pricing_loader = USEquityPricingLoader.without_fx(
            self.bcolz_equity_daily_bar_reader,
            adjustment_reader,
        )

        results = pricing_loader.load_adjusted_array(
            domain=US_EQUITIES,
            columns=columns,
            dates=query_days,
            sids=self.sids,
            mask=np.ones((len(query_days), len(self.sids)), dtype=bool),
        )
        closes, volumes = map(getitem(results), columns)

        expected_baseline_closes = expected_bar_values_2d(
            shifted_query_days,
            self.sids,
            self.asset_info,
            "close",
        )
        expected_baseline_volumes = expected_bar_values_2d(
            shifted_query_days,
            self.sids,
            self.asset_info,
            "volume",
        )

        # AdjustedArrays should yield the same data as the expected baseline.
        for windowlen in range(1, len(query_days) + 1):
            for offset, window in enumerate(closes.traverse(windowlen)):
                assert_array_equal(
                    expected_baseline_closes[offset : offset + windowlen],
                    window,
                )

            for offset, window in enumerate(volumes.traverse(windowlen)):
                assert_array_equal(
                    expected_baseline_volumes[offset : offset + windowlen],
                    window,
                )

        # Verify that we checked up to the longest possible window.
        with pytest.raises(WindowLengthTooLong):
            closes.traverse(windowlen + 1)
        with pytest.raises(WindowLengthTooLong):
            volumes.traverse(windowlen + 1)

    def apply_adjustments(self, dates, assets, baseline_values, adjustments):
        min_date, max_date = dates[[0, -1]]
        # HACK: Simulate the coercion to float64 we do in adjusted_array.  This
        # should be removed when AdjustedArray properly supports
        # non-floating-point types.
        orig_dtype = baseline_values.dtype
        values = baseline_values.astype(np.float64).copy()
        for eff_date_secs, ratio, sid in adjustments.itertuples(index=False):
            eff_date = seconds_to_timestamp(eff_date_secs)
            # Don't apply adjustments that aren't in the current date range.
            if eff_date not in dates:
                continue
            eff_date_loc = dates.get_loc(eff_date)
            asset_col = assets.get_loc(sid)
            # Apply ratio multiplicatively to the asset column on all rows less
            # than or equal adjustment effective date.
            values[: eff_date_loc + 1, asset_col] *= ratio
        return values.astype(orig_dtype)

    def test_read_with_adjustments(self):
        columns = [USEquityPricing.high, USEquityPricing.volume]
        query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP)
        # Our expected results for each day are based on values from the
        # previous day.
        shifted_query_days = self.calendar_days_between(
            TEST_QUERY_START,
            TEST_QUERY_STOP,
            shift=-1,
        )

        pricing_loader = USEquityPricingLoader.without_fx(
            self.bcolz_equity_daily_bar_reader,
            self.adjustment_reader,
        )

        results = pricing_loader.load_adjusted_array(
            domain=US_EQUITIES,
            columns=columns,
            dates=query_days,
            sids=pd.Index(np.arange(1, 7), dtype="int64"),
            mask=np.ones((len(query_days), 6), dtype=bool),
        )
        highs, volumes = map(getitem(results), columns)

        expected_baseline_highs = expected_bar_values_2d(
            shifted_query_days,
            self.sids,
            self.asset_info,
            "high",
        )
        expected_baseline_volumes = expected_bar_values_2d(
            shifted_query_days,
            self.sids,
            self.asset_info,
            "volume",
        )

        # At each point in time, the AdjustedArrays should yield the baseline
        # with all adjustments up to that date applied.
        for windowlen in range(1, len(query_days) + 1):
            for offset, window in enumerate(highs.traverse(windowlen)):
                baseline = expected_baseline_highs[offset : offset + windowlen]
                baseline_dates = query_days[offset : offset + windowlen]
                expected_adjusted_highs = self.apply_adjustments(
                    baseline_dates,
                    self.sids,
                    baseline,
                    # Apply all adjustments.
                    pd.concat([SPLITS, MERGERS, DIVIDENDS_EXPECTED], ignore_index=True),
                )
                assert_allclose(expected_adjusted_highs, window)

            for offset, window in enumerate(volumes.traverse(windowlen)):
                baseline = expected_baseline_volumes[offset : offset + windowlen]
                baseline_dates = query_days[offset : offset + windowlen]
                # Apply only splits and invert the ratio.
                adjustments = SPLITS.copy()
                adjustments.ratio = 1 / adjustments.ratio

                expected_adjusted_volumes = self.apply_adjustments(
                    baseline_dates,
                    self.sids,
                    baseline,
                    adjustments,
                )
                # FIXME: Make AdjustedArray properly support integral types.
                assert_array_equal(
                    expected_adjusted_volumes,
                    window.astype(np.uint32),
                )

        # Verify that we checked up to the longest possible window.
        with pytest.raises(WindowLengthTooLong):
            highs.traverse(windowlen + 1)
        with pytest.raises(WindowLengthTooLong):
            volumes.traverse(windowlen + 1)
