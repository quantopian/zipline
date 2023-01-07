import logging
import pytest
import numpy as np
import pandas as pd

from zipline.data.adjustments import (
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
)
from zipline.data.in_memory_daily_bars import InMemoryDailyBarReader
from zipline.testing import parameter_space
from zipline.testing.predicates import (
    assert_frame_equal,
    assert_series_equal,
)
from zipline.testing.fixtures import (
    WithInstanceTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
)


class TestSQLiteAdjustmentsWriter(
    WithTradingCalendars, WithInstanceTmpDir, ZiplineTestCase
):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def init_instance_fixtures(self):
        super(TestSQLiteAdjustmentsWriter, self).init_instance_fixtures()
        self.db_path = self.instance_tmpdir.getpath("adjustments.db")

    def writer(self, session_bar_reader):
        return self.enter_instance_context(
            SQLiteAdjustmentWriter(
                self.db_path,
                session_bar_reader,
                overwrite=True,
            ),
        )

    def component_dataframes(self, convert_dates=True):
        with SQLiteAdjustmentReader(self.db_path) as r:
            return r.unpack_db_to_component_dfs(convert_dates=convert_dates)

    def empty_in_memory_reader(self, dates, sids):
        nan_frame = pd.DataFrame(
            np.nan,
            index=dates,
            columns=sids,
        )
        frames = {key: nan_frame for key in ("open", "high", "low", "close", "volume")}

        return InMemoryDailyBarReader(
            frames,
            self.trading_calendar,
            currency_codes=pd.Series(index=sids, data="USD"),
        )

    def writer_without_pricing(self, dates, sids):
        return self.writer(self.empty_in_memory_reader(dates, sids))

    def in_memory_reader_for_close(self, close):
        nan_frame = pd.DataFrame(
            np.nan,
            index=close.index,
            columns=close.columns,
        )
        frames = {"close": close}
        for key in "open", "high", "low", "volume":
            frames[key] = nan_frame
        return InMemoryDailyBarReader(
            frames,
            self.trading_calendar,
            currency_codes=pd.Series(index=close.columns, data="USD"),
        )

    def writer_from_close(self, close):
        return self.writer(self.in_memory_reader_for_close(close))

    def assert_all_empty(self, dfs):
        for k, v in dfs.items():
            assert len(v) == 0, f"{k} dataframe should be empty"

    def test_calculate_dividend_ratio(self):
        first_date_ix = 200
        dates = self.trading_calendar.sessions[first_date_ix : first_date_ix + 3]

        before_pricing_data = dates[0] - self.trading_calendar.day
        one_day_past_pricing_data = dates[-1] + self.trading_calendar.day

        ten_days_past_pricing_data = dates[-1] + self.trading_calendar.day * 10

        def T(n):
            # return dates[n].tz_localize("UTC")
            return dates[n]

        close = pd.DataFrame(
            [
                [10.0, 0.5, 30.0],  # noqa
                [9.5, 0.4, np.nan],  # noqa
                [15.0, 0.6, np.nan],
            ],  # noqa
            columns=[0, 1, 2],
            index=dates,
        )

        dividends = pd.DataFrame(
            [
                # ex_date of >=0 means that we cannot get the previous day's
                # close, so we should not expect to see this dividend in the
                # output
                [0, before_pricing_data, 10],
                [0, T(0), 10],
                # previous price was 0.4, meaning the dividend amount
                # is greater than or equal to price and the ratio would be
                # negative. we should warn and drop this row
                [1, T(1), 0.51],
                # previous price was 0.4, meaning the dividend amount
                # is exactly equal to price and the ratio would be 0.
                # we should warn and drop this row
                [1, T(2), 0.4],
                # previous price is nan, so we cannot compute the ratio.
                # we should warn and drop this row
                [2, T(2), 10],
                # previous price was 10, expected ratio is 0.95
                [0, T(1), 0.5],
                # previous price was 0.4, expected ratio is 0.9
                [1, T(2), 0.04],
                # we shouldn't crash in the process of warning/dropping this
                # row even though it is past the range of `dates`
                [2, one_day_past_pricing_data, 0.1],
                [2, ten_days_past_pricing_data, 0.1],
            ],
            columns=["sid", "ex_date", "amount"],
        )

        # give every extra date field a unique date so that we can make sure
        # they appear unchanged in the dividends payouts
        ix = first_date_ix
        for col in "declared_date", "record_date", "pay_date":
            extra_dates = self.trading_calendar.sessions[ix : ix + len(dividends)]
            ix += len(dividends)
            dividends[col] = extra_dates

        self.writer_from_close(close).write(dividends=dividends)
        dfs = self.component_dataframes()
        dividend_payouts = dfs.pop("dividend_payouts")
        dividend_ratios = dfs.pop("dividends")
        self.assert_all_empty(dfs)

        payout_sort_key = ["sid", "ex_date", "amount"]
        dividend_payouts = dividend_payouts.sort_values(payout_sort_key)
        dividend_payouts = dividend_payouts.reset_index(drop=True)

        expected_dividend_payouts = dividend_payouts.sort_values(
            payout_sort_key,
        )
        expected_dividend_payouts.reset_index(drop=True, inplace=True)

        assert_frame_equal(dividend_payouts, expected_dividend_payouts)

        expected_dividend_ratios = pd.DataFrame(
            [
                [T(1), 0.95, 0],
                [T(2), 0.90, 1],
            ],
            columns=["effective_date", "ratio", "sid"],
        )
        dividend_ratios.sort_values(
            ["effective_date", "sid"],
            inplace=True,
        )
        dividend_ratios.reset_index(drop=True, inplace=True)
        assert_frame_equal(dividend_ratios, expected_dividend_ratios)

        with self._caplog.at_level(logging.WARNING):
            assert (
                "Couldn't compute ratio for dividend sid=2, ex_date=1990-10-18, amount=10.000"
                in self._caplog.messages
            )
            assert (
                "Couldn't compute ratio for dividend sid=2, ex_date=1990-10-19, amount=0.100"
                in self._caplog.messages
            )

            assert (
                "Couldn't compute ratio for dividend sid=2, ex_date=1990-11-01, amount=0.100"
                in self._caplog.messages
            )

            assert (
                "Dividend ratio <= 0 for dividend sid=1, ex_date=1990-10-17, amount=0.510"
                in self._caplog.messages
            )

            assert (
                "Dividend ratio <= 0 for dividend sid=1, ex_date=1990-10-18, amount=0.400"
                in self._caplog.messages
            )

    def _test_identity(self, name):
        sids = np.arange(5)

        # tx_convert makes tz-naive
        dates = self.trading_calendar.sessions

        def T(n):
            return dates[n]

        sort_key = ["effective_date", "sid", "ratio"]
        input_ = pd.DataFrame(
            [
                [T(0), 0.1, 1],
                [T(1), 2.0, 1],
                [T(0), 0.1, 2],
                [T(4), 2.0, 2],
                [T(8), 2.4, 2],
            ],
            columns=["effective_date", "ratio", "sid"],
        ).sort_values(sort_key)

        self.writer_without_pricing(dates, sids).write(**{name: input_})
        dfs = self.component_dataframes()

        output = dfs.pop(name).sort_values(sort_key)
        self.assert_all_empty(dfs)
        # from nose.tools import set_trace;set_trace()
        assert_frame_equal(input_, output)

    def test_splits(self):
        self._test_identity("splits")

    def test_mergers(self):
        self._test_identity("mergers")

    def test_stock_dividends(self):
        sids = np.arange(5)
        dates = self.trading_calendar.sessions

        def T(n):
            return dates[n]

        sort_key = ["sid", "ex_date", "payment_sid", "ratio"]
        input_ = pd.DataFrame(
            [
                [0, T(0), 1.5, 1],
                [0, T(1), 0.5, 2],
                # the same asset has two stock dividends for different assets on
                # the same day
                [1, T(0), 1, 2],
                [1, T(0), 1.2, 3],
            ],
            columns=["sid", "ex_date", "ratio", "payment_sid"],
        ).sort_values(sort_key)

        # give every extra date field a unique date so that we can make sure
        # they appear unchanged in the dividends payouts
        ix = 0
        for col in "declared_date", "record_date", "pay_date":
            extra_dates = dates[ix : ix + len(input_)]
            ix += len(input_)
            input_[col] = extra_dates

        self.writer_without_pricing(dates, sids).write(stock_dividends=input_)
        dfs = self.component_dataframes()

        output = dfs.pop("stock_dividend_payouts").sort_values(sort_key)
        self.assert_all_empty(dfs)

        assert_frame_equal(output, input_[sorted(input_.columns)])

    @parameter_space(convert_dates=[True, False])
    def test_empty_frame_dtypes(self, convert_dates):
        """Test that dataframe dtypes are preserved for empty tables."""

        sids = np.arange(5)
        dates = self.trading_calendar.sessions

        if convert_dates:
            date_dtype = np.dtype("M8[ns]")
        else:
            date_dtype = np.dtype("int64")

        # Write all empty frames.
        self.writer_without_pricing(dates, sids).write()

        dfs = self.component_dataframes(convert_dates)

        for df in dfs.values():
            assert len(df) == 0

        for key in "splits", "mergers", "dividends":
            result = dfs[key].dtypes
            expected = pd.Series(
                {
                    "effective_date": date_dtype,
                    "ratio": np.dtype("float64"),
                    "sid": np.dtype("int64"),
                }
            ).sort_index()
            assert_series_equal(result, expected)

        result = dfs["dividend_payouts"].dtypes
        expected = pd.Series(
            {
                "sid": np.dtype("int64"),
                "ex_date": date_dtype,
                "declared_date": date_dtype,
                "record_date": date_dtype,
                "pay_date": date_dtype,
                "amount": np.dtype("float64"),
            }
        ).sort_index()

        assert_series_equal(result, expected)

        result = dfs["stock_dividend_payouts"].dtypes
        expected = pd.Series(
            {
                "sid": np.dtype("int64"),
                "ex_date": date_dtype,
                "declared_date": date_dtype,
                "record_date": date_dtype,
                "pay_date": date_dtype,
                "payment_sid": np.dtype("int64"),
                "ratio": np.dtype("float64"),
            }
        ).sort_index()

        assert_series_equal(result, expected)
