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
import logging
import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from zipline.data.data_portal import DataPortal
from zipline.errors import (
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
    InvalidBenchmarkAsset,
)
from zipline.sources.benchmark_source import BenchmarkSource
from zipline.testing import (
    MockDailyBarReader,
    create_minute_bar_data,
    tmp_bcolz_equity_minute_bar_reader,
)
from zipline.testing.fixtures import (
    WithDataPortal,
    WithSimParams,
    WithTradingCalendars,
    ZiplineTestCase,
)
from zipline.testing.predicates import assert_equal
from zipline.utils.run_algo import BenchmarkSpec


@pytest.fixture(scope="class")
def set_test_benchmark_spec(request, with_asset_finder):
    ASSET_FINDER_COUNTRY_CODE = "??"
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-12-29")
    request.cls.START_DATE = START_DATE
    request.cls.END_DATE = END_DATE

    zero_returns_index = pd.date_range(
        request.cls.START_DATE,
        request.cls.END_DATE,
        freq="D",
        tz="utc",
    )
    request.cls.zero_returns = pd.Series(index=zero_returns_index, data=0.0)

    equities = pd.DataFrame.from_dict(
        {
            1: {
                "symbol": "A",
                "start_date": START_DATE,
                "end_date": END_DATE + pd.Timedelta(days=1),
                "exchange": "TEST",
            },
            2: {
                "symbol": "B",
                "start_date": START_DATE,
                "end_date": END_DATE + pd.Timedelta(days=1),
                "exchange": "TEST",
            },
        },
        orient="index",
    )

    equities = equities
    exchange_names = [df["exchange"] for df in (equities,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    request.cls.asset_finder = with_asset_finder(
        **dict(equities=equities, exchanges=exchanges)
    )


class TestBenchmark(
    WithDataPortal, WithSimParams, WithTradingCalendars, ZiplineTestCase
):
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-12-29")

    @classmethod
    def make_equity_info(cls):
        return pd.DataFrame.from_dict(
            {
                1: {
                    "symbol": "A",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE + pd.Timedelta(days=1),
                    "exchange": "TEST",
                },
                2: {
                    "symbol": "B",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE + pd.Timedelta(days=1),
                    "exchange": "TEST",
                },
                3: {
                    "symbol": "C",
                    "start_date": pd.Timestamp("2006-05-26"),
                    "end_date": pd.Timestamp("2006-08-09"),
                    "exchange": "TEST",
                },
                4: {
                    "symbol": "D",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE + pd.Timedelta(days=1),
                    "exchange": "TEST",
                },
            },
            orient="index",
        )

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return MockDailyBarReader(
            dates=cls.trading_calendar.sessions_in_range(
                cls.START_DATE,
                cls.END_DATE,
            ),
        )

    @classmethod
    def make_stock_dividends_data(cls):
        declared_date = cls.sim_params.sessions[45]
        ex_date = cls.sim_params.sessions[50]
        record_date = pay_date = cls.sim_params.sessions[55]
        return pd.DataFrame(
            {
                "sid": np.array([4], dtype=np.uint32),
                "payment_sid": np.array([5], dtype=np.uint32),
                "ratio": np.array([2], dtype=np.float64),
                "declared_date": np.array([declared_date], dtype="datetime64[ns]"),
                "ex_date": np.array([ex_date], dtype="datetime64[ns]"),
                "record_date": np.array([record_date], dtype="datetime64[ns]"),
                "pay_date": np.array([pay_date], dtype="datetime64[ns]"),
            }
        )

    def test_normal(self):
        days_to_use = self.sim_params.sessions[1:]

        source = BenchmarkSource(
            self.asset_finder.retrieve_asset(1),
            self.trading_calendar,
            days_to_use,
            self.data_portal,
        )

        # should be the equivalent of getting the price history, then doing
        # a pct_change on it
        manually_calculated = self.data_portal.get_history_window(
            [1],
            days_to_use[-1],
            len(days_to_use),
            "1d",
            "close",
            "daily",
        )[1].pct_change()

        # compare all the fields except the first one, for which we don't have
        # data in manually_calculated
        for idx, day in enumerate(days_to_use[1:]):
            assert source.get_value(day) == manually_calculated[idx + 1]

        # compare a slice of the data
        assert_series_equal(
            source.get_range(days_to_use[1], days_to_use[10]), manually_calculated[1:11]
        )

    def test_asset_not_trading(self):
        benchmark = self.asset_finder.retrieve_asset(3)
        benchmark_start = benchmark.start_date
        benchmark_end = benchmark.end_date

        expected_msg = (
            f"Equity(3 [C]) does not exist on {self.sim_params.sessions[1]}. "
            f"It started trading on {benchmark_start}."
        )
        with pytest.raises(
            BenchmarkAssetNotAvailableTooEarly, match=re.escape(expected_msg)
        ):
            BenchmarkSource(
                benchmark,
                self.trading_calendar,
                self.sim_params.sessions[1:],
                self.data_portal,
            )

        expected_msg = (
            f"Equity(3 [C]) does not exist on {self.sim_params.sessions[-1]}. "
            f"It stopped trading on {benchmark_end}."
        )
        with pytest.raises(
            BenchmarkAssetNotAvailableTooLate, match=re.escape(expected_msg)
        ):
            BenchmarkSource(
                benchmark,
                self.trading_calendar,
                self.sim_params.sessions[120:],
                self.data_portal,
            )

    def test_asset_IPOed_same_day(self):
        # gotta get some minute data up in here.
        # add sid 4 for a couple of days
        minutes = self.trading_calendar.sessions_minutes(
            self.sim_params.sessions[0], self.sim_params.sessions[5]
        )

        tmp_reader = tmp_bcolz_equity_minute_bar_reader(
            self.trading_calendar,
            self.trading_calendar.sessions,
            create_minute_bar_data(minutes, [2]),
        )
        with tmp_reader as reader:
            data_portal = DataPortal(
                self.asset_finder,
                self.trading_calendar,
                first_trading_day=reader.first_trading_day,
                equity_minute_reader=reader,
                equity_daily_reader=self.bcolz_equity_daily_bar_reader,
                adjustment_reader=self.adjustment_reader,
            )

            source = BenchmarkSource(
                self.asset_finder.retrieve_asset(2),
                self.trading_calendar,
                self.sim_params.sessions,
                data_portal,
            )

            days_to_use = self.sim_params.sessions

            # first value should be 0.0, coming from daily data
            assert round(abs(0.0 - source.get_value(days_to_use[0])), 7) == 0

            manually_calculated = data_portal.get_history_window(
                [2],
                days_to_use[-1],
                len(days_to_use),
                "1d",
                "close",
                "daily",
            )[2].pct_change()

            for idx, day in enumerate(days_to_use[1:]):
                assert source.get_value(day) == manually_calculated[idx + 1]

    def test_no_stock_dividends_allowed(self):
        # try to use sid(4) as benchmark, should blow up due to the presence
        # of a stock dividend

        err_msg = (
            "Equity(4 [D]) cannot be used as the benchmark "
            "because it has a stock dividend on 2006-03-16 "
            "00:00:00.  Choose another asset to use as the "
            "benchmark."
        )

        with pytest.raises(InvalidBenchmarkAsset, match=re.escape(err_msg)):
            BenchmarkSource(
                self.asset_finder.retrieve_asset(4),
                self.trading_calendar,
                self.sim_params.sessions,
                self.data_portal,
            )


@pytest.mark.usefixtures("set_test_benchmark_spec")
class TestBenchmarkSpec:
    def resolve_spec(self, spec):
        return spec.resolve(self.asset_finder, self.START_DATE, self.END_DATE)

    def test_no_benchmark(self, caplog):
        """Test running with no benchmark provided.

        We should have no benchmark sid and have a returns series of all zeros.
        """
        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=None,
            benchmark_symbol=None,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        assert sid is None
        assert returns is None

        expected = [
            "No benchmark configured. Assuming algorithm calls set_benchmark.",
            "Pass --benchmark-sid, --benchmark-symbol, or --benchmark-file to set a source of benchmark returns.",  # noqa
            "Pass --no-benchmark to use a dummy benchmark of zero returns.",
        ]

        with caplog.at_level(logging.WARNING):
            assert_equal(caplog.messages, expected)

    def test_no_benchmark_explicitly_disabled(self, caplog):
        """Test running with no benchmark provided, with no_benchmark flag."""
        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=True,
            benchmark_sid=None,
            benchmark_symbol=None,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        assert sid is None
        assert_series_equal(returns, self.zero_returns)

        expected = []
        with caplog.at_level(logging.WARNING):
            assert_equal(caplog.messages, expected)

    @pytest.mark.parametrize("case", [("A", 1), ("B", 2)])
    def test_benchmark_symbol(self, case, caplog):
        """Test running with no benchmark provided, with no_benchmark flag."""
        symbol, expected_sid = case

        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=None,
            benchmark_symbol=symbol,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        assert_equal(sid, expected_sid)
        assert returns is None

        expected = []
        with caplog.at_level(logging.WARNING):
            assert_equal(caplog.messages, expected)

    @pytest.mark.parametrize("input_sid", [1, 2])
    def test_benchmark_sid(self, input_sid, caplog):
        """Test running with no benchmark provided, with no_benchmark flag."""
        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=input_sid,
            benchmark_symbol=None,
            benchmark_file=None,
        )

        sid, returns = self.resolve_spec(spec)

        assert_equal(sid, input_sid)
        assert returns is None

        expected = []
        with caplog.at_level(logging.WARNING):
            assert_equal(caplog.messages, expected)

    def test_benchmark_file(self, tmp_path, caplog):
        """Test running with a benchmark file."""

        csv_file_path = tmp_path / "b.csv"
        with open(csv_file_path, "w") as csv_file:
            csv_file.write(
                "date,return\n"
                "2020-01-03 00:00:00+00:00,-0.1\n"
                "2020-01-06 00:00:00+00:00,0.333\n"
                "2020-01-07 00:00:00+00:00,0.167\n"
                "2020-01-08 00:00:00+00:00,0.143\n"
                "2020-01-09 00:00:00+00:00,6.375\n"
            )

        spec = BenchmarkSpec.from_cli_params(
            no_benchmark=False,
            benchmark_sid=None,
            benchmark_symbol=None,
            benchmark_file=csv_file_path,
        )

        sid, returns = self.resolve_spec(spec)

        assert sid is None

        expected_returns = pd.Series(
            {
                pd.Timestamp("2020-01-03"): -0.1,
                pd.Timestamp("2020-01-06"): 0.333,
                pd.Timestamp("2020-01-07"): 0.167,
                pd.Timestamp("2020-01-08"): 0.143,
                pd.Timestamp("2020-01-09"): 6.375,
            }
        )

        assert_series_equal(returns, expected_returns, check_names=False)

        expected = []
        with caplog.at_level(logging.WARNING):
            assert_equal(caplog.messages, expected)
