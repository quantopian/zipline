#
# Copyright 2018 Quantopian, Inc.
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

import datetime

import numpy as np
import pandas as pd
import pytest

from zipline.finance.metrics import _ClassicRiskMetrics as ClassicRiskMetrics
from zipline.finance.trading import SimulationParameters
from zipline.utils import factory

RETURNS_BASE = 0.01
RETURNS = [RETURNS_BASE] * 251

BENCHMARK_BASE = 0.005
BENCHMARK = [BENCHMARK_BASE] * 251
DECIMAL_PLACES = 8

PERIODS = [
    "one_month",
    "three_month",
    "six_month",
    "twelve_month",
]


@pytest.fixture(scope="class")
def set_test_risk(request, set_trading_calendar):
    request.cls.trading_calendar = set_trading_calendar
    request.cls.start_session = pd.Timestamp("2006-01-01")
    request.cls.end_session = request.cls.trading_calendar.minute_to_session(
        pd.Timestamp("2006-12-31"), direction="previous"
    )
    request.cls.sim_params = SimulationParameters(
        start_session=request.cls.start_session,
        end_session=request.cls.end_session,
        trading_calendar=request.cls.trading_calendar,
    )
    request.cls.algo_returns = factory.create_returns_from_list(
        RETURNS, request.cls.sim_params
    )
    request.cls.benchmark_returns = factory.create_returns_from_list(
        BENCHMARK, request.cls.sim_params
    )
    request.cls.metrics = ClassicRiskMetrics.risk_report(
        algorithm_returns=request.cls.algo_returns,
        benchmark_returns=request.cls.benchmark_returns,
        algorithm_leverages=pd.Series(0.0, index=request.cls.algo_returns.index),
    )


@pytest.mark.usefixtures("set_test_risk", "with_benchmark_returns")
class TestRisk:
    def test_factory(self):
        returns = [0.1] * 100
        r_objects = factory.create_returns_from_list(returns, self.sim_params)
        assert r_objects.index[-1] <= pd.Timestamp("2006-12-31")

    def test_drawdown(self):
        for period in PERIODS:
            assert all(x["max_drawdown"] == 0 for x in self.metrics[period])

    def test_benchmark_returns_06(self):
        for period, _period_len in zip(PERIODS, [1, 3, 6, 12]):
            np.testing.assert_almost_equal(
                [x["benchmark_period_return"] for x in self.metrics[period]],
                [
                    (1 + BENCHMARK_BASE) ** x["trading_days"] - 1
                    for x in self.metrics[period]
                ],
                DECIMAL_PLACES,
            )

    def test_trading_days(self):
        assert [x["trading_days"] for x in self.metrics["twelve_month"]] == [251]
        assert [x["trading_days"] for x in self.metrics["one_month"]] == [
            20,
            19,
            23,
            19,
            22,
            22,
            20,
            23,
            20,
            22,
            21,
            20,
        ]

    def test_benchmark_volatility(self):
        # Volatility is calculated by a empyrical function so testing
        # of period volatility will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        for period in PERIODS:
            assert all(
                isinstance(x["benchmark_volatility"], float)
                for x in self.metrics[period]
            )

    def test_algorithm_returns(self):
        for period in PERIODS:
            np.testing.assert_almost_equal(
                [x["algorithm_period_return"] for x in self.metrics[period]],
                [
                    (1 + RETURNS_BASE) ** x["trading_days"] - 1
                    for x in self.metrics[period]
                ],
                DECIMAL_PLACES,
            )

    def test_algorithm_volatility(self):
        # Volatility is calculated by a empyrical function so testing
        # of period volatility will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        for period in PERIODS:
            assert all(
                isinstance(x["algo_volatility"], float) for x in self.metrics[period]
            )

    def test_algorithm_sharpe(self):
        # The sharpe ratio is calculated by a empyrical function so testing
        # of period sharpe ratios will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        for period in PERIODS:
            assert all(isinstance(x["sharpe"], float) for x in self.metrics[period])

    def test_algorithm_sortino(self):
        # The sortino ratio is calculated by a empyrical function so testing
        # of period sortino ratios will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        for period in PERIODS:
            assert all(
                isinstance(x["sortino"], float) or x["sortino"] is None
                for x in self.metrics[period]
            )

    def test_algorithm_beta(self):
        # Beta is calculated by a empyrical function so testing
        # of period beta will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        for period in PERIODS:
            assert all(
                isinstance(x["beta"], float) or x["beta"] is None
                for x in self.metrics[period]
            )

    def test_algorithm_alpha(self):
        # Alpha is calculated by a empyrical function so testing
        # of period alpha will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        for period in PERIODS:
            assert all(
                isinstance(x["alpha"], float) or x["alpha"] is None
                for x in self.metrics[period]
            )

    def test_treasury_returns(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = ClassicRiskMetrics.risk_report(
            algorithm_returns=returns,
            benchmark_returns=self.benchmark_returns,
            algorithm_leverages=pd.Series(0.0, index=returns.index),
        )

        # These values are all expected to be zero because we explicity zero
        # out the treasury period returns as they are no longer actually used.
        for period in PERIODS:
            assert [x["treasury_period_return"] for x in metrics[period]] == [
                0.0
            ] * len(metrics[period])

    def test_benchmarkrange(self):
        start_session = pd.Timestamp("2008-01-01")

        end_session = self.trading_calendar.minute_to_session(
            pd.Timestamp("2010-01-01"), direction="previous"
        )

        sim_params = SimulationParameters(
            start_session=start_session,
            end_session=end_session,
            trading_calendar=self.trading_calendar,
        )

        returns = factory.create_returns_from_range(sim_params)
        metrics = ClassicRiskMetrics.risk_report(
            algorithm_returns=returns,
            # use returns from the fixture to ensure that we have enough data.
            benchmark_returns=self.BENCHMARK_RETURNS,
            algorithm_leverages=pd.Series(0.0, index=returns.index),
        )

        self.check_metrics(metrics, 24, start_session)

    def test_partial_month(self):

        start_session = self.trading_calendar.minute_to_session(
            pd.Timestamp("1993-02-01")
        )

        # 1992 and 1996 were leap years
        total_days = 365 * 5 + 2
        end_session = start_session + datetime.timedelta(days=total_days)
        sim_params90s = SimulationParameters(
            start_session=start_session,
            end_session=end_session,
            trading_calendar=self.trading_calendar,
        )

        returns = factory.create_returns_from_range(sim_params90s)
        returns = returns[:-10]  # truncate the returns series to end mid-month
        metrics = ClassicRiskMetrics.risk_report(
            algorithm_returns=returns,
            # use returns from the fixture to ensure that we have enough data.
            benchmark_returns=self.BENCHMARK_RETURNS,
            algorithm_leverages=pd.Series(0.0, index=returns.index),
        )
        total_months = 60
        self.check_metrics(metrics, total_months, start_session)

    def check_metrics(self, metrics, total_months, start_date):
        """
        confirm that the right number of riskmetrics were calculated for each
        window length.
        """
        for period, length in zip(PERIODS, [1, 3, 6, 12]):
            self.assert_range_length(metrics[period], total_months, length, start_date)

    def assert_month(self, start_month, actual_end_month):
        if start_month == 1:
            expected_end_month = 12
        else:
            expected_end_month = start_month - 1

        assert expected_end_month == actual_end_month

    def assert_range_length(self, col, total_months, period_length, start_date):
        if period_length > total_months:
            assert not col
        else:
            period_end = pd.Timestamp(col[-1]["period_label"], tz="utc")
            assert len(col) == total_months - (period_length - 1), (
                "mismatch for total months - expected:{total_months}/"
                "actual:{actual}, period:{period_length}, "
                "start:{start_date}, calculated end:{end}"
            ).format(
                total_months=total_months,
                period_length=period_length,
                start_date=start_date,
                end=period_end,
                actual=len(col),
            )
            self.assert_month(start_date.month, period_end.month)

    def test_algorithm_leverages(self):
        # Max leverage for an algorithm with 'None' as leverage is 0.
        for period, expected_len in zip(PERIODS, [12, 10, 7, 1]):
            assert [x["max_leverage"] for x in self.metrics[period]] == [
                0.0
            ] * expected_len

        test_period = ClassicRiskMetrics.risk_metric_period(
            start_session=self.start_session,
            end_session=self.end_session,
            algorithm_returns=self.algo_returns,
            benchmark_returns=self.benchmark_returns,
            algorithm_leverages=pd.Series([0.01, 0.02, 0.03]),
        )

        # This return period has a list instead of None for algorithm_leverages
        # Confirm that max_leverage is set to the max of those values
        assert test_period["max_leverage"] == 0.03

    def test_sharpe_value_when_null(self):
        # Sharpe is displayed as '0.0' instead of np.nan
        null_returns = factory.create_returns_from_list([0.0] * 251, self.sim_params)
        test_period = ClassicRiskMetrics.risk_metric_period(
            start_session=self.start_session,
            end_session=self.end_session,
            algorithm_returns=null_returns,
            benchmark_returns=self.benchmark_returns,
            algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index),
        )
        assert test_period["sharpe"] == 0.0

    def test_sharpe_value_when_benchmark_null(self):
        # Sharpe is displayed as '0.0' instead of np.nan
        null_returns = factory.create_returns_from_list([0.0] * 251, self.sim_params)
        test_period = ClassicRiskMetrics.risk_metric_period(
            start_session=self.start_session,
            end_session=self.end_session,
            algorithm_returns=null_returns,
            benchmark_returns=null_returns,
            algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index),
        )
        assert test_period["sharpe"] == 0.0

    def test_representation(self):
        test_period = ClassicRiskMetrics.risk_metric_period(
            start_session=self.start_session,
            end_session=self.end_session,
            algorithm_returns=self.algo_returns,
            benchmark_returns=self.benchmark_returns,
            algorithm_leverages=pd.Series(0.0, index=self.algo_returns.index),
        )
        metrics = {
            "algorithm_period_return",
            "benchmark_period_return",
            "treasury_period_return",
            "period_label",
            "excess_return",
            "trading_days",
            "benchmark_volatility",
            "algo_volatility",
            "sharpe",
            "sortino",
            "beta",
            "alpha",
            "max_drawdown",
            "max_leverage",
        }

        assert set(test_period) == metrics
