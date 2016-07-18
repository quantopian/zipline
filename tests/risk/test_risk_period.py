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

import datetime
import calendar
import pandas as pd
import numpy as np

import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.finance.trading import SimulationParameters
from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase

from zipline.finance.risk.period import RiskMetricsPeriod

RETURNS_BASE = 0.01
RETURNS = [RETURNS_BASE] * 251

BENCHMARK_BASE = 0.005
BENCHMARK = [BENCHMARK_BASE] * 251
DECIMAL_PLACES = 8


class TestRisk(WithTradingEnvironment, ZiplineTestCase):

    def init_instance_fixtures(self):
        super(TestRisk, self).init_instance_fixtures()
        self.start_session = pd.Timestamp("2006-01-01", tz='UTC')
        self.end_session = self.trading_calendar.minute_to_session_label(
            pd.Timestamp("2006-12-31", tz='UTC'),
            direction="previous"
        )
        self.sim_params = SimulationParameters(
            start_session=self.start_session,
            end_session=self.end_session,
            trading_calendar=self.trading_calendar,
        )
        self.algo_returns = factory.create_returns_from_list(
            RETURNS,
            self.sim_params
        )
        self.benchmark_returns = factory.create_returns_from_list(
            BENCHMARK,
            self.sim_params
        )
        self.metrics = risk.RiskReport(
            self.algo_returns,
            self.sim_params,
            benchmark_returns=self.benchmark_returns,
            trading_calendar=self.trading_calendar,
            treasury_curves=self.env.treasury_curves,
        )

    def test_factory(self):
        returns = [0.1] * 100
        r_objects = factory.create_returns_from_list(returns, self.sim_params)
        self.assertTrue(r_objects.index[-1] <=
                        pd.Timestamp('2006-12-31', tz='UTC'))

    def test_drawdown(self):
        np.testing.assert_equal(
            all(x.max_drawdown == 0 for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(x.max_drawdown == 0 for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(x.max_drawdown == 0 for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(x.max_drawdown == 0 for x in self.metrics.year_periods),
            True)

    def test_benchmark_returns_06(self):
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics.month_periods],
            [(1 + BENCHMARK_BASE) ** len(x.benchmark_returns) - 1
             for x in self.metrics.month_periods],
            DECIMAL_PLACES)
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics.three_month_periods],
            [(1 + BENCHMARK_BASE) ** len(x.benchmark_returns) - 1
             for x in self.metrics.three_month_periods],
            DECIMAL_PLACES)
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics.six_month_periods],
            [(1 + BENCHMARK_BASE) ** len(x.benchmark_returns) - 1
             for x in self.metrics.six_month_periods],
            DECIMAL_PLACES)
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics.year_periods],
            [(1 + BENCHMARK_BASE) ** len(x.benchmark_returns) - 1
             for x in self.metrics.year_periods],
            DECIMAL_PLACES)

    def test_trading_days(self):
        self.assertEqual([x.num_trading_days
                          for x in self.metrics.year_periods],
                         [251])
        self.assertEqual([x.num_trading_days
                          for x in self.metrics.month_periods],
                         [20, 19, 23, 19, 22, 22, 20, 23, 20, 22, 21, 20])

    def test_benchmark_volatility(self):
        # Volatility is calculated by a empyrical function so testing
        # of period volatility will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.benchmark_volatility, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.benchmark_volatility, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.benchmark_volatility, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.benchmark_volatility, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_returns(self):
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics.month_periods],
            [(1 + RETURNS_BASE) ** len(x.algorithm_returns) - 1
             for x in self.metrics.month_periods],
            DECIMAL_PLACES)
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics.three_month_periods],
            [(1 + RETURNS_BASE) ** len(x.algorithm_returns) - 1
             for x in self.metrics.three_month_periods],
            DECIMAL_PLACES)
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics.six_month_periods],
            [(1 + RETURNS_BASE) ** len(x.algorithm_returns) - 1
             for x in self.metrics.six_month_periods],
            DECIMAL_PLACES)
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics.year_periods],
            [(1 + RETURNS_BASE) ** len(x.algorithm_returns) - 1
             for x in self.metrics.year_periods],
            DECIMAL_PLACES)

    def test_algorithm_volatility(self):
        # Volatility is calculated by a empyrical function so testing
        # of period volatility will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.algorithm_volatility, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.algorithm_volatility, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.algorithm_volatility, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.algorithm_volatility, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_sharpe(self):
        # The sharpe ratio is calculated by a empyrical function so testing
        # of period sharpe ratios will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.sharpe, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.sharpe, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.sharpe, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.sharpe, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_downside_risk(self):
        # Downside risk is calculated by a empyrical function so testing
        # of period downside risk will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.downside_risk, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.downside_risk, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.downside_risk, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.downside_risk, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_sortino(self):
        # The sortino ratio is calculated by a empyrical function so testing
        # of period sortino ratios will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.sortino, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.sortino, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.sortino, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.sortino, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_information(self):
        # The information ratio is calculated by a empyrical function
        # testing of period information ratio will be limited to determine
        # if the value is numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.information, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.information, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.information, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.information, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_beta(self):
        # Beta is calculated by a empyrical function so testing
        # of period beta will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.beta, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.beta, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.beta, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.beta, float)
                for x in self.metrics.year_periods),
            True)

    def test_algorithm_alpha(self):
        # Alpha is calculated by a empyrical function so testing
        # of period alpha will be limited to determine if the value is
        # numerical. This tests for its existence and format.
        np.testing.assert_equal(
            all(isinstance(x.alpha, float)
                for x in self.metrics.month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.alpha, float)
                for x in self.metrics.three_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.alpha, float)
                for x in self.metrics.six_month_periods),
            True)
        np.testing.assert_equal(
            all(isinstance(x.alpha, float)
                for x in self.metrics.year_periods),
            True)

    def test_treasury_returns(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = risk.RiskReport(returns, self.sim_params,
                                  trading_calendar=self.trading_calendar,
                                  treasury_curves=self.env.treasury_curves,
                                  benchmark_returns=self.env.benchmark_returns)
        self.assertEqual([round(x.treasury_period_return, 4)
                          for x in metrics.month_periods],
                         [0.0037,
                          0.0034,
                          0.0039,
                          0.0038,
                          0.0040,
                          0.0037,
                          0.0043,
                          0.0043,
                          0.0038,
                          0.0044,
                          0.0043,
                          0.004])

        self.assertEqual([round(x.treasury_period_return, 4)
                          for x in metrics.three_month_periods],
                         [0.0114,
                          0.0116,
                          0.0122,
                          0.0125,
                          0.0129,
                          0.0127,
                          0.0123,
                          0.0128,
                          0.0125,
                          0.0127])
        self.assertEqual([round(x.treasury_period_return, 4)
                          for x in metrics.six_month_periods],
                         [0.0260,
                          0.0257,
                          0.0258,
                          0.0252,
                          0.0259,
                          0.0256,
                          0.0257])

        self.assertEqual([round(x.treasury_period_return, 4)
                          for x in metrics.year_periods],
                         [0.0500])

    def test_benchmarkrange(self):
        start_session = self.trading_calendar.minute_to_session_label(
            pd.Timestamp("2008-01-01", tz='UTC')
        )

        end_session = self.trading_calendar.minute_to_session_label(
            pd.Timestamp("2010-01-01", tz='UTC'), direction="previous"
        )

        sim_params = SimulationParameters(
            start_session=start_session,
            end_session=end_session,
            trading_calendar=self.trading_calendar,
        )

        returns = factory.create_returns_from_range(sim_params)
        metrics = risk.RiskReport(returns, self.sim_params,
                                  trading_calendar=self.trading_calendar,
                                  treasury_curves=self.env.treasury_curves,
                                  benchmark_returns=self.env.benchmark_returns)

        self.check_metrics(metrics, 24, start_session)

    def test_partial_month(self):

        start_session = self.trading_calendar.minute_to_session_label(
            pd.Timestamp("1991-01-01", tz='UTC')
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
        metrics = risk.RiskReport(returns, sim_params90s,
                                  trading_calendar=self.trading_calendar,
                                  treasury_curves=self.env.treasury_curves,
                                  benchmark_returns=self.env.benchmark_returns)
        total_months = 60
        self.check_metrics(metrics, total_months, start_session)

    def check_metrics(self, metrics, total_months, start_date):
        """
        confirm that the right number of riskmetrics were calculated for each
        window length.
        """
        self.assert_range_length(
            metrics.month_periods,
            total_months,
            1,
            start_date
        )

        self.assert_range_length(
            metrics.three_month_periods,
            total_months,
            3,
            start_date
        )

        self.assert_range_length(
            metrics.six_month_periods,
            total_months,
            6,
            start_date
        )

        self.assert_range_length(
            metrics.year_periods,
            total_months,
            12,
            start_date
        )

    def assert_last_day(self, period_end):
        # 30 days has september, april, june and november
        if period_end.month in [9, 4, 6, 11]:
            self.assertEqual(period_end.day, 30)
        # all the rest have 31, except for february
        elif(period_end.month != 2):
            self.assertEqual(period_end.day, 31)
        else:
            if calendar.isleap(period_end.year):
                self.assertEqual(period_end.day, 29)
            else:
                self.assertEqual(period_end.day, 28)

    def assert_month(self, start_month, actual_end_month):
        if start_month == 1:
            expected_end_month = 12
        else:
            expected_end_month = start_month - 1

        self.assertEqual(expected_end_month, actual_end_month)

    def assert_range_length(self, col, total_months,
                            period_length, start_date):
        if (period_length > total_months):
            self.assertEqual(len(col), 0)
        else:
            self.assertEqual(
                len(col),
                total_months - (period_length - 1),
                "mismatch for total months - \
                expected:{total_months}/actual:{actual}, \
                period:{period_length}, start:{start_date}, \
                calculated end:{end}".format(total_months=total_months,
                                             period_length=period_length,
                                             start_date=start_date,
                                             end=col[-1]._end_session,
                                             actual=len(col))
            )
            self.assert_month(start_date.month, col[-1]._end_session.month)
            self.assert_last_day(col[-1]._end_session)

    def test_algorithm_leverages(self):
        # Max leverage for an algorithm with 'None' as leverage is 0.
        np.testing.assert_equal(
            [x.max_leverage for x in self.metrics.month_periods],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(
            [x.max_leverage for x in self.metrics.three_month_periods],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(
            [x.max_leverage for x in self.metrics.six_month_periods],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(
            [x.max_leverage for x in self.metrics.year_periods],
            [0.0])

    def test_returns_beyond_treasury(self):
        # The last treasury value is used when return dates go beyond
        # treasury curve data
        treasury_curves = self.env.treasury_curves
        treasury = treasury_curves[treasury_curves.index < self.start_session]

        test_period = RiskMetricsPeriod(
            start_session=self.start_session,
            end_session=self.end_session,
            returns=self.algo_returns,
            benchmark_returns=self.benchmark_returns,
            trading_calendar=self.trading_calendar,
            treasury_curves=treasury,
            algorithm_leverages=[.01, .02, .03]
        )
        assert test_period.treasury_curves.equals(treasury[-1:])
        # This return period has a list instead of None for algorithm_leverages
        # Confirm that max_leverage is set to the max of those values
        assert test_period.max_leverage == .03

    def test_index_mismatch_exception(self):
        # An exception is raised when returns and benchmark returns
        # have indexes that do not match
        bench_params = SimulationParameters(
            start_session=pd.Timestamp("2006-02-01", tz='UTC'),
            end_session=pd.Timestamp("2006-02-28", tz='UTC'),
            trading_calendar=self.trading_calendar,
        )
        benchmark = factory.create_returns_from_list(
            [BENCHMARK_BASE]*19,
            bench_params
        )
        with np.testing.assert_raises(Exception):
            RiskMetricsPeriod(
                start_session=self.start_session,
                end_session=self.end_session,
                returns=self.algo_returns,
                benchmark_returns=benchmark,
                trading_calendar=self.trading_calendar,
                treasury_curves=self.env.treasury_curves,
            )

    def test_sharpe_value_when_null(self):
        # Sharpe is displayed as '0.0' instead of np.nan
        null_returns = factory.create_returns_from_list(
            [0.0]*251,
            self.sim_params
        )
        test_period = RiskMetricsPeriod(
            start_session=self.start_session,
            end_session=self.end_session,
            returns=null_returns,
            benchmark_returns=self.benchmark_returns,
            trading_calendar=self.trading_calendar,
            treasury_curves=self.env.treasury_curves,
        )
        assert test_period.sharpe == 0.0

    def test_representation(self):
        test_period = RiskMetricsPeriod(
            start_session=self.start_session,
            end_session=self.end_session,
            returns=self.algo_returns,
            benchmark_returns=self.benchmark_returns,
            trading_calendar=self.trading_calendar,
            treasury_curves=self.env.treasury_curves,
        )
        metrics = [
            "algorithm_period_returns",
            "benchmark_period_returns",
            "excess_return",
            "num_trading_days",
            "benchmark_volatility",
            "algorithm_volatility",
            "sharpe",
            "sortino",
            "information",
            "beta",
            "alpha",
            "max_drawdown",
            "max_leverage",
            "algorithm_returns",
            "benchmark_returns",
        ]
        representation = test_period.__repr__()

        assert all([metric in representation for metric in metrics])
