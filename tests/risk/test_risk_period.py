#
# Copyright 2013 Quantopian, Inc.
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

import unittest
import datetime
import calendar
import numpy as np
import pytz
import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.finance.trading import SimulationParameters

from . import answer_key
from . answer_key import AnswerKey

ANSWER_KEY = AnswerKey()

RETURNS = ANSWER_KEY.RETURNS


class TestRisk(unittest.TestCase):

    def setUp(self):

        start_date = datetime.datetime(
            year=2006,
            month=1,
            day=1,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)
        end_date = datetime.datetime(
            year=2006, month=12, day=31, tzinfo=pytz.utc)

        self.sim_params = SimulationParameters(
            period_start=start_date,
            period_end=end_date
        )

        self.algo_returns_06 = factory.create_returns_from_list(
            RETURNS,
            self.sim_params
        )

        self.benchmark_returns_06 = \
            answer_key.RETURNS_DATA['Benchmark Returns']

        self.metrics_06 = risk.RiskReport(
            self.algo_returns_06,
            self.sim_params,
            benchmark_returns=self.benchmark_returns_06,
        )

        start_08 = datetime.datetime(
            year=2008,
            month=1,
            day=1,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)

        end_08 = datetime.datetime(
            year=2008,
            month=12,
            day=31,
            tzinfo=pytz.utc
        )
        self.sim_params08 = SimulationParameters(
            period_start=start_08,
            period_end=end_08
        )

    def tearDown(self):
        return

    def test_factory(self):
        returns = [0.1] * 100
        r_objects = factory.create_returns_from_list(returns, self.sim_params)
        self.assertTrue(r_objects.index[-1] <=
                        datetime.datetime(
                            year=2006, month=12, day=31, tzinfo=pytz.utc))

    def test_drawdown(self):
        returns = factory.create_returns_from_list(
            [1.0, -0.5, 0.8, .17, 1.0, -0.1, -0.45], self.sim_params)
        # 200, 100, 180, 210.6, 421.2, 379.8, 208.494
        metrics = risk.RiskMetricsPeriod(returns.index[0],
                                         returns.index[-1],
                                         returns)
        self.assertEqual(metrics.max_drawdown, 0.505)

    def test_benchmark_returns_06(self):

        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics_06.month_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_RETURNS['Monthly'])
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_RETURNS['3-Month'])
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_RETURNS['6-month'])
        np.testing.assert_almost_equal(
            [x.benchmark_period_returns
             for x in self.metrics_06.year_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_RETURNS['year'])

    def test_trading_days_06(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = risk.RiskReport(returns, self.sim_params)
        self.assertEqual([x.num_trading_days for x in metrics.year_periods],
                         [251])
        self.assertEqual([x.num_trading_days for x in metrics.month_periods],
                         [20, 19, 23, 19, 22, 22, 20, 23, 20, 22, 21, 20])

    def test_benchmark_volatility_06(self):

        np.testing.assert_almost_equal(
            [x.benchmark_volatility
             for x in self.metrics_06.month_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_VOLATILITY['Monthly'])
        np.testing.assert_almost_equal(
            [x.benchmark_volatility
             for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_VOLATILITY['3-Month'])
        np.testing.assert_almost_equal(
            [x.benchmark_volatility
             for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_VOLATILITY['6-month'])
        np.testing.assert_almost_equal(
            [x.benchmark_volatility
             for x in self.metrics_06.year_periods],
            ANSWER_KEY.BENCHMARK_PERIOD_VOLATILITY['year'])

    def test_algorithm_returns_06(self):
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_RETURNS['Monthly'])
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_RETURNS['3-Month'])
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_RETURNS['6-month'])
        np.testing.assert_almost_equal(
            [x.algorithm_period_returns
             for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_RETURNS['year'])

    def test_algorithm_volatility_06(self):
        np.testing.assert_almost_equal(
            [x.algorithm_volatility
             for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_VOLATILITY['Monthly'])
        np.testing.assert_almost_equal(
            [x.algorithm_volatility
             for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_VOLATILITY['3-Month'])
        np.testing.assert_almost_equal(
            [x.algorithm_volatility
             for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_VOLATILITY['6-month'])
        np.testing.assert_almost_equal(
            [x.algorithm_volatility
             for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_VOLATILITY['year'])

    def test_algorithm_sharpe_06(self):
        np.testing.assert_almost_equal(
            [x.sharpe for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SHARPE['Monthly'])
        np.testing.assert_almost_equal(
            [x.sharpe for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SHARPE['3-Month'])
        np.testing.assert_almost_equal(
            [x.sharpe for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SHARPE['6-month'])
        np.testing.assert_almost_equal(
            [x.sharpe for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SHARPE['year'])

    def test_algorithm_downside_risk_06(self):
        np.testing.assert_almost_equal(
            [x.downside_risk for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_DOWNSIDE_RISK['Monthly'],
            decimal=4)
        np.testing.assert_almost_equal(
            [x.downside_risk for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_DOWNSIDE_RISK['3-Month'],
            decimal=4)
        np.testing.assert_almost_equal(
            [x.downside_risk for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_DOWNSIDE_RISK['6-month'],
            decimal=4)
        np.testing.assert_almost_equal(
            [x.downside_risk for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_DOWNSIDE_RISK['year'],
            decimal=4)

    def test_algorithm_sortino_06(self):
        np.testing.assert_almost_equal(
            [x.sortino for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SORTINO['Monthly'],
            decimal=3)

        np.testing.assert_almost_equal(
            [x.sortino for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SORTINO['3-Month'],
            decimal=3)

        np.testing.assert_almost_equal(
            [x.sortino for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SORTINO['6-month'],
            decimal=3)

        np.testing.assert_almost_equal(
            [x.sortino for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_SORTINO['year'],
            decimal=3)

    def test_algorithm_information_06(self):
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.month_periods],
                         [0.131,
                          -0.11,
                          -0.067,
                          0.136,
                          0.301,
                          -0.387,
                          0.107,
                          -0.032,
                          -0.058,
                          0.069,
                          0.095,
                          -0.123])
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.013,
                          -0.009,
                          0.111,
                          -0.014,
                          -0.017,
                          -0.108,
                          0.011,
                          -0.004,
                          0.032,
                          0.011])
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-0.013,
                          -0.014,
                          -0.003,
                          -0.002,
                          -0.011,
                          -0.041,
                          0.011])
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.year_periods],
                         [-0.001])

    def test_algorithm_beta_06(self):
        np.testing.assert_almost_equal(
            [x.beta for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BETA['Monthly'])
        np.testing.assert_almost_equal(
            [x.beta for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BETA['3-Month'])
        np.testing.assert_almost_equal(
            [x.beta for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BETA['6-month'])
        np.testing.assert_almost_equal(
            [x.beta for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BETA['year'])

    def test_algorithm_alpha_06(self):
        np.testing.assert_almost_equal(
            [x.alpha for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_ALPHA['Monthly'])
        np.testing.assert_almost_equal(
            [x.alpha for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_ALPHA['3-Month'])
        np.testing.assert_almost_equal(
            [x.alpha for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_ALPHA['6-month'])
        np.testing.assert_almost_equal(
            [x.alpha for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_ALPHA['year'])

    # FIXME: Covariance is not matching excel precisely enough to run the test.
    # Month 4 seems to be the problem. Variance is disabled
    # just to avoid distraction - it is much closer than covariance
    # and can probably pass with 6 significant digits instead of 7.
    # re-enable variance, alpha, and beta tests once this is resolved
    def test_algorithm_covariance_06(self):
        np.testing.assert_almost_equal(
            [x.algorithm_covariance for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_COVARIANCE['Monthly'])
        np.testing.assert_almost_equal(
            [x.algorithm_covariance
             for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_COVARIANCE['3-Month'])
        np.testing.assert_almost_equal(
            [x.algorithm_covariance
             for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_COVARIANCE['6-month'])
        np.testing.assert_almost_equal(
            [x.algorithm_covariance
             for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_COVARIANCE['year'])

    def test_benchmark_variance_06(self):
        np.testing.assert_almost_equal(
            [x.benchmark_variance
             for x in self.metrics_06.month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BENCHMARK_VARIANCE['Monthly'])
        np.testing.assert_almost_equal(
            [x.benchmark_variance
             for x in self.metrics_06.three_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BENCHMARK_VARIANCE['3-Month'])
        np.testing.assert_almost_equal(
            [x.benchmark_variance
             for x in self.metrics_06.six_month_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BENCHMARK_VARIANCE['6-month'])
        np.testing.assert_almost_equal(
            [x.benchmark_variance
             for x in self.metrics_06.year_periods],
            ANSWER_KEY.ALGORITHM_PERIOD_BENCHMARK_VARIANCE['year'])

    def test_benchmark_returns_08(self):
        returns = factory.create_returns_from_range(self.sim_params08)
        metrics = risk.RiskReport(returns, self.sim_params08)

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.month_periods],
                         [-0.061,
                          -0.035,
                          -0.006,
                          0.048,
                          0.011,
                          -0.086,
                          -0.01,
                          0.012,
                          -0.091,
                          -0.169,
                          -0.075,
                          0.008])

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.three_month_periods],
                         [-0.099,
                          0.005,
                          0.052,
                          -0.032,
                          -0.085,
                          -0.084,
                          -0.089,
                          -0.236,
                          -0.301,
                          -0.226])

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.six_month_periods],
                         [-0.128,
                          -0.081,
                          -0.036,
                          -0.118,
                          -0.301,
                          -0.36,
                          -0.294])

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.year_periods],
                         [-0.385])

    def test_trading_days_08(self):
        returns = factory.create_returns_from_range(self.sim_params08)
        metrics = risk.RiskReport(returns, self.sim_params08)
        self.assertEqual([x.num_trading_days for x in metrics.year_periods],
                         [253])

        self.assertEqual([x.num_trading_days for x in metrics.month_periods],
                         [21, 20, 20, 22, 21, 21, 22, 21, 21, 23, 19, 22])

    def test_benchmark_volatility_08(self):
        returns = factory.create_returns_from_range(self.sim_params08)
        metrics = risk.RiskReport(returns, self.sim_params08)

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.month_periods],
                         [0.07,
                          0.058,
                          0.082,
                          0.054,
                          0.041,
                          0.057,
                          0.068,
                          0.06,
                          0.157,
                          0.244,
                          0.195,
                          0.145])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.three_month_periods],
                         [0.12,
                          0.113,
                          0.105,
                          0.09,
                          0.098,
                          0.107,
                          0.179,
                          0.293,
                          0.344,
                          0.34])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.six_month_periods],
                         [0.15,
                          0.149,
                          0.15,
                          0.2,
                          0.308,
                          0.36,
                          0.383])
        # TODO: ugly, but I can't get the rounded float to match.
        # maybe we need a different test that checks the
        # difference between the numbers
        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.year_periods],
                         [0.411])

    def test_treasury_returns_06(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = risk.RiskReport(returns, self.sim_params)
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
        self.check_year_range(
            datetime.datetime(
                year=2008, month=1, day=1, tzinfo=pytz.utc),
            2)

    def test_partial_month(self):

        start = datetime.datetime(
            year=1991,
            month=1,
            day=1,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)

        # 1992 and 1996 were leap years
        total_days = 365 * 5 + 2
        end = start + datetime.timedelta(days=total_days)
        sim_params90s = SimulationParameters(
            period_start=start,
            period_end=end
        )

        returns = factory.create_returns_from_range(sim_params90s)
        returns = returns[:-10]  # truncate the returns series to end mid-month
        metrics = risk.RiskReport(returns, sim_params90s)
        total_months = 60
        self.check_metrics(metrics, total_months, start)

    def check_year_range(self, start_date, years):
        sim_params = SimulationParameters(
            period_start=start_date,
            period_end=start_date.replace(year=(start_date.year + years))
        )
        returns = factory.create_returns_from_range(sim_params)
        metrics = risk.RiskReport(returns, self.sim_params)
        total_months = years * 12
        self.check_metrics(metrics, total_months, start_date)

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
        if(period_length > total_months):
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
                                             end=col[-1].end_date,
                                             actual=len(col))
            )
            self.assert_month(start_date.month, col[-1].end_date.month)
            self.assert_last_day(col[-1].end_date)
