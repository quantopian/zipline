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
import pytz
import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.finance.trading import SimulationParameters


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

        self.metrics_06 = risk.RiskReport(
            self.algo_returns_06,
            self.sim_params
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
        self.assertTrue(r_objects[-1].date <=
                        datetime.datetime(
                            year=2006, month=12, day=31, tzinfo=pytz.utc))

    def test_drawdown(self):
        returns = factory.create_returns_from_list(
            [1.0, -0.5, 0.8, .17, 1.0, -0.1, -0.45], self.sim_params)
        #200, 100, 180, 210.6, 421.2, 379.8, 208.494
        metrics = risk.RiskMetricsBatch(returns[0].date,
                                        returns[-1].date,
                                        returns)
        self.assertEqual(metrics.max_drawdown, 0.505)

    def test_benchmark_returns_06(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = risk.RiskReport(returns, self.sim_params)
        self.assertEqual([round(x.benchmark_period_returns, 4)
                          for x in metrics.month_periods],
                         [0.0255,
                          0.0004,
                          0.0110,
                          0.0057,
                          -0.0290,
                          0.0021,
                          0.0061,
                          0.0221,
                          0.0247,
                          0.0324,
                          0.0189,
                          0.0139])
        self.assertEqual([round(x.benchmark_period_returns, 4)
                          for x in metrics.three_month_periods],
                         [0.0372,
                          0.0171,
                          -0.0128,
                          -0.0214,
                          -0.0211,
                          0.0305,
                          0.0537,
                          0.0813,
                          0.0780,
                          0.0666])
        self.assertEqual([round(x.benchmark_period_returns, 4)
                          for x in metrics.six_month_periods],
                         [0.015,
                          -0.0043,
                          0.0173,
                          0.0311,
                          0.0586,
                          0.1108,
                          0.1239])
        self.assertEqual([round(x.benchmark_period_returns, 4)
                          for x in metrics.year_periods],
                         [0.1407])

    def test_trading_days_06(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = risk.RiskReport(returns, self.sim_params)
        self.assertEqual([x.num_trading_days for x in metrics.year_periods],
                         [251])
        self.assertEqual([x.num_trading_days for x in metrics.month_periods],
                         [20, 19, 23, 19, 22, 22, 20, 23, 20, 22, 21, 20])

    def test_benchmark_volatility_06(self):
        returns = factory.create_returns_from_range(self.sim_params)
        metrics = risk.RiskReport(returns, self.sim_params)
        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.month_periods],
                         [0.031,
                          0.026,
                          0.024,
                          0.025,
                          0.037,
                          0.047,
                          0.039,
                          0.022,
                          0.022,
                          0.021,
                          0.025,
                          0.019])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.three_month_periods],
                         [0.047,
                          0.043,
                          0.050,
                          0.064,
                          0.070,
                          0.064,
                          0.049,
                          0.037,
                          0.039,
                          0.037])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.six_month_periods],
                         [0.079,
                          0.082,
                          0.081,
                          0.081,
                          0.08,
                          0.074,
                          0.061])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.year_periods],
                         [0.100])

    def test_algorithm_returns_06(self):
        self.assertEqual([round(x.algorithm_period_returns, 3)
                          for x in self.metrics_06.month_periods],
                         [0.101,
                          -0.062,
                          -0.041,
                          0.092,
                          0.135,
                          -0.25,
                          0.076,
                          -0.003,
                          -0.024,
                          0.072,
                          0.063,
                          -0.071])

        self.assertEqual([round(x.algorithm_period_returns, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.009,
                          -0.017,
                          0.188,
                          -0.071,
                          -0.085,
                          -0.196,
                          0.047,
                          0.043,
                          0.112,
                          0.058])

        self.assertEqual([round(x.algorithm_period_returns, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-0.08,
                          -0.101,
                          -0.044,
                          -0.027,
                          -0.045,
                          -0.106,
                          0.108])

        self.assertEqual([round(x.algorithm_period_returns, 3)
                          for x in self.metrics_06.year_periods],
                         [0.02])

    def test_algorithm_volatility_06(self):
        self.assertEqual([round(x.algorithm_volatility, 3)
                          for x in self.metrics_06.month_periods],
                         [0.137,
                          0.12,
                          0.13,
                          0.142,
                          0.128,
                          0.14,
                          0.141,
                          0.118,
                          0.143,
                          0.144,
                          0.117,
                          0.135])

        self.assertEqual([round(x.algorithm_volatility, 3)
                          for x in self.metrics_06.three_month_periods],
                         [0.222,
                          0.224,
                          0.229,
                          0.243,
                          0.243,
                          0.235,
                          0.23,
                          0.231,
                          0.231,
                          0.227])

        self.assertEqual([round(x.algorithm_volatility, 3)
                          for x in self.metrics_06.six_month_periods],
                         [0.328,
                          0.329,
                          0.329,
                          0.333,
                          0.334,
                          0.329,
                          0.321])

        self.assertEqual([round(x.algorithm_volatility, 3)
                          for x in self.metrics_06.year_periods],
                         [0.458])

    def test_algorithm_sharpe_06(self):
        self.assertEqual([round(x.sharpe, 3)
                          for x in self.metrics_06.month_periods],
                         [0.711,
                          -0.541,
                          -0.348,
                          0.625,
                          1.017,
                          -1.809,
                          0.508,
                          -0.062,
                          -0.193,
                          0.467,
                          0.502,
                          -0.557])

        self.assertEqual([round(x.sharpe, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.094,
                          -0.129,
                          0.769,
                          -0.342,
                          -0.402,
                          -0.888,
                          0.153,
                          0.131,
                          0.432,
                          0.2])

        self.assertEqual([round(x.sharpe, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-0.322,
                          -0.383,
                          -0.213,
                          -0.156,
                          -0.213,
                          -0.398,
                          0.257])

        self.assertEqual([round(x.sharpe, 3)
                          for x in self.metrics_06.year_periods],
                         [-0.066])

    def test_algorithm_sortino_06(self):
        self.assertEqual([round(x.sortino, 3)
                          for x in self.metrics_06.month_periods],
                         [4.491,
                          -2.842,
                          -2.052,
                          3.898,
                          7.023,
                          -8.532,
                          3.079,
                          -0.354,
                          -1.125,
                          3.009,
                          3.277,
                          -3.122])
        self.assertEqual([round(x.sortino, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.769,
                          -1.043,
                          6.677,
                          -2.77,
                          -3.209,
                          -6.769,
                          1.253,
                          1.085,
                          3.659,
                          1.674])
        self.assertEqual([round(x.sortino, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-2.728,
                          -3.258,
                          -1.84,
                          -1.366,
                          -1.845,
                          -3.415,
                          2.238])
        self.assertEqual([round(x.sortino, 3)
                          for x in self.metrics_06.year_periods],
                         [-0.524])

    def test_algorithm_information_06(self):
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.month_periods],
                         [0.131,
                          -0.11,
                          -0.067,
                          0.144,
                          0.298,
                          -0.391,
                          0.106,
                          -0.034,
                          -0.058,
                          0.068,
                          0.09,
                          -0.125])
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.013,
                          -0.006,
                          0.113,
                          -0.012,
                          -0.02,
                          -0.11,
                          0.01,
                          -0.005,
                          0.03,
                          0.009])
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-0.013,
                          -0.013,
                          -0.003,
                          -0.002,
                          -0.013,
                          -0.042,
                          0.009])
        self.assertEqual([round(x.information, 3)
                          for x in self.metrics_06.year_periods],
                         [-0.002])

    def dtest_algorithm_beta_06(self):
        self.assertEqual([round(x.beta, 3)
                          for x in self.metrics_06.month_periods],
                         [0.553,
                          0.583,
                          -2.168,
                          -0.548,
                          1.463,
                          -0.322,
                          -1.38,
                          1.473,
                          -1.315,
                          -0.7,
                          0.352,
                          -2.002])

        self.assertEqual([round(x.beta, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.075,
                          -0.637,
                          0.124,
                          0.186,
                          -0.204,
                          -0.497,
                          -0.867,
                          -0.173,
                          -0.499,
                          -0.563])

        self.assertEqual([round(x.beta, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-0.075,
                          -0.637,
                          0.124,
                          0.186,
                          -0.204,
                          -0.497,
                          -0.867,
                          -0.173,
                          -0.499,
                          -0.563])
        self.assertEqual([round(x.beta, 3)
                          for x in self.metrics_06.year_periods], [-0.219])

    def dtest_algorithm_alpha_06(self):
        self.assertEqual([round(x.alpha, 3)
                          for x in self.metrics_06.month_periods],
                         [0.085,
                          -0.063,
                          -0.03,
                          0.093,
                          0.182,
                          -0.255,
                          0.073,
                          -0.032,
                          0,
                          0.086,
                          0.054,
                          -0.058])

        self.assertEqual([round(x.alpha, 3)
                          for x in self.metrics_06.three_month_periods],
                         [-0.051,
                          -0.021,
                          0.179,
                          -0.077,
                          -0.106,
                          -0.202,
                          0.069,
                          0.042,
                          0.13,
                          0.073])

        self.assertEqual([round(x.alpha, 3)
                          for x in self.metrics_06.six_month_periods],
                         [-0.105,
                          -0.135,
                          -0.072,
                          -0.051,
                          -0.066,
                          -0.094,
                          0.152])
        self.assertEqual([round(x.alpha, 3)
                          for x in self.metrics_06.year_periods],
                         [-0.011])

    # FIXME: Covariance is not matching excel precisely enough to run the test.
    # Month 4 seems to be the problem. Variance is disabled
    # just to avoid distraction - it is much closer than covariance
    # and can probably pass with 6 significant digits instead of 7.
    #re-enable variance, alpha, and beta tests once this is resolved
    def dtest_algorithm_covariance_06(self):
        metric = self.metrics_06.month_periods[3]
        print repr(metric)
        print "----"
        self.assertEqual([round(x.algorithm_covariance, 7)
                          for x in self.metrics_06.month_periods],
                         [0.0000289,
                          0.0000222,
                          -0.0000554,
                          -0.0000192,
                          0.0000954,
                          -0.0000333,
                          -0.0001111,
                          0.0000322,
                          -0.0000349,
                          -0.0000143,
                          0.0000108,
                          -0.0000386])

        self.assertEqual([round(x.algorithm_covariance, 7)
                          for x in self.metrics_06.three_month_periods],
                         [-0.0000026,
                          -0.0000189,
                          0.0000049,
                          0.0000121,
                          -0.0000158,
                          -0.000031,
                          -0.0000336,
                          -0.0000036,
                          -0.0000119,
                          -0.0000122])

        self.assertEqual([round(x.algorithm_covariance, 7)
                          for x in self.metrics_06.six_month_periods],
                         [0.000005,
                          -0.0000172,
                          -0.0000142,
                          -0.0000102,
                          -0.0000089,
                          -0.0000207,
                          -0.0000229])

        self.assertEqual([round(x.algorithm_covariance, 7)
                          for x in self.metrics_06.year_periods],
                         [-8.75273E-06])

    def dtest_benchmark_variance_06(self):
        self.assertEqual([round(x.benchmark_variance, 7)
                          for x in self.metrics_06.month_periods],
                         [0.0000496,
                          0.000036,
                          0.0000244,
                          0.0000332,
                          0.0000623,
                          0.0000989,
                          0.0000765,
                          0.0000209,
                          0.0000252,
                          0.0000194,
                          0.0000292,
                          0.0000183])

        self.assertEqual([round(x.benchmark_variance, 7)
                          for x in self.metrics_06.three_month_periods],
                         [0.0000351,
                          0.0000298,
                          0.0000395,
                          0.0000648,
                          0.0000773,
                          0.0000625,
                          0.0000387,
                          0.0000211,
                          0.0000238,
                          0.0000217])

        self.assertEqual([round(x.benchmark_variance, 7)
                          for x in self.metrics_06.six_month_periods],
                         [0.0000499,
                          0.0000538,
                          0.0000508,
                          0.0000517,
                          0.0000492,
                          0.0000432,
                          0.00003])

        self.assertEqual([round(x.benchmark_variance, 7)
                          for x in self.metrics_06.year_periods],
                         [0.0000399])

    def test_benchmark_returns_08(self):
        returns = factory.create_returns_from_range(self.sim_params08)
        metrics = risk.RiskReport(returns, self.sim_params08)

        monthly = [round(x.benchmark_period_returns, 3)
                   for x in metrics.month_periods]

        self.assertEqual(monthly,
                         [-0.051,
                          -0.039,
                          0.001,
                          0.043,
                          0.011,
                          -0.075,
                          -0.007,
                          0.026,
                          -0.093,
                          -0.160,
                          -0.072,
                          0.009])

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.three_month_periods],
                         [-0.087,
                          0.003,
                          0.055,
                          -0.026,
                          -0.072,
                          -0.058,
                          -0.075,
                          -0.218,
                          -0.293,
                          -0.214])

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.six_month_periods],
                         [-0.110,
                          -0.069,
                          -0.006,
                          -0.099,
                          -0.274,
                          -0.334,
                          -0.273])

        self.assertEqual([round(x.benchmark_period_returns, 3)
                          for x in metrics.year_periods],
                         [-0.353])

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
                         [0.069,
                          0.056,
                          0.080,
                          0.049,
                          0.040,
                          0.052,
                          0.068,
                          0.055,
                          0.150,
                          0.230,
                          0.188,
                          0.137])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.three_month_periods],
                         [0.118,
                          0.108,
                          0.101,
                          0.083,
                          0.094,
                          0.102,
                          0.172,
                          0.277,
                          0.328,
                          0.323])

        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.six_month_periods],
                         [0.144,
                          0.143,
                          0.143,
                          0.190,
                          0.292,
                          0.342,
                          0.364])
        # TODO: ugly, but I can't get the rounded float to match.
        # maybe we need a different test that checks the
        # difference between the numbers
        self.assertEqual([round(x.benchmark_volatility, 3)
                          for x in metrics.year_periods],
                         [0.391])

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

        #1992 and 1996 were leap years
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
        #30 days has september, april, june and november
        if period_end.month in [9, 4, 6, 11]:
            self.assertEqual(period_end.day, 30)
        #all the rest have 31, except for february
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

RETURNS = [
    0.0093,
    -0.0193,
    0.0351,
    0.0396,
    0.0338,
    -0.0211,
    0.0389,
    0.0326,
    -0.0137,
    -0.0411,
    -0.0032,
    0.0149,
    0.0133,
    0.0348,
    0.042,
    -0.0455,
    0.0262,
    -0.0461,
    0.0021,
    -0.0273,
    -0.0429,
    0.0427,
    -0.0104,
    0.0346,
    -0.0311,
    0.0003,
    0.0211,
    0.0248,
    -0.0215,
    0.004,
    0.0267,
    0.0029,
    -0.0369,
    0.0057,
    0.0298,
    -0.0179,
    -0.0361,
    -0.0401,
    -0.0123,
    -0.005,
    0.0203,
    -0.041,
    0.0011,
    0.0118,
    0.0103,
    -0.0184,
    -0.0437,
    0.0411,
    -0.0242,
    -0.0054,
    -0.0039,
    -0.0273,
    -0.0075,
    0.0064,
    -0.0376,
    0.0424,
    0.0399,
    0.019,
    0.0236,
    -0.0284,
    -0.0341,
    0.0266,
    0.05,
    0.0069,
    -0.0442,
    -0.016,
    0.0173,
    0.0348,
    -0.0404,
    -0.0068,
    -0.0376,
    0.0356,
    0.0043,
    -0.0481,
    -0.0134,
    0.0257,
    0.0442,
    0.0234,
    0.0394,
    0.0376,
    -0.0147,
    -0.0098,
    0.0474,
    -0.0102,
    0.0138,
    0.0286,
    0.0347,
    0.0279,
    -0.0067,
    0.0462,
    -0.0432,
    0.0247,
    0.0174,
    -0.0305,
    -0.0317,
    -0.0068,
    0.0264,
    -0.0257,
    -0.0328,
    0.0092,
    0.0288,
    -0.002,
    0.0288,
    0.028,
    -0.0093,
    0.0178,
    -0.0365,
    -0.0086,
    -0.0133,
    -0.0309,
    0.0473,
    -0.0149,
    0.0378,
    -0.0316,
    -0.0292,
    -0.0453,
    -0.0451,
    0.0093,
    0.0397,
    -0.0361,
    -0.0168,
    -0.0494,
    -0.0143,
    -0.0405,
    -0.0349,
    0.0069,
    0.0378,
    -0.0233,
    -0.0492,
    0.018,
    -0.0386,
    0.0339,
    0.0119,
    0.0454,
    0.0118,
    -0.011,
    -0.0254,
    0.0266,
    -0.0366,
    -0.0211,
    0.0399,
    0.0307,
    0.035,
    -0.0402,
    0.0304,
    -0.0031,
    0.0256,
    0.0134,
    -0.0019,
    -0.0235,
    -0.0058,
    -0.0117,
    0.0051,
    -0.0451,
    -0.0466,
    -0.0124,
    0.0283,
    -0.0499,
    0.0318,
    -0.0028,
    0.0203,
    0.005,
    0.0085,
    0.0048,
    0.0277,
    0.0159,
    -0.0149,
    0.035,
    0.0404,
    -0.01,
    0.0377,
    0.0302,
    0.0046,
    -0.0328,
    -0.0469,
    0.0071,
    -0.0382,
    -0.0214,
    0.0429,
    0.0145,
    -0.0279,
    -0.0172,
    0.0423,
    0.041,
    -0.0183,
    0.0137,
    -0.0412,
    -0.0348,
    0.0302,
    0.0248,
    0.0051,
    -0.0298,
    -0.0103,
    -0.0333,
    -0.0399,
    0.0485,
    -0.0166,
    0.0384,
    0.0259,
    -0.0163,
    0.0357,
    0.0308,
    -0.0386,
    0.0481,
    -0.0446,
    -0.0282,
    -0.0037,
    0.0202,
    0.0216,
    0.0113,
    0.0194,
    0.0392,
    0.0016,
    0.0268,
    -0.0155,
    -0.027,
    0.02,
    0.0216,
    -0.0009,
    0.022,
    0.0,
    0.041,
    0.0133,
    -0.0382,
    0.0495,
    -0.0221,
    -0.0329,
    -0.0033,
    -0.0089,
    -0.0129,
    -0.0252,
    0.048,
    -0.0307,
    -0.0357,
    0.0033,
    -0.0412,
    -0.0407,
    0.0455,
    0.0159,
    -0.0051,
    -0.0274,
    -0.0213,
    0.0361,
    0.0051,
    -0.0378,
    0.0084,
    0.0066,
    -0.0103,
    -0.0037,
    0.0478,
    -0.0278]
