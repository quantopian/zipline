#
# Copyright 2014 Quantopian, Inc.
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
import numpy as np
import pandas as pd
import pytz
import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.finance.trading import SimulationParameters

from . import answer_key
ANSWER_KEY = answer_key.ANSWER_KEY


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
            year=2006, month=12, day=29, tzinfo=pytz.utc)

        self.sim_params = SimulationParameters(
            period_start=start_date,
            period_end=end_date
        )

        self.algo_returns_06 = factory.create_returns_from_list(
            answer_key.ALGORITHM_RETURNS.values,
            self.sim_params
        )

        self.cumulative_metrics_06 = risk.RiskMetricsCumulative(
            self.sim_params)

        for dt, returns in answer_key.RETURNS_DATA.iterrows():
            self.cumulative_metrics_06.update(dt,
                                              returns['Algorithm Returns'],
                                              returns['Benchmark Returns'],
                                              {'leverage': 0.0})

    def test_algorithm_volatility_06(self):
        algo_vol_answers = answer_key.RISK_CUMULATIVE.volatility
        for dt, value in algo_vol_answers.iteritems():
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.metrics.algorithm_volatility[dt],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_sharpe_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.sharpe.iteritems():
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.metrics.sharpe[dt],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_downside_risk_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.downside_risk.iteritems():
            np.testing.assert_almost_equal(
                value,
                self.cumulative_metrics_06.metrics.downside_risk[dt],
                err_msg="Mismatch at %s" % (dt,))

    def test_sortino_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.sortino.iteritems():
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.metrics.sortino[dt],
                value,
                decimal=4,
                err_msg="Mismatch at %s" % (dt,))

    def test_information_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.information.iteritems():
            np.testing.assert_almost_equal(
                value,
                self.cumulative_metrics_06.metrics.information[dt],
                err_msg="Mismatch at %s" % (dt,))

    def test_alpha_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.alpha.iteritems():
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.metrics.alpha[dt],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_beta_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.beta.iteritems():
            np.testing.assert_almost_equal(
                value,
                self.cumulative_metrics_06.metrics.beta[dt],
                err_msg="Mismatch at %s" % (dt,))

    def test_max_drawdown_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.max_drawdown.iteritems():
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.max_drawdowns[dt],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_drawdown_duration(self):
        returns = factory.create_returns_from_list(
            [1.0, -0.5, 0.8, .17, 1.0, -0.1, -0.45], self.sim_params)
        # 200, 100, 180, 210.6, 421.2, 379.8, 208.494
        drawdown_duration = pd.Series(
            [0, 2, 3, 3, 3, 3, 3], index=returns.index)

        metrics = risk.RiskMetricsCumulative(self.sim_params)

        for dt, value in returns.iteritems():
            metrics.update(dt,
                           value,
                           0,  # benchmark used so set to zero
                           {'leverage': 0.0})

        for dt, value in drawdown_duration.iteritems():
            self.assertEqual(metrics.max_drawdown_durations[dt], value)
