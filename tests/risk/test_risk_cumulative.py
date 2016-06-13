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

import datetime
import numpy as np
import pytz
import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase

from zipline.finance.trading import SimulationParameters
from . import answer_key
ANSWER_KEY = answer_key.ANSWER_KEY


class TestRisk(WithTradingEnvironment, ZiplineTestCase):

    def init_instance_fixtures(self):
        super(TestRisk, self).init_instance_fixtures()

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
            period_end=end_date,
            trading_schedule=self.trading_schedule,
        )

        self.algo_returns_06 = factory.create_returns_from_list(
            answer_key.ALGORITHM_RETURNS.values,
            self.sim_params
        )

        self.cumulative_metrics_06 = risk.RiskMetricsCumulative(
            self.sim_params,
            treasury_curves=self.env.treasury_curves,
            trading_schedule=self.trading_schedule,
        )

        for dt, returns in answer_key.RETURNS_DATA.iterrows():
            self.cumulative_metrics_06.update(dt,
                                              returns['Algorithm Returns'],
                                              returns['Benchmark Returns'],
                                              0.0)

    def test_algorithm_volatility_06(self):
        algo_vol_answers = answer_key.RISK_CUMULATIVE.volatility
        for dt, value in algo_vol_answers.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.algorithm_volatility[dt_loc],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_sharpe_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.sharpe.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.sharpe[dt_loc],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_downside_risk_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.downside_risk.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                value,
                self.cumulative_metrics_06.downside_risk[dt_loc],
                err_msg="Mismatch at %s" % (dt,))

    def test_sortino_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.sortino.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.sortino[dt_loc],
                value,
                decimal=4,
                err_msg="Mismatch at %s" % (dt,))

    def test_information_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.information.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                value,
                self.cumulative_metrics_06.information[dt_loc],
                err_msg="Mismatch at %s" % (dt,))

    def test_alpha_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.alpha.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.alpha[dt_loc],
                value,
                err_msg="Mismatch at %s" % (dt,))

    def test_beta_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.beta.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                value,
                self.cumulative_metrics_06.beta[dt_loc],
                err_msg="Mismatch at %s" % (dt,))

    def test_max_drawdown_06(self):
        for dt, value in answer_key.RISK_CUMULATIVE.max_drawdown.iteritems():
            dt_loc = self.cumulative_metrics_06.cont_index.get_loc(dt)
            np.testing.assert_almost_equal(
                self.cumulative_metrics_06.max_drawdowns[dt_loc],
                value,
                err_msg="Mismatch at %s" % (dt,))
