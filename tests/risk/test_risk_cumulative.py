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

import numpy as np
import pandas as pd
import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase

from zipline.finance.trading import SimulationParameters

RETURNS_BASE = 0.01
RETURNS = [RETURNS_BASE] * 251

BENCHMARK_BASE = 0.005
BENCHMARK = [BENCHMARK_BASE] * 251
DECIMAL_PLACES = 8


class TestRisk(WithTradingEnvironment, ZiplineTestCase):

    def init_instance_fixtures(self):
        super(TestRisk, self).init_instance_fixtures()

        start_session = pd.Timestamp("2006-01-01", tz='UTC')
        end_session = pd.Timestamp("2006-12-29", tz='UTC')

        self.sim_params = SimulationParameters(
            start_session=start_session,
            end_session=end_session,
            trading_calendar=self.trading_calendar,
        )
        self.algo_returns = factory.create_returns_from_list(
            RETURNS,
            self.sim_params
        )
        self.cumulative_metrics = risk.RiskMetricsCumulative(
            self.sim_params,
            treasury_curves=self.env.treasury_curves,
            trading_calendar=self.trading_calendar,
        )
        for dt, returns in self.algo_returns.iteritems():
            self.cumulative_metrics.update(
                dt,
                returns,
                BENCHMARK_BASE,
                0.0
            )

    def test_algorithm_volatility(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.algorithm_volatility)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.algorithm_volatility),
            True
        )

    def test_sharpe(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.sharpe)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.sharpe),
            True)

    def test_downside_risk(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.downside_risk)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.downside_risk),
            True)

    def test_sortino(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.sortino)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.sortino),
            True)

    def test_information(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.information)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.information),
            True)

    def test_alpha(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.alpha)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.alpha),
            True)

    def test_beta(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.beta)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.beta),
            True)

    def test_max_drawdown(self):
        np.testing.assert_equal(
            len(self.algo_returns),
            len(self.cumulative_metrics.max_drawdowns)
        )
        np.testing.assert_equal(
            all(isinstance(x, float)
                for x in self.cumulative_metrics.max_drawdowns),
            True)

    def test_representation(self):
        assert all([metric in self.cumulative_metrics.__repr__() for metric in
                   self.cumulative_metrics.METRIC_NAMES])
