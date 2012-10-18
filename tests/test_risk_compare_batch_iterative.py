#
# Copyright 2012 Quantopian, Inc.
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
import pytz

import numpy as np

import zipline.finance.risk as risk
from zipline.utils import factory

from zipline.finance.trading import TradingEnvironment
from test_risk import RETURNS


class RiskCompareIterativeToBatch(unittest.TestCase):
    """
    Assert that RiskMetricsIterative and RiskMetricsBatch
    behave in the same way.
    """

    def setUp(self):
        self.start_date = datetime.datetime(
            year=2006,
            month=1,
            day=1,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)
        self.end_date = datetime.datetime(
            year=2006, month=12, day=31, tzinfo=pytz.utc)
        self.benchmark_returns, self.treasury_curves = \
        factory.load_market_data()

        self.trading_env = TradingEnvironment(
            self.benchmark_returns,
            self.treasury_curves,
            period_start=self.start_date,
            period_end=self.end_date,
            capital_base=1000.0
        )

        self.oneday = datetime.timedelta(days=1)

    def test_risk_metrics_returns(self):
        risk_metrics_refactor = risk.RiskMetricsIterative(
            self.start_date, self.trading_env)

        todays_date = self.start_date

        cur_returns = []
        for i, ret in enumerate(RETURNS):
            todays_return_obj = risk.DailyReturn(
                todays_date,
                ret
            )

            cur_returns.append(todays_return_obj)

            try:
                risk_metrics_original = risk.RiskMetricsBatch(
                    start_date=self.start_date,
                    end_date=todays_date + self.oneday,
                    returns=cur_returns,
                    trading_environment=self.trading_env
                )
            except Exception as e:
                #assert that when original raises exception, same
                #exception is raised by risk_metrics_refactor
                np.testing.assert_raises(
                    type(e), risk_metrics_refactor.update, ret, self.oneday)
                continue

            risk_metrics_refactor.update(ret, self.oneday)

            todays_date += self.oneday

            self.assertEqual(
                risk_metrics_original.start_date,
                risk_metrics_refactor.start_date)
            self.assertEqual(
                risk_metrics_original.end_date,
                risk_metrics_refactor.end_date)
            self.assertEqual(
                risk_metrics_original.treasury_duration,
                risk_metrics_refactor.treasury_duration)
            self.assertEqual(
                risk_metrics_original.treasury_curve,
                risk_metrics_refactor.treasury_curve)
            self.assertEqual(
                risk_metrics_original.treasury_period_return,
                risk_metrics_refactor.treasury_period_return)
            self.assertEqual(
                risk_metrics_original.benchmark_returns,
                risk_metrics_refactor.benchmark_returns)
            self.assertEqual(
                risk_metrics_original.algorithm_returns,
                risk_metrics_refactor.algorithm_returns)
            risk_original_dict = risk_metrics_original.to_dict()
            risk_refactor_dict = risk_metrics_refactor.to_dict()
            self.assertEqual(set(risk_original_dict.keys()),
                             set(risk_refactor_dict.keys()))

            err_msg_format = \
"In update step {iter}: {measure} should be {truth} but is {returned}!"

            for measure in risk_original_dict.iterkeys():
                if measure == 'max_drawdown':
                    np.testing.assert_almost_equal(
                        risk_refactor_dict[measure],
                        risk_original_dict[measure],
                        err_msg=err_msg_format.format(
                            iter=i,
                            measure=measure,
                            truth=risk_original_dict[measure],
                            returned=risk_refactor_dict[measure]))
                else:
                    np.testing.assert_equal(
                        risk_original_dict[measure],
                        risk_refactor_dict[measure],
                        err_msg_format.format(
                            iter=i,
                            measure=measure,
                            truth=risk_original_dict[measure],
                            returned=risk_refactor_dict[measure])
                    )
