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

import numbers
import unittest
import datetime
import pytz

import numpy as np
import pandas as pd

import zipline.finance.risk as risk
import zipline.finance.trading as trading
from zipline.finance.trading import SimulationParameters
from zipline.protocol import DailyReturn

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

    def test_risk_metrics_returns(self):
        trading.environment = trading.TradingEnvironment()
        # Advance start date to first date in the trading calendar
        if trading.environment.is_trading_day(self.start_date):
            start_date = self.start_date
        else:
            start_date = trading.environment.next_trading_day(self.start_date)

        self.all_benchmark_returns = pd.Series({
            x.date: x.returns
            for x in trading.environment.benchmark_returns
            if x.date >= self.start_date
        })

        start_index = trading.environment.trading_days.searchsorted(start_date)
        end_date = trading.environment.trading_days[
            start_index + len(RETURNS)]

        sim_params = SimulationParameters(start_date, end_date)

        risk_metrics_refactor = risk.RiskMetricsIterative(sim_params)
        todays_date = start_date

        cur_returns = []
        for i, ret in enumerate(RETURNS):

            todays_return_obj = DailyReturn(
                todays_date,
                ret
            )
            cur_returns.append(todays_return_obj)

            # Move forward day counter to next trading day
            todays_date = trading.environment.next_trading_day(todays_date)

            try:
                risk_metrics_original = risk.RiskMetricsBatch(
                    start_date=start_date,
                    end_date=todays_date,
                    returns=cur_returns
                )
            except Exception as e:
                #assert that when original raises exception, same
                #exception is raised by risk_metrics_refactor
                np.testing.assert_raises(
                    type(e),
                    risk_metrics_refactor.update,
                    todays_date,
                    self.all_benchmark_returns[todays_return_obj.date]
                )
                continue

            risk_metrics_refactor.update(
                todays_date,
                ret,
                self.all_benchmark_returns[todays_return_obj.date])

            self.assertEqual(
                risk_metrics_original.start_date,
                risk_metrics_refactor.start_date)
            self.assertEqual(
                risk_metrics_original.end_date,
                risk_metrics_refactor.algorithm_returns.index[-1])
            self.assertEqual(
                risk_metrics_original.treasury_period_return,
                risk_metrics_refactor.treasury_period_return)
            np.testing.assert_allclose(
                risk_metrics_original.benchmark_returns,
                risk_metrics_refactor.benchmark_returns,
                rtol=0.001
            )
            np.testing.assert_allclose(
                risk_metrics_original.algorithm_returns,
                risk_metrics_refactor.algorithm_returns,
                rtol=0.001
            )
            risk_original_dict = risk_metrics_original.to_dict()
            risk_refactor_dict = risk_metrics_refactor.to_dict()
            self.assertEqual(set(risk_original_dict.keys()),
                             set(risk_refactor_dict.keys()))

            err_msg_format = """\
"In update step {iter}: {measure} should be {truth} but is {returned}!"""

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
                    if isinstance(risk_original_dict[measure], numbers.Real):
                        np.testing.assert_allclose(
                            risk_original_dict[measure],
                            risk_refactor_dict[measure],
                            rtol=0.001,
                            err_msg=err_msg_format.format(
                                iter=i,
                                measure=measure,
                                truth=risk_original_dict[measure],
                                returned=risk_refactor_dict[measure])
                        )
                    else:
                        np.testing.assert_equal(
                            risk_original_dict[measure],
                            risk_refactor_dict[measure],
                            err_msg=err_msg_format.format(
                                iter=i,
                                measure=measure,
                                truth=risk_original_dict[measure],
                                returned=risk_refactor_dict[measure])
                        )
