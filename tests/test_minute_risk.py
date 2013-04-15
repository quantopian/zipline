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
import pytz

from zipline.finance.trading import SimulationParameters
from zipline.finance import risk


class TestMinuteRisk(unittest.TestCase):

    def setUp(self):

        start_date = datetime.datetime(
            year=2006,
            month=1,
            day=3,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)
        end_date = datetime.datetime(
            year=2006, month=1, day=3, tzinfo=pytz.utc)

        self.sim_params = SimulationParameters(
            period_start=start_date,
            period_end=end_date
        )
        self.sim_params.emission_rate = 'minute'

    def test_minute_risk(self):

        risk_metrics = risk.RiskMetricsIterative(self.sim_params)

        first_dt = self.sim_params.first_open
        second_dt = self.sim_params.first_open + datetime.timedelta(minutes=1)

        risk_metrics.update(first_dt, 1.0, 2.0)

        self.assertEquals(1, len(risk_metrics.alpha))

        risk_metrics.update(second_dt, 3.0, 4.0)

        self.assertEquals(2, len(risk_metrics.alpha))
