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
from datetime import timedelta

from zipline.assets.synthetic import make_simple_equity_info
from zipline.test_algorithms import (
    ExceptionAlgorithm,
    DivByZeroAlgorithm,
    SetPortfolioAlgorithm,
)
from zipline.testing.fixtures import (
    WithDataPortal,
    ZiplineTestCase,
)

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class ExceptionTestCase(WithDataPortal, ZiplineTestCase):
    SIM_PARAMS_END = None
    SIM_PARAMS_NUM_DAYS = 4

    sid = 133

    @classmethod
    def make_equities_info(cls):
        return make_simple_equity_info(
            [cls.sid],
            cls.SIM_PARAMS_START,
            cls.SIM_PARAMS_START + timedelta(days=7),
        )

    def test_exception_in_handle_data(self):
        algo = ExceptionAlgorithm('handle_data',
                                  self.sid,
                                  sim_params=self.sim_params,
                                  env=self.env)

        with self.assertRaises(Exception) as ctx:
            algo.run(self.data_portal)

        self.assertEqual(str(ctx.exception), 'Algo exception in handle_data')

    def test_zerodivision_exception_in_handle_data(self):
        algo = DivByZeroAlgorithm(self.sid,
                                  sim_params=self.sim_params,
                                  env=self.env)

        with self.assertRaises(ZeroDivisionError):
            algo.run(self.data_portal)

    def test_set_portfolio(self):
        """
        Are we protected against overwriting an algo's portfolio?
        """
        algo = SetPortfolioAlgorithm(self.sid,
                                     sim_params=self.sim_params,
                                     env=self.env)

        with self.assertRaises(AttributeError):
            algo.run(self.data_portal)
