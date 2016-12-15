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
import pandas as pd

from zipline.test_algorithms import (
    ExceptionAlgorithm,
    DivByZeroAlgorithm,
    SetPortfolioAlgorithm,
)
from zipline.testing.fixtures import (
    WithDataPortal,
    WithSimParams,
    ZiplineTestCase,
)

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class ExceptionTestCase(WithDataPortal, WithSimParams, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    START_DATE = pd.Timestamp('2006-01-07', tz='utc')

    sid, = ASSET_FINDER_EQUITY_SIDS = 133,

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
