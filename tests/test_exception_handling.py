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

from unittest import TestCase
from testfixtures import TempDirectory

from zipline.finance.trading import TradingEnvironment
from zipline.test_algorithms import (
    ExceptionAlgorithm,
    DivByZeroAlgorithm,
    SetPortfolioAlgorithm,
)
from zipline.testing import (
    setup_logger,
    teardown_logger
)
import zipline.utils.factory as factory
from zipline.testing.core import create_data_portal

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class ExceptionTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sid = 133
        cls.env = TradingEnvironment()
        cls.env.write_data(equities_identifiers=[cls.sid])

        cls.tempdir = TempDirectory()

        cls.sim_params = factory.create_simulation_parameters(
            num_days=4,
            env=cls.env
        )

        cls.data_portal = create_data_portal(
            env=cls.env,
            tempdir=cls.tempdir,
            sim_params=cls.sim_params,
            sids=[cls.sid]
        )

        setup_logger(cls)

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()
        teardown_logger(cls)

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
