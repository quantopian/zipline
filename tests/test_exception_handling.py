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

import zipline.utils.simfactory as simfactory
import zipline.utils.factory as factory

from zipline.test_algorithms import (
    ExceptionAlgorithm,
    DivByZeroAlgorithm,
    SetPortfolioAlgorithm,
)
from zipline.finance.slippage import FixedSlippage
from zipline.finance.trading import TradingEnvironment


from zipline.utils.test_utils import (
    drain_zipline,
    setup_logger,
    teardown_logger,
    ExceptionSource,
)

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class ExceptionTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.env.write_data(equities_identifiers=[133])

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def setUp(self):
        self.zipline_test_config = {
            'sid': 133,
            'slippage': FixedSlippage()
        }
        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

    def test_datasource_exception(self):
        self.zipline_test_config['trade_source'] = ExceptionSource()
        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config
        )

        with self.assertRaises(ZeroDivisionError):
            output, _ = drain_zipline(self, zipline)

    def test_exception_in_handle_data(self):
        # Simulation
        # ----------
        self.zipline_test_config['algorithm'] = \
            ExceptionAlgorithm(
                'handle_data',
                self.zipline_test_config['sid'],
                sim_params=factory.create_simulation_parameters(),
                env=self.env
        )

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config
        )

        with self.assertRaises(Exception) as ctx:
            output, _ = drain_zipline(self, zipline)

        self.assertEqual(str(ctx.exception), 'Algo exception in handle_data')

    def test_zerodivision_exception_in_handle_data(self):

        # Simulation
        # ----------
        self.zipline_test_config['algorithm'] = \
            DivByZeroAlgorithm(
                self.zipline_test_config['sid'],
                sim_params=factory.create_simulation_parameters(),
                env=self.env
        )

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config
        )

        with self.assertRaises(ZeroDivisionError):
            output, _ = drain_zipline(self, zipline)

    def test_set_portfolio(self):
        """
        Are we protected against overwriting an algo's portfolio?
        """

        # Simulation
        # ----------
        self.zipline_test_config['algorithm'] = \
            SetPortfolioAlgorithm(
                self.zipline_test_config['sid'],
                sim_params=factory.create_simulation_parameters(),
                env=self.env
        )

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config
        )

        with self.assertRaises(AttributeError):
            output, _ = drain_zipline(self, zipline)
