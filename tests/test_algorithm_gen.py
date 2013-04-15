#!/usr/bin/python
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
from nose.tools import timed

from datetime import datetime

import pytz
import zipline.finance.trading as trading
from zipline.algorithm import TradingAlgorithm
from zipline.finance import slippage
from zipline.utils import factory
from zipline.utils.test_utils import (
    setup_logger,
    teardown_logger
)

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class RecordDateSlippage(slippage.FixedSlippage):
    def __init__(self, spread):
        super(RecordDateSlippage, self).__init__(spread=spread)
        self.latest_date = None

    def simulate(self, event, open_orders):
        self.latest_date = event['datetime']
        result = super(RecordDateSlippage, self).simulate(event, open_orders)
        return result


class TestAlgo(TradingAlgorithm):

    def __init__(self, asserter, *args, **kwargs):
        super(TestAlgo, self).__init__(*args, **kwargs)
        self.asserter = asserter

    def initialize(self, window_length=100):
        self.latest_date = None

        self.set_slippage(RecordDateSlippage(spread=0.05))
        self.stocks = [8229]
        self.ordered = False

    def handle_data(self, data):
        self.latest_date = self.get_datetime()

        if not self.ordered:
            for stock in self.stocks:
                self.order(stock, 100)

            self.ordered = True

        else:

            self.asserter.assertGreaterEqual(
                self.latest_date,
                self.slippage.latest_date
            )


class AlgorithmGeneratorTestCase(TestCase):
    def setUp(self):
        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

    def test_lse_algorithm(self):

        lse = trading.TradingEnvironment(
            bm_symbol='^FTSE',
            exchange_tz='Europe/London'
        )

        with lse:

            sim_params = factory.create_simulation_parameters(
                start=datetime(2012, 5, 1, tzinfo=pytz.utc),
                end=datetime(2012, 6, 30, tzinfo=pytz.utc)
            )
            algo = TestAlgo(self, sim_params=sim_params)
            trade_source = factory.create_daily_trade_source(
                [8229],
                200,
                sim_params
            )
            algo.set_sources([trade_source])

            gen = algo.get_generator()
            results = list(gen)
            self.assertEqual(len(results), 42)
            # May 7, 2012 was an LSE holiday, confirm the 4th trading
            # day was May 8.
            self.assertEqual(results[4]['daily_perf']['period_open'],
                             datetime(2012, 5, 8, 8, 31, tzinfo=pytz.utc))

    @timed(DEFAULT_TIMEOUT)
    def test_generator_dates(self):
        """
        Ensure the pipeline of generators are in sync, at least as far as
        their current dates.
        """
        sim_params = factory.create_simulation_parameters(
            start=datetime(2011, 7, 30, tzinfo=pytz.utc),
            end=datetime(2012, 7, 30, tzinfo=pytz.utc)
        )
        algo = TestAlgo(self, sim_params=sim_params)
        trade_source = factory.create_daily_trade_source(
            [8229],
            200,
            sim_params
        )
        algo.set_sources([trade_source])

        gen = algo.get_generator()
        self.assertTrue(list(gen))

        self.assertTrue(algo.slippage.latest_date)
        self.assertTrue(algo.latest_date)

    @timed(DEFAULT_TIMEOUT)
    def test_progress(self):
        """
        Ensure the pipeline of generators are in sync, at least as far as
        their current dates.
        """
        sim_params = factory.create_simulation_parameters(
            start=datetime(2008, 1, 1, tzinfo=pytz.utc),
            end=datetime(2008, 1, 5, tzinfo=pytz.utc)
        )
        algo = TestAlgo(self, sim_params=sim_params)
        trade_source = factory.create_daily_trade_source(
            [8229],
            3,
            sim_params
        )
        algo.set_sources([trade_source])

        gen = algo.get_generator()
        results = list(gen)
        self.assertEqual(results[-2]['progress'], 1.0)
