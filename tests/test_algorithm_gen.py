#
# Copyright 2014 Quantopian, Inc.
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
from nose.tools import (
    timed,
    nottest
)

from datetime import datetime
import pandas as pd

import pytz
from zipline.finance import trading
from zipline.algorithm import TradingAlgorithm
from zipline.finance import slippage
from zipline.utils import factory
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.test_utils import (
    setup_logger,
    teardown_logger
)
from zipline.protocol import (
    Event,
    DATASOURCE_TYPE
)

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class RecordDateSlippage(slippage.FixedSlippage):
    def __init__(self, spread):
        super(RecordDateSlippage, self).__init__(spread=spread)
        self.latest_date = None

    def simulate(self, event, open_orders):
        self.latest_date = event.dt
        result = super(RecordDateSlippage, self).simulate(event, open_orders)
        return result


class TestAlgo(TradingAlgorithm):

    def __init__(self, asserter, *args, **kwargs):
        super(TestAlgo, self).__init__(*args, **kwargs)
        self.asserter = asserter

    def initialize(self, window_length=100):
        self.latest_date = None

        self.set_slippage(RecordDateSlippage(spread=0.05))
        self.stocks = [self.sid(8229)]
        self.ordered = False
        self.num_bars = 0

    def handle_data(self, data):
        self.num_bars += 1
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

    @nottest
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
            algo = TestAlgo(self, identifiers=[8229], sim_params=sim_params)
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
        algo = TestAlgo(self, identifiers=[8229], sim_params=sim_params)
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
    def test_handle_data_on_market(self):
        """
        Ensure that handle_data is only called on market minutes.

        i.e. events that come in at midnight should be processed at market
        open.
        """
        from zipline.finance.trading import SimulationParameters
        sim_params = SimulationParameters(
            period_start=datetime(2012, 7, 30, tzinfo=pytz.utc),
            period_end=datetime(2012, 7, 30, tzinfo=pytz.utc),
            data_frequency='minute'
        )
        algo = TestAlgo(self, identifiers=[8229], sim_params=sim_params)

        midnight_custom_source = [Event({
            'custom_field': 42.0,
            'sid': 'custom_data',
            'source_id': 'TestMidnightSource',
            'dt': pd.Timestamp('2012-07-30', tz='UTC'),
            'type': DATASOURCE_TYPE.CUSTOM
        })]
        minute_event_source = [Event({
            'volume': 100,
            'price': 200.0,
            'high': 210.0,
            'open_price': 190.0,
            'low': 180.0,
            'sid': 8229,
            'source_id': 'TestMinuteEventSource',
            'dt': pd.Timestamp('2012-07-30 9:31 AM', tz='US/Eastern').
            tz_convert('UTC'),
            'type': DATASOURCE_TYPE.TRADE
        })]

        algo.set_sources([midnight_custom_source, minute_event_source])

        gen = algo.get_generator()
        # Consume the generator
        list(gen)

        # Though the events had different time stamps, handle data should
        # have only been called once, at the market open.
        self.assertEqual(algo.num_bars, 1)

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

    def test_benchmark_times_match_market_close_for_minutely_data(self):
        """
        Benchmark dates should be adjusted so that benchmark events are
        emitted at the end of each trading day when working with minutely
        data.
        Verification relies on the fact that there are no trades so
        algo.datetime should be equal to the last benchmark time.
        See https://github.com/quantopian/zipline/issues/241
        """
        sim_params = create_simulation_parameters(num_days=1,
                                                  data_frequency='minute')
        algo = TestAlgo(self, sim_params=sim_params, identifiers=[8229])
        algo.run(source=[], overwrite_sim_params=False)
        self.assertEqual(algo.datetime, sim_params.last_close)
