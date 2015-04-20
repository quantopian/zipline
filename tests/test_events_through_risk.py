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

import numpy as np

from zipline.finance.trading import SimulationParameters
from zipline.finance import trading
from zipline.algorithm import TradingAlgorithm
from zipline.protocol import (
    Event,
    DATASOURCE_TYPE
)


class BuyAndHoldAlgorithm(TradingAlgorithm):

    SID_TO_BUY_AND_HOLD = 1

    def initialize(self):
        self.holding = False

    def handle_data(self, data):
        if not self.holding:
            self.order(self.sid(self.SID_TO_BUY_AND_HOLD), 100)
            self.holding = True


class TestEventsThroughRisk(unittest.TestCase):

    def test_daily_buy_and_hold(self):

        start_date = datetime.datetime(
            year=2006,
            month=1,
            day=3,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)
        end_date = datetime.datetime(
            year=2006,
            month=1,
            day=5,
            hour=0,
            minute=0,
            tzinfo=pytz.utc)

        sim_params = SimulationParameters(
            period_start=start_date,
            period_end=end_date,
            data_frequency='daily',
            emission_rate='daily'
        )

        algo = BuyAndHoldAlgorithm(
            identifiers=[1],
            sim_params=sim_params)

        first_date = datetime.datetime(2006, 1, 3, tzinfo=pytz.utc)
        second_date = datetime.datetime(2006, 1, 4, tzinfo=pytz.utc)
        third_date = datetime.datetime(2006, 1, 5, tzinfo=pytz.utc)

        trade_bar_data = [
            Event({
                'open_price': 10,
                'close_price': 15,
                'price': 15,
                'volume': 1000,
                'sid': 1,
                'dt': first_date,
                'source_id': 'test-trade-source',
                'type': DATASOURCE_TYPE.TRADE
            }),
            Event({
                'open_price': 15,
                'close_price': 20,
                'price': 20,
                'volume': 2000,
                'sid': 1,
                'dt': second_date,
                'source_id': 'test_list',
                'type': DATASOURCE_TYPE.TRADE
            }),
            Event({
                'open_price': 20,
                'close_price': 15,
                'price': 15,
                'volume': 1000,
                'sid': 1,
                'dt': third_date,
                'source_id': 'test_list',
                'type': DATASOURCE_TYPE.TRADE
            }),
        ]
        benchmark_data = [
            Event({
                'returns': 0.1,
                'dt': first_date,
                'source_id': 'test-benchmark-source',
                'type': DATASOURCE_TYPE.BENCHMARK
            }),
            Event({
                'returns': 0.2,
                'dt': second_date,
                'source_id': 'test-benchmark-source',
                'type': DATASOURCE_TYPE.BENCHMARK
            }),
            Event({
                'returns': 0.4,
                'dt': third_date,
                'source_id': 'test-benchmark-source',
                'type': DATASOURCE_TYPE.BENCHMARK
            }),
        ]

        algo.benchmark_return_source = benchmark_data
        algo.set_sources(list([trade_bar_data]))
        gen = algo._create_generator(sim_params)

        # TODO: Hand derive these results.
        #       Currently, the output from the time of this writing to
        #       at least be an early warning against changes.
        expected_algorithm_returns = {
            first_date: 0.0,
            second_date: -0.000350,
            third_date: -0.050018
        }

        # TODO: Hand derive these results.
        #       Currently, the output from the time of this writing to
        #       at least be an early warning against changes.
        expected_sharpe = {
            first_date: np.nan,
            second_date: -22.322677,
            third_date: -9.353741
        }

        for bar in gen:
            current_dt = algo.datetime
            crm = algo.perf_tracker.cumulative_risk_metrics

            np.testing.assert_almost_equal(
                crm.algorithm_returns[current_dt],
                expected_algorithm_returns[current_dt],
                decimal=6)

            np.testing.assert_almost_equal(
                crm.metrics.sharpe[current_dt],
                expected_sharpe[current_dt],
                decimal=6,
                err_msg="Mismatch at %s" % (current_dt,))

    def test_minute_buy_and_hold(self):
        with trading.TradingEnvironment():
            start_date = datetime.datetime(
                year=2006,
                month=1,
                day=3,
                hour=0,
                minute=0,
                tzinfo=pytz.utc)
            end_date = datetime.datetime(
                year=2006,
                month=1,
                day=5,
                hour=0,
                minute=0,
                tzinfo=pytz.utc)

            sim_params = SimulationParameters(
                period_start=start_date,
                period_end=end_date,
                emission_rate='daily',
                data_frequency='minute')

            algo = BuyAndHoldAlgorithm(
                identifiers=[1],
                sim_params=sim_params)

            first_date = datetime.datetime(2006, 1, 3, tzinfo=pytz.utc)
            first_open, first_close = \
                trading.environment.get_open_and_close(first_date)

            second_date = datetime.datetime(2006, 1, 4, tzinfo=pytz.utc)
            second_open, second_close = \
                trading.environment.get_open_and_close(second_date)

            third_date = datetime.datetime(2006, 1, 5, tzinfo=pytz.utc)
            third_open, third_close = \
                trading.environment.get_open_and_close(third_date)

            benchmark_data = [
                Event({
                    'returns': 0.1,
                    'dt': first_close,
                    'source_id': 'test-benchmark-source',
                    'type': DATASOURCE_TYPE.BENCHMARK
                }),
                Event({
                    'returns': 0.2,
                    'dt': second_close,
                    'source_id': 'test-benchmark-source',
                    'type': DATASOURCE_TYPE.BENCHMARK
                }),
                Event({
                    'returns': 0.4,
                    'dt': third_close,
                    'source_id': 'test-benchmark-source',
                    'type': DATASOURCE_TYPE.BENCHMARK
                }),
            ]

            trade_bar_data = [
                Event({
                    'open_price': 10,
                    'close_price': 15,
                    'price': 15,
                    'volume': 1000,
                    'sid': 1,
                    'dt': first_open,
                    'source_id': 'test-trade-source',
                    'type': DATASOURCE_TYPE.TRADE
                }),
                Event({
                    'open_price': 10,
                    'close_price': 15,
                    'price': 15,
                    'volume': 1000,
                    'sid': 1,
                    'dt': first_open + datetime.timedelta(minutes=10),
                    'source_id': 'test-trade-source',
                    'type': DATASOURCE_TYPE.TRADE
                }),
                Event({
                    'open_price': 15,
                    'close_price': 20,
                    'price': 20,
                    'volume': 2000,
                    'sid': 1,
                    'dt': second_open,
                    'source_id': 'test-trade-source',
                    'type': DATASOURCE_TYPE.TRADE
                }),
                Event({
                    'open_price': 15,
                    'close_price': 20,
                    'price': 20,
                    'volume': 2000,
                    'sid': 1,
                    'dt': second_open + datetime.timedelta(minutes=10),
                    'source_id': 'test-trade-source',
                    'type': DATASOURCE_TYPE.TRADE
                }),
                Event({
                    'open_price': 20,
                    'close_price': 15,
                    'price': 15,
                    'volume': 1000,
                    'sid': 1,
                    'dt': third_open,
                    'source_id': 'test-trade-source',
                    'type': DATASOURCE_TYPE.TRADE
                }),
                Event({
                    'open_price': 20,
                    'close_price': 15,
                    'price': 15,
                    'volume': 1000,
                    'sid': 1,
                    'dt': third_open + datetime.timedelta(minutes=10),
                    'source_id': 'test-trade-source',
                    'type': DATASOURCE_TYPE.TRADE
                }),
            ]

            algo.benchmark_return_source = benchmark_data
            algo.set_sources(list([trade_bar_data]))
            gen = algo._create_generator(sim_params)

            crm = algo.perf_tracker.cumulative_risk_metrics

            first_msg = next(gen)

            self.assertIsNotNone(first_msg,
                                 "There should be a message emitted.")

            # Protects against bug where the positions appeared to be
            # a day late, because benchmarks were triggering
            # calculations before the events for the day were
            # processed.
            self.assertEqual(1, len(algo.portfolio.positions), "There should "
                             "be one position after the first day.")

            self.assertEquals(
                0,
                crm.metrics.algorithm_volatility[algo.datetime.date()],
                "On the first day algorithm volatility does not exist.")

            second_msg = next(gen)

            self.assertIsNotNone(second_msg, "There should be a message "
                                 "emitted.")

            self.assertEqual(1, len(algo.portfolio.positions),
                             "Number of positions should stay the same.")

            # TODO: Hand derive. Current value is just a canary to
            # detect changes.
            np.testing.assert_almost_equal(
                0.050022510129558301,
                crm.algorithm_returns[-1],
                decimal=6)

            third_msg = next(gen)

            self.assertEqual(1, len(algo.portfolio.positions),
                             "Number of positions should stay the same.")

            self.assertIsNotNone(third_msg, "There should be a message "
                                 "emitted.")

            # TODO: Hand derive. Current value is just a canary to
            # detect changes.
            np.testing.assert_almost_equal(
                -0.047639464532418657,
                crm.algorithm_returns[-1],
                decimal=6)
