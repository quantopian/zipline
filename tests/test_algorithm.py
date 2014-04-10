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
from datetime import timedelta
import numpy as np
import pandas as pd
from mock import MagicMock

from zipline.utils.test_utils import setup_logger
import zipline.utils.factory as factory
import zipline.utils.simfactory as simfactory

from zipline.test_algorithms import (TestRegisterTransformAlgorithm,
                                     RecordAlgorithm,
                                     TestOrderAlgorithm,
                                     TestOrderInstantAlgorithm,
                                     TestOrderValueAlgorithm,
                                     TestTargetAlgorithm,
                                     TestOrderPercentAlgorithm,
                                     TestTargetPercentAlgorithm,
                                     TestTargetValueAlgorithm,
                                     EmptyPositionsAlgorithm,
                                     initialize_noop,
                                     handle_data_noop,
                                     initialize_api,
                                     handle_data_api,
                                     noop_algo,
                                     api_algo,
                                     api_symbol_algo,
                                     call_all_order_methods,
                                     record_variables,
                                     record_float_magic
                                     )

from zipline.utils.test_utils import drain_zipline, assert_single_position

from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource,
                             RandomWalkSource)

from zipline.transforms import MovingAverage
from zipline.finance.trading import SimulationParameters
from zipline.utils.api_support import set_algo_instance
from zipline.algorithm import TradingAlgorithm


class TestRecordAlgorithm(TestCase):
    def setUp(self):
        self.sim_params = factory.create_simulation_parameters(num_days=4)
        trade_history = factory.create_trade_history(
            133,
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            self.sim_params
        )

        self.source = SpecificEquityTrades(event_list=trade_history)
        self.df_source, self.df = \
            factory.create_test_df_source(self.sim_params)

    def test_record_incr(self):
        algo = RecordAlgorithm(
            sim_params=self.sim_params,
            data_frequency='daily')
        output = algo.run(self.source)

        np.testing.assert_array_equal(output['incr'].values,
                                      range(1, len(output) + 1))


class TestTransformAlgorithm(TestCase):
    def setUp(self):
        setup_logger(self)
        self.sim_params = factory.create_simulation_parameters(num_days=4)
        setup_logger(self)

        trade_history = factory.create_trade_history(
            133,
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            self.sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        self.df_source, self.df = \
            factory.create_test_df_source(self.sim_params)

        self.panel_source, self.panel = \
            factory.create_test_panel_source(self.sim_params)

    def test_source_as_input(self):
        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[133]
        )
        algo.run(self.source)
        self.assertEqual(len(algo.sources), 1)
        assert isinstance(algo.sources[0], SpecificEquityTrades)

    def test_multi_source_as_input_no_start_end(self):
        algo = TestRegisterTransformAlgorithm(
            sids=[133]
        )

        with self.assertRaises(AssertionError):
            algo.run([self.source, self.df_source])

    def test_multi_source_as_input(self):
        sim_params = SimulationParameters(
            self.df.index[0],
            self.df.index[-1]
        )
        algo = TestRegisterTransformAlgorithm(
            sim_params=sim_params,
            sids=[0, 1, 133]
        )
        algo.run([self.source, self.df_source])
        self.assertEqual(len(algo.sources), 2)

    def test_df_as_input(self):
        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[0, 1]
        )
        algo.run(self.df)
        assert isinstance(algo.sources[0], DataFrameSource)

    def test_panel_as_input(self):
        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[0, 1])
        algo.run(self.panel)
        assert isinstance(algo.sources[0], DataPanelSource)

    def test_run_twice(self):
        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[0, 1]
        )

        res1 = algo.run(self.df)
        res2 = algo.run(self.df)

        np.testing.assert_array_equal(res1, res2)

    def test_transform_registered(self):
        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[133]
        )

        algo.run(self.source)
        assert 'mavg' in algo.registered_transforms
        assert algo.registered_transforms['mavg']['args'] == (['price'],)
        assert algo.registered_transforms['mavg']['kwargs'] == \
            {'window_length': 2, 'market_aware': True}
        assert algo.registered_transforms['mavg']['class'] is MovingAverage

    def test_data_frequency_setting(self):
        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            data_frequency='daily'
        )
        self.assertEqual(algo.data_frequency, 'daily')
        self.assertEqual(algo.annualizer, 250)

        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            data_frequency='minute'
        )
        self.assertEqual(algo.data_frequency, 'minute')
        self.assertEqual(algo.annualizer, 250 * 6 * 60)

        algo = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            data_frequency='minute',
            annualizer=10
        )
        self.assertEqual(algo.data_frequency, 'minute')
        self.assertEqual(algo.annualizer, 10)

    def test_order_methods(self):
        AlgoClasses = [TestOrderAlgorithm,
                       TestOrderValueAlgorithm,
                       TestTargetAlgorithm,
                       TestOrderPercentAlgorithm,
                       TestTargetPercentAlgorithm,
                       TestTargetValueAlgorithm]

        for AlgoClass in AlgoClasses:
            algo = AlgoClass(
                sim_params=self.sim_params,
                data_frequency='daily'
            )
            algo.run(self.df)

    def test_order_instant(self):
        algo = TestOrderInstantAlgorithm(sim_params=self.sim_params,
                                         data_frequency='daily',
                                         instant_fill=True)

        algo.run(self.df)

    def test_minute_data(self):
        source = RandomWalkSource(freq='minute',
                                  start=pd.Timestamp('2000-1-1',
                                                     tz='UTC'),
                                  end=pd.Timestamp('2000-1-1',
                                                   tz='UTC'))
        algo = TestOrderInstantAlgorithm(sim_params=self.sim_params,
                                         data_frequency='minute',
                                         instant_fill=True)
        algo.run(source)


class TestPositions(TestCase):

    def setUp(self):
        setup_logger(self)
        self.sim_params = factory.create_simulation_parameters(num_days=4)
        setup_logger(self)

        trade_history = factory.create_trade_history(
            1,
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            self.sim_params
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        self.df_source, self.df = \
            factory.create_test_df_source(self.sim_params)

        self.panel_source, self.panel = \
            factory.create_test_panel_source(self.sim_params)

    def test_empty_portfolio(self):
        algo = EmptyPositionsAlgorithm(sim_params=self.sim_params,
                                       data_frequency='daily')

        daily_stats = algo.run(self.df)

        expected_position_count = [
            0,  # Before entering the first position
            1,  # After entering, exiting on this date
            0,  # After exiting
            0,
        ]

        for i, expected in enumerate(expected_position_count):
            self.assertEqual(daily_stats.ix[i]['num_positions'],
                             expected)


class TestAlgoScript(TestCase):
    def setUp(self):
        days = 251
        self.sim_params = factory.create_simulation_parameters(num_days=days)
        setup_logger(self)

        trade_history = factory.create_trade_history(
            133,
            [10.0] * days,
            [100] * days,
            timedelta(days=1),
            self.sim_params
        )

        self.source = SpecificEquityTrades(event_list=trade_history)
        self.df_source, self.df = \
            factory.create_test_df_source(self.sim_params)

        self.zipline_test_config = {
            'sid': 0,
        }

    def test_noop(self):
        algo = TradingAlgorithm(initialize=initialize_noop,
                                handle_data=handle_data_noop)
        algo.run(self.df)

    def test_noop_string(self):
        algo = TradingAlgorithm(script=noop_algo)
        algo.run(self.df)

    def test_api_calls(self):
        algo = TradingAlgorithm(initialize=initialize_api,
                                handle_data=handle_data_api)
        algo.run(self.df)

    def test_api_calls_string(self):
        algo = TradingAlgorithm(script=api_algo)
        algo.run(self.df)

    def test_api_symbol(self):
        algo = TradingAlgorithm(script=api_symbol_algo)
        algo.run(self.df)

    def test_fixed_slippage(self):
        # verify order -> transaction -> portfolio position.
        # --------------
        test_algo = TradingAlgorithm(
            script="""
from zipline.api import (slippage,
                         commission,
                         set_slippage,
                         set_commission,
                         order,
                         record)

def initialize(context):
    model = slippage.FixedSlippage(spread=0.10)
    set_slippage(model)
    set_commission(commission.PerTrade(100.00))
    context.count = 1
    context.incr = 0

def handle_data(context, data):
    if context.incr < context.count:
        order(0, -1000)
    record(price=data[0].price)

    context.incr += 1""",
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        self.zipline_test_config['algorithm'] = test_algo
        self.zipline_test_config['trade_count'] = 200

        # this matches the value in the algotext initialize
        # method, and will be used inside assert_single_position
        # to confirm we have as many transactions as orders we
        # placed.
        self.zipline_test_config['order_count'] = 1

        # self.zipline_test_config['transforms'] = \
        #     test_algo.transform_visitor.transforms.values()

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config)

        output, _ = assert_single_position(self, zipline)

        # confirm the slippage and commission on a sample
        # transaction
        recorded_price = output[1]['daily_perf']['recorded_vars']['price']
        transaction = output[1]['daily_perf']['transactions'][0]
        self.assertEqual(100.0, transaction['commission'])
        expected_spread = 0.05
        expected_commish = 0.10
        expected_price = recorded_price - expected_spread - expected_commish
        self.assertEqual(expected_price, transaction['price'])

    def test_volshare_slippage(self):
        # verify order -> transaction -> portfolio position.
        # --------------
        test_algo = TradingAlgorithm(
            script="""
from zipline.api import *

def initialize(context):
    model = slippage.VolumeShareSlippage(
                            volume_limit=.3,
                            price_impact=0.05
                       )
    set_slippage(model)
    set_commission(commission.PerShare(0.02))
    context.count = 2
    context.incr = 0

def handle_data(context, data):
    if context.incr < context.count:
        # order small lots to be sure the
        # order will fill in a single transaction
        order(0, 5000)
    record(price=data[0].price)
    record(volume=data[0].volume)
    record(incr=context.incr)
    context.incr += 1
    """,
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        self.zipline_test_config['algorithm'] = test_algo
        self.zipline_test_config['trade_count'] = 100

        # 67 will be used inside assert_single_position
        # to confirm we have as many transactions as expected.
        # The algo places 2 trades of 5000 shares each. The trade
        # events have volume ranging from 100 to 950. The volume cap
        # of 0.3 limits the trade volume to a range of 30 - 316 shares.
        # The spreadsheet linked below calculates the total position
        # size over each bar, and predicts 67 txns will be required
        # to fill the two orders. The number of bars and transactions
        # differ because some bars result in multiple txns. See
        # spreadsheet for details:
# https://www.dropbox.com/s/ulrk2qt0nrtrigb/Volume%20Share%20Worksheet.xlsx
        self.zipline_test_config['expected_transactions'] = 67

        # self.zipline_test_config['transforms'] = \
        #     test_algo.transform_visitor.transforms.values()

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config)
        output, _ = assert_single_position(self, zipline)

        # confirm the slippage and commission on a sample
        # transaction
        per_share_commish = 0.02
        perf = output[1]
        transaction = perf['daily_perf']['transactions'][0]
        commish = transaction['amount'] * per_share_commish
        self.assertEqual(commish, transaction['commission'])
        self.assertEqual(2.029, transaction['price'])

    def test_algo_record_vars(self):
        test_algo = TradingAlgorithm(
            script=record_variables,
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        self.zipline_test_config['algorithm'] = test_algo
        self.zipline_test_config['trade_count'] = 200

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config)
        output, _ = drain_zipline(self, zipline)
        self.assertEqual(len(output), 252)
        incr = []
        for o in output[:200]:
            incr.append(o['daily_perf']['recorded_vars']['incr'])

        np.testing.assert_array_equal(incr, range(1, 201))

    def test_algo_record_allow_mock(self):
        """
        Test that values from "MagicMock"ed methods can be passed to record.

        Relevant for our basic/validation and methods like history, which
        will end up returning a MagicMock instead of a DataFrame.
        """
        test_algo = TradingAlgorithm(
            script=record_variables,
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        test_algo.record(foo=MagicMock())

    def _algo_record_float_magic_should_pass(self, var_type):
        test_algo = TradingAlgorithm(
            script=record_float_magic % var_type,
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        self.zipline_test_config['algorithm'] = test_algo
        self.zipline_test_config['trade_count'] = 200

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config)
        output, _ = drain_zipline(self, zipline)
        self.assertEqual(len(output), 252)
        incr = []
        for o in output[:200]:
            incr.append(o['daily_perf']['recorded_vars']['data'])
        np.testing.assert_array_equal(incr, [np.nan] * 200)

    def test_algo_record_nan(self):
        self._algo_record_float_magic_should_pass('nan')

    def test_order_methods(self):
        """Only test that order methods can be called without error.
        Correct filling of orders is tested in zipline.
        """
        test_algo = TradingAlgorithm(
            script=call_all_order_methods,
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        self.zipline_test_config['algorithm'] = test_algo
        self.zipline_test_config['trade_count'] = 200

        zipline = simfactory.create_test_zipline(
            **self.zipline_test_config)

        output, _ = drain_zipline(self, zipline)


class TestHistory(TestCase):
    def test_history(self):
        history_algo = """
from zipline.api import history, add_history

def initialize(context):
    add_history(10, '1d', 'price')

def handle_data(context, data):
    df = history(10, '1d', 'price')
"""
        start = pd.Timestamp('1991-01-01', tz='UTC')
        end = pd.Timestamp('1991-01-15', tz='UTC')
        source = RandomWalkSource(start=start,
                                  end=end)
        algo = TradingAlgorithm(script=history_algo, data_frequency='minute')
        output = algo.run(source)
        self.assertIsNot(output, None)
