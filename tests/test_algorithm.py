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
from datetime import timedelta
import numpy as np

from zipline.utils.test_utils import setup_logger
import zipline.utils.factory as factory
from zipline.test_algorithms import (TestRegisterTransformAlgorithm,
                                     RecordAlgorithm)
from zipline.sources import (SpecificEquityTrades,
                             DataFrameSource,
                             DataPanelSource)
from zipline.transforms import MovingAverage
from zipline.finance.trading import SimulationParameters


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
