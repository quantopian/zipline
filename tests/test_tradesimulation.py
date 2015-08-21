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
import pandas as pd

from nose_parameterized import parameterized
from six.moves import range
from unittest import TestCase
from zipline import TradingAlgorithm
from zipline.test_algorithms import NoopAlgorithm
from zipline.utils import factory


class BeforeTradingAlgorithm(TradingAlgorithm):
    def __init__(self, *args, **kwargs):
        self.before_trading_at = []
        super(BeforeTradingAlgorithm, self).__init__(*args, **kwargs)

    def before_trading_start(self, data):
        self.before_trading_at.append(self.datetime)


FREQUENCIES = {'daily': 0, 'minute': 1}  # daily is less frequent than minute


class TestTradeSimulation(TestCase):

    def test_minutely_emissions_generate_performance_stats_for_last_day(self):
        params = factory.create_simulation_parameters(num_days=1,
                                                      data_frequency='minute',
                                                      emission_rate='minute')
        algo = NoopAlgorithm(sim_params=params)
        algo.run(source=[], overwrite_sim_params=False)
        self.assertEqual(algo.perf_tracker.day_count, 1.0)

    @parameterized.expand([('%s_%s_%s' % (num_days, freq, emission_rate),
                            num_days, freq, emission_rate)
                           for freq in FREQUENCIES
                           for emission_rate in FREQUENCIES
                           for num_days in range(1, 4)
                           if FREQUENCIES[emission_rate] <= FREQUENCIES[freq]])
    def test_before_trading_start(self, test_name, num_days, freq,
                                  emission_rate):
        params = factory.create_simulation_parameters(
            num_days=num_days, data_frequency=freq,
            emission_rate=emission_rate)

        algo = BeforeTradingAlgorithm(sim_params=params)
        algo.run(source=[], overwrite_sim_params=False)

        self.assertEqual(algo.perf_tracker.day_count, num_days)
        self.assertTrue(params.trading_days.equals(
            pd.DatetimeIndex(algo.before_trading_at)),
            "Expected %s but was %s."
            % (params.trading_days, algo.before_trading_at))
