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
from zipline.test_algorithms import NoopAlgorithm
from zipline.utils import factory


class TestTradeSimulation(TestCase):

    def test_minutely_emissions_generate_performance_stats_for_last_day(self):
        params = factory.create_simulation_parameters(num_days=1,
                                                      data_frequency='minute',
                                                      emission_rate='minute')
        algo = NoopAlgorithm(sim_params=params)
        algo.run(source=[])
        self.assertEqual(algo.perf_tracker.day_count, 1.0)
