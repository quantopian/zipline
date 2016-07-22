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
from datetime import time

import pandas as pd
from mock import patch

from nose_parameterized import parameterized
from six.moves import range
from unittest import TestCase
from zipline import TradingAlgorithm
from zipline.sources.benchmark_source import BenchmarkSource
from zipline.test_algorithms import NoopAlgorithm
from zipline.utils import factory
from zipline.testing.core import FakeDataPortal
from zipline.utils.calendars.trading_calendar import days_at_time


class BeforeTradingAlgorithm(TradingAlgorithm):
    def __init__(self, *args, **kwargs):
        self.before_trading_at = []
        super(BeforeTradingAlgorithm, self).__init__(*args, **kwargs)

    def before_trading_start(self, data):
        self.before_trading_at.append(self.datetime)

    def handle_data(self, data):
        pass


FREQUENCIES = {'daily': 0, 'minute': 1}  # daily is less frequent than minute


class TestTradeSimulation(TestCase):

    def fake_minutely_benchmark(self, dt):
        return 0.01

    def test_minutely_emissions_generate_performance_stats_for_last_day(self):
        params = factory.create_simulation_parameters(num_days=1,
                                                      data_frequency='minute',
                                                      emission_rate='minute')
        with patch.object(BenchmarkSource, "get_value",
                          self.fake_minutely_benchmark):
            algo = NoopAlgorithm(sim_params=params)
            algo.run(FakeDataPortal())
            self.assertEqual(len(algo.perf_tracker.sim_params.sessions), 1)

    @parameterized.expand([('%s_%s_%s' % (num_sessions, freq, emission_rate),
                            num_sessions, freq, emission_rate)
                           for freq in FREQUENCIES
                           for emission_rate in FREQUENCIES
                           for num_sessions in range(1, 4)
                           if FREQUENCIES[emission_rate] <= FREQUENCIES[freq]])
    def test_before_trading_start(self, test_name, num_days, freq,
                                  emission_rate):
        params = factory.create_simulation_parameters(
            num_days=num_days, data_frequency=freq,
            emission_rate=emission_rate)

        def fake_benchmark(self, dt):
            return 0.01

        with patch.object(BenchmarkSource, "get_value",
                          self.fake_minutely_benchmark):
            algo = BeforeTradingAlgorithm(sim_params=params)
            algo.run(FakeDataPortal())

            self.assertEqual(
                len(algo.perf_tracker.sim_params.sessions),
                num_days
            )

            bts_minutes = days_at_time(
                params.sessions, time(8, 45), "US/Eastern"
            )

            self.assertTrue(
                bts_minutes.equals(
                    pd.DatetimeIndex(algo.before_trading_at)
                ),
                "Expected %s but was %s." % (params.sessions,
                                             algo.before_trading_at))
