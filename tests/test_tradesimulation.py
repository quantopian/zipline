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

from zipline.gens.sim_engine import BEFORE_TRADING_START_BAR

from zipline.finance.asset_restrictions import NoRestrictions
from zipline.finance import metrics
from zipline.finance.trading import SimulationParameters
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.testing.core import parameter_space
import zipline.testing.fixtures as zf


class TestBeforeTradingStartTiming(
    zf.WithMakeAlgo, zf.WithTradingSessions, zf.ZiplineTestCase
):

    ASSET_FINDER_EQUITY_SIDS = (1,)
    BENCHMARK_SID = 1
    # These dates are chosen to cross a DST transition.
    #      March 2016
    # Su Mo Tu We Th Fr Sa
    #        1  2  3  4  5
    #  6  7  8  9 10 11 12
    # 13 14 15 16 17 18 19
    # 20 21 22 23 24 25 26
    # 27 28 29 30 31
    START_DATE = pd.Timestamp("2016-03-10")
    END_DATE = pd.Timestamp("2016-03-15")

    @parameter_space(
        num_sessions=[1, 2, 3],
        data_frequency=["daily", "minute"],
        emission_rate=["daily", "minute"],
        __fail_fast=True,
    )
    def test_before_trading_start_runs_at_8_45(
        self, num_sessions, data_frequency, emission_rate
    ):
        bts_times = []

        def initialize(algo, data):
            pass

        def before_trading_start(algo, data):
            bts_times.append(algo.get_datetime())

        sim_params = SimulationParameters(
            # start at index 1 so we have an extra day to calculate benchmark
            # returns.
            start_session=self.nyse_sessions[1],
            end_session=self.nyse_sessions[num_sessions],
            data_frequency=data_frequency,
            emission_rate=emission_rate,
            trading_calendar=self.trading_calendar,
        )

        self.run_algorithm(
            before_trading_start=before_trading_start,
            sim_params=sim_params,
        )

        assert len(bts_times) == num_sessions
        expected_times = [
            pd.Timestamp("2016-03-11 8:45", tz="US/Eastern").tz_convert("UTC"),
            pd.Timestamp("2016-03-14 8:45", tz="US/Eastern").tz_convert("UTC"),
            pd.Timestamp("2016-03-15 8:45", tz="US/Eastern").tz_convert("UTC"),
        ]
        assert bts_times == expected_times[:num_sessions]


class BeforeTradingStartsOnlyClock:
    def __init__(self, bts_minute):
        self.bts_minute = bts_minute

    def __iter__(self):
        yield self.bts_minute, BEFORE_TRADING_START_BAR


class TestBeforeTradingStartSimulationDt(zf.WithMakeAlgo, zf.ZiplineTestCase):

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_bts_simulation_dt(self):
        code = """
def initialize(context):
    pass
"""
        algo = self.make_algo(script=code, metrics=metrics.load("none"))
        algo.metrics_tracker = algo._create_metrics_tracker()
        benchmark_source = algo._create_benchmark_source()
        algo.metrics_tracker.handle_start_of_simulation(benchmark_source)

        dt = pd.Timestamp("2016-08-04 9:13:14", tz="US/Eastern")
        algo_simulator = AlgorithmSimulator(
            algo,
            self.sim_params,
            self.data_portal,
            BeforeTradingStartsOnlyClock(dt),
            benchmark_source,
            NoRestrictions(),
        )

        # run through the algo's simulation
        list(algo_simulator.transform())

        # since the clock only ever emitted a single before_trading_start
        # event, we can check that the simulation_dt was properly set
        assert dt == algo_simulator.simulation_dt
