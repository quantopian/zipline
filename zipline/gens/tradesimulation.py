#
# Copyright 2015 Quantopian, Inc.
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
from logbook import Logger, Processor
from pandas.tslib import normalize_date
from zipline.protocol import BarData
from zipline.utils.api_support import ZiplineAPI

from zipline.gens.sim_engine import (
    DATA_AVAILABLE,
    ONCE_A_DAY,
    CALC_DAILY_PERFORMANCE,
    CALC_MINUTE_PERFORMANCE
)


log = Logger('Trade Simulation')


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algo, sim_params, data_portal, clock, benchmark_source):

        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params
        self.env = algo.trading_environment
        self.data_portal = data_portal

        # ==============
        # Algo Setup
        # ==============
        self.algo = algo
        self.algo_start = normalize_date(self.sim_params.first_open)

        # ==============
        # Snapshot Setup
        # ==============

        # The algorithm's data as of our most recent event.
        # We want an object that will have empty objects as default
        # values on missing keys.
        self.current_data = BarData(data_portal=self.data_portal)

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None

        self.clock = clock

        self.benchmark_source = benchmark_source

        # =============
        # Logging Setup
        # =============

        # Processor function for injecting the algo_dt into
        # user prints/logs.
        def inject_algo_dt(record):
            if 'algo_dt' not in record.extra:
                record.extra['algo_dt'] = self.simulation_dt
        self.processor = Processor(inject_algo_dt)

    def transform(self):
        """
        Main generator work loop.
        """
        algo = self.algo
        algo.data_portal = self.data_portal
        handle_data = algo.event_manager.handle_data
        current_data = self.current_data

        data_portal = self.data_portal

        blotter = self.algo.blotter
        blotter.data_portal = data_portal

        # can't cache a pointer to algo.perf_tracker because we're not
        # guaranteed that the algo doesn't swap out perf trackers during
        # its lifetime.
        perf_process_order = self.algo.perf_tracker.process_order
        perf_process_txn = self.algo.perf_tracker.process_transaction
        algo.perf_tracker.position_tracker.data_portal = data_portal

        def inner_loop(dt_to_use):
            # called every tick (minute or day).

            data_portal.current_dt = dt_to_use
            self.simulation_dt = dt_to_use
            algo.on_dt_changed(dt_to_use)

            new_transactions = blotter.process_open_orders(dt_to_use)
            for transaction in new_transactions:
                perf_process_txn(transaction)

                # since this order was modified, record it
                order = blotter.orders[transaction.order_id]
                perf_process_order(order)

            handle_data(algo, current_data, dt_to_use)

            # grab any new orders from the blotter, then clear the list.
            # this includes cancelled orders.
            new_orders = blotter.new_orders
            blotter.new_orders = []

            # if we have any new orders, record them so that we know
            # in what perf period they were placed.
            if new_orders:
                for new_order in new_orders:
                    perf_process_order(new_order)

        def once_a_day(midnight_dt):
            # set all the timestamps
            self.simulation_dt = midnight_dt
            algo.on_dt_changed(midnight_dt)
            data_portal.current_day = midnight_dt

            # call before trading start
            algo.before_trading_start(current_data)

            perf_tracker = algo.perf_tracker

            # handle any splits that impact any positions or any open orders.
            sids_we_care_about = \
                list(set(list(perf_tracker.position_tracker.positions.keys()) +
                         list(blotter.open_orders.keys())))

            if len(sids_we_care_about) > 0:
                splits = data_portal.get_splits(sids_we_care_about,
                                                midnight_dt)
                if len(splits) > 0:
                    blotter.process_splits(splits)
                    perf_tracker.position_tracker.handle_splits(splits)

        with self.processor, ZiplineAPI(self.algo):
            for dt, action in self.clock:
                if action == DATA_AVAILABLE:
                    inner_loop(dt)
                elif action == ONCE_A_DAY:
                    once_a_day(dt)
                elif action == CALC_DAILY_PERFORMANCE:
                    algo.perf_tracker.all_benchmark_returns[dt] = \
                        self.benchmark_source.get_value(dt)
                    yield self._get_daily_message(dt, algo, algo.perf_tracker)
                elif action == CALC_MINUTE_PERFORMANCE:
                    algo.perf_tracker.all_benchmark_returns[dt] = \
                        self.benchmark_source.get_value(dt)
                    minute_msg, daily_msg = \
                        self._get_minute_message(dt, algo, algo.perf_tracker)

                    yield minute_msg

                    if daily_msg:
                        yield daily_msg

        risk_message = algo.perf_tracker.handle_simulation_end()
        yield risk_message

    @staticmethod
    def _get_daily_message(dt, algo, perf_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = perf_tracker.handle_market_close_daily(dt)
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message

    @staticmethod
    def _get_minute_message(dt, algo, perf_tracker):
        """
        Get a perf message for the given datetime.
        """
        rvars = algo.recorded_vars

        minute_message, daily_message = perf_tracker.handle_minute_close(dt)
        minute_message['minute_perf']['recorded_vars'] = rvars

        if daily_message:
            daily_message["daily_perf"]["recorded_vars"] = rvars

        return minute_message, daily_message


