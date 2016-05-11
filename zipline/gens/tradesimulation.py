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
from contextlib2 import ExitStack
from logbook import Logger, Processor
from pandas.tslib import normalize_date
from zipline.protocol import BarData
from zipline.utils.api_support import ZiplineAPI
from six import viewkeys

from zipline.gens.sim_engine import (
    BAR,
    DAY_START,
    DAY_END,
    MINUTE_END
)

log = Logger('Trade Simulation')


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
                 universe_func):

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

        # This object is the way that user algorithms interact with OHLCV data,
        # fetcher data, and some API methods like `data.can_trade`.
        self.current_data = self._create_bar_data(universe_func)

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None
        self.previous_dt = self.algo_start

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

    def get_simulation_dt(self):
        return self.simulation_dt

    def _create_bar_data(self, universe_func):
        return BarData(
            data_portal=self.data_portal,
            simulation_dt_func=self.get_simulation_dt,
            data_frequency=self.sim_params.data_frequency,
            universe_func=universe_func
        )

    def transform(self):
        """
        Main generator work loop.
        """
        algo = self.algo

        def every_bar(dt_to_use, current_data=self.current_data,
                      handle_data=algo.event_manager.handle_data):
            # called every tick (minute or day).

            self.simulation_dt = dt_to_use
            algo.on_dt_changed(dt_to_use)

            blotter = algo.blotter
            perf_tracker = algo.perf_tracker

            # handle any transactions and commissions coming out new orders
            # placed in the last bar
            new_transactions, new_commissions, closed_orders = \
                blotter.get_transactions(current_data)

            blotter.prune_orders(closed_orders)

            for transaction in new_transactions:
                perf_tracker.process_transaction(transaction)

                # since this order was modified, record it
                order = blotter.orders[transaction.order_id]
                perf_tracker.process_order(order)

            if new_commissions:
                for commission in new_commissions:
                    perf_tracker.process_commission(commission)

            handle_data(algo, current_data, dt_to_use)

            # grab any new orders from the blotter, then clear the list.
            # this includes cancelled orders.
            new_orders = blotter.new_orders
            blotter.new_orders = []

            # if we have any new orders, record them so that we know
            # in what perf period they were placed.
            if new_orders:
                for new_order in new_orders:
                    perf_tracker.process_order(new_order)

            self.algo.portfolio_needs_update = True
            self.algo.account_needs_update = True
            self.algo.performance_needs_update = True

        def once_a_day(midnight_dt, current_data=self.current_data,
                       data_portal=self.data_portal):
            # Get the positions before updating the date so that prices are
            # fetched for trading close instead of midnight
            positions = algo.perf_tracker.position_tracker.positions
            position_assets = algo.asset_finder.retrieve_all(positions)

            # set all the timestamps
            self.simulation_dt = midnight_dt
            algo.on_dt_changed(midnight_dt)

            # we want to wait until the clock rolls over to the next day
            # before cleaning up expired assets.
            self._cleanup_expired_assets(midnight_dt, position_assets)

            perf_tracker = algo.perf_tracker

            # handle any splits that impact any positions or any open orders.
            assets_we_care_about = \
                viewkeys(perf_tracker.position_tracker.positions) | \
                viewkeys(algo.blotter.open_orders)

            if assets_we_care_about:
                splits = data_portal.get_splits(assets_we_care_about,
                                                midnight_dt)
                if splits:
                    algo.blotter.process_splits(splits)
                    perf_tracker.position_tracker.handle_splits(splits)

            # call before trading start
            algo.before_trading_start(current_data)

        def handle_benchmark(date, benchmark_source=self.benchmark_source):
            algo.perf_tracker.all_benchmark_returns[date] = \
                benchmark_source.get_value(date)

        def on_exit():
            self.benchmark_source = self.current_data = self.data_portal = None

        with ExitStack() as stack:
            stack.callback(on_exit)
            stack.enter_context(self.processor)
            stack.enter_context(ZiplineAPI(self.algo))

            if algo.data_frequency == 'minute':
                def execute_order_cancellation_policy():
                    algo.blotter.execute_cancel_policy(DAY_END)
            else:
                def execute_order_cancellation_policy():
                    pass

            for dt, action in self.clock:
                if action == BAR:
                    every_bar(dt)
                elif action == DAY_START:
                    once_a_day(dt)
                elif action == DAY_END:
                    # End of the day.
                    if algo.perf_tracker.emission_rate == 'daily':
                        handle_benchmark(normalize_date(dt))
                    execute_order_cancellation_policy()

                    yield self._get_daily_message(dt, algo, algo.perf_tracker)
                elif action == MINUTE_END:
                    handle_benchmark(dt)
                    minute_msg = \
                        self._get_minute_message(dt, algo, algo.perf_tracker)

                    yield minute_msg

        risk_message = algo.perf_tracker.handle_simulation_end()
        yield risk_message

    def _cleanup_expired_assets(self, dt, position_assets):
        """
        Clear out any assets that have expired before starting a new sim day.

        Performs two functions:

        1. Finds all assets for which we have open orders and clears any
           orders whose assets are on or after their auto_close_date.

        2. Finds all assets for which we have positions and generates
           close_position events for any assets that have reached their
           auto_close_date.
        """
        algo = self.algo

        def past_auto_close_date(asset):
            acd = asset.auto_close_date
            return acd is not None and acd <= dt

        # Remove positions in any sids that have reached their auto_close date.
        assets_to_clear = \
            [asset for asset in position_assets if past_auto_close_date(asset)]
        perf_tracker = algo.perf_tracker
        data_portal = self.data_portal
        for asset in assets_to_clear:
            perf_tracker.process_close_position(asset, dt, data_portal)

        # Remove open orders for any sids that have reached their
        # auto_close_date.
        blotter = algo.blotter
        assets_to_cancel = \
            set([asset for asset in blotter.open_orders
                 if past_auto_close_date(asset)])
        for asset in assets_to_cancel:
            blotter.cancel_all_orders_for_asset(asset)

    def _get_daily_message(self, dt, algo, perf_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = perf_tracker.handle_market_close(
            dt, self.data_portal,
        )
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message

    def _get_minute_message(self, dt, algo, perf_tracker):
        """
        Get a perf message for the given datetime.
        """
        rvars = algo.recorded_vars

        minute_message = perf_tracker.handle_minute_close(
            dt, self.data_portal,
        )

        minute_message['minute_perf']['recorded_vars'] = rvars
        return minute_message
