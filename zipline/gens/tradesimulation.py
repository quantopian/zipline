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
import math
import numpy as np
import pandas as pd

from logbook import Logger, Processor
from pandas.tslib import normalize_date

from zipline.protocol import BarData

from zipline.gens.sim_engine import DayEngine

from zipline.utils.api_support import ZiplineAPI

log = Logger('Trade Simulation')


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algo, sim_params, data_portal):

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
        # FIXME
        # benchmark_iter = None
        # if hasattr(self.algo, "benchmark_iter"):
        #     benchmark_iter = iter(self.algo.benchmark_iter)

        self.current_data = BarData(data_portal=self.data_portal)

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None

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
        sim_params = algo.sim_params
        trading_days = sim_params.trading_days
        env = self.env
        trading_o_and_c = env.open_and_closes.ix[trading_days]
        market_opens = trading_o_and_c['market_open'].values.astype(
            'datetime64[ns]').astype(np.int64)
        market_closes = trading_o_and_c['market_close'].values.astype(
            'datetime64[ns]').astype(np.int64)
        handle_data = algo.event_manager.handle_data
        current_data = self.current_data

        perf_tracker = self.algo.perf_tracker
        perf_tracker_benchmark_returns = perf_tracker.all_benchmark_returns
        data_portal = self.data_portal

        day_engine = DayEngine(market_opens, market_closes)

        blotter = self.algo.blotter
        blotter.slippage_func.data_portal = data_portal

        perf_process_order = self.algo.perf_tracker.process_order
        perf_process_txn = self.algo.perf_tracker.process_transaction
        perf_tracker.position_tracker.data_portal = data_portal

        all_trading_days = self.env.trading_days
        all_trading_days = all_trading_days[all_trading_days.slice_indexer(
            '2002-01-02')]
        first_trading_day_idx = all_trading_days.searchsorted(trading_days[0])

        def inner_loop(dt_to_use):
            self.on_dt_changed(dt_to_use)

            new_transactions = blotter.process_open_orders(dt_to_use)
            for transaction in new_transactions:
                perf_process_txn(transaction)

            # update the portfolio, so that if the user does
            # context.portfolio.positions, it's accurate
            perf_tracker.get_portfolio(True)

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

        with self.processor.threadbound(), ZiplineAPI(self.algo):
            if self.sim_params.data_frequency == "daily":
                for day_idx, trading_day in enumerate(trading_days):
                    data_portal.current_dt = trading_day
                    data_portal.current_day = trading_day

                    inner_loop(trading_day)

                    # Update benchmark before getting market close.
                    perf_tracker_benchmark_returns[trading_day] = 0.001
                    # use the last dt as market close
                    yield self.get_message(trading_day)

                    algo.portfolio_needs_update = True
                    algo.account_needs_update = True
                    algo.performance_needs_update = True
            else:
                for day_idx, trading_day in enumerate(trading_days):
                    day_offset = (day_idx + first_trading_day_idx) * 390
                    minutes = pd.DatetimeIndex(day_engine.
                                               market_minutes(day_idx),
                                               tz='UTC')
                    for minute_idx, minute in enumerate(minutes):
                        data_portal.current_dt = minute.value
                        blotter.current_dt = minute
                        data_portal.current_day = trading_day
                        data_portal.cur_data_offset = day_offset + minute_idx

                        inner_loop(minute)

                    # Update benchmark before getting market close.
                    perf_tracker_benchmark_returns[trading_day] = 0.001
                    # use the last dt as market close
                    yield self.get_message(minute)

                    algo.portfolio_needs_update = True
                    algo.account_needs_update = True
                    algo.performance_needs_update = True

        risk_message = self.algo.perf_tracker.handle_simulation_end()
        yield risk_message

    # FIXME nobody calls this
    def _call_before_trading_start(self, dt):
        dt = normalize_date(dt)
        self.simulation_dt = dt
        self.on_dt_changed(dt)
        self.algo.before_trading_start()

    def on_dt_changed(self, dt):
        if self.algo.datetime != dt:
            self.algo.on_dt_changed(dt)

    def get_message(self, dt):
        """
        Get a perf message for the given datetime.
        """
        # Ensure that updated_portfolio has been called at least once for this
        # dt before we emit a perf message.  This is a no-op if
        # updated_portfolio has already been called this dt.
        self.algo.updated_portfolio()
        self.algo.updated_account()

        rvars = self.algo.recorded_vars
        if self.algo.perf_tracker.emission_rate == 'daily':
            perf_message = \
                self.algo.perf_tracker.handle_market_close_daily()
            perf_message['daily_perf']['recorded_vars'] = rvars
            return perf_message

        elif self.algo.perf_tracker.emission_rate == 'minute':
            self.algo.perf_tracker.handle_minute_close(dt)
            perf_message = self.algo.perf_tracker.to_dict()
            perf_message['minute_perf']['recorded_vars'] = rvars
            return perf_message

    def generate_messages(self, dt):
        """
        Generator that yields perf messages for the given datetime.
        """
        # Ensure that updated_portfolio has been called at least once for this
        # dt before we emit a perf message.  This is a no-op if
        # updated_portfolio has already been called this dt.
        self.algo.updated_portfolio()
        self.algo.updated_account()

        rvars = self.algo.recorded_vars
        if self.algo.perf_tracker.emission_rate == 'daily':
            perf_message = \
                self.algo.perf_tracker.handle_market_close_daily()
            perf_message['daily_perf']['recorded_vars'] = rvars
            yield perf_message

        elif self.algo.perf_tracker.emission_rate == 'minute':
            # close the minute in the tracker, and collect the daily message if
            # the minute is the close of the trading day
            minute_message, daily_message = \
                self.algo.perf_tracker.handle_minute_close(dt)

            # collect and yield the minute's perf message
            minute_message['minute_perf']['recorded_vars'] = rvars
            yield minute_message

            # if there was a daily perf message, collect and yield it
            if daily_message:
                daily_message['daily_perf']['recorded_vars'] = rvars
                yield daily_message
