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
import numpy as np
import pandas as pd

from logbook import Logger, Processor
from pandas.tslib import normalize_date
from zipline.errors import (
    BenchmarkAssetNotAvailableTooEarly,
    BenchmarkAssetNotAvailableTooLate,
    InvalidBenchmarkAsset)

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
        blotter.data_portal = data_portal

        perf_process_order = self.algo.perf_tracker.process_order
        perf_process_txn = self.algo.perf_tracker.process_transaction
        perf_tracker.position_tracker.data_portal = data_portal

        all_trading_days = self.env.trading_days
        all_trading_days = all_trading_days[all_trading_days.slice_indexer(
            '2002-01-02')]
        first_trading_day_idx = all_trading_days.searchsorted(trading_days[0])

        benchmark_series = self._prepare_benchmark_series(
            algo.benchmark_sid, env, trading_days, data_portal
        )

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
            if self.sim_params.data_frequency == "daily":
                for day_idx, trading_day in enumerate(trading_days):
                    once_a_day(trading_day)
                    inner_loop(trading_day)

                    # Update benchmark before getting market close.
                    perf_tracker_benchmark_returns[trading_day] = \
                        benchmark_series.loc[trading_day]

                    yield self.get_message(trading_day, algo, perf_tracker)
            else:
                for day_idx, trading_day in enumerate(trading_days):
                    once_a_day(trading_day)

                    day_offset = (day_idx + first_trading_day_idx) * 390
                    minutes = pd.DatetimeIndex(day_engine.
                                               market_minutes(day_idx),
                                               tz='UTC')
                    for minute_idx, minute in enumerate(minutes):
                        data_portal.cur_data_offset = day_offset + minute_idx
                        inner_loop(minute)

                    # Update benchmark before getting market close.
                    perf_tracker_benchmark_returns[trading_day] = \
                        benchmark_series.loc[trading_day]

                    yield self.get_message(minute, algo, perf_tracker)

        risk_message = perf_tracker.handle_simulation_end()
        yield risk_message

    def get_message(self, dt, algo, perf_tracker):
        """
        Get a perf message for the given datetime.
        """
        rvars = algo.recorded_vars
        if perf_tracker.emission_rate == 'daily':
            perf_message = \
                perf_tracker.handle_market_close_daily()
            perf_message['daily_perf']['recorded_vars'] = rvars
            return perf_message

        elif perf_tracker.emission_rate == 'minute':
            perf_tracker.handle_minute_close(dt)
            perf_message = perf_tracker.to_dict()
            perf_message['minute_perf']['recorded_vars'] = rvars
            return perf_message

    @staticmethod
    def _prepare_benchmark_series(sid, env, trading_days, data_portal):
        """
        Internal method that precalculates the benchmark return series for
        use in the simulation.

        Parameters
        ----------
        algo: TradingAlgorithm

        env: TradingEnvironment

        trading_days: pd.DateTimeIndex

        data_portal: DataPortal

        Notes
        -----
        If the benchmark asset started trading after the simulation start,
        or finished trading before the simulation end, exceptions are raised.

        If the benchmark asset started trading the same day as the simulation
        start, the first available minute price on that day is used instead
        of the previous close.

        We use history to get an adjusted price history for each day's close,
        as of the look-back date (the last day of the simulation).  Prices are
        fully adjusted for dividends, splits, and mergers.

        Returns
        -------
        A pd.Series, indexed by trading day, whose values represent the %
        change from close to close.
        """
        if sid is None:
            # get benchmark info from trading environment
            return env.benchmark_returns[trading_days[0]:trading_days[-1]]
        else:
            # check if this security has a stock dividend.  if so, raise an
            # error suggesting that the user pick a different asset to use
            # as benchmark.
            stock_dividends = \
                data_portal.get_stock_dividends(sid, trading_days)

            if len(stock_dividends) > 0:
                raise InvalidBenchmarkAsset(
                    sid=str(sid),
                    dt=stock_dividends[0]["ex_date"]
                )

            benchmark_asset = env.asset_finder.retrieve_asset(sid)
            if benchmark_asset.start_date > trading_days[0]:
                # the asset started trading after the first simulation day
                raise BenchmarkAssetNotAvailableTooEarly(
                    sid=str(sid),
                    dt=trading_days[0],
                    start_dt=benchmark_asset.start_date
                )

            if benchmark_asset.end_date < trading_days[-1]:
                # the asset stopped trading before the last simulation day
                raise BenchmarkAssetNotAvailableTooLate(
                    sid=str(sid),
                    dt=trading_days[0],
                    end_dt=benchmark_asset.end_date
                )

            # get the window of close prices for benchmark_sid from the last
            # trading day of the simulation, going up to one day before the
            # simulation start day (so that we can get the % change on day 1)
            benchmark_series = data_portal.get_history_window(
                [sid],
                trading_days[-1],
                bar_count=len(trading_days) + 1,
                frequency="1d",
                field="close"
            )[sid]

            # now, we need to check if we can safely go use the
            # one-day-before-sim-start value, by seeing if the asset was
            # trading that day.
            trading_day_before_sim_start = \
                env.previous_trading_day(trading_days[0])

            if benchmark_asset.start_date > trading_day_before_sim_start:
                # we can't go back one day before sim start, because the asset
                # didn't start trading until the same day as the sim start.
                # instead, we'll use the first available minute value of the
                # first sim day.
                minutes_in_first_day = \
                    env.market_minutes_for_day(trading_days[0])

                # get a minute history window of the first day
                minute_window = data_portal.get_history_window(
                    [sid],
                    minutes_in_first_day[-1],
                    bar_count=len(minutes_in_first_day),
                    frequency="1m",
                    field="close_price"
                )[sid]

                # find the first non-zero value
                value_to_use = minute_window[minute_window != 0][0]
                benchmark_series[0] = value_to_use

            return benchmark_series.pct_change()[1:]
