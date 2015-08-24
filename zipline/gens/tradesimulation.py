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

from contextlib2 import ExitStack

from logbook import Logger, Processor
from pandas.tslib import normalize_date

from zipline.utils.api_support import ZiplineAPI

from zipline.finance.trading import NoFurtherDataError
from zipline.protocol import (
    BarData,
    SIDData,
    DATASOURCE_TYPE
)

log = Logger('Trade Simulation')


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algo, sim_params):

        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params

        # ==============
        # Algo Setup
        # ==============
        self.algo = algo
        self.algo_start = normalize_date(self.sim_params.first_open)
        self.env = algo.trading_environment

        # ==============
        # Snapshot Setup
        # ==============

        # The algorithm's data as of our most recent event.
        # We want an object that will have empty objects as default
        # values on missing keys.
        self.current_data = BarData()

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

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        # Initialize the mkt_close
        mkt_open = self.algo.perf_tracker.market_open
        mkt_close = self.algo.perf_tracker.market_close

        # inject the current algo
        # snapshot time to any log record generated.

        with ExitStack() as stack:
            stack.enter_context(self.processor)
            stack.enter_context(ZiplineAPI(self.algo))

            data_frequency = self.sim_params.data_frequency

            self._call_before_trading_start(mkt_open)

            for date, snapshot in stream_in:

                self.simulation_dt = date
                self.on_dt_changed(date)

                # If we're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                if date < self.algo_start:
                    for event in snapshot:
                        if event.type == DATASOURCE_TYPE.SPLIT:
                            self.algo.blotter.process_split(event)

                        elif event.type == DATASOURCE_TYPE.TRADE:
                            self.update_universe(event)
                            self.algo.perf_tracker.process_trade(event)
                        elif event.type == DATASOURCE_TYPE.CUSTOM:
                            self.update_universe(event)

                else:
                    messages = self._process_snapshot(
                        date,
                        snapshot,
                        self.algo.instant_fill,
                    )
                    # Perf messages are only emitted if the snapshot contained
                    # a benchmark event.
                    for message in messages:
                        yield message

                    # When emitting minutely, we need to call
                    # before_trading_start before the next trading day begins
                    if date == mkt_close:
                        if mkt_close <= self.algo.perf_tracker.last_close:
                            before_last_close = \
                                mkt_close < self.algo.perf_tracker.last_close
                            try:
                                mkt_open, mkt_close = \
                                    self.env.next_open_and_close(mkt_close)

                            except NoFurtherDataError:
                                # If at the end of backtest history,
                                # skip advancing market close.
                                pass

                            if before_last_close:
                                self._call_before_trading_start(mkt_open)

                    elif data_frequency == 'daily':
                        next_day = self.env.next_trading_day(date)

                        if next_day is not None and \
                           next_day < self.algo.perf_tracker.last_close:
                            self._call_before_trading_start(next_day)

                    self.algo.portfolio_needs_update = True
                    self.algo.account_needs_update = True
                    self.algo.performance_needs_update = True

            risk_message = self.algo.perf_tracker.handle_simulation_end()
            yield risk_message

    def _process_snapshot(self, dt, snapshot, instant_fill):
        """
        Process a stream of events corresponding to a single datetime, possibly
        returning a perf message to be yielded.

        If @instant_fill = True, we delay processing of events until after the
        user's call to handle_data, and we process the user's placed orders
        before the snapshot's events.  Note that this introduces a lookahead
        bias, since the user effectively is effectively placing orders that are
        filled based on trades that happened prior to the call the handle_data.

        If @instant_fill = False, we process Trade events before calling
        handle_data.  This means that orders are filled based on trades
        occurring in the next snapshot.  This is the more conservative model,
        and as such it is the default behavior in TradingAlgorithm.
        """

        # Flags indicating whether we saw any events of type TRADE and type
        # BENCHMARK.  Respectively, these control whether or not handle_data is
        # called for this snapshot and whether we emit a perf message for this
        # snapshot.
        any_trade_occurred = False
        benchmark_event_occurred = False

        if instant_fill:
            events_to_be_processed = []

        # Assign process events to variables to avoid attribute access in
        # innermost loops.
        #
        # Done here, to allow for perf_tracker or blotter to be swapped out
        # or changed in between snapshots.
        perf_process_trade = self.algo.perf_tracker.process_trade
        perf_process_transaction = self.algo.perf_tracker.process_transaction
        perf_process_order = self.algo.perf_tracker.process_order
        perf_process_benchmark = self.algo.perf_tracker.process_benchmark
        perf_process_split = self.algo.perf_tracker.process_split
        perf_process_dividend = self.algo.perf_tracker.process_dividend
        perf_process_commission = self.algo.perf_tracker.process_commission
        perf_process_close_position = \
            self.algo.perf_tracker.process_close_position
        blotter_process_trade = self.algo.blotter.process_trade
        blotter_process_benchmark = self.algo.blotter.process_benchmark

        # Containers for the snapshotted events, so that the events are
        # processed in a predictable order, without relying on the sorted order
        # of the individual sources.

        # There is only one benchmark per snapshot, will be set to the current
        # benchmark iff it occurs.
        benchmark = None
        # trades and customs are initialized as a list since process_snapshot
        # is most often called on market bars, which could contain trades or
        # custom events.
        trades = []
        customs = []
        closes = []

        # splits and dividends are processed once a day.
        #
        # The avoidance of creating the list every time this is called is more
        # to attempt to show that this is the infrequent case of the method,
        # since the performance benefit from deferring the list allocation is
        # marginal.  splits list will be allocated when a split occurs in the
        # snapshot.
        splits = None
        # dividends list will be allocated when a dividend occurs in the
        # snapshot.
        dividends = None

        for event in snapshot:
            if event.type == DATASOURCE_TYPE.TRADE:
                trades.append(event)
            elif event.type == DATASOURCE_TYPE.BENCHMARK:
                benchmark = event
            elif event.type == DATASOURCE_TYPE.SPLIT:
                if splits is None:
                    splits = []
                splits.append(event)
            elif event.type == DATASOURCE_TYPE.CUSTOM:
                customs.append(event)
            elif event.type == DATASOURCE_TYPE.DIVIDEND:
                if dividends is None:
                    dividends = []
                dividends.append(event)
            elif event.type == DATASOURCE_TYPE.CLOSE_POSITION:
                closes.append(event)
            else:
                raise log.warn("Unrecognized event=%s".format(event))

        # Handle benchmark first.
        #
        # Internal broker implementation depends on the benchmark being
        # processed first so that transactions and commissions reported from
        # the broker can be injected.
        if benchmark is not None:
            benchmark_event_occurred = True
            perf_process_benchmark(benchmark)
            for txn, order in blotter_process_benchmark(benchmark):
                if txn.type == DATASOURCE_TYPE.TRANSACTION:
                    perf_process_transaction(txn)
                elif txn.type == DATASOURCE_TYPE.COMMISSION:
                    perf_process_commission(txn)
                perf_process_order(order)

        for trade in trades:
            self.update_universe(trade)
            any_trade_occurred = True
            if instant_fill:
                events_to_be_processed.append(trade)
            else:
                for txn, order in blotter_process_trade(trade):
                    if txn.type == DATASOURCE_TYPE.TRANSACTION:
                        perf_process_transaction(txn)
                    elif txn.type == DATASOURCE_TYPE.COMMISSION:
                        perf_process_commission(txn)
                    perf_process_order(order)
                perf_process_trade(trade)

        for custom in customs:
            self.update_universe(custom)

        for close in closes:
            self.update_universe(close)
            perf_process_close_position(close)

        if splits is not None:
            for split in splits:
                # process_split is not assigned to a variable since it is
                # called rarely compared to the other event processors.
                self.algo.blotter.process_split(split)
                perf_process_split(split)

        if dividends is not None:
            for dividend in dividends:
                perf_process_dividend(dividend)

        if any_trade_occurred:
            new_orders = self._call_handle_data()
            for order in new_orders:
                perf_process_order(order)

        if instant_fill:
            # Now that handle_data has been called and orders have been placed,
            # process the event stream to fill user orders based on the events
            # from this snapshot.
            for trade in events_to_be_processed:
                for txn, order in blotter_process_trade(trade):
                    if txn is not None:
                        perf_process_transaction(txn)
                    if order is not None:
                        perf_process_order(order)
                perf_process_trade(trade)

        if benchmark_event_occurred:
            return self.generate_messages(dt)
        else:
            return ()

    def _call_handle_data(self):
        """
        Call the user's handle_data, returning any orders placed by the algo
        during the call.
        """
        self.algo.event_manager.handle_data(
            self.algo,
            self.current_data,
            self.simulation_dt,
        )
        orders = self.algo.blotter.new_orders
        self.algo.blotter.new_orders = []
        return orders

    def _call_before_trading_start(self, dt):
        dt = normalize_date(dt)
        self.simulation_dt = dt
        self.on_dt_changed(dt)
        self.algo.before_trading_start(self.current_data)

    def on_dt_changed(self, dt):
        if self.algo.datetime != dt:
            self.algo.on_dt_changed(dt)

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

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
        # Update our knowledge of this event's sid
        # rather than use if event.sid in ..., just trying
        # and handling the exception is significantly faster
        try:
            sid_data = self.current_data[event.sid]
        except KeyError:
            sid_data = self.current_data[event.sid] = SIDData(event.sid)

        sid_data.__dict__.update(event.__dict__)
