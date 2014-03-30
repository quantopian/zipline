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
from logbook import Logger, Processor

from zipline.finance import trading
from zipline.protocol import (
    BarData,
    SIDData,
    DATASOURCE_TYPE
)
from zipline.gens.utils import hash_args

log = Logger('Trade Simulation')


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def get_hash(self):
        """
        There should only ever be one TSC in the system, so
        we don't bother passing args into the hash.
        """
        return self.__class__.__name__ + hash_args()

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
        self.algo_start = self.sim_params.first_open
        self.algo_start = self.algo_start.replace(hour=0, minute=0,
                                                  second=0,
                                                  microsecond=0)

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

    @property
    def perf_key(self):
        return self.EMISSION_TO_PERF_KEY_MAP[
            self.algo.perf_tracker.emission_rate]

    def process_event(self, event):
        process_trade = self.algo.blotter.process_trade
        for txn, order in process_trade(event):
            self.algo.perf_tracker.process_event(txn)
            self.algo.perf_tracker.process_event(order)
        self.algo.perf_tracker.process_event(event)

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        # Initialize the mkt_close
        mkt_open = self.algo.perf_tracker.market_open
        mkt_close = self.algo.perf_tracker.market_close

        # inject the current algo
        # snapshot time to any log record generated.
        with self.processor.threadbound():
            updated = False
            bm_updated = False
            for date, snapshot in stream_in:
                self.algo.set_datetime(date)
                self.simulation_dt = date
                self.algo.perf_tracker.set_date(date)
                self.algo.blotter.set_date(date)
                # If we're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                if date < self.algo_start:
                    for event in snapshot:
                        if event.type == DATASOURCE_TYPE.SPLIT:
                            self.algo.blotter.process_split(event)

                        if event.type in (DATASOURCE_TYPE.TRADE,
                                          DATASOURCE_TYPE.CUSTOM):
                            self.update_universe(event)
                        self.algo.perf_tracker.process_event(event)

                else:
                    if self.algo.instant_fill:
                        events = []

                    for event in snapshot:
                        if event.type == DATASOURCE_TYPE.TRADE:
                            self.update_universe(event)
                            updated = True

                        elif event.type == DATASOURCE_TYPE.BENCHMARK:
                            self.algo.set_datetime(event.dt)
                            bm_updated = True

                        elif event.type == DATASOURCE_TYPE.CUSTOM:
                            self.update_universe(event)

                        elif event.type == DATASOURCE_TYPE.SPLIT:
                            self.algo.blotter.process_split(event)

                        # If we are instantly filling orders we process
                        # them after handle_data().
                        if not self.algo.instant_fill:
                            self.process_event(event)
                        else:
                            events.append(event)

                    # Send the current state of the universe
                    # to the user's algo.
                    if updated:
                        self.algo.handle_data(self.current_data)
                        updated = False

                    # run orders placed in the algorithm call
                    # above through perf tracker before emitting
                    # the perf packet, so that the perf includes
                    # placed orders
                    for order in self.algo.blotter.new_orders:
                        self.algo.perf_tracker.process_event(order)
                    self.algo.blotter.new_orders = []

                    # If we are instantly filling we execute orders
                    # in this iteration rather than the next.
                    if self.algo.instant_fill:
                        for event in events:
                            self.process_event(event)

                    # The benchmark is our internal clock. When it
                    # updates, we need to emit a performance message.
                    if bm_updated:
                        bm_updated = False
                        self.algo.updated_portfolio()
                        yield self.get_message(date)

                    # When emitting minutely, we re-iterate the day as a
                    # packet with the entire days performance rolled up.
                    if self.algo.perf_tracker.emission_rate == 'minute':
                        if date == mkt_close:
                            daily_rollup = self.algo.perf_tracker.to_dict(
                                emission_type='daily'
                            )
                            daily_rollup['daily_perf']['recorded_vars'] = \
                                self.algo.recorded_vars
                            yield daily_rollup
                            tp = self.algo.perf_tracker.todays_performance
                            tp.rollover()
                            if mkt_close <= self.algo.perf_tracker.last_close:
                                try:
                                    mkt_open, mkt_close = \
                                        trading.environment.\
                                        next_open_and_close(
                                            mkt_close
                                        )
                                except trading.NoFurtherDataError:
                                    # If at the end of backtest history,
                                    # skip advancing market close.
                                    pass
                                self.algo.perf_tracker.handle_intraday_close(
                                    mkt_open, mkt_close)

                    self.algo.portfolio_needs_update = True

            risk_message = self.algo.perf_tracker.handle_simulation_end()
            yield risk_message

    def get_message(self, date):
        rvars = self.algo.recorded_vars
        if self.algo.perf_tracker.emission_rate == 'daily':
            perf_message = \
                self.algo.perf_tracker.handle_market_close()
            perf_message['daily_perf']['recorded_vars'] = rvars
            return perf_message

        elif self.algo.perf_tracker.emission_rate == 'minute':
            self.algo.perf_tracker.handle_minute_close(date)
            perf_message = self.algo.perf_tracker.to_dict()
            perf_message['minute_perf']['recorded_vars'] = rvars
            return perf_message

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
            sid_data = self.current_data[event.sid] = SIDData()
        sid_data.__dict__.update(event.__dict__)
