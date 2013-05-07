#
# Copyright 2013 Quantopian, Inc.
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
import itertools
from itertools import chain
from logbook import Logger, Processor

import zipline.finance.trading as trading
from zipline.protocol import BarData, DATASOURCE_TYPE
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
        self.snapshot_dt = None

        # =============
        # Logging Setup
        # =============

        # Processor function for injecting the algo_dt into
        # user prints/logs.
        def inject_algo_dt(record):
            if not 'algo_dt' in record.extra:
                record.extra['algo_dt'] = self.snapshot_dt
        self.processor = Processor(inject_algo_dt)

    @property
    def perf_key(self):
        return self.EMISSION_TO_PERF_KEY_MAP[
            self.algo.perf_tracker.emission_rate]

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        # Initialize the mkt_close
        mkt_close = self.algo.perf_tracker.market_close
        # Set the simulation date to be the first event we see.
        peek_date, peek_snapshot = next(stream_in)
        self.simulation_dt = peek_date

        # Stitch back together the generator by placing the peeked
        # event back in front
        stream = itertools.chain([(peek_date, peek_snapshot)],
                                 stream_in)

        # inject the current algo
        # snapshot time to any log record generated.
        with self.processor.threadbound():

            updated = False
            bm_updated = False
            for date, snapshot in stream:
                self.algo.perf_tracker.set_date(date)
                self.algo.blotter.set_date(date)
                # If we're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                if date < self.algo_start:
                    for event in snapshot:
                        if event.type in (DATASOURCE_TYPE.TRADE,
                                          DATASOURCE_TYPE.CUSTOM):
                            self.update_universe(event)
                        self.algo.perf_tracker.process_event(event)

                else:

                    for event in snapshot:
                        if event.type in (DATASOURCE_TYPE.TRADE,
                                          DATASOURCE_TYPE.CUSTOM):
                            self.update_universe(event)
                            updated = True
                        if event.type == DATASOURCE_TYPE.BENCHMARK:
                            bm_updated = True
                        txns, orders = self.algo.blotter.process_trade(event)
                        for data in chain(txns, orders, [event]):
                            self.algo.perf_tracker.process_event(data)

                    # Update our portfolio.
                    self.algo.set_portfolio(
                        self.algo.perf_tracker.get_portfolio()
                    )

                    # Send the current state of the universe
                    # to the user's algo.
                    if updated:
                        self.simulate_snapshot(date)
                        updated = False

                        # run orders placed in the algorithm call
                        # above through perf tracker before emitting
                        # the perf packet, so that the perf includes
                        # placed orders
                        for order in self.algo.blotter.new_orders:
                            self.algo.perf_tracker.process_event(order)
                        self.algo.blotter.new_orders = []

                    # The benchmark is our internal clock. When it
                    # updates, we need to emit a performance message.
                    if bm_updated:
                        bm_updated = False
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
                            if mkt_close < self.algo.perf_tracker.last_close:
                                mkt_close = self.get_next_close(mkt_close)

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

    def get_next_close(self, mkt_close):
        if mkt_close >= trading.environment.last_trading_day:
            return self.sim_params.last_close
        else:
            return trading.environment.next_open_and_close(mkt_close)[1]

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
        # Update our knowledge of this event's sid
        sid_data = self.current_data[event.sid]
        sid_data.__dict__.update(event.__dict__)

    def simulate_snapshot(self, date):
        """
        Run the user's algo against our current snapshot and update
        the algo's simulated time.
        """
        # Needs to be set so that we inject the proper date into algo
        # log/print lines.
        self.snapshot_dt = date
        self.algo.set_datetime(self.snapshot_dt)
        # Update the simulation time.
        self.simulation_dt = date
        self.algo.handle_data(self.current_data)
