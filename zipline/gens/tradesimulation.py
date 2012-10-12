#
# Copyright 2012 Quantopian, Inc.
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
from collections import defaultdict

from datetime import datetime
from itertools import groupby
from operator import attrgetter

from zipline import ndict

from zipline.finance.trading import TransactionSimulator
from zipline.finance.performance import PerformanceTracker
from zipline.gens.utils import hash_args

log = Logger('Trade Simulation')


class TradeSimulationClient(object):
    """
    Generator-style class that takes the expected output of a merge, a
    user algorithm, a trading environment, and a simulator slippage as
    arguments.  Pipes the merge stream through a TransactionSimulator
    and a PerformanceTracker, which keep track of the current state of
    our algorithm's simulated universe. Results are fed to the user's
    algorithm, which directly inserts transactions into the
    TransactionSimulator's order book.

    TransactionSimulator maintains a dictionary from sids to the
    as-yet unfilled orders placed by the user's algorithm.  As trade
    events arrive, if the algorithm has open orders against the
    trade's sid, the simulator will fill orders up to 25% of market
    cap.  Applied transactions are added to a txn field on the event
    and forwarded to PerformanceTracker. The txn field is set to None
    on non-trade events and events that do not match any open orders.

    PerformanceTracker receives the updated event messages from
    TransactionSimulator, maintaining a set of daily and cumulative
    performance metrics for the algorithm.  The tracker removes the
    txn field from each event it receives, replacing it with a
    portfolio field to be fed into the user algo. At the end of each
    trading day, the PerformanceTracker also generates a daily
    performance report, which is appended to event's perf_report
    field.

    Fully processed events are fed to AlgorithmSimulator, which
    batches together events with the same dt field into a single
    snapshot to be fed to the algo. The portfolio object is repeatedly
    overwritten so that only the most recent snapshot of the universe
    is sent to the algo.
    """

    def __init__(self, algo, environment):

        self.algo = algo
        self.environment = environment

        self.ordering_client = TransactionSimulator()
        self.perf_tracker = PerformanceTracker(self.environment)

        self.algo_start = self.environment.first_open
        self.algo_sim = AlgorithmSimulator(
            self.ordering_client,
            self.algo,
            self.algo_start
        )

    def get_hash(self):
        """
        There should only ever be one TSC in the system, so
        we don't bother passing args into the hash.
        """
        return self.__class__.__name__ + hash_args()

    def simulate(self, stream_in):
        """
        Main generator work loop.
        """

        # Simulate filling any open orders made by the previous run of
        # the user's algorithm.  Fills the Transaction field on any
        # event that results in a filled order.
        with_filled_orders = self.ordering_client.transform(stream_in)

        # Pipe the events with transactions to perf. This will remove
        # the TRANSACTION field added by TransactionSimulator and replace it
        # with a portfolio field to be passed to the user's
        # algorithm. Also adds a perf_message field which is usually
        # none, but contains an update message once per day.
        with_portfolio = self.perf_tracker.transform(with_filled_orders)

        # Pass the messages from perf to the user's algorithm for simulation.
        # Events are batched by dt so that the algo handles all events for a
        # given timestamp at one one go.
        performance_messages = self.algo_sim.transform(with_portfolio)

        # The algorithm will yield a daily_results message (as
        # calculated by the performance tracker) at the end of each
        # day.  It will also yield a risk report at the end of the
        # simulation.
        for message in performance_messages:
            yield message


class AlgorithmSimulator(object):

    def __init__(self,
                 order_book,
                 algo,
                 algo_start):

        # ==========
        # Algo Setup
        # ==========

        # We extract the order book from the txn client so that
        # the algo can place new orders.
        self.order_book = order_book

        self.algo = algo
        self.algo_start = algo_start

        # Monkey patch the user algorithm to place orders in the
        # TransactionSimulator's order book and use our logger.
        self.algo.set_order(self.order)

        # ==============
        # Snapshot Setup
        # ==============

        # The algorithm's universe as of our most recent event.
        # We want an ndict that will have empty ndicts as default
        # values on missing keys.
        self.universe = ndict(internal=defaultdict(ndict))

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
            record.extra['algo_dt'] = self.snapshot_dt
        self.processor = Processor(inject_algo_dt)

    def order(self, sid, amount):
        """
        Closure to pass into the user's algo to allow placing orders
        into the transaction simulator's dict of open orders.
        """
        order = ndict({
            'dt': self.simulation_dt,
            'sid': sid,
            'amount': int(amount),
            'filled': 0
        })

        # Tell the user if they try to buy 0 shares of something.
        if order.amount == 0:
            zero_message = "Requested to trade zero shares of {sid}".format(
                sid=order.sid
            )
            log.debug(zero_message)
            # Don't bother placing orders for 0 shares.
            return

        # Add non-zero orders to the order book.
        # !!!IMPORTANT SIDE-EFFECT!!!
        # This modifies the internal state of the transaction
        # simulator so that it can fill the placed order when it
        # receives its next message.
        self.order_book.place_order(order)

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        # inject the current algo
        # snapshot time to any log record generated.
        with self.processor.threadbound():
            # Group together events with the same dt field. This depends on the
            # events already being sorted.
            for date, snapshot in groupby(stream_in, attrgetter('dt')):
                # Set the simulation date to be the first event we see.
                # This should only occur once, at the start of the test.
                if self.simulation_dt is None:
                    self.simulation_dt = date

                # Done message has the risk report, so we yield before exiting.
                if date == 'DONE':
                    for event in snapshot:
                        yield event.perf_message
                    raise StopIteration

                # We're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                elif date < self.algo_start:
                    for event in snapshot:
                        del event['perf_message']
                        self.update_universe(event)

                # The algo has taken so long to process events that
                # its simulated time is later than the event time.
                # Update the universe and yield any perf messages
                # encountered, but don't call handle_data.
                elif date < self.simulation_dt:
                    for event in snapshot:
                        # Only yield if we have something interesting to say.
                        if event.perf_message is not None:
                            yield event.perf_message
                        # Delete the message before updating,
                        # so we don't send it to the user.
                        del event['perf_message']
                        self.update_universe(event)

                # Regular snapshot.  Update the universe and send a snapshot
                # to handle data.
                else:
                    for event in snapshot:
                        # Only yield if we have something interesting to say.
                        if event.perf_message is not None:
                            yield event.perf_message
                        del event['perf_message']

                        self.update_universe(event)

                    # Send the current state of the universe
                    # to the user's algo.
                    self.simulate_snapshot(date)

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
        # Update our portfolio.
        self.algo.set_portfolio(event.portfolio)

        # Update our knowledge of this event's sid
        for field in event.keys():
            self.universe[event.sid][field] = event[field]

    def simulate_snapshot(self, date):
        """
        Run the user's algo against our current snapshot and update
        the algo's simulated time.
        """
        # Needs to be set so that we inject the proper date into algo
        # log/print lines.
        self.snapshot_dt = date
        start_tic = datetime.now()
        self.algo.handle_data(self.universe)
        stop_tic = datetime.now()

        # How long did you take?
        delta = stop_tic - start_tic

        # Update the simulation time.
        self.simulation_dt = date + delta
