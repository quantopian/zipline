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

from logbook import Logger, Processor
from collections import defaultdict

from zipline import ndict
from zipline.protocol import SIDData

from zipline.finance.trading import TransactionSimulator
from zipline.finance.performance import PerformanceTracker
from zipline.gens.utils import hash_args

log = Logger('Trade Simulation')


class Order(object):
    def __init__(self, dt, sid, amount, stop=None, limit=None, filled=0):
        """
        @dt - datetime.datetime that the order was placed
        @sid - stock sid of the order
        @amount - the number of shares to buy/sell
                  a positive sign indicates a buy
                  a negative sign indicates a sell
        @filled - how many shares of the order have been filled so far
        """
        self.dt = dt
        self.sid = sid
        self.amount = amount
        self.filled = filled
        self.stop = stop
        self.limit = limit

    def __getitem__(self, name):
        return self.__dict__[name]


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

    def __init__(self, algo, sim_params):

        self.algo = algo
        self.sim_params = sim_params

        self.ordering_client = TransactionSimulator()
        self.perf_tracker = PerformanceTracker(self.sim_params)

        self.algo_start = self.sim_params.first_open
        self.algo_sim = AlgorithmSimulator(
            self.ordering_client,
            self.perf_tracker,
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
        # algorithm. Also adds a perf_messages field which is usually
        # empty, but contains update messages once per day.
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

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'intraday_perf',
        'daily': 'daily_perf'
    }

    def __init__(self,
                 order_book,
                 perf_tracker,
                 algo,
                 algo_start):

        # ==========
        # Algo Setup
        # ==========

        # We extract the order book from the txn client so that
        # the algo can place new orders.
        self.order_book = order_book
        self.perf_tracker = perf_tracker

        self.perf_key = self.EMISSION_TO_PERF_KEY_MAP[
            perf_tracker.emission_rate]

        self.algo = algo
        self.algo_start = algo_start.replace(hour=0, minute=0,
                                             second=0,
                                             microsecond=0)

        # Monkey patch the user algorithm to place orders in the
        # TransactionSimulator's order book and use our logger.
        self.algo.set_order(self.order)

        # ==============
        # Snapshot Setup
        # ==============

        # The algorithm's universe as of our most recent event.
        # We want an ndict that will have empty objects as default
        # values on missing keys.
        self.universe = ndict(internal=defaultdict(SIDData))

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

    def order(self, sid, amount, limit_price=None, stop_price=None):

        # something could be done with amount to further divide
        # between buy by share count OR buy shares up to a dollar amount
        # numeric == share count  AND  "$dollar.cents" == cost amount

        """
        amount > 0 :: Buy/Cover
        amount < 0 :: Sell/Short
        Market order:    order(sid,amount)
        Limit order:     order(sid,amount, limit_price)
        Stop order:      order(sid,amount, None, stop_price)
        StopLimit order: order(sid,amount, limit_price, stop_price)
        """

        # just validates amount and passes rest on to TransactionSimulator
        # Tell the user if they try to buy 0 shares of something.
        if amount == 0:
            zero_message = "Requested to trade zero shares of {psid}".format(
                psid=sid
            )
            log.debug(zero_message)
            # Don't bother placing orders for 0 shares.
            return

        order = Order(**{
            'dt': self.simulation_dt,
            'sid': sid,
            'amount': int(amount),
            'filled': 0,
            'stop': stop_price,
            'limit': limit_price
        })

        # Add non-zero orders to the order book.
        # !!!IMPORTANT SIDE-EFFECT!!!
        # This modifies the internal state of the transaction
        # simulator so that it can fill the placed order when it
        # receives its next message.
        err_str = self.order_book.place_order(order)
        if err_str is not None and len(err_str) > 0:
            # error, trade was not placed, log it out
            log.debug(err_str)

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
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

            for date, snapshot in stream:
                # We're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                if date < self.algo_start:
                    for event in snapshot:
                        del event['perf_messages']
                        self.update_universe(event)

                # Regular snapshot.  Update the universe and send a snapshot
                # to handle data.
                else:
                    for event in snapshot:
                        for perf_message in event.perf_messages:
                            # append current values of recorded vars
                            # to emitted message
                            perf_message[self.perf_key]['recorded_vars'] =\
                                self.algo.recorded_vars
                            yield perf_message
                        del event['perf_messages']

                        self.update_universe(event)

                    # Send the current state of the universe
                    # to the user's algo.
                    self.simulate_snapshot(date)

            perf_messages, risk_message = \
                self.perf_tracker.handle_simulation_end()

            for message in perf_messages:
                message[self.perf_key]['recorded_vars'] =\
                    self.algo.recorded_vars
                yield message

            yield risk_message

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
        # Update our portfolio.
        self.algo.set_portfolio(event.portfolio)
        # the portfolio is modified by each event passed into the
        # performance tracker (prices and amounts can change).
        # Performance tracker sends back an up-to-date portfolio
        # with each event. However, we provide the portfolio to
        # the algorithm via a setter method, rather than as part
        # of the event data sent to handle_data. To avoid
        # confusion, we remove it from the event here.
        del event.portfolio
        # Update our knowledge of this event's sid
        sid_data = self.universe[event.sid]
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
        self.algo.handle_data(self.universe)

        # Update the simulation time.
        self.simulation_dt = date
