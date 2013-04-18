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
import math
import uuid
import numpy as np

from copy import copy
from itertools import chain
from logbook import Logger, Processor
from collections import defaultdict

from zipline import ndict
from zipline.protocol import SIDData, DATASOURCE_TYPE
from zipline.finance.performance import PerformanceTracker
from zipline.gens.utils import hash_args

from zipline.finance.slippage import (
    VolumeShareSlippage,
    transact_partial,
    check_order_triggers
)
from zipline.finance.commission import PerShare
import zipline.utils.math_utils as zp_math

log = Logger('Trade Simulation')

from zipline.utils.protocol_utils import Enum

ORDER_STATUS = Enum(
    'OPEN',
    'FILLED'
)


class Blotter(object):

    def __init__(self):
        self.transact = transact_partial(VolumeShareSlippage(), PerShare())
        # these orders are aggregated by sid
        self.open_orders = defaultdict(list)
        # keep a dict of orders by their own id
        self.orders = {}
        # holding orders that have come in since the last
        # event.
        self.new_orders = []

    def place_order(self, order):
        # initialized filled field.
        order.filled = 0
        self.open_orders[order.sid].append(order)
        self.orders[order.id] = order
        self.new_orders.append(order)

    def transform(self, stream_in):
        """
        Main generator work loop.
        """
        for date, snapshot in stream_in:
            # relay any orders placed in prior snapshot
            # handling and reset the internal holding pen
            if self.new_orders:
                yield date, self.new_orders
                self.new_orders = []
            results = []

            for event in snapshot:
                results.append(event)
                # We only fill transactions on trade events.
                if event.type == DATASOURCE_TYPE.TRADE:
                    txns, modified_orders = self.process_trade(event)
                    results.extend(chain(txns, modified_orders))

            yield date, results

    def process_trade(self, trade_event):
        if zp_math.tolerant_equals(trade_event.volume, 0):
            # there are zero volume trade_events bc some stocks trade
            # less frequently than once per minute.
            return [], []

        if trade_event.sid in self.open_orders:
            orders = self.open_orders[trade_event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
            # Only use orders for the current day or before
            current_orders = filter(
                lambda o: o.dt <= trade_event.dt,
                orders)
        else:
            return [], []

        txns = self.transact(trade_event, current_orders)
        for txn in txns:
            self.orders[txn.order_id].filled += txn.amount
            # mark the last_modified date of the order to match
            self.orders[txn.order_id].last_modified_dt = txn.dt

        modified_orders = [order for order
                           in self.open_orders[trade_event.sid]
                           if order.last_modified_dt == trade_event.dt]
        for order in modified_orders:
            if not order.open:
                del self.orders[order.id]

        # update the open orders for the trade_event's sid
        self.open_orders[trade_event.sid] = \
            [order for order
             in self.open_orders[trade_event.sid]
             if order.open]

        return txns, modified_orders


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
        # get a string representation of the uuid.
        self.id = self.make_id()
        self.dt = dt
        self.last_modified_dt = dt
        self.sid = sid
        self.amount = amount
        self.filled = filled
        self.status = ORDER_STATUS.OPEN
        self.stop = stop
        self.limit = limit
        self.stop_reached = False
        self.limit_reached = False
        self.direction = math.copysign(1, self.amount)
        self.type = DATASOURCE_TYPE.ORDER

    def make_id(self):
        return uuid.uuid4().get_hex()

    def to_dict(self):
        py = copy(self.__dict__)
        for field in ['type', 'direction']:
            del py[field]
        return py

    def check_triggers(self, event):
        """
        Update internal state based on price triggers and the
        trade event's price.
        """
        stop_reached, limit_reached = \
            check_order_triggers(self, event)
        if (stop_reached, limit_reached) \
                != (self.stop_reached, self.limit_reached):
            self.last_modified_dt = event.dt
        self.stop_reached = stop_reached
        self.limit_reached = limit_reached

    @property
    def open(self):
        remainder = self.amount - self.filled
        if remainder != 0:
            self.status = ORDER_STATUS.OPEN
        else:
            self.status = ORDER_STATUS.FILLED

        return self.status == ORDER_STATUS.OPEN

    @property
    def triggered(self):
        """
        For a market order, True.
        For a stop order, True IFF stop_reached.
        For a limit order, True IFF limit_reached.
        For a stop-limit order, True IFF (stp_reached AND limit_reached)
        """
        if self.stop and not self.stop_reached:
            return False

        if self.limit and not self.limit_reached:
            return False

        return True

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

    def __init__(self, algo, sim_params, blotter=None):

        self.algo = algo
        self.sim_params = sim_params

        if not blotter:
            self.blotter = Blotter()

        self.perf_tracker = PerformanceTracker(self.sim_params)

        self.algo_start = self.sim_params.first_open
        self.algo_sim = AlgorithmSimulator(
            self.blotter,
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
        with_filled_orders = self.blotter.transform(stream_in)

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
                 blotter,
                 perf_tracker,
                 algo,
                 algo_start):

        # ==========
        # Algo Setup
        # ==========

        # We extract the order book from the txn client so that
        # the algo can place new orders.
        self.blotter = blotter
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
        self.algo.set_order_value(self.order_value)

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
        Market order:    order(sid, amount)
        Limit order:     order(sid, amount, limit_price)
        Stop order:      order(sid, amount, None, stop_price)
        StopLimit order: order(sid, amount, limit_price, stop_price)
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
        # This modifies the internal state of the blotter
        # so that it can fill the placed order when it
        # receives its next message.
        self.blotter.place_order(order)

        return order.id

    def order_value(self, sid, value, limit_price=None, stop_price=None):
        """
        Place an order by desired value rather than desired number of shares.
        If the requested sid is found in the universe, the requested value is
        divided by its price to imply the number of shares to transact.

        value > 0 :: Buy/Cover
        value < 0 :: Sell/Short
        Market order:    order(sid, value)
        Limit order:     order(sid, value, limit_price)
        Stop order:      order(sid, value, None, stop_price)
        StopLimit order: order(sid, value, limit_price, stop_price)
        """
        last_price = self.universe[sid].price
        if not np.allclose(last_price, 0):
            amount = value / last_price
            return self.order(sid, amount, limit_price, stop_price)

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

            if self.perf_tracker.emission_rate == 'daily':
                for message in perf_messages:
                    message[self.perf_key]['recorded_vars'] =\
                        self.algo.recorded_vars
                    yield message

            # When emitting minutely, it is still useful to have a final
            # packet with the entire days performance rolled up.
            if self.perf_tracker.emission_rate == 'minute':
                daily_rollup = self.perf_tracker.to_dict(
                    emission_type='daily'
                )
                daily_rollup['daily_perf']['recorded_vars'] = \
                    self.algo.recorded_vars
                yield daily_rollup

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
