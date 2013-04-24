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
            results = []

            for event in snapshot:
                results.append(event)
                # We only fill transactions on trade events.
                if event.type == DATASOURCE_TYPE.TRADE:
                    txns, modified_orders = self.process_trade(event)
                    results.extend(chain(txns, modified_orders))

            yield date, results

    def process_trade(self, trade_event):
        if trade_event.type != DATASOURCE_TYPE.TRADE:
            return [], []

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
            # mark the date of the order to match the transaction
            # that is filling it.
            self.orders[txn.order_id].dt = txn.dt

        modified_orders = [order for order
                           in self.open_orders[trade_event.sid]
                           if order.dt == trade_event.dt]
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
        self.created = dt
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
            self.dt = event.dt
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
        # Pass the messages from perf to the user's algorithm for simulation.
        # Events are batched by dt so that the algo handles all events for a
        # given timestamp at one one go.
        performance_messages = self.algo_sim.transform(stream_in)

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

            updated = False
            bm_updated = False
            for date, snapshot in stream:
                self.perf_tracker.set_date(date)
                # If we're still in the warmup period.  Use the event to
                # update our universe, but don't yield any perf messages,
                # and don't send a snapshot to handle_data.
                if date < self.algo_start:
                    for event in snapshot:
                        if event.type in (DATASOURCE_TYPE.TRADE,
                                          DATASOURCE_TYPE.CUSTOM):
                            self.update_universe(event)
                        self.perf_tracker.process_event(event)

                else:

                    for event in snapshot:
                        if event.type in (DATASOURCE_TYPE.TRADE,
                                          DATASOURCE_TYPE.CUSTOM):
                            self.update_universe(event)
                            updated = True
                        if event.type == DATASOURCE_TYPE.BENCHMARK:
                            bm_updated = True
                        txns, orders = self.blotter.process_trade(event)
                        for data in chain([event], txns, orders):
                            self.perf_tracker.process_event(data)

                    # Update our portfolio.
                    self.algo.set_portfolio(self.perf_tracker.get_portfolio())

                    # Send the current state of the universe
                    # to the user's algo.
                    if updated:
                        self.simulate_snapshot(date)
                        updated = False

                        # run orders placed in the algorithm call
                        # above through perf tracker before emitting
                        # the perf packet, so that the perf includes
                        # placed orders
                        for order in self.blotter.new_orders:
                            self.perf_tracker.process_event(order)
                        self.blotter.new_orders = []

                    # The benchmark is our internal clock. When it
                    # updates, we need to emit a performance message.
                    if bm_updated:
                        bm_updated = False
                        yield self.get_message(date)

            risk_message = self.perf_tracker.handle_simulation_end()

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

    def get_message(self, date):
        rvars = self.algo.recorded_vars
        if self.perf_tracker.emission_rate == 'daily':
            perf_message = \
                self.perf_tracker.handle_market_close()
            perf_message['daily_perf']['recorded_vars'] = rvars
            return perf_message

        elif self.perf_tracker.emission_rate == 'minute':
            self.perf_tracker.handle_minute_close(date)
            perf_message = self.perf_tracker.to_dict()
            perf_message['intraday_perf']['recorded_vars'] = rvars
            return perf_message

    def update_universe(self, event):
        """
        Update the universe with new event information.
        """
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
        # Update the simulation time.
        self.simulation_dt = date
        self.algo.handle_data(self.universe)
