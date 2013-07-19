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
import math
import uuid

from copy import copy
from logbook import Logger
from collections import defaultdict

import zipline.errors
import zipline.protocol as zp

from zipline.finance.slippage import (
    VolumeShareSlippage,
    transact_partial,
    check_order_triggers
)
from zipline.finance.commission import PerShare
import zipline.utils.math_utils as zp_math

log = Logger('Blotter')

from zipline.utils.protocol_utils import Enum

ORDER_STATUS = Enum(
    'OPEN',
    'FILLED',
    'CANCELLED'
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
        self.current_dt = None
        self.max_shares = int(1e+11)

    def __repr__(self):
        return """
{class_name}(
    transact_partial={transact_partial},
    open_orders={open_orders},
    orders={orders},
    new_orders={new_orders},
    current_dt={current_dt})
""".strip().format(class_name=self.__class__.__name__,
                   transact_partial=self.transact.args,
                   open_orders=self.open_orders,
                   orders=self.orders,
                   new_orders=self.new_orders,
                   current_dt=self.current_dt)

    def set_date(self, dt):
        self.current_dt = dt

    def order(self, sid, amount, limit_price, stop_price, order_id=None):

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

        # Fractional shares are not supported.
        amount = int(amount)

        # just validates amount and passes rest on to TransactionSimulator
        # Tell the user if they try to buy 0 shares of something.
        if amount == 0:
            zero_message = "Requested to trade zero shares of {psid}".format(
                psid=sid
            )
            log.debug(zero_message)
            # Don't bother placing orders for 0 shares.
            return
        elif amount > self.max_shares:
            # Arbitrary limit of 100 billion (US) shares will never be
            # exceeded except by a buggy algorithm.
            raise OverflowError('Can\'t order more than %d shares' %
                                self.max_shares)

        order = Order(**{
            'dt': self.current_dt,
            'sid': sid,
            'amount': amount,
            'filled': 0,
            'stop': stop_price,
            'limit': limit_price,
            'id': order_id
        })

        # initialized filled field.
        order.filled = 0
        self.open_orders[order.sid].append(order)
        self.orders[order.id] = order
        self.new_orders.append(order)

        return order.id

    def cancel(self, order_id):
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]
        if cur_order.open:
            order_list = self.open_orders[cur_order.sid]
            if cur_order in order_list:
                order_list.remove(cur_order)

            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.status = ORDER_STATUS.CANCELLED
            cur_order.dt = self.current_dt
            # we want this order's new status to be relayed out
            # along with newly placed orders.
            self.new_orders.append(cur_order)

    def process_split(self, split_event):
        if split_event.sid not in self.open_orders:
            return

        orders_to_modify = self.open_orders[split_event.sid]
        for order in orders_to_modify:
            order.handle_split(split_event)

    def process_trade(self, trade_event):
        if trade_event.type != zp.DATASOURCE_TYPE.TRADE:
            return

        if zp_math.tolerant_equals(trade_event.volume, 0):
            # there are zero volume trade_events bc some stocks trade
            # less frequently than once per minute.
            return

        if trade_event.sid in self.open_orders:
            orders = self.open_orders[trade_event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
            # Only use orders for the current day or before
            current_orders = filter(
                lambda o: o.dt <= trade_event.dt,
                orders)
        else:
            return

        for order, txn in self.transact(trade_event, current_orders):
            if txn.amount == 0:
                raise zipline.errors.TransactionWithNoAmount(txn=txn)
            if math.copysign(1, txn.amount) != order.direction:
                raise zipline.errors.TransactionWithWrongDirection(
                    txn=txn, order=order)
            if abs(txn.amount) > abs(self.orders[txn.order_id].amount):
                raise zipline.errors.TransactionVolumeExceedsOrder(
                    txn=txn, order=order)

            order.filled += txn.amount
            # mark the date of the order to match the transaction
            # that is filling it.
            order.dt = txn.dt

            yield txn, order

        # update the open orders for the trade_event's sid
        self.open_orders[trade_event.sid] = \
            [order for order
             in self.open_orders[trade_event.sid]
             if order.open]


class Order(object):
    def __init__(self, dt, sid, amount, stop=None, limit=None, filled=0,
                 id=None):
        """
        @dt - datetime.datetime that the order was placed
        @sid - stock sid of the order
        @amount - the number of shares to buy/sell
                  a positive sign indicates a buy
                  a negative sign indicates a sell
        @filled - how many shares of the order have been filled so far
        """
        # get a string representation of the uuid.
        self.id = id or self.make_id()
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
        self.type = zp.DATASOURCE_TYPE.ORDER

    def make_id(self):
        return uuid.uuid4().hex

    def to_dict(self):
        py = copy(self.__dict__)
        for field in ['type', 'direction']:
            del py[field]
        return py

    def to_api_obj(self):
        pydict = self.to_dict()
        obj = zp.Order(initial_values=pydict)
        return obj

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

    def handle_split(self, split_event):
        ratio = split_event.ratio

        # update the amount, limit_price, and stop_price
        # by the split's ratio

        # info here: http://finra.complinet.com/en/display/display_plain.html?
        # rbid=2403&element_id=8950&record_id=12208&print=1

        # if we have an open order for 100 shares at $20, and we get
        # a 3:1 split, we now want to have an open order for 33 shares at $60
        # for the amount, we round down to the nearest whole share
        self.amount = int(self.amount * ratio)

        if self.limit:
            self.limit = round(self.limit / ratio, 2)

        if self.stop:
            self.stop = round(self.stop / ratio, 2)

    @property
    def open(self):
        if self.status == ORDER_STATUS.CANCELLED:
            return False

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
        For a stop-limit order, True IFF (stop_reached AND limit_reached)
        """
        if self.stop and not self.stop_reached:
            return False

        if self.limit and not self.limit_reached:
            return False

        return True

    @property
    def open_amount(self):
        return self.amount - self.filled
