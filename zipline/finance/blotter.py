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
import math
import uuid

from copy import copy
from logbook import Logger
from collections import defaultdict

from six import text_type

import zipline.errors
import zipline.protocol as zp

from zipline.finance.slippage import (
    VolumeShareSlippage,
    transact_partial,
    check_order_triggers
)
from zipline.finance.commission import PerShare

log = Logger('Blotter')

from zipline.utils.protocol_utils import Enum

ORDER_STATUS = Enum(
    'OPEN',
    'FILLED',
    'CANCELLED',
    'REJECTED',
    'HELD',
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

    def order(self, sid, amount, style, order_id=None):

        # something could be done with amount to further divide
        # between buy by share count OR buy shares up to a dollar amount
        # numeric == share count  AND  "$dollar.cents" == cost amount

        """
        amount > 0 :: Buy/Cover
        amount < 0 :: Sell/Short
        Market order:    order(sid, amount)
        Limit order:     order(sid, amount, style=LimitOrder(limit_price))
        Stop order:      order(sid, amount, style=StopOrder(stop_price))
        StopLimit order: order(sid, amount, style=StopLimitOrder(limit_price,
                               stop_price))
        """
        if amount == 0:
            # Don't bother placing orders for 0 shares.
            return
        elif amount > self.max_shares:
            # Arbitrary limit of 100 billion (US) shares will never be
            # exceeded except by a buggy algorithm.
            raise OverflowError("Can't order more than %d shares" %
                                self.max_shares)

        is_buy = (amount > 0)
        order = Order(
            dt=self.current_dt,
            sid=sid,
            amount=amount,
            stop=style.get_stop_price(is_buy),
            limit=style.get_limit_price(is_buy),
            id=order_id
        )

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
            cur_order.cancel()
            cur_order.dt = self.current_dt
            # we want this order's new status to be relayed out
            # along with newly placed orders.
            self.new_orders.append(cur_order)

    def reject(self, order_id, reason=''):
        """
        Mark the given order as 'rejected', which is functionally similar to
        cancelled. The distinction is that rejections are involuntary (and
        usually include a message from a broker indicating why the order was
        rejected) while cancels are typically user-driven.
        """
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]

        order_list = self.open_orders[cur_order.sid]
        if cur_order in order_list:
            order_list.remove(cur_order)

        if cur_order in self.new_orders:
            self.new_orders.remove(cur_order)
        cur_order.reject(reason=reason)
        cur_order.dt = self.current_dt
        # we want this order's new status to be relayed out
        # along with newly placed orders.
        self.new_orders.append(cur_order)

    def hold(self, order_id, reason=''):
        """
        Mark the order with order_id as 'held'. Held is functionally similar
        to 'open'. When a fill (full or partial) arrives, the status
        will automatically change back to open/filled as necessary.
        """
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]
        if cur_order.open:
            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.hold(reason=reason)
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

        if trade_event.sid not in self.open_orders:
            return

        if trade_event.volume < 1:
            # there are zero volume trade_events bc some stocks trade
            # less frequently than once per minute.
            return

        orders = self.open_orders[trade_event.sid]
        orders = sorted(orders, key=lambda o: o.dt)
        # Only use orders for the current day or before
        current_orders = filter(
            lambda o: o.dt <= trade_event.dt,
            orders)

        for txn, order in self.process_transactions(trade_event,
                                                    current_orders):
            yield txn, order

        # update the open orders for the trade_event's sid
        self.open_orders[trade_event.sid] = \
            [order for order
             in self.open_orders[trade_event.sid]
             if order.open]

    def process_transactions(self, trade_event, current_orders):
        for order, txn in self.transact(trade_event, current_orders):
            if txn.type == zp.DATASOURCE_TYPE.COMMISSION:
                order.commission = (order.commission or 0.0) + txn.cost
            else:
                if txn.amount == 0:
                    raise zipline.errors.TransactionWithNoAmount(txn=txn)
                if math.copysign(1, txn.amount) != order.direction:
                    raise zipline.errors.TransactionWithWrongDirection(
                        txn=txn, order=order)
                if abs(txn.amount) > abs(self.orders[txn.order_id].amount):
                    raise zipline.errors.TransactionVolumeExceedsOrder(
                        txn=txn, order=order)

                order.filled += txn.amount
                if txn.commission is not None:
                    order.commission = ((order.commission or 0.0)
                                        + txn.commission)

            # mark the date of the order to match the transaction
            # that is filling it.
            order.dt = txn.dt

            yield txn, order

    def _get_state(self):

        state_to_save = ['new_orders', 'open_orders', 'orders', '_status']

        state_dict = {k: self.__dict__[k] for k in state_to_save
                      if k in self.__dict__}

        return 'Blotter', state_dict

    def _set_state(self, saved_state):
        self.__dict__.update(saved_state)


class Order(object):
    def __init__(self, dt, sid, amount, stop=None, limit=None, filled=0,
                 commission=None, id=None):
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
        self.reason = None
        self.created = dt
        self.sid = sid
        self.amount = amount
        self.filled = filled
        self.commission = commission
        self._status = ORDER_STATUS.OPEN
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
        for field in ['type', 'direction', '_status']:
            del py[field]
        py['status'] = self.status
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
        stop_reached, limit_reached, sl_stop_reached = \
            check_order_triggers(self, event)
        if (stop_reached, limit_reached) \
                != (self.stop_reached, self.limit_reached):
            self.dt = event.dt
        self.stop_reached = stop_reached
        self.limit_reached = limit_reached
        if sl_stop_reached:
            # Change the STOP LIMIT order into a LIMIT order
            self.stop = None

    def handle_split(self, split_event):
        ratio = split_event.ratio

        # update the amount, limit_price, and stop_price
        # by the split's ratio

        # info here: http://finra.complinet.com/en/display/display_plain.html?
        # rbid=2403&element_id=8950&record_id=12208&print=1

        # new_share_amount = old_share_amount / ratio
        # new_price = old_price * ratio

        self.amount = int(self.amount / ratio)

        if self.limit is not None:
            self.limit = round(self.limit * ratio, 2)

        if self.stop is not None:
            self.stop = round(self.stop * ratio, 2)

    @property
    def status(self):
        if not self.open_amount:
            return ORDER_STATUS.FILLED
        elif self._status == ORDER_STATUS.HELD and self.filled:
            return ORDER_STATUS.OPEN
        else:
            return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def cancel(self):
        self.status = ORDER_STATUS.CANCELLED

    def reject(self, reason=''):
        self.status = ORDER_STATUS.REJECTED
        self.reason = reason

    def hold(self, reason=''):
        self.status = ORDER_STATUS.HELD
        self.reason = reason

    @property
    def open(self):
        return self.status in [ORDER_STATUS.OPEN, ORDER_STATUS.HELD]

    @property
    def triggered(self):
        """
        For a market order, True.
        For a stop order, True IFF stop_reached.
        For a limit order, True IFF limit_reached.
        """
        if self.stop is not None and not self.stop_reached:
            return False

        if self.limit is not None and not self.limit_reached:
            return False

        return True

    def _get_state(self):
        """
        Return a serialized version of the order.
        """
        state_dict = {}
        for k, v in self.__dict__.iteritems():
            if (not k.startswith('_')):
                state_dict[k] = v

        state_dict['_status'] = self._status

        return 'Order', state_dict

    def _set_state(self, saved_state):
        """
        Reconstruct this order from saved_state.
        """
        self.__dict__.update(saved_state)

    @property
    def open_amount(self):
        return self.amount - self.filled

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%s)" % self.to_dict().__repr__()

    def __unicode__(self):
        """
        Unicode representation for this object.
        """
        return text_type(repr(self))
