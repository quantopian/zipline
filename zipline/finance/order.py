#
# Copyright 2016 Quantopian, Inc.
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

from six import text_type

import zipline.protocol as zp
from zipline.assets import Asset
from zipline.utils.enum import enum
from zipline.utils.input_validation import expect_types

ORDER_STATUS = enum(
    'OPEN',
    'FILLED',
    'CANCELLED',
    'REJECTED',
    'HELD',
)

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3

ORDER_FIELDS_TO_IGNORE = {'type', 'direction', '_status', 'asset'}


class Order(object):
    # using __slots__ to save on memory usage.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    __slots__ = ["id", "dt", "reason", "created", "asset", "amount", "filled",
                 "commission", "_status", "stop", "limit", "stop_reached",
                 "limit_reached", "direction", "type", "broker_order_id"]

    @expect_types(asset=Asset)
    def __init__(self, dt, asset, amount, stop=None, limit=None, filled=0,
                 commission=0, id=None):
        """
        @dt - datetime.datetime that the order was placed
        @asset - asset for the order.
        @amount - the number of shares to buy/sell
                  a positive sign indicates a buy
                  a negative sign indicates a sell
        @filled - how many shares of the order have been filled so far
        """

        # get a string representation of the uuid.
        self.id = self.make_id() if id is None else id
        self.dt = dt
        self.reason = None
        self.created = dt
        self.asset = asset
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
        self.broker_order_id = None

    def make_id(self):
        return uuid.uuid4().hex

    def to_dict(self):
        dct = {name: getattr(self, name)
               for name in self.__slots__
               if name not in ORDER_FIELDS_TO_IGNORE}

        if self.broker_order_id is None:
            del dct['broker_order_id']

        # Adding 'sid' for backwards compatibility with downstream consumers.
        dct['sid'] = self.asset
        dct['status'] = self.status

        return dct

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset

    def to_api_obj(self):
        pydict = self.to_dict()
        obj = zp.Order(initial_values=pydict)
        return obj

    def check_triggers(self, price, dt):
        """
        Update internal state based on price triggers and the
        trade event's price.
        """
        stop_reached, limit_reached, sl_stop_reached = \
            self.check_order_triggers(price)
        if (stop_reached, limit_reached) \
                != (self.stop_reached, self.limit_reached):
            self.dt = dt
        self.stop_reached = stop_reached
        self.limit_reached = limit_reached
        if sl_stop_reached:
            # Change the STOP LIMIT order into a LIMIT order
            self.stop = None

    def check_order_triggers(self, current_price):
        """
        Given an order and a trade event, return a tuple of
        (stop_reached, limit_reached).
        For market orders, will return (False, False).
        For stop orders, limit_reached will always be False.
        For limit orders, stop_reached will always be False.
        For stop limit orders a Boolean is returned to flag
        that the stop has been reached.

        Orders that have been triggered already (price targets reached),
        the order's current values are returned.
        """
        if self.triggered:
            return (self.stop_reached, self.limit_reached, False)

        stop_reached = False
        limit_reached = False
        sl_stop_reached = False

        order_type = 0

        if self.amount > 0:
            order_type |= BUY
        else:
            order_type |= SELL

        if self.stop is not None:
            order_type |= STOP

        if self.limit is not None:
            order_type |= LIMIT

        if order_type == BUY | STOP | LIMIT:
            if current_price >= self.stop:
                sl_stop_reached = True
                if current_price <= self.limit:
                    limit_reached = True
        elif order_type == SELL | STOP | LIMIT:
            if current_price <= self.stop:
                sl_stop_reached = True
                if current_price >= self.limit:
                    limit_reached = True
        elif order_type == BUY | STOP:
            if current_price >= self.stop:
                stop_reached = True
        elif order_type == SELL | STOP:
            if current_price <= self.stop:
                stop_reached = True
        elif order_type == BUY | LIMIT:
            if current_price <= self.limit:
                limit_reached = True
        elif order_type == SELL | LIMIT:
            # This is a SELL LIMIT order
            if current_price >= self.limit:
                limit_reached = True

        return (stop_reached, limit_reached, sl_stop_reached)

    def handle_split(self, ratio):
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
