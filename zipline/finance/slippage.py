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
from __future__ import division

import abc

import math

from copy import copy
from functools import partial

from six import with_metaclass

from zipline.protocol import DATASOURCE_TYPE
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3


def check_order_triggers(order, event):
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
    if order.triggered:
        return (order.stop_reached, order.limit_reached, False)

    stop_reached = False
    limit_reached = False
    sl_stop_reached = False

    order_type = 0

    if order.amount > 0:
        order_type |= BUY
    else:
        order_type |= SELL

    if order.stop is not None:
        order_type |= STOP

    if order.limit is not None:
        order_type |= LIMIT

    if order_type == BUY | STOP | LIMIT:
        if event.price >= order.stop:
            sl_stop_reached = True
            if event.price <= order.limit:
                limit_reached = True
    elif order_type == SELL | STOP | LIMIT:
        if event.price <= order.stop:
            sl_stop_reached = True
            if event.price >= order.limit:
                limit_reached = True
    elif order_type == BUY | STOP:
        if event.price >= order.stop:
            stop_reached = True
    elif order_type == SELL | STOP:
        if event.price <= order.stop:
            stop_reached = True
    elif order_type == BUY | LIMIT:
        if event.price <= order.limit:
            limit_reached = True
    elif order_type == SELL | LIMIT:
        # This is a SELL LIMIT order
        if event.price >= order.limit:
            limit_reached = True

    return (stop_reached, limit_reached, sl_stop_reached)


def transact_stub(slippage, commission, event, open_orders):
    """
    This is intended to be wrapped in a partial, so that the
    slippage and commission models can be enclosed.
    """
    for order, transaction in slippage(event, open_orders):
        if transaction and transaction.amount != 0:
            direction = math.copysign(1, transaction.amount)
            per_share, total_commission = commission.calculate(transaction)
            transaction.price += per_share * direction
            transaction.commission = total_commission
        yield order, transaction


def transact_partial(slippage, commission):
    return partial(transact_stub, slippage, commission)


class Transaction(object):

    def __init__(self, sid, amount, dt, price, order_id, commission=None):
        self.sid = sid
        self.amount = amount
        self.dt = dt
        self.price = price
        self.order_id = order_id
        self.commission = commission
        self.type = DATASOURCE_TYPE.TRANSACTION

    def __getitem__(self, name):
        return self.__dict__[name]

    def to_dict(self):
        py = copy(self.__dict__)
        del py['type']
        return py

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Transaction saved state is too old.")

        self.__dict__.update(state)


def create_transaction(event, dt, order, price, amount):

    # floor the amount to protect against non-whole number orders
    # TODO: Investigate whether we can add a robust check in blotter
    # and/or tradesimulation, as well.
    amount_magnitude = int(abs(amount))

    if amount_magnitude < 1:
        raise Exception("Transaction magnitude must be at least 1.")

    transaction = Transaction(
        sid=event.sid,
        amount=int(amount),
        dt=dt,
        price=price,
        order_id=order.id
    )

    return transaction


class LiquidityExceeded(Exception):
    pass


class SlippageModel(with_metaclass(abc.ABCMeta)):

    @property
    def volume_for_bar(self):
        return self._volume_for_bar

    @abc.abstractproperty
    def process_order(self, event, order):
        pass

    def simulate(self, event, current_orders, dt):

        self._volume_for_bar = 0

        for order in current_orders:

            if order.open_amount == 0:
                continue

            order.check_triggers(event)
            if not order.triggered:
                continue

            try:
                txn = self.process_order(event, order, dt)
            except LiquidityExceeded:
                break

            if txn:
                self._volume_for_bar += abs(txn.amount)
                yield order, txn

    def __call__(self, event, current_orders, dt, **kwargs):
        return self.simulate(event, current_orders, dt, **kwargs)


class VolumeShareSlippage(SlippageModel):

    def __init__(self,
                 volume_limit=.25,
                 price_impact=0.1):

        self.volume_limit = volume_limit
        self.price_impact = price_impact

    def __repr__(self):
        return """
{class_name}(
    volume_limit={volume_limit},
    price_impact={price_impact})
""".strip().format(class_name=self.__class__.__name__,
                   volume_limit=self.volume_limit,
                   price_impact=self.price_impact)

    def process_order(self, event, order, dt):

        volume = self.data_portal.get_current_price_data(order.sid, 'volume')
        price = self.data_portal.get_current_price_data(order.sid, 'close')
        max_volume = self.volume_limit * volume

        # price impact accounts for the total volume of transactions
        # created against the current minute bar
        remaining_volume = max_volume - self.volume_for_bar
        if remaining_volume < 1:
            # we can't fill any more transactions
            raise LiquidityExceeded()

        # the current order amount will be the min of the
        # volume available in the bar or the open amount.
        cur_volume = int(min(remaining_volume, abs(order.open_amount)))

        if cur_volume < 1:
            return

        # tally the current amount into our total amount ordered.
        # total amount will be used to calculate price impact
        total_volume = self.volume_for_bar + cur_volume

        volume_share = min(total_volume / volume,
                           self.volume_limit)

        simulated_impact = volume_share ** 2 \
            * math.copysign(self.price_impact, order.direction) \
            * price
        impacted_price = price + simulated_impact

        if order.limit:
            # this is tricky! if an order with a limit price has reached
            # the limit price, we will try to fill the order. do not fill
            # these shares if the impacted price is worse than the limit
            # price. return early to avoid creating the transaction.

            # buy order is worse if the impacted price is greater than
            # the limit price. sell order is worse if the impacted price
            # is less than the limit price
            if (order.direction > 0 and impacted_price > order.limit) or \
                    (order.direction < 0 and impacted_price < order.limit):
                return

        return create_transaction(
            order.sid,
            dt,
            order,
            impacted_price,
            math.copysign(cur_volume, order.direction)
        )

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("VolumeShareSlippage saved state is too old.")

        self.__dict__.update(state)


class FixedSlippage(SlippageModel):

    def __init__(self, spread=0.0):
        """
        Use the fixed slippage model, which will just add/subtract
        a specified spread spread/2 will be added on buys and subtracted
        on sells per share
        """
        self.spread = spread

    def process_order(self, event, order):
        return create_transaction(
            event,
            order,
            event.price + (self.spread / 2.0 * order.direction),
            order.amount,
        )

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("FixedSlippage saved state is too old.")

        self.__dict__.update(state)
