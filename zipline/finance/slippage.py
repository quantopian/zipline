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
import pytz
import math

from copy import copy
from functools import partial
from zipline.protocol import DATASOURCE_TYPE
import zipline.utils.math_utils as zp_math

from logbook import Processor


def check_order_triggers(order, event):
    """
    Given an order and a trade event, return a tuple of
    (stop_reached, limit_reached).
    For market orders, will return (False, False).
    For stop orders, limit_reached will always be False.
    For limit orders, stop_reached will always be False.

    Orders that have been triggered already (price targets reached),
    the order's current values are returned.
    """
    if order.triggered:
        return (order.stop_reached, order.limit_reached)

    stop_reached = False
    limit_reached = False
    # if the stop price is reached, simply set stop_reached
    if order.stop is not None:
        if (order.direction * (event.price - order.stop) <= 0):
            # convert stop -> limit or market
            stop_reached = True

    # if the limit price is reached, we execute this order at
    # (event.price + simulated_impact)
    # we skip this order with a continue when the limit is not reached
    if order.limit is not None:
        # if limit conditions not met, then continue
        if (order.direction * (event.price - order.limit) <= 0):
            limit_reached = True

    return (stop_reached, limit_reached)


def transact_stub(slippage, commission, event, open_orders):
    """
    This is intended to be wrapped in a partial, so that the
    slippage and commission models can be enclosed.
    """
    def inject_algo_dt(record):
        if not 'algo_dt' in record.extra:
            record.extra['algo_dt'] = event['dt']

    with Processor(inject_algo_dt).threadbound():

        transactions = slippage.simulate(event, open_orders)

        for transaction in transactions:
            if (
                transaction
                and not
                zp_math.tolerant_equals(transaction.amount, 0)
            ):
                direction = math.copysign(1, transaction.amount)
                per_share, total_commission = commission.calculate(transaction)
                transaction.price = transaction.price + (per_share * direction)
                transaction.commission = total_commission
        return transactions


def transact_partial(slippage, commission):
    return partial(transact_stub, slippage, commission)


class Transaction(object):

    def __init__(self, sid, amount, dt, price, order_id=None, commission=None):
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


def create_transaction(sid, amount, price, dt, order_id):

    txn = {
        'sid': sid,
        'amount': int(amount),
        'dt': dt,
        'price': price,
        'order_id': order_id
    }

    transaction = Transaction(**txn)
    return transaction


class VolumeShareSlippage(object):

    def __init__(self,
                 volume_limit=.25,
                 price_impact=0.1):

        self.volume_limit = volume_limit
        self.price_impact = price_impact

    def simulate(self, event, current_orders):

        dt = event.dt
        simulated_impact = 0.0
        max_volume = self.volume_limit * event.volume
        total_volume = 0

        txns = []
        for order in current_orders:

            open_amount = order.amount - order.filled

            if zp_math.tolerant_equals(open_amount, 0):
                continue

            order.check_triggers(event)
            if not order.triggered:
                continue

            # price impact accounts for the total volume of transactions
            # created against the current minute bar
            remaining_volume = max_volume - total_volume
            if (
                remaining_volume <= 0
                or
                zp_math.tolerant_equals(remaining_volume, 0)
            ):
                # we can't fill any more transactions
                return txns

            # the current order amount will be the min of the
            # volume available in the bar or the open amount.
            cur_amount = min(remaining_volume, abs(open_amount))
            cur_amount = cur_amount * order.direction
            # tally the current amount into our total amount ordered.
            # total amount will be used to calculate price impact
            total_volume = total_volume + order.direction * cur_amount

            volume_share = min(order.direction * (total_volume) / event.volume,
                               self.volume_limit)

            simulated_impact = (volume_share) ** 2 \
                * self.price_impact * order.direction * event.price

            if order.direction * cur_amount > 0:
                txn = create_transaction(
                    event.sid,
                    cur_amount,
                    # In the future, we may want to change the next line
                    # for limit pricing
                    event.price + simulated_impact,
                    dt.replace(tzinfo=pytz.utc),
                    order.id
                )

                txns.append(txn)

        return txns


class FixedSlippage(object):

    def __init__(self, spread=0.0):
        """
        Use the fixed slippage model, which will just add/subtract
        a specified spread spread/2 will be added on buys and subtracted
        on sells per share
        """
        self.spread = spread

    def simulate(self, event, orders):

        txns = []
        for order in orders:
            # TODO: what if we have 2 orders, one for 100 shares long,
            # and one for 100 shares short
            # such as in a hedging scenario?

            order.check_triggers(event)
            if not order.triggered:
                continue

            if zp_math.tolerant_equals(order.amount, 0):
                return txns

            txn = create_transaction(
                event.sid,
                order.amount,
                event.price + (self.spread / 2.0 * order.direction),
                event.dt.replace(tzinfo=pytz.utc),
                order.id
            )

            # mark the date of the order to match the transaction
            order.dt = event.dt
            txns.append(txn)
        return txns
