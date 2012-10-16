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
from datetime import timedelta

import pytz
import math

from functools import partial

from zipline.utils.protocol_utils import ndict


def transact_stub(slippage, commission, event, open_orders):
    """
    This is intended to be wrapped in a partial, so that the
    slippage and commission models can be enclosed.
    """
    transaction = slippage.simulate(event, open_orders)
    if transaction and transaction.amount != 0:
        direction = abs(transaction.amount) / transaction.amount
        per_share, total_commission = commission.calculate(transaction)
        transaction.price = transaction.price + (per_share * direction)
        transaction.commission = total_commission
    return transaction


def transact_partial(slippage, commission):
    return partial(transact_stub, slippage, commission)


def create_transaction(sid, amount, price, dt):

    txn = {'sid': sid,
           'amount': int(amount),
           'dt': dt,
           'price': price,
          }

    transaction = ndict(txn)
    return transaction


class VolumeShareSlippage(object):

    def __init__(self,
                 volume_limit=.25,
                 price_impact=0.1,
                 delay=timedelta(minutes=1)):

        self.volume_limit = volume_limit
        self.price_impact = price_impact
        self.delay = delay

    def simulate(self, event, open_orders):

        if(event.volume == 0):
            #there are zero volume events bc some stocks trade
            #less frequently than once per minute.
            return None

        if event.sid in open_orders:
            orders = open_orders[event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
            # Only use orders for the current day or before
            current_orders = filter(
                lambda o: o.dt + self.delay <= event.dt,
                orders)
        else:
            return None

        dt = event.dt
        total_order = 0
        simulated_amount = 0
        simulated_impact = 0.0
        direction = 1.0

        for order in current_orders:

            open_amount = order.amount - order.filled

            if(open_amount != 0):
                direction = open_amount / math.fabs(open_amount)
            else:
                direction = 1

            desired_order = total_order + open_amount

            volume_share = min(direction * (desired_order) / event.volume,
                               self.volume_limit)
            simulated_amount = int(volume_share * event.volume * direction)
            simulated_impact = (volume_share) ** 2 \
            * self.price_impact * direction * event.price

            order.filled += (simulated_amount - total_order)
            total_order = simulated_amount

            # we cap the volume share at configured % of a trade
            if volume_share == self.volume_limit:
                break

        filled_orders = [x for x in orders
                         if abs(x.amount - x.filled) > 0
                         and x.dt.day >= event.dt.day]

        open_orders[event.sid] = filled_orders

        if simulated_amount != 0:
            return create_transaction(
                event.sid,
                simulated_amount,
                event.price + simulated_impact,
                dt.replace(tzinfo=pytz.utc),
            )


class FixedSlippage(object):

    def __init__(self, spread=0.0):
        """
        Use the fixed slippage model, which will just add/subtract
        a specified spread spread/2 will be added on buys and subtracted
        on sells per share
        """
        self.spread = spread

    def simulate(self, event, open_orders):
        if event.sid in open_orders:
            orders = open_orders[event.sid]
            orders = sorted(orders, key=lambda o: o.dt)
        else:
            return None

        amount = 0
        for order in orders:
            amount += order.amount

        if(amount == 0):
            return

        direction = amount / math.fabs(amount)

        txn = create_transaction(
            event.sid,
            amount,
            event.price + (self.spread / 2.0 * direction),
            event.dt
        )

        open_orders[event.sid] = []

        return txn
