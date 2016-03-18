#
# Copyright 2015 Quantopian, Inc.
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
from six import with_metaclass

from zipline.finance.transaction import create_transaction

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3


class LiquidityExceeded(Exception):
    pass


DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT = 0.025


class SlippageModel(with_metaclass(abc.ABCMeta)):
    def __init__(self):
        self._volume_for_bar = 0

    @property
    def volume_for_bar(self):
        return self._volume_for_bar

    @abc.abstractproperty
    def process_order(self, bar_data, price, volume, order):
        pass

    def simulate(self, bar_data, asset, orders_for_asset):
        self._volume_for_bar = 0
        volume = bar_data.current(asset, "volume")

        if volume == 0:
            return

        price = bar_data.current(asset, "price")
        dt = bar_data.current_dt

        for order in orders_for_asset:
            if order.open_amount == 0:
                continue

            order.check_triggers(price, dt)
            if not order.triggered:
                continue

            try:
                txn = self.process_order(bar_data, price, volume, order)
            except LiquidityExceeded:
                break

            if txn:
                self._volume_for_bar += abs(txn.amount)
                yield order, txn

    def __call__(self, bar_data, asset, current_orders):
        return self.simulate(bar_data, asset, current_orders)


class VolumeShareSlippage(SlippageModel):

    def __init__(self, volume_limit=DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT,
                 price_impact=0.1):

        self.volume_limit = volume_limit
        self.price_impact = price_impact

        super(VolumeShareSlippage, self).__init__()

    def __repr__(self):
        return """
{class_name}(
    volume_limit={volume_limit},
    price_impact={price_impact})
""".strip().format(class_name=self.__class__.__name__,
                   volume_limit=self.volume_limit,
                   price_impact=self.price_impact)

    def process_order(self, bar_data, price, volume, order):

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
            order,
            bar_data.current_dt,
            impacted_price,
            math.copysign(cur_volume, order.direction)
        )


class FixedSlippage(SlippageModel):

    def __init__(self, spread=0.0):
        """
        Use the fixed slippage model, which will just add/subtract
        a specified spread spread/2 will be added on buys and subtracted
        on sells per share
        """
        self.spread = spread

    def process_order(self, bar_data, price, volume, order):
        return create_transaction(
            order,
            bar_data.current_dt,
            price + (self.spread / 2.0 * order.direction),
            order.amount,
        )
