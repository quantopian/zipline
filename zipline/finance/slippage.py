#
# Copyright 2017 Quantopian, Inc.
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
from six import with_metaclass, iteritems

from pandas import isnull

from zipline.finance.transaction import create_transaction

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3


class LiquidityExceeded(Exception):
    pass


DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT = 0.025


def fill_price_worse_than_limit_price(fill_price, order):
    """
    Checks whether the fill price is worse than the order's limit price.

    Parameters
    ----------
    fill_price: float
        The price to check.

    order: zipline.finance.order.Order
        The order whose limit price to check.

    Returns
    -------
    bool: Whether the fill price is above the limit price (for a buy) or below
    the limit price (for a sell).
    """
    if order.limit:
        # this is tricky! if an order with a limit price has reached
        # the limit price, we will try to fill the order. do not fill
        # these shares if the impacted price is worse than the limit
        # price. return early to avoid creating the transaction.

        # buy order is worse if the impacted price is greater than
        # the limit price. sell order is worse if the impacted price
        # is less than the limit price
        if (order.direction > 0 and fill_price > order.limit) or \
                (order.direction < 0 and fill_price < order.limit):
            return True

    return False


class SlippageModel(with_metaclass(abc.ABCMeta)):
    """Abstract interface for defining a slippage model.
    """
    def __init__(self):
        self._volume_for_bar = 0

    @property
    def volume_for_bar(self):
        return self._volume_for_bar

    @abc.abstractproperty
    def process_order(self, data, order):
        """Process how orders get filled.

        Parameters
        ----------
        data : BarData
            The data for the given bar.
        order : Order
            The order to simulate.

        Returns
        -------
        execution_price : float
            The price to execute the trade at.
        execution_volume : int
            The number of shares that could be filled. This may not be all
            the shares ordered in which case the order will be filled over
            multiple bars.
        """
        pass

    def simulate(self, data, asset, orders_for_asset):
        self._volume_for_bar = 0
        volume = data.current(asset, "volume")

        if volume == 0:
            return

        # can use the close price, since we verified there's volume in this
        # bar.
        price = data.current(asset, "close")

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END
        dt = data.current_dt

        for order in orders_for_asset:
            if order.open_amount == 0:
                continue

            order.check_triggers(price, dt)
            if not order.triggered:
                continue

            txn = None

            try:
                execution_price, execution_volume = \
                    self.process_order(data, order)

                if execution_price is not None:
                    txn = create_transaction(
                        order,
                        data.current_dt,
                        execution_price,
                        execution_volume
                    )

            except LiquidityExceeded:
                break

            if txn:
                self._volume_for_bar += abs(txn.amount)
                yield order, txn

    def __call__(self, bar_data, asset, current_orders):
        return self.simulate(bar_data, asset, current_orders)

    def __eq__(self, other):
        return self.asdict() == other.asdict()

    def __hash__(self):
        return hash((
            type(self),
            tuple(sorted(iteritems(self.asdict())))
        ))

    def asdict(self):
        return self.__dict__


class VolumeShareSlippage(SlippageModel):
    """Model slippage as a function of the volume of shares traded.
    """

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

    def process_order(self, data, order):
        volume = data.current(order.asset, "volume")

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
            return None, None

        # tally the current amount into our total amount ordered.
        # total amount will be used to calculate price impact
        total_volume = self.volume_for_bar + cur_volume

        volume_share = min(total_volume / volume,
                           self.volume_limit)

        price = data.current(order.asset, "close")

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END

        simulated_impact = volume_share ** 2 \
            * math.copysign(self.price_impact, order.direction) \
            * price
        impacted_price = price + simulated_impact

        if fill_price_worse_than_limit_price(impacted_price, order):
            return None, None

        return (
            impacted_price,
            math.copysign(cur_volume, order.direction)
        )


class FixedSlippage(SlippageModel):
    """Model slippage as a fixed spread.

    Parameters
    ----------
    spread : float, optional
        spread / 2 will be added to buys and subtracted from sells.
    """

    def __init__(self, spread=0.0):
        self.spread = spread

    def process_order(self, data, order):
        price = data.current(order.asset, "close")

        return (
            price + (self.spread / 2.0 * order.direction),
            order.amount
        )
