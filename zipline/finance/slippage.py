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


class CloseStage(object):
    """Simulate bar as 3 sequential stages Open, HighLow, Close at which orders
    can be triggered or executed. A StopLimit order can be triggered at
    one stage and executed at a later stage if the slippage model supports
    multiple stages. This fits well with tracking volume so that trades at
    later stages can be rejected and the ones in earlier executed.
    """
    limits_enhance = True

    def get_trigger_field(self, order):
        """Determine which price field to use for deciding order triggered
         state. This field is not necesarily the one execution price is
        based on"""
        return 'close'


class HLStage(CloseStage):
    limits_enhance = False

    def get_trigger_field(self, order):
        if order.stop is None:
            if order.limit is None:
                return None
            else:
                return 'low' if order.direction > 0 else 'high'
        else:
            return 'high' if order.direction > 0 else 'low'


class OpenStage(object):
    limits_enhance = True

    def get_trigger_field(self, order):
        return 'open'


class SlippageModel(with_metaclass(abc.ABCMeta)):
    """Abstract interface for defining a slippage model.
    """
    def __init__(self):
        self._volume_for_bar = 0
        self.bar_stages = CloseStage(),

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
        if data.current(asset, "volume") == 0:
            return

        dt = data.current_dt
        for order in orders_for_asset:
            order.stop_stage = None

        for self.bar_stage in self.bar_stages:
            for order in orders_for_asset:
                if order.open_amount == 0:
                    continue

                trigger_field = self.bar_stage.get_trigger_field(order)
                if trigger_field is None:
                    continue
                price = data.current(asset, trigger_field)
                order.check_triggers(price, dt)
                # try to execute limit in the bar that
                # triggered stop
                if order.stop_ghost is not None:
                    order.stop_stage = self.bar_stage
                    if not self.bar_stage.limits_enhance:
                        order.check_triggers(order.stop_ghost, dt)

                if not order.triggered:
                    order.stop_ghost = None
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

    def get_impact_base(self, data, order):
        impact_field = 'close' if order.stop or order.stop_ghost \
            else self.bar_stage.get_trigger_field(order)
        return data.current(order.asset, impact_field)

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

        impact_base = self.get_impact_base(data, order)
        if impact_base is None:
            return None, None
        simulated_impact = volume_share ** 2 \
            * math.copysign(self.price_impact, order.direction) \
            * impact_base
        impacted_price = impact_base + simulated_impact

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
                # TODO: if impacted_price is worse than limit, solve the
                # impacted_price equation for impacted_price = limit and find
                # the volume that can be fulfilled within the limit and
                # return it
                return None, None

        use_limit_price = order.limit and (
            order.stop_stage is not self.bar_stage
            if order.stop_stage
            else not self.bar_stage.limits_enhance)
        order.stop_ghost = None  # ignore for later bars
        return (
            order.limit if use_limit_price else impacted_price,
            math.copysign(cur_volume, order.direction)
        )


class HLCVolumeSlippage(VolumeShareSlippage):
    """Considers High, Low for limit and stop orders. Execute limit orders at
     limit price instead of close price. Market orders execute at Close."""

    def __init__(self):
        super(HLCVolumeSlippage, self).__init__()
        self.bar_stages = HLStage(), CloseStage()

    def get_impact_base(self, data, order):
        # prevent triggered stop orders far away from bar range to execute at
        # impossible favorable price
        stop = order.stop or order.stop_ghost
        if stop and not self.bar_stage.limits_enhance:
            trigger_field = 'low' if order.direction > 0 else 'high'
            price = data.current(order.asset, trigger_field)
            is_possible = (stop >= price) if order.direction > 0 \
                else (stop <= price)
            return stop if is_possible else None
        else:
            trigger_field = self.bar_stage.get_trigger_field(order)
            price = data.current(order.asset, trigger_field)
            return price


class OHLVolumeSlippage(VolumeShareSlippage):
    """Considers all OHLC prices. Market orders execute at open price. Limits
     execute at market price if marketable else at limit if between low and
      high. Stop orders execute at open or stop price"""
    def __init__(self):
        super(OHLVolumeSlippage, self).__init__()
        self.bar_stages = OpenStage(), HLStage(), CloseStage()

    def get_impact_base(self, data, order):
        return not self.bar_stage.limits_enhance \
            and (order.stop or order.stop_ghost) \
            or data.current(order.asset,
                            self.bar_stage.get_trigger_field(order))


class FixedSlippage(SlippageModel):
    """Model slippage as a fixed spread.

    Parameters
    ----------
    spread : float, optional
        spread / 2 will be added to buys and subtracted from sells.
    """

    def __init__(self, spread=0.0):
        self.spread = spread
        super(FixedSlippage, self).__init__()

    def process_order(self, data, order):
        price = data.current(order.asset, "close")

        return (
            price + (self.spread / 2.0 * order.direction),
            order.amount
        )
