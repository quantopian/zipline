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
from copy import copy
import math

from six import with_metaclass
from zipline.finance.transaction import create_transaction

from zipline.utils.serialization_utils import VERSION_LABEL


class LiquidityExceeded(Exception):
    pass


class SlippageModel(with_metaclass(abc.ABCMeta)):
    def __init__(self):
        self.data_portal = None
        self._volume_for_bar = 0

    @property
    def volume_for_bar(self):
        return self._volume_for_bar

    @abc.abstractproperty
    def process_order(self, order, dt):
        pass

    def simulate(self, current_orders, dt):
        self._volume_for_bar = 0

        for order in current_orders:
            if order.open_amount == 0:
                continue

            price = self.data_portal.get_current_price_data(
                order.sid, 'close', dt)
            order.check_triggers(price, dt)
            if not order.triggered:
                continue

            try:
                txn = self.process_order(order, dt)
            except LiquidityExceeded:
                break

            if txn:
                self._volume_for_bar += abs(txn.amount)
                yield order, txn

    def __call__(self, current_orders, dt, **kwargs):
        return self.simulate(current_orders, dt, **kwargs)


class VolumeShareSlippage(SlippageModel):

    def __init__(self, volume_limit=0.25, price_impact=0.1):
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

    def process_order(self, order, dt):
        volume = self.data_portal.get_current_price_data(
            order.sid, 'volume', dt)
        price = self.data_portal.get_current_price_data(
            order.sid, 'close', dt)
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

    def process_order(self, order, dt):
        price = self.data_portal.get_current_price_data(order.sid, "close")

        return create_transaction(
            order.sid,
            dt,
            order,
            price + (self.spread / 2.0 * order.direction),
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
