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

import abc

from sys import float_info

from six import with_metaclass

import zipline.utils.math_utils as zp_math


def round_for_minimum_price_variation(x, is_buy,
                                      diff=(0.0095 - .005)):
    """
    On an order to buy, between .05 below to .95 above a penny, use that penny.
    On an order to sell, between .95 below to .05 above a penny, use that
    penny.
    buy: [<X-1>.0095, X.0195) -> round to X.01,
    sell: (<X-1>.0005, X.0105] -> round to X.01
    """
    # Subtracting an epsilon from diff to enforce the open-ness of the upper
    # bound on buys and the lower bound on sells.  Using the actual system
    # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
    epsilon = float_info.epsilon * 10
    diff = diff - epsilon

    # relies on rounding half away from zero, unlike numpy's bankers' rounding
    rounded = round(x - (diff if is_buy else -diff), 2)
    if zp_math.tolerant_equals(rounded, 0.0):
        return 0.0
    return rounded


class ExecutionStyle(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class representing a modification to a standard order.
    """

    @abc.abstractmethod
    def get_limit_price(self, is_buy):
        """
        Get the limit price for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_stop_price(self, is_buy):
        """
        Get the stop price for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplemented


class MarketOrder(ExecutionStyle):
    """
    Class encapsulating an order to be placed at the current market price.
    """

    def __init__(self):
        pass

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, _is_buy):
        return None


class LimitOrder(ExecutionStyle):
    """
    Execution style representing an order to be executed at a price equal to or
    better than a specified limit price.
    """
    def __init__(self, limit_price):
        """
        Store the given price.
        """
        if limit_price < 0:
            raise ValueError("Can't place a limit with a negative price.")
        self.limit_price = limit_price

    def get_limit_price(self, is_buy):
        return round_for_minimum_price_variation(self.limit_price, is_buy)

    def get_stop_price(self, _is_buy):
        return None


class StopOrder(ExecutionStyle):
    """
    Execution style representing an order to be placed once the market price
    reaches a specified stop price.
    """
    def __init__(self, stop_price):
        """
        Store the given price.
        """
        if stop_price < 0:
            raise ValueError(
                "Can't place a stop order with a negative price."
            )
        self.stop_price = stop_price

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, is_buy):
        return round_for_minimum_price_variation(self.stop_price, is_buy)


class StopLimitOrder(ExecutionStyle):
    """
    Execution style representing a limit order to be placed with a specified
    limit price once the market reaches a specified stop price.
    """
    def __init__(self, limit_price, stop_price):
        """
        Store the given prices
        """
        if limit_price < 0:
            raise ValueError(
                "Can't place a limit with a negative price."
            )
        if stop_price < 0:
            raise ValueError(
                "Can't place a stop order with a negative price."
            )

        self.limit_price = limit_price
        self.stop_price = stop_price

    def get_limit_price(self, is_buy):
        return round_for_minimum_price_variation(self.limit_price, is_buy)

    def get_stop_price(self, is_buy):
        return round_for_minimum_price_variation(self.stop_price, is_buy)
