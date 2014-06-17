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

from zipline.utils import tradingcalendar


class ExecutionStyle(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class representing a modification to a standard order.
    """

    _exchange = None

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

    @property
    def exchange(self):
        """
        The exchange to which this order should be routed.
        """
        return self._exchange


class MarketOrder(ExecutionStyle):
    """
    Class encapsulating an order to be placed at the current market price.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, _is_buy):
        return None


class LimitOrder(ExecutionStyle):
    """
    Execution style representing an order to be executed at a price equal to or
    better than a specified limit price.
    """
    def __init__(self, limit_price, exchange=None):
        """
        Store the given price.
        """
        if limit_price < 0:
            raise ValueError("Can't place a limit with a negative price.")
        self.limit_price = limit_price
        self._exchange = exchange

    def get_limit_price(self, is_buy):
        return asymmetric_round_price_to_penny(self.limit_price, is_buy)

    def get_stop_price(self, _is_buy):
        return None


class StopOrder(ExecutionStyle):
    """
    Execution style representing an order to be placed once the market price
    reaches a specified stop price.
    """
    def __init__(self, stop_price, exchange=None):
        """
        Store the given price.
        """
        if stop_price < 0:
            raise ValueError(
                "Can't place a stop order with a negative price."
            )
        self.stop_price = stop_price
        self._exchange = exchange

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, is_buy):
        return asymmetric_round_price_to_penny(self.stop_price, not is_buy)


class StopLimitOrder(ExecutionStyle):
    """
    Execution style representing a limit order to be placed with a specified
    limit price once the market reaches a specified stop price.
    """
    def __init__(self, limit_price, stop_price, exchange=None):
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
        self._exchange = exchange

    def get_limit_price(self, is_buy):
        return asymmetric_round_price_to_penny(self.limit_price, is_buy)

    def get_stop_price(self, is_buy):
        return asymmetric_round_price_to_penny(self.stop_price, not is_buy)


class TimeInForceModel(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class representing when an order is valid

    Types supported by IB
        DAY: will cancel at EOD
        GTC: Good till cancelled
        IOC: Immediate-or-Cancel; Any portion not filled as soon as it becomes
             available in the market is canceled.
        GTD: Good till date
    """

    _time_in_force = None

    @abc.abstractmethod
    def get_good_till_dt(self):
        """
        Get the last valid datetime for this order
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_good_after_dt(self):
        """
        Get the first valid datetime for this order
        """
        raise NotImplemented

    @abc.abstractmethod
    def within_valid_date_range(self, dt):
        raise NotImplemented

    def __repr__(self):
        return str(self._time_in_force)


class DayOrder(TimeInForceModel):

    def __init__(self, algo_dt):
        self._time_in_force = 'DAY'
        self.good_after_dt = algo_dt
        open_and_closes = tradingcalendar.open_and_closes
        dt = tradingcalendar.canonicalize_datetime(self.good_after_dt)
        idx = open_and_closes.index.searchsorted(dt)
        self.good_till_dt = open_and_closes.ix[idx]['market_close']

    def get_good_till_dt(self):
        return self.good_till_dt

    def get_good_after_dt(self):
        return self.good_after_dt

    def within_valid_date_range(self, dt):
        return (dt >= self.good_after_dt) and (dt <= self.good_till_dt)


class GoodTillCancelled(TimeInForceModel):

    def __init__(self):
        self._time_in_force = 'GTC'

    def get_good_till_dt(self):
        return None

    def get_good_after_dt(self):
        return None

    def within_valid_date_range(self, dt):
        return True


class ImmediateOrCancel(TimeInForceModel):

    def __init__(self, algo_dt):
        self._time_in_force = 'IOC'
        self.good_after_dt = algo_dt

    def get_good_till_dt(self):
        # Use algo_dt + one time step? Hmmmm
        pass

    def get_good_after_dt(self):
        return self.good_after_dt

    def within_valid_date_range(self, dt):
        return (dt >= self.good_after_dt) and (dt <= self.good_till_dt)


class GoodTillDate(TimeInForceModel):

    def __init__(self, algo_dt, good_till_dt):
        self._time_in_force = 'GTD'
        self.good_after_dt = algo_dt
        self.good_till_dt = good_till_dt

    def get_good_till_dt(self):
        return self.good_till_dt

    def get_good_after_dt(self):
        return self.good_after_dt

    def within_valid_date_range(self, dt):
        return (dt >= self.good_after_dt) and (dt <= self.good_till_dt)


class GoodBetweenDates(TimeInForceModel):
    ''' Not supported by IB, but they have fields for good after/before  '''
    def __init__(self, good_after_dt, good_till_dt):
        self._time_in_force = ''
        self.good_after_dt = good_after_dt
        self.good_till_dt = good_till_dt

    def get_good_till_dt(self):
        return self.good_till_dt

    def get_good_after_dt(self):
        return self.good_after_dt

    def within_valid_date_range(self, dt):
        return (dt >= self.good_after_dt) and (dt <= self.good_till_dt)


def asymmetric_round_price_to_penny(price, prefer_round_down,
                                    diff=(0.0095 - .005)):
    """
    Asymmetric rounding function for adjusting prices to two places in a way
    that "improves" the price.  For limit prices, this means preferring to
    round down on buys and preferring to round up on sells.  For stop prices,
    it means the reverse.

    If prefer_round_down == True:
        When .05 below to .95 above a penny, use that penny.
    If prefer_round_down == False:
        When .95 below to .05 above a penny, use that penny.

    In math-speak:
    If prefer_round_down: [<X-1>.0095, X.0195) -> round to X.01.
    If not prefer_round_down: (<X-1>.0005, X.0105] -> round to X.01.
    """
    # Subtracting an epsilon from diff to enforce the open-ness of the upper
    # bound on buys and the lower bound on sells.  Using the actual system
    # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
    epsilon = float_info.epsilon * 10
    diff = diff - epsilon

    # relies on rounding half away from zero, unlike numpy's bankers' rounding
    rounded = round(price - (diff if prefer_round_down else -diff), 2)
    if zp_math.tolerant_equals(rounded, 0.0):
        return 0.0
    return rounded
