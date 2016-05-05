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

DEFAULT_PER_SHARE_COST = 0.0075         # 0.75 cents per share
DEFAULT_MINIMUM_COST_PER_TRADE = 1.0    # $1 per trade


class PerShare(object):
    """Calculates a commission for a transaction based on a per
    share cost with an optional minimum cost per trade.

    Parameters
    ----------
    cost : float, optional
        The amount of commissions paid per share traded.
    min_trade_cost : optional
        The minimum amount of commisions paid per trade.
    """

    def __init__(self,
                 cost=DEFAULT_PER_SHARE_COST,
                 min_trade_cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        """
        Cost parameter is the cost of a trade per-share. $0.03
        means three cents per share, which is a very conservative
        (quite high) for per share costs.
        min_trade_cost parameter is the minimum trade cost
        regardless of the number of shares traded (e.g. $1.00).
        """
        self.cost = float(cost)
        self.min_trade_cost = None if min_trade_cost is None\
            else float(min_trade_cost)

    def __repr__(self):
        return "{class_name}(cost={cost}, min trade cost={min_trade_cost})"\
            .format(class_name=self.__class__.__name__,
                    cost=self.cost,
                    min_trade_cost=self.min_trade_cost)

    def calculate(self, transaction):
        """
        returns a tuple of:
        (per share commission, total transaction commission)
        """
        commission = abs(transaction.amount * self.cost)
        if self.min_trade_cost is None:
            return self.cost, commission
        else:
            commission = max(commission, self.min_trade_cost)
            return abs(commission / transaction.amount), commission


class PerTrade(object):
    """Calculates a commission for a transaction based on a per
    trade cost.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commisions paid per trade.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        """
        Cost parameter is the cost of a trade, regardless of
        share count. $5.00 per trade is fairly typical of
        discount brokers.
        """
        # Cost needs to be floating point so that calculation using division
        # logic does not floor to an integer.
        self.cost = float(cost)

    def calculate(self, transaction):
        """
        returns a tuple of:
        (per share commission, total transaction commission)
        """
        if transaction.amount == 0:
            return 0.0, 0.0

        return abs(self.cost / transaction.amount), self.cost


class PerDollar(object):
    """Calculates a commission for a transaction based on a per
    dollar cost.

    Parameters
    ----------
    cost : float, optional
        The amount of commissions paid per dollar traded.
    """

    def __init__(self, cost=0.0015):
        """
        Cost parameter is the cost of a trade per-dollar. 0.0015
        on $1 million means $1,500 commission (=1,000,000 x 0.0015)
        """
        self.cost = float(cost)

    def __repr__(self):
        return "{class_name}(cost={cost})".format(
            class_name=self.__class__.__name__,
            cost=self.cost)

    def calculate(self, transaction):
        """
        returns a tuple of:
        (per share commission, total transaction commission)
        """
        cost_per_share = transaction.price * self.cost
        return cost_per_share, abs(transaction.amount) * cost_per_share
