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

import bisect
import numpy as np

DEFAULT_PER_SHARE_COST = 0.0075         # 0.75 cents per share
DEFAULT_MINIMUM_COST_PER_TRADE = 1.0    # $1 per trade


class PerShare(object):
    """
    Calculates a commission for a transaction based on a per
    share cost with an optional minimum cost per trade.
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
    """
    Calculates a commission for a transaction based on a per
    trade cost.
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
    """
    Calculates a commission for a transaction based on a per
    dollar cost.
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


class PerShareWithOptions(object):
    """
    Calculates a commission for a transaction based on a per
    share cost with an optional minimum cost per trade.
    Applies different set of fees to equities and underliers.
    """

    def __init__(self, equity_cost=DEFAULT_PER_SHARE_COST,
                 equity_min_trade_cost=DEFAULT_MINIMUM_COST_PER_TRADE,
                 option_costs=None,
                 option_min_trade_cost=DEFAULT_MINIMUM_COST_PER_TRADE,
                 equity_sids=None, option_sids=None):
        """
        Args:
            equity_cost: cost of an equity trade per-share
            equity_min_trade_cost: the minimum trade cost for equities
            option_costs: dictionary specifying per-share commissions in the
                          option commission tiers defined by IB.
                          For a description of IB option commission tiers, see
                          the link below. The dictionary is keyed by a 2d
                          tuple containing the minimum and maximum option
                          premium. Values of the dict are the per-contract
                          commission for an option with premium between these
                          two minimum and maximum values. For example, the
                          following dict:
                                tier_dict = {}
                                tier_dict[(0.10,  np.inf)] = 0.70
                                tier_dict[(0.05,    0.10)] = 0.50
                                tier_dict[(-np.inf, 0.05)] = 0.25
                          indicates that an option with a premium >= $0.10
                          has a per-contract commission of $0.70. An option
                          with a premium between $0.05 and $0.10 has a
                          commission of $0.50 per-contract, and an option
                          with a premium less than $0.05 has a per-contract
                          commission of $0.25.
            option_min_trade_cost: minimum trade cost for options
            equity_sids: list of integer sids to apply equity commissions
            option_sids: list of integer sids to apply option commissions

            https://gdcdyn.interactivebrokers.com/en/index.php?f=commission&p=options1
        """
        self.equity_cost = float(equity_cost)

        self.equity_min_trade_cost = None if equity_min_trade_cost is None\
            else float(equity_min_trade_cost)

        if option_costs is not None:
            assert isinstance(option_costs, dict)
        self.option_costs = option_costs
        self.option_min_trade_cost = None if option_min_trade_cost is None\
            else float(option_min_trade_cost)

        # check that all sids are integers
        assert isinstance(equity_sids, list)
        assert isinstance(option_sids, list)
        for sid in equity_sids:
            assert(isinstance(sid, int))
        for sid in option_sids:
            assert(isinstance(sid, int))

        # sort for later searching with bisect
        self.equity_sids = sorted(equity_sids)
        self.option_sids = sorted(option_sids)

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
        assert isinstance(transaction.sid, int)
        sid = transaction.sid

        if _list_index(self.equity_sids, sid) is not None:
            cost = self.equity_cost
            min_trade_cost = self.equity_min_trade_cost
        elif _list_index(self.option_sids, sid) is not None:
            cost = _lookup_option_cost(self.option_costs, transaction.price)
            min_trade_cost = self.option_min_trade_cost
        else:
            raise Exception('Unable to find sid = %i in either the'
                            ' equities or options.' % sid)

        commission = abs(transaction.amount * cost)
        if min_trade_cost is None:
            return cost, commission
        else:
            commission = max(commission, min_trade_cost)
            return abs(commission / transaction.amount), commission

    @staticmethod
    def ib_default_tier():
        """
        Interactive Brokers option commissions tier for < 10,000
        monthly contracts traded.
            https://gdcdyn.interactivebrokers.com/en/index.php?f=commission&p=options1
        """
        tier_dict = {}
        tier_dict[(0.10, np.inf)] = 0.70
        tier_dict[(0.05, 0.10)] = 0.50
        tier_dict[(-np.inf, 0.05)] = 0.25
        return tier_dict


def _lookup_option_cost(tier_dict, premium):
    """
    Returns the per share option commission for a given premium.
    """
    if tier_dict is None:
        return 0.0
    for (min_val, max_val) in tier_dict:
        if min_val <= premium < max_val:
            return tier_dict[(min_val, max_val)]
    raise Exception('Failed to lookup tiered option cost.')


def _list_index(a, x):
    """
    Find index of `x` in sorted list `a`.
    Returns None if `x` is not in `a`.
    """
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return None
