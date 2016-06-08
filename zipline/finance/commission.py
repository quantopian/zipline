#
# Copyright 2016 Quantopian, Inc.
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

from abc import abstractmethod
from six import with_metaclass

DEFAULT_PER_SHARE_COST = 0.0075         # 0.75 cents per share
DEFAULT_MINIMUM_COST_PER_TRADE = 1.0    # $1 per trade


class CommissionModel(with_metaclass(abc.ABCMeta)):
    """
    Abstract commission model interface.

    Commission models are responsible for accepting order/transaction pairs and
    calculating how much commission should be charged to an algorithm's account
    on each transaction.
    """

    @abstractmethod
    def calculate(self, order, transaction):
        """
        Calculate the amount of commission to charge on ``order`` as a result
        of ``transaction``.

        Parameters
        ----------
        order : zipline.finance.order.Order
            The order being processed.

            The ``commission`` field of ``order`` is a float indicating the
            amount of commission already charged on this order.

        transaction : zipline.finance.transaction.Transaction
            The transaction being processed. A single order may generate
            multiple transactions if there isn't enough volume in a given bar
            to fill the full amount requested in the order.

        Returns
        -------
        amount_charged : float
            The additional commission, in dollars, that we should attribute to
            this order.
        """
        raise NotImplementedError('calculate')


class PerShare(CommissionModel):
    """
    Calculates a commission for a transaction based on a per share cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost : float, optional
        The amount of commissions paid per share traded.
    min_trade_cost : optional
        The minimum amount of commissions paid per trade.
    """

    def __init__(self,
                 cost=DEFAULT_PER_SHARE_COST,
                 min_trade_cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        self.cost_per_share = float(cost)
        self.min_trade_cost = min_trade_cost

    def __repr__(self):
        return "{class_name}(cost_per_share={cost_per_share}, " \
               "min_trade_cost={min_trade_cost})" \
            .format(class_name=self.__class__.__name__,
                    cost_per_share=self.cost_per_share,
                    min_trade_cost=self.min_trade_cost)

    def calculate(self, order, transaction):
        """
        If there is a minimum commission:
            If the order hasn't had a commission paid yet, pay the minimum
            commission.

            If the order has paid a commission, start paying additional
            commission once the minimum commission has been reached.

        If there is no minimum commission:
            Pay commission based on number of shares in the transaction.
        """
        additional_commission = abs(transaction.amount * self.cost_per_share)

        if self.min_trade_cost is None:
            # no min trade cost, so just return the cost for this transaction
            return additional_commission

        if order.commission == 0:
            # no commission paid yet, pay at least the minimum
            return max(self.min_trade_cost, additional_commission)
        else:
            # we've already paid some commission, so figure out how much we
            # would be paying if we only counted per share.
            per_share_total = \
                (order.filled * self.cost_per_share) + additional_commission

            if per_share_total < self.min_trade_cost:
                # if we haven't hit the minimum threshold yet, don't pay
                # additional commission
                return 0
            else:
                # we've exceeded the threshold, so pay more commission.
                return per_share_total - order.commission


class PerTrade(CommissionModel):
    """
    Calculates a commission for a transaction based on a per trade cost.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commissions paid per trade.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_TRADE):
        """
        Cost parameter is the cost of a trade, regardless of share count.
        $5.00 per trade is fairly typical of discount brokers.
        """
        # Cost needs to be floating point so that calculation using division
        # logic does not floor to an integer.
        self.cost = float(cost)

    def calculate(self, order, transaction):
        """
        If the order hasn't had a commission paid yet, pay the fixed
        commission.
        """
        if order.commission == 0:
            # if the order hasn't had a commission attributed to it yet,
            # that's what we need to pay.
            return self.cost
        else:
            # order has already had commission attributed, so no more
            # commission.
            return 0.0


class PerDollar(CommissionModel):
    """
    Calculates a commission for a transaction based on a per trade cost.

    Parameters
    ----------
    cost : float
        The flat amount of commissions paid per trade.
    """

    def __init__(self, cost=0.0015):
        """
        Cost parameter is the cost of a trade per-dollar. 0.0015
        on $1 million means $1,500 commission (=1M * 0.0015)
        """
        self.cost_per_dollar = float(cost)

    def __repr__(self):
        return "{class_name}(cost_per_dollar={cost})".format(
            class_name=self.__class__.__name__,
            cost=self.cost_per_dollar)

    def calculate(self, order, transaction):
        """
        Pay commission based on dollar value of shares.
        """
        cost_per_share = transaction.price * self.cost_per_dollar
        return abs(transaction.amount) * cost_per_share
