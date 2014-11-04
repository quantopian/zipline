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


class PerShare(object):
    """
    Calculates a commission for a transaction based on a per
    share cost with an optional minimum cost per trade.
    """

    def __init__(self, cost=0.03, min_trade_cost=None):
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

    def _get_state(self):
        state_dict = {}
        for k, v in self.__dict__.iteritems():
            if (not k.startswith('_')):
                state_dict[k] = v

        return 'PerShare', state_dict

    def _set_state(self, saved_state):
        self.__dict__.update(saved_state)


class PerTrade(object):
    """
    Calculates a commission for a transaction based on a per
    trade cost.
    """

    def __init__(self, cost=5.0):
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

    def _get_state(self):
        state_dict = {}
        for k, v in self.__dict__.iteritems():
            if (not k.startswith('_')):
                state_dict[k] = v

        return 'PerTrade', state_dict

    def _set_state(self, saved_state):
        self.__dict__.update(saved_state)


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

    def _get_state(self):
        state_dict = {}
        for k, v in self.__dict__.iteritems():
            if (not k.startswith('_')):
                state_dict[k] = v

        return 'PerDollar', state_dict

    def _set_state(self, saved_state):
        self.__dict__.update(saved_state)
