#
# Copyright 2013 Quantopian, Inc.
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
import pandas as pd

from .utils.enum import enum

from zipline._protocol import BarData  # noqa


# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = enum(
    'AS_TRADED_EQUITY',
    'MERGER',
    'SPLIT',
    'DIVIDEND',
    'TRADE',
    'TRANSACTION',
    'ORDER',
    'EMPTY',
    'DONE',
    'CUSTOM',
    'BENCHMARK',
    'COMMISSION',
    'CLOSE_POSITION'
)

# Expected fields/index values for a dividend Series.
DIVIDEND_FIELDS = [
    'declared_date',
    'ex_date',
    'gross_amount',
    'net_amount',
    'pay_date',
    'payment_sid',
    'ratio',
    'sid',
]
# Expected fields/index values for a dividend payment Series.
DIVIDEND_PAYMENT_FIELDS = [
    'id',
    'payment_sid',
    'cash_amount',
    'share_count',
]


class Event(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__ = initial_values

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        delattr(self, name)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        return hasattr(other, '__dict__') and self.__dict__ == other.__dict__

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_series(self, index=None):
        return pd.Series(self.__dict__, index=index)


class Order(Event):
    pass


class Portfolio(object):

    def __init__(self):
        self.capital_used = 0.0
        self.starting_cash = 0.0
        self.portfolio_value = 0.0
        self.pnl = 0.0
        self.returns = 0.0
        self.cash = 0.0
        self.positions = Positions()
        self.start_date = None
        self.positions_value = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)


class Account(object):
    '''
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    '''

    def __init__(self):
        self.settled_cash = 0.0
        self.accrued_interest = 0.0
        self.buying_power = float('inf')
        self.equity_with_loan = 0.0
        self.total_positions_value = 0.0
        self.regt_equity = 0.0
        self.regt_margin = float('inf')
        self.initial_margin_requirement = 0.0
        self.maintenance_margin_requirement = 0.0
        self.available_funds = 0.0
        self.excess_liquidity = 0.0
        self.cushion = 0.0
        self.day_trades_remaining = float('inf')
        self.leverage = 0.0
        self.net_leverage = 0.0
        self.net_liquidation = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Account({0})".format(self.__dict__)


class Position(object):

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0
        self.last_sale_date = None

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Position({0})".format(self.__dict__)


class Positions(dict):

    def __missing__(self, key):
        pos = Position(key)
        return pos
