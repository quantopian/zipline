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
from copy import copy

import pandas as pd

from . utils.protocol_utils import Enum

from pandas.tslib import normalize_date
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = Enum(
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

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        # Have to convert to primitive dict
        state_dict['positions'] = dict(self.positions)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Portfolio saved state is too old.")

        self.positions = Positions()
        self.positions.update(state.pop('positions'))

        self.__dict__.update(state)


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

    def __getstate__(self):

        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Account saved state is too old.")

        self.__dict__.update(state)


class Position(object):

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Position({0})".format(self.__dict__)

    def __getstate__(self):
        state_dict = copy(self.__dict__)

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION

        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("Protocol Position saved state is too old.")

        self.__dict__.update(state)


class Positions(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos


class BarData(object):
    """
    Holds the event data for all sids for a given dt.

    This is what is passed as `data` to the `handle_data` function.
    """

    def __init__(self, data_portal, simulator):
        self.data_portal = data_portal or {}
        self.simulator = simulator
        self._views = {}

    @property
    def simulation_dt(self):
        return self.simulator.simulation_dt

    def _get_equity_price_view(self, asset):
        """
        Returns a DataPortalSidView for the given asset.  Used to support the
        data[sid(N)] public API.  Not needed if DataPortal is used standalone.

        Parameters
        ----------
        asset : Asset
            Asset that is being queried.

        Returns
        -------
        DataPortalSidView: Accessor into the given asset's data.
        """
        try:
            view = self._views[asset]
        except KeyError:
            view = self._views[asset] = \
                SidView(asset, self.data_portal, self)

        return view

    def __getitem__(self, name):
        return self._get_equity_price_view(name)

    def __iter__(self):
        raise TypeError('%r object is not iterable'
                        % self.__class__.__name__)

    @property
    def fetcher_assets(self):
        return self.data_portal.get_fetcher_assets(
            normalize_date(self.simulation_dt)
        )


class SidView(object):
    def __init__(self, asset, data_portal, bar_data):
        self.asset = asset
        self.data_portal = data_portal
        self.bar_data = bar_data

    def __getattr__(self, column):
        return self.data_portal.get_spot_value(
            self.asset, column, self.bar_data.simulation_dt)

    def __contains__(self, column):
        return self.data_portal.contains(self.asset, column)

    def __getitem__(self, column):
        return self.__getattr__(column)
