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

from six import iteritems, iterkeys
import pandas as pd
from pandas.tseries.tools import normalize_date
import numpy as np

from . utils.protocol_utils import Enum
from . utils.math_utils import nanstd, nanmean, nansum

from zipline.utils.algo_instance import get_algo_instance
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


def dividend_payment(data=None):
    """
    Take a dictionary whose values are in DIVIDEND_PAYMENT_FIELDS and return a
    series representing the payment of a dividend.

    Ids are assigned to each historical dividend in
    PerformanceTracker.update_dividends. They are guaranteed to be unique
    integers with the context of a single simulation. If @data is non-empty, a
    id is required to identify the historical dividend associated with this
    payment.

    Additionally, if @data is non-empty, either data['cash_amount'] should be
    nonzero or data['payment_sid'] should be an asset identifier and
    data['share_count'] should be nonzero.

    The returned Series is given its id value as a name so that concatenating
    payments results in a DataFrame indexed by id.  (Note, however, that the
    name value is not used to construct an index when this series is returned
    by function passed to `DataFrame.apply`.  In such a case, pandas preserves
    the index of the DataFrame on which `apply` is being called.)
    """
    return pd.Series(
        data=data,
        name=data['id'] if data is not None else None,
        index=DIVIDEND_PAYMENT_FIELDS,
        dtype=object,
    )


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


# class Order(Event):
#     pass


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

    Note: Many methods are analogues of dictionary because of historical
    usage of what this replaced as a dictionary subclass.
    """

    def __init__(self, data=None, data_portal=None):
        self._data = data or {}
        self._contains_override = None
        self._factor_matrix = None
        self._factor_matrix_expires = pd.Timestamp(0, tz='UTC')

        self.data_portal = data_portal or {}

    @property
    def factors(self):
        algo = get_algo_instance()
        today = normalize_date(algo.get_datetime())
        if today > self._factor_matrix_expires:
            self._factor_matrix, self._factor_matrix_expires = \
                algo.compute_factor_matrix(today)
        try:
            return self._factor_matrix.loc[today]
        except KeyError:
            # This happens if no assets passed our filters on a given day.
            return pd.DataFrame(
                index=[],
                columns=self._factor_matrix.columns,
            )

    def __contains__(self, name):
        return self.data_portal.is_currently_alive(name)

    def has_key(self, name):
        """
        DEPRECATED: __contains__ is preferred, but this method is for
        compatibility with existing algorithms.
        """
        return name in self

    def __setitem__(self, name, value):
        # No longer supported.
        pass

    def __getitem__(self, name):
        return self.data_portal.get_equity_price_view(name)

    def __delitem__(self, name):
        # No longer supported.
        pass

    def __iter__(self):
        for sid, data in iteritems(self._data):
            # Allow contains override to filter out sids.
            if sid in self:
                if len(data):
                    yield sid

    def iterkeys(self):
        # Allow contains override to filter out sids.
        return (sid for sid in iterkeys(self._data) if sid in self)

    def keys(self):
        # Allow contains override to filter out sids.
        return list(self.iterkeys())

    def itervalues(self):
        return (value for _sid, value in self.iteritems())

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        return ((sid, value) for sid, value
                in iteritems(self._data)
                if sid in self)

    def items(self):
        return list(self.iteritems())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self._data)
