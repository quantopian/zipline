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
from warnings import warn

import pandas as pd

from .assets import Asset, Future
from .utils.enum import enum
from ._protocol import BarData  # noqa


class MutableView(object):
    """A mutable view over an "immutable" object.

    Parameters
    ----------
    ob : any
        The object to take a view over.
    """
    # add slots so we don't accidentally add attributes to the view instead of
    # ``ob``
    __slots__ = ('_mutable_view_ob',)

    def __init__(self, ob):
        object.__setattr__(self, '_mutable_view_ob', ob)

    def __getattr__(self, attr):
        return getattr(self._mutable_view_ob, attr)

    def __setattr__(self, attr, value):
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self._mutable_view_ob)


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
            self.__dict__.update(initial_values)

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


def _deprecated_getitem_method(name, attrs):
    """Create a deprecated ``__getitem__`` method that tells users to use
    getattr instead.

    Parameters
    ----------
    name : str
        The name of the object in the warning message.
    attrs : iterable[str]
        The set of allowed attributes.

    Returns
    -------
    __getitem__ : callable[any, str]
        The ``__getitem__`` method to put in the class dict.
    """
    attrs = frozenset(attrs)
    msg = (
        "'{name}[{attr!r}]' is deprecated, please use"
        " '{name}.{attr}' instead"
    )

    def __getitem__(self, key):
        """``__getitem__`` is deprecated, please use attribute access instead.
        """
        warn(msg.format(name=name, attr=key), DeprecationWarning, stacklevel=2)
        if key in attrs:
            return getattr(self, key)
        raise KeyError(key)

    return __getitem__


class Order(Event):
    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'order', {
            'dt',
            'sid',
            'amount',
            'stop',
            'limit',
            'id',
            'filled',
            'commission',
            'stop_reached',
            'limit_reached',
            'created',
        },
    )


def asset_multiplier(asset):
    return asset.multiplier if isinstance(asset, Future) else 1


class Portfolio(object):
    """The portfolio at a given time.

    Parameters
    ----------
    start_date : pd.Timestamp
        The start date for the period being recorded.
    capital_base : float
        The starting value for the portfolio. This will be used as the starting
        cash, current cash, and portfolio value.
    """

    def __init__(self, start_date=None, capital_base=0.0):
        self_ = MutableView(self)
        self_.cash_flow = 0.0
        self_.starting_cash = capital_base
        self_.portfolio_value = capital_base
        self_.pnl = 0.0
        self_.returns = 0.0
        self_.cash = capital_base
        self_.positions = Positions()
        self_.start_date = start_date
        self_.positions_value = 0.0
        self_.positions_exposure = 0.0

    @property
    def capital_used(self):
        return self.cash_flow

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Portfolio objects')

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'portfolio', {
            'capital_used',
            'starting_cash',
            'portfolio_value',
            'pnl',
            'returns',
            'cash',
            'positions',
            'start_date',
            'positions_value',
        },
    )

    @property
    def current_portfolio_weights(self):
        """
        Compute each asset's weight in the portfolio by calculating its held
        value divided by the total value of all positions.

        Each equity's value is its price times the number of shares held. Each
        futures contract's value is its unit price times number of shares held
        times the multiplier.
        """
        position_values = pd.Series({
            asset: (
                position.last_sale_price *
                position.amount *
                asset_multiplier(asset)
            )
            for asset, position in self.positions.items()
        })
        return position_values / self.portfolio_value


class Account(object):
    """
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    """

    def __init__(self, portfolio):
        self_ = MutableView(self)
        self_.settled_cash = 0.0
        self_.accrued_interest = 0.0
        self_.buying_power = float('inf')
        self_.equity_with_loan = 0.0
        self_.total_positions_value = 0.0
        self_.total_positions_exposure = 0.0
        self_.regt_equity = 0.0
        self_.regt_margin = float('inf')
        self_.initial_margin_requirement = 0.0
        self_.maintenance_margin_requirement = 0.0
        self_.available_funds = 0.0
        self_.excess_liquidity = 0.0
        self_.cushion = 0.0
        self_.day_trades_remaining = float('inf')
        self_.leverage = 0.0
        self_.net_leverage = 0.0
        self_.net_liquidation = 0.0

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Account objects')

    def __repr__(self):
        return "Account({0})".format(self.__dict__)

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'account', {
            'settled_cash',
            'accrued_interest',
            'buying_power',
            'equity_with_loan',
            'total_positions_value',
            'total_positions_exposure',
            'regt_equity',
            'regt_margin',
            'initial_margin_requirement',
            'maintenance_margin_requirement',
            'available_funds',
            'excess_liquidity',
            'cushion',
            'day_trades_remaining',
            'leverage',
            'net_leverage',
            'net_liquidation',
        },
    )


class InnerPosition(object):
    """The real values of a position.

    This exists to be owned by both a
    :class:`zipline.finance.position.Position` and a
    :class:`zipline.protocol.Position` at the same time without a cycle.
    """
    __slots__ = (
        'asset',
        'amount',
        'cost_basis',
        'last_sale_price',
        'last_sale_date',
    )

    def __init__(self,
                 asset,
                 amount=0,
                 cost_basis=0.0,
                 last_sale_price=0.0,
                 last_sale_date=None):
        self.asset = asset
        self.amount = amount
        self.cost_basis = cost_basis  # per share
        self.last_sale_price = last_sale_price
        self.last_sale_date = last_sale_date

    def __repr__(self):
        return (
            '%s(asset=%r, amount=%r, cost_basis=%r,'
            ' last_sale_price=%r, last_sale_date=%r)' % (
                type(self).__name__,
                self.asset,
                self.amount,
                self.cost_basis,
                self.last_sale_price,
                self.last_sale_date,
            )
        )


class Position(object):
    __slots__ = ('_underlying_position',)

    def __init__(self, underlying_position):
        object.__setattr__(self, '_underlying_position', underlying_position)

    def __getattr__(self, attr):
        return getattr(self._underlying_position, attr)

    def __setattr__(self, attr, value):
        raise AttributeError('cannot mutate Position objects')

    @property
    def sid(self):
        # for backwards compatibility
        return self.asset

    def __repr__(self):
        return 'Position(%r)' % {
            k: getattr(self, k)
            for k in (
                'asset',
                'amount',
                'cost_basis',
                'last_sale_price',
                'last_sale_date',
            )
        }

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'position', {
            'sid',
            'amount',
            'cost_basis',
            'last_sale_price',
            'last_sale_date',
        },
    )


# Copied from Position and renamed.  This is used to handle cases where a user
# does something like `context.portfolio.positions[100]` instead of
# `context.portfolio.positions[sid(100)]`.
class _DeprecatedSidLookupPosition(object):
    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0
        self.last_sale_date = None

    def __repr__(self):
        return "_DeprecatedSidLookupPosition({0})".format(self.__dict__)

    # If you are adding new attributes, don't update this set. This method
    # is deprecated to normal attribute access so we don't want to encourage
    # new usages.
    __getitem__ = _deprecated_getitem_method(
        'position', {
            'sid',
            'amount',
            'cost_basis',
            'last_sale_price',
            'last_sale_date',
        },
    )


class Positions(dict):
    def __missing__(self, key):
        if isinstance(key, Asset):
            return Position(InnerPosition(key))
        elif isinstance(key, int):
            warn("Referencing positions by integer is deprecated."
                 " Use an asset instead.")
        else:
            warn("Position lookup expected a value of type Asset but got {0}"
                 " instead.".format(type(key).__name__))

        return _DeprecatedSidLookupPosition(key)
