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
import pandas as pd

from .assets import Asset
from enum import IntEnum
from ._protocol import BarData, InnerPosition  # noqa


class MutableView:
    """A mutable view over an "immutable" object.

    Parameters
    ----------
    ob : any
        The object to take a view over.
    """

    # add slots so we don't accidentally add attributes to the view instead of
    # ``ob``
    __slots__ = ("_mutable_view_ob",)

    def __init__(self, ob):
        object.__setattr__(self, "_mutable_view_ob", ob)

    def __getattr__(self, attr):
        return getattr(self._mutable_view_ob, attr)

    def __setattr__(self, attr, value):
        vars(self._mutable_view_ob)[attr] = value

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self._mutable_view_ob)


# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = IntEnum(
    "DATASOURCE_TYPE",
    [
        "AS_TRADED_EQUITY",
        "MERGER",
        "SPLIT",
        "DIVIDEND",
        "TRADE",
        "TRANSACTION",
        "ORDER",
        "EMPTY",
        "DONE",
        "CUSTOM",
        "BENCHMARK",
        "COMMISSION",
        "CLOSE_POSITION",
    ],
    start=0,
)

# Expected fields/index values for a dividend Series.
DIVIDEND_FIELDS = [
    "declared_date",
    "ex_date",
    "gross_amount",
    "net_amount",
    "pay_date",
    "payment_sid",
    "ratio",
    "sid",
]
# Expected fields/index values for a dividend payment Series.
DIVIDEND_PAYMENT_FIELDS = [
    "id",
    "payment_sid",
    "cash_amount",
    "share_count",
]


class Event:
    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__.update(initial_values)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        return hasattr(other, "__dict__") and self.__dict__ == other.__dict__

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)

    def to_series(self, index=None):
        return pd.Series(self.__dict__, index=index)


class Order(Event):
    pass


class Portfolio:
    """Object providing read-only access to current portfolio state.

    Parameters
    ----------
    start_date : pd.Timestamp
        The start date for the period being recorded.
    capital_base : float
        The starting value for the portfolio. This will be used as the starting
        cash, current cash, and portfolio value.

    Attributes
    ----------
    positions : zipline.protocol.Positions
        Dict-like object containing information about currently-held positions.
    cash : float
        Amount of cash currently held in portfolio.
    portfolio_value : float
        Current liquidation value of the portfolio's holdings.
        This is equal to ``cash + sum(shares * price)``
    starting_cash : float
        Amount of cash in the portfolio at the start of the backtest.
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
        raise AttributeError("cannot mutate Portfolio objects")

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)

    @property
    def current_portfolio_weights(self):
        """
        Compute each asset's weight in the portfolio by calculating its held
        value divided by the total value of all positions.

        Each equity's value is its price times the number of shares held. Each
        futures contract's value is its unit price times number of shares held
        times the multiplier.
        """
        position_values = pd.Series(
            {
                asset: (
                    position.last_sale_price * position.amount * asset.price_multiplier
                )
                for asset, position in self.positions.items()
            },
            dtype=float,
        )
        return position_values / self.portfolio_value


class Account:
    """
    The account object tracks information about the trading account. The
    values are updated as the algorithm runs and its keys remain unchanged.
    If connected to a broker, one can update these values with the trading
    account values as reported by the broker.
    """

    def __init__(self):
        self_ = MutableView(self)
        self_.settled_cash = 0.0
        self_.accrued_interest = 0.0
        self_.buying_power = float("inf")
        self_.equity_with_loan = 0.0
        self_.total_positions_value = 0.0
        self_.total_positions_exposure = 0.0
        self_.regt_equity = 0.0
        self_.regt_margin = float("inf")
        self_.initial_margin_requirement = 0.0
        self_.maintenance_margin_requirement = 0.0
        self_.available_funds = 0.0
        self_.excess_liquidity = 0.0
        self_.cushion = 0.0
        self_.day_trades_remaining = float("inf")
        self_.leverage = 0.0
        self_.net_leverage = 0.0
        self_.net_liquidation = 0.0

    def __setattr__(self, attr, value):
        raise AttributeError("cannot mutate Account objects")

    def __repr__(self):
        return "Account({0})".format(self.__dict__)


class Position:
    """
    A position held by an algorithm.

    Attributes
    ----------
    asset : zipline.assets.Asset
        The held asset.
    amount : int
        Number of shares held. Short positions are represented with negative
        values.
    cost_basis : float
        Average price at which currently-held shares were acquired.
    last_sale_price : float
        Most recent price for the position.
    last_sale_date : pd.Timestamp
        Datetime at which ``last_sale_price`` was last updated.
    """

    __slots__ = ("_underlying_position",)

    def __init__(self, underlying_position):
        object.__setattr__(self, "_underlying_position", underlying_position)

    def __getattr__(self, attr):
        return getattr(self._underlying_position, attr)

    def __setattr__(self, attr, value):
        raise AttributeError("cannot mutate Position objects")

    @property
    def sid(self):
        # for backwards compatibility
        return self.asset

    def __repr__(self):
        return "Position(%r)" % {
            k: getattr(self, k)
            for k in (
                "asset",
                "amount",
                "cost_basis",
                "last_sale_price",
                "last_sale_date",
            )
        }


class Positions(dict):
    """A dict-like object containing the algorithm's current positions."""

    def __missing__(self, key):
        if isinstance(key, Asset):
            return Position(InnerPosition(key))

        raise ValueError(
            "Position lookup expected a value of type Asset"
            f" but got {type(key).__name__} instead"
        )
