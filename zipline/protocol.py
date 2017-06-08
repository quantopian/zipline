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
from functools import partial
from warnings import warn

from empyrical import conditional_value_at_risk
import numpy as np
import pandas as pd

from zipline._protocol import BarData  # noqa
from zipline.assets import Asset, Future
from zipline.errors import InsufficientHistoricalData
from zipline.utils.cache import ExpiringCache
from zipline.utils.enum import enum
from zipline.utils.input_validation import expect_types


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

DEFAULT_EXPECTED_SHORTFALL_LOOKBACK_DAYS = 504
DEFAULT_EXPECTED_SHORTFALL_MINIMUM_DAYS = 252
DEFAULT_EXPECTED_SHORTFALL_CUTOFF = 0.05


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
            return self.__dict__[key]
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


def asset_returns_for_expected_shortfall(assets,
                                         benchmark,
                                         data_portal,
                                         end_date,
                                         lookback_days):
    if benchmark is not None:
        # If the algorithm held a position in the benchmark asset, include it
        # in the expected shortfall calculation. Otherwise, just use it as a
        # filler for missing values.
        if benchmark in assets:
            include_benchmark = True
        else:
            assets.append(benchmark)
            include_benchmark = False

    # NOTE: Using the simulation calendar here is based on the assumption that
    # this calendar runs on the union of all trading days of all asset classes.
    # For example, when doing history pricing calls for both equities and
    # futures, we require equity holidays that are not future holidays to be
    # forward filled. This keeps the dates aligned when computing returns. It
    # just so happens that the us_futures calendar is a strict superset of the
    # NYSE calendar, making this assumption true.
    prices = data_portal.get_history_window(
       assets=assets,
       end_dt=end_date,
       bar_count=lookback_days,
       frequency='1d',
       field='price',
       data_frequency='daily',
    )

    # Get returns values for all assets for the entirety of the simulation.
    asset_returns = prices.pct_change().iloc[1:]

    # Any assets that came into existence after the start date of the
    # simulation have their missing returns values proxied with the benchmark's
    # returns values.
    if benchmark is not None:
        benchmark_returns = asset_returns[benchmark].values
        filler_df = pd.DataFrame(
            np.tile(
                benchmark_returns[:, np.newaxis],
                (1, len(asset_returns.columns)),
            ),
            index=asset_returns.index,
            columns=asset_returns.columns,
        )
        asset_returns.fillna(filler_df, inplace=True)
        if not include_benchmark:
            asset_returns.drop(benchmark, axis=1, inplace=True)

    return asset_returns.fillna(0)


class Portfolio(object):

    def __init__(self, data_portal, current_dt_callback, benchmark_asset=None):
        self.capital_used = 0.0
        self.starting_cash = 0.0
        self.portfolio_value = 0.0
        self.pnl = 0.0
        self.returns = 0.0
        self.cash = 0.0
        self.positions = Positions()
        self.start_date = None
        self.positions_value = 0.0

        self.benchmark_asset = benchmark_asset

        self._data_portal = data_portal
        self._current_dt_callback = current_dt_callback
        self._expiring_cache = ExpiringCache()

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
    def current_date(self):
        return self._current_dt_callback()

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

    def asset_for_history_call(self, asset, date):
        if isinstance(asset, Future):
            # Infer the offset of the given future by comparing it to the
            # upcoming closing contract according to the given date.
            asset_finder = self._data_portal.asset_finder
            offset = asset_finder.offset_of_contract(asset, date)
            return asset_finder.get_continuous_future(
                root_symbol=asset.root_symbol,
                offset=offset,
                roll_style='volume',
                adjustment='mul',
            )
        return asset

    def expected_shortfall(
            self,
            lookback_days=DEFAULT_EXPECTED_SHORTFALL_LOOKBACK_DAYS,
            cutoff=DEFAULT_EXPECTED_SHORTFALL_CUTOFF):
        """
        Function for computing expected shortfall (also known as CVaR, or
        Conditional Value at Risk) for the portfolio according to the assets
        currently held and their respective weight in the portfolio.

        Parameters
        ----------

        lookback_days : int, optional
            The number of days of asset returns history to use.
        cutoff : float, optional
            The percentile cutoff to use for finding the worst returns values.

        Returns
        -------
        expected_shortfall : float
            The expected shortfall of the current portfolio.

        Raises
        ------
        InsufficientHistoricalData
            Raised if there is less than 'lookback_days' days worth of asset
            data available from the portfolio's current date.
        """
        data_portal = self._data_portal
        current_date = self.current_date
        benchmark = self.benchmark_asset
        calendar = data_portal.trading_calendar

        try:
            return self._expiring_cache.get('expected_shortfall', current_date)
        except KeyError:
            pass

        # If we do not have enough data to look back on then the expected
        # shortfall calculation will not be reliable, so instead of returning
        # NaN, raise an exception alerting the user that this method cannot be
        # called over the given simulation dates.
        data_start = data_portal.first_available_session
        num_days_of_data = calendar.session_distance(data_start, current_date)
        if num_days_of_data < lookback_days:
            suggested_start_date = data_start + (calendar.day * lookback_days)
            raise InsufficientHistoricalData(
                method_name='expected_shortfall',
                lookback_days=lookback_days,
                suggested_start_date=suggested_start_date.date(),
            )

        # Series mapping each asset to its portfolio weight.
        weights = self.current_portfolio_weights()

        convert_futures = partial(
            self.asset_for_history_call, date=current_date,
        )
        assets = list(map(convert_futures, weights.index))
        asset_returns = asset_returns_for_expected_shortfall(
            assets, benchmark, data_portal, current_date, lookback_days,
        )

        expected_shortfall = conditional_value_at_risk(
            returns=asset_returns.dot(weights.values), cutoff=cutoff,
        )
        self._expiring_cache.set(
            'expected_shortfall', expected_shortfall, current_date,
        )
        return expected_shortfall


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
        self.total_positions_exposure = 0.0
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


class Position(object):
    @expect_types(asset=Asset)
    def __init__(self, asset):
        self.asset = asset
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0
        self.last_sale_date = None

    @property
    def sid(self):
        # for backwards compatibility
        return self.asset

    def __repr__(self):
        return "Position({0})".format(self.__dict__)

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
            return Position(key)
        elif isinstance(key, int):
            warn("Referencing positions by integer is deprecated."
                 " Use an asset instead.")
        else:
            warn("Position lookup expected a value of type Asset but got {0}"
                 " instead.".format(type(key).__name__))

        return _DeprecatedSidLookupPosition(key)
