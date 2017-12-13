#
# Copyright 2017 Quantopian, Inc.
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
from __future__ import division

from abc import abstractmethod
import math

import numpy as np
from pandas import isnull
from six import with_metaclass
from toolz import merge

from zipline.assets import Equity, Future
from zipline.errors import HistoryWindowStartsBeforeData
from zipline.finance.constants import ROOT_SYMBOL_TO_ETA
from zipline.finance.shared import AllowedAssetMarker, FinancialModelMeta
from zipline.finance.transaction import create_transaction
from zipline.utils.cache import ExpiringCache
from zipline.utils.dummy import DummyMapping
from zipline.utils.input_validation import expect_strictly_bounded

SELL = 1 << 0
BUY = 1 << 1
STOP = 1 << 2
LIMIT = 1 << 3

SQRT_252 = math.sqrt(252)

DEFAULT_EQUITY_VOLUME_SLIPPAGE_BAR_LIMIT = 0.025
DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT = 0.05


class LiquidityExceeded(Exception):
    pass


def fill_price_worse_than_limit_price(fill_price, order):
    """
    Checks whether the fill price is worse than the order's limit price.

    Parameters
    ----------
    fill_price: float
        The price to check.

    order: zipline.finance.order.Order
        The order whose limit price to check.

    Returns
    -------
    bool: Whether the fill price is above the limit price (for a buy) or below
    the limit price (for a sell).
    """
    if order.limit:
        # this is tricky! if an order with a limit price has reached
        # the limit price, we will try to fill the order. do not fill
        # these shares if the impacted price is worse than the limit
        # price. return early to avoid creating the transaction.

        # buy order is worse if the impacted price is greater than
        # the limit price. sell order is worse if the impacted price
        # is less than the limit price
        if (order.direction > 0 and fill_price > order.limit) or \
                (order.direction < 0 and fill_price < order.limit):
            return True

    return False


class SlippageModel(with_metaclass(FinancialModelMeta)):
    """Abstract interface for defining a slippage model.
    """

    # Asset types that are compatible with the given model.
    allowed_asset_types = (Equity, Future)

    def __init__(self):
        self._volume_for_bar = 0

    @property
    def volume_for_bar(self):
        return self._volume_for_bar

    @abstractmethod
    def process_order(self, data, order):
        """Process how orders get filled.

        Parameters
        ----------
        data : BarData
            The data for the given bar.
        order : Order
            The order to simulate.

        Returns
        -------
        execution_price : float
            The price to execute the trade at.
        execution_volume : int
            The number of shares that could be filled. This may not be all
            the shares ordered in which case the order will be filled over
            multiple bars.
        """
        pass

    def simulate(self, data, asset, orders_for_asset):
        self._volume_for_bar = 0
        volume = data.current(asset, "volume")

        if volume == 0:
            return

        # can use the close price, since we verified there's volume in this
        # bar.
        price = data.current(asset, "close")

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END
        dt = data.current_dt

        for order in orders_for_asset:
            if order.open_amount == 0:
                continue

            order.check_triggers(price, dt)
            if not order.triggered:
                continue

            txn = None

            try:
                execution_price, execution_volume = \
                    self.process_order(data, order)

                if execution_price is not None:
                    txn = create_transaction(
                        order,
                        data.current_dt,
                        execution_price,
                        execution_volume
                    )

            except LiquidityExceeded:
                break

            if txn:
                self._volume_for_bar += abs(txn.amount)
                yield order, txn

    def asdict(self):
        return self.__dict__


class EquitySlippageModel(with_metaclass(AllowedAssetMarker, SlippageModel)):
    """
    Base class for slippage models which only support equities.
    """
    allowed_asset_types = (Equity,)


class FutureSlippageModel(with_metaclass(AllowedAssetMarker, SlippageModel)):
    """
    Base class for slippage models which only support futures.
    """
    allowed_asset_types = (Future,)


class VolumeShareSlippage(SlippageModel):
    """
    Model slippage as a function of the volume of contracts traded.
    """
    def __init__(self, volume_limit=DEFAULT_EQUITY_VOLUME_SLIPPAGE_BAR_LIMIT,
                 price_impact=0.1):

        super(VolumeShareSlippage, self).__init__()

        self.volume_limit = volume_limit
        self.price_impact = price_impact

    def __repr__(self):
        return """
{class_name}(
    volume_limit={volume_limit},
    price_impact={price_impact})
""".strip().format(class_name=self.__class__.__name__,
                   volume_limit=self.volume_limit,
                   price_impact=self.price_impact)

    def process_order(self, data, order):
        volume = data.current(order.asset, "volume")

        max_volume = self.volume_limit * volume

        # price impact accounts for the total volume of transactions
        # created against the current minute bar
        remaining_volume = max_volume - self.volume_for_bar
        if remaining_volume < 1:
            # we can't fill any more transactions
            raise LiquidityExceeded()

        # the current order amount will be the min of the
        # volume available in the bar or the open amount.
        cur_volume = int(min(remaining_volume, abs(order.open_amount)))

        if cur_volume < 1:
            return None, None

        # tally the current amount into our total amount ordered.
        # total amount will be used to calculate price impact
        total_volume = self.volume_for_bar + cur_volume

        volume_share = min(total_volume / volume,
                           self.volume_limit)

        price = data.current(order.asset, "close")

        # BEGIN
        #
        # Remove this block after fixing data to ensure volume always has
        # corresponding price.
        if isnull(price):
            return
        # END

        simulated_impact = volume_share ** 2 \
            * math.copysign(self.price_impact, order.direction) \
            * price
        impacted_price = price + simulated_impact

        if fill_price_worse_than_limit_price(impacted_price, order):
            return None, None

        return (
            impacted_price,
            math.copysign(cur_volume, order.direction)
        )


class FixedSlippage(SlippageModel):
    """
    Model slippage as a fixed spread.

    Parameters
    ----------
    spread : float, optional
        spread / 2 will be added to buys and subtracted from sells.
    """
    def __init__(self, spread=0.0):
        super(FixedSlippage, self).__init__()
        self.spread = spread

    def __repr__(self):
        return '{class_name}(spread={spread})'.format(
            class_name=self.__class__.__name__, spread=self.spread,
        )

    def process_order(self, data, order):
        price = data.current(order.asset, "close")

        return (
            price + (self.spread / 2.0 * order.direction),
            order.amount
        )


class MarketImpactBase(SlippageModel):
    """
    Base class for slippage models which compute a simulated price impact
    according to a history lookback.
    """

    NO_DATA_VOLATILITY_SLIPPAGE_IMPACT = 10.0 / 10000

    def __init__(self):
        super(MarketImpactBase, self).__init__()
        self._window_data_cache = ExpiringCache()

    @abstractmethod
    def get_txn_volume(self, data, order):
        """
        Return the number of shares we would like to order in this minute.

        Parameters
        ----------
        data : BarData
        order : Order

        Return
        ------
        int : the number of shares
        """
        raise NotImplementedError('get_txn_volume')

    @abstractmethod
    def get_simulated_impact(self,
                             order,
                             current_price,
                             current_volume,
                             txn_volume,
                             mean_volume,
                             volatility):
        """
        Calculate simulated price impact.

        Parameters
        ----------
        order : The order being processed.
        current_price : Current price of the asset being ordered.
        current_volume : Volume of the asset being ordered for the current bar.
        txn_volume : Number of shares/contracts being ordered.
        mean_volume : Trailing ADV of the asset.
        volatility : Annualized daily volatility of volume.

        Return
        ------
        int : impact on the current price.
        """
        raise NotImplementedError('get_simulated_impact')

    def process_order(self, data, order):
        if order.open_amount == 0:
            return None, None

        minute_data = data.current(order.asset, ['volume', 'high', 'low'])
        mean_volume, volatility = self._get_window_data(data, order.asset, 20)

        # Price to use is the average of the minute bar's open and close.
        price = np.mean([minute_data['high'], minute_data['low']])

        volume = minute_data['volume']
        if not volume:
            return None, None

        txn_volume = int(
            min(self.get_txn_volume(data, order), abs(order.open_amount))
        )

        # If the computed transaction volume is zero or a decimal value, 'int'
        # will round it down to zero. In that case just bail.
        if txn_volume == 0:
            return None, None

        if mean_volume == 0 or np.isnan(volatility):
            # If this is the first day the contract exists or there is no
            # volume history, default to a conservative estimate of impact.
            simulated_impact = price * self.NO_DATA_VOLATILITY_SLIPPAGE_IMPACT
        else:
            simulated_impact = self.get_simulated_impact(
                order=order,
                current_price=price,
                current_volume=volume,
                txn_volume=txn_volume,
                mean_volume=mean_volume,
                volatility=volatility,
            )

        impacted_price = \
            price + math.copysign(simulated_impact, order.direction)

        if fill_price_worse_than_limit_price(impacted_price, order):
            return None, None

        return impacted_price, math.copysign(txn_volume, order.direction)

    def _get_window_data(self, data, asset, window_length):
        """
        Internal utility method to return the trailing mean volume over the
        past 'window_length' days, and volatility of close prices for a
        specific asset.

        Parameters
        ----------
        data : The BarData from which to fetch the daily windows.
        asset : The Asset whose data we are fetching.
        window_length : Number of days of history used to calculate the mean
            volume and close price volatility.

        Returns
        -------
        (mean volume, volatility)
        """
        try:
            values = self._window_data_cache.get(asset, data.current_session)
        except KeyError:
            try:
                # Add a day because we want 'window_length' complete days,
                # excluding the current day.
                volume_history = data.history(
                    asset, 'volume', window_length + 1, '1d',
                )
                close_history = data.history(
                    asset, 'close', window_length + 1, '1d',
                )
            except HistoryWindowStartsBeforeData:
                # If there is not enough data to do a full history call, return
                # values as if there was no data.
                return 0, np.NaN

            # Exclude the first value of the percent change array because it is
            # always just NaN.
            close_volatility = close_history[:-1].pct_change()[1:].std(
                skipna=False,
            )
            values = {
                'volume': volume_history[:-1].mean(),
                'close': close_volatility * SQRT_252,
            }
            self._window_data_cache.set(asset, values, data.current_session)

        return values['volume'], values['close']


class VolatilityVolumeShare(MarketImpactBase):
    """
    Model slippage for futures contracts according to the following formula:

        new_price = price + (price * MI / 10000),

    where 'MI' is market impact, which is defined as:

        MI = eta * sigma * sqrt(psi)

    Eta is a constant which varies by root symbol.
    Sigma is 20-day annualized volatility.
    Psi is the volume traded in the given bar divided by 20-day ADV.

    Parameters
    ----------
    volume_limit : float
        Maximum percentage (as a decimal) of a bar's total volume that can be
        traded.
    eta : float or dict
        Constant used in the market impact formula. If given a float, the eta
        for all futures contracts is the same. If given a dictionary, it must
        map root symbols to the eta for contracts of that symbol.
    """

    NO_DATA_VOLATILITY_SLIPPAGE_IMPACT = 7.5 / 10000
    allowed_asset_types = (Future,)

    def __init__(self, volume_limit, eta=ROOT_SYMBOL_TO_ETA):
        super(VolatilityVolumeShare, self).__init__()
        self.volume_limit = volume_limit

        # If 'eta' is a constant, use a dummy mapping to treat it as a
        # dictionary that always returns the same value.
        # NOTE: This dictionary does not handle unknown root symbols, so it may
        # be worth revisiting this behavior.
        if isinstance(eta, (int, float)):
            self._eta = DummyMapping(float(eta))
        else:
            # Eta is a dictionary. If the user's dictionary does not provide a
            # value for a certain contract, fall back on the pre-defined eta
            # values per root symbol.
            self._eta = merge(ROOT_SYMBOL_TO_ETA, eta)

    def __repr__(self):
        if isinstance(self._eta, DummyMapping):
            # Eta is a constant, so extract it.
            eta = self._eta['dummy key']
        else:
            eta = '<varies>'
        return '{class_name}(volume_limit={volume_limit}, eta={eta})'.format(
            class_name=self.__class__.__name__,
            volume_limit=self.volume_limit,
            eta=eta,
        )

    def get_simulated_impact(self,
                             order,
                             current_price,
                             current_volume,
                             txn_volume,
                             mean_volume,
                             volatility):
        eta = self._eta[order.asset.root_symbol]
        psi = txn_volume / mean_volume

        market_impact = eta * volatility * math.sqrt(psi)

        # We divide by 10,000 because this model computes to basis points.
        # To convert from bps to % we need to divide by 100, then again to
        # convert from % to fraction.
        return (current_price * market_impact) / 10000

    def get_txn_volume(self, data, order):
        volume = data.current(order.asset, 'volume')
        return volume * self.volume_limit


class FixedBasisPointsSlippage(SlippageModel):
    """
    Model slippage as a fixed percentage of fill price. Executes the full
    order immediately.

    Orders to buy will be filled at: `price + (price * basis_points * 0.0001)`.
    Orders to sell will be filled at:
        `price - (price * basis_points * 0.0001)`.

    Parameters
    ----------
    basis_points : float, optional
        Number of basis points of slippage to apply on each execution.

    volume_limit : float, optional
        fraction of the trading volume that can be filled each minute.
    """
    @expect_strictly_bounded(
        basis_points=(0, None),
        volume_limit=(0, None),
        __funcname='FixedBasisPointsSlippage',
    )
    def __init__(self, basis_points=5, volume_limit=0.1):
        super(FixedBasisPointsSlippage, self).__init__()
        self.basis_points = basis_points
        self.percentage = self.basis_points / 10000.0
        self.volume_limit = volume_limit

    def __repr__(self):
        return """
{class_name}(
    basis_points={basis_points},
    volume_limit={volume_limit},
)
""".strip().format(
            class_name=self.__class__.__name__,
            basis_points=self.basis_points,
            volume_limit=self.volume_limit,
        )

    def process_order(self, data, order):

        volume = data.current(order.asset, "volume")
        max_volume = int(self.volume_limit * volume)

        price = data.current(order.asset, "close")
        shares_to_fill = min(abs(order.open_amount),
                             max_volume - self.volume_for_bar)

        return (
            price + price * (self.percentage * order.direction),
            shares_to_fill * order.direction
        )
