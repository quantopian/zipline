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

from __future__ import division

import logbook
import numpy as np
from collections import namedtuple
from math import isnan
from zipline.finance.performance.position import Position
from zipline.finance.transaction import Transaction

try:
    # optional cython based OrderedDict
    from cyordereddict import OrderedDict
except ImportError:
    from collections import OrderedDict
from six import iteritems, itervalues

import zipline.protocol as zp
from zipline.assets import (
    Equity, Future
)
from zipline.errors import PositionTrackerMissingAssetFinder
from . position import positiondict

log = logbook.Logger('Performance')


PositionStats = namedtuple('PositionStats',
                           ['net_exposure',
                            'gross_value',
                            'gross_exposure',
                            'short_value',
                            'short_exposure',
                            'shorts_count',
                            'long_value',
                            'long_exposure',
                            'longs_count',
                            'net_value'])


def calc_position_values(amounts,
                         last_sale_prices,
                         value_multipliers):
    iter_amount_price_multiplier = zip(
        amounts,
        last_sale_prices,
        itervalues(value_multipliers),
    )
    return [
        price * amount * multiplier for
        price, amount, multiplier in iter_amount_price_multiplier
    ]


def calc_net(values):
    # Returns 0.0 if there are no values.
    return sum(values, np.float64())


def calc_position_exposures(amounts,
                            last_sale_prices,
                            exposure_multipliers):
    iter_amount_price_multiplier = zip(
        amounts,
        last_sale_prices,
        itervalues(exposure_multipliers),
    )
    return [
        price * amount * multiplier for
        price, amount, multiplier in iter_amount_price_multiplier
    ]


def calc_long_value(position_values):
    return sum(i for i in position_values if i > 0)


def calc_short_value(position_values):
    return sum(i for i in position_values if i < 0)


def calc_long_exposure(position_exposures):
    return sum(i for i in position_exposures if i > 0)


def calc_short_exposure(position_exposures):
    return sum(i for i in position_exposures if i < 0)


def calc_longs_count(position_exposures):
    return sum(1 for i in position_exposures if i > 0)


def calc_shorts_count(position_exposures):
    return sum(1 for i in position_exposures if i < 0)


def calc_gross_exposure(long_exposure, short_exposure):
    return long_exposure + abs(short_exposure)


def calc_gross_value(long_value, short_value):
    return long_value + abs(short_value)


class PositionTracker(object):

    def __init__(self, asset_finder, data_portal, data_frequency):
        self.asset_finder = asset_finder

        # FIXME really want to avoid storing a data portal here,
        # but the path to get to maybe_create_close_position_transaction
        # is long and tortuous
        self._data_portal = data_portal

        # sid => position object
        self.positions = positiondict()
        # Arrays for quick calculations of positions value
        self._position_value_multipliers = OrderedDict()
        self._position_exposure_multipliers = OrderedDict()
        self._unpaid_dividends = {}
        self._unpaid_stock_dividends = {}
        self._positions_store = zp.Positions()

        self.data_frequency = data_frequency

    def _update_asset(self, sid):
        try:
            self._position_value_multipliers[sid]
            self._position_exposure_multipliers[sid]
        except KeyError:
            # Check if there is an AssetFinder
            if self.asset_finder is None:
                raise PositionTrackerMissingAssetFinder()

            # Collect the value multipliers from applicable sids
            asset = self.asset_finder.retrieve_asset(sid)
            if isinstance(asset, Equity):
                self._position_value_multipliers[sid] = 1
                self._position_exposure_multipliers[sid] = 1
            if isinstance(asset, Future):
                self._position_value_multipliers[sid] = 0
                self._position_exposure_multipliers[sid] = asset.multiplier

    def update_positions(self, positions):
        # update positions in batch
        self.positions.update(positions)
        for sid, pos in iteritems(positions):
            self._update_asset(sid)

    def update_position(self, sid, amount=None, last_sale_price=None,
                        last_sale_date=None, cost_basis=None):
        if sid not in self.positions:
            position = Position(sid)
            self.positions[sid] = position
        else:
            position = self.positions[sid]

        if amount is not None:
            position.amount = amount
            self._update_asset(sid=sid)
        if last_sale_price is not None:
            position.last_sale_price = last_sale_price
        if last_sale_date is not None:
            position.last_sale_date = last_sale_date
        if cost_basis is not None:
            position.cost_basis = cost_basis

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        sid = txn.sid

        if sid not in self.positions:
            position = Position(sid)
            self.positions[sid] = position
        else:
            position = self.positions[sid]

        position.update(txn)

        if position.amount == 0:
            # if this position now has 0 shares, remove it from our internal
            # bookkeeping.
            del self.positions[sid]

            try:
                # if this position exists in our user-facing dictionary,
                # remove it as well.
                del self._positions_store[sid]
            except KeyError:
                pass

        self._update_asset(sid)

    def handle_commission(self, sid, cost):
        # Adjust the cost basis of the stock if we own it
        if sid in self.positions:
            self.positions[sid].adjust_commission_cost_basis(sid, cost)

    def handle_splits(self, splits):
        """
        Processes a list of splits by modifying any positions as needed.

        Parameters
        ----------
        splits: list
            A list of splits.  Each split is a tuple of (sid, ratio).

        Returns
        -------
        int: The leftover cash from fractional sahres after modifying each
            position.
        """
        total_leftover_cash = 0

        for split in splits:
            sid = split[0]
            if sid in self.positions:
                # Make the position object handle the split. It returns the
                # leftover cash from a fractional share, if there is any.
                position = self.positions[sid]
                leftover_cash = position.handle_split(sid, split[1])
                self._update_asset(split[0])
                total_leftover_cash += leftover_cash

        return total_leftover_cash

    def earn_dividends(self, dividends, stock_dividends):
        """
        Given a list of dividends whose ex_dates are all the next trading day,
        calculate and store the cash and/or stock payments to be paid on each
        dividend's pay date.

        Parameters
        ----------
        dividends: iterable of (asset, amount, pay_date) namedtuples

        stock_dividends: iterable of (asset, payment_asset, ratio, pay_date)
            namedtuples.
        """
        for dividend in dividends:
            # Store the earned dividends so that they can be paid on the
            # dividends' pay_dates.
            div_owed = self.positions[dividend.asset].earn_dividend(dividend)
            try:
                self._unpaid_dividends[dividend.pay_date].append(div_owed)
            except KeyError:
                self._unpaid_dividends[dividend.pay_date] = [div_owed]

        for stock_dividend in stock_dividends:
            div_owed = \
                self.positions[stock_dividend.asset].earn_stock_dividend(
                    stock_dividend)
            try:
                self._unpaid_stock_dividends[stock_dividend.pay_date].\
                    append(div_owed)
            except KeyError:
                self._unpaid_stock_dividends[stock_dividend.pay_date] = \
                    [div_owed]

    def pay_dividends(self, next_trading_day):
        """
        Returns a cash payment based on the dividends that should be paid out
        according to the accumulated bookkeeping of earned, unpaid, and stock
        dividends.
        """
        net_cash_payment = 0.0

        try:
            payments = self._unpaid_dividends[next_trading_day]
            # Mark these dividends as paid by dropping them from our unpaid
            del self._unpaid_dividends[next_trading_day]
        except KeyError:
            payments = []

        # representing the fact that we're required to reimburse the owner of
        # the stock for any dividends paid while borrowing.
        for payment in payments:
            net_cash_payment += payment['amount']

        # Add stock for any stock dividends paid.  Again, the values here may
        # be negative in the case of short positions.
        try:
            stock_payments = self._unpaid_stock_dividends[next_trading_day]
        except:
            stock_payments = []

        for stock_payment in stock_payments:
            payment_asset = stock_payment['payment_asset']
            share_count = stock_payment['share_count']
            # note we create a Position for stock dividend if we don't
            # already own the asset
            if payment_asset in self.positions:
                position = self.positions[payment_asset]
            else:
                position = self.positions[payment_asset] = \
                    Position(payment_asset)

            position.amount += share_count
            self._update_asset(payment_asset)

        return net_cash_payment

    def maybe_create_close_position_transaction(self, asset, dt):
        if not self.positions.get(asset):
            return None

        amount = self.positions.get(asset).amount
        price = self._data_portal.get_spot_value(
            asset, 'price', dt, self.data_frequency)

        # Get the last traded price if price is no longer available
        if isnan(price):
            price = self.positions.get(asset).last_sale_price

        txn = Transaction(
            sid=asset,
            amount=(-1 * amount),
            dt=dt,
            price=price,
            commission=0,
            order_id=None,
        )
        return txn

    def get_positions(self):

        positions = self._positions_store

        for sid, pos in iteritems(self.positions):

            if pos.amount == 0:
                # Clear out the position if it has become empty since the last
                # time get_positions was called.  Catching the KeyError is
                # faster than checking `if sid in positions`, and this can be
                # potentially called in a tight inner loop.
                try:
                    del positions[sid]
                except KeyError:
                    pass
                continue

            # Note that this will create a position if we don't currently have
            # an entry
            position = positions[sid]
            position.amount = pos.amount
            position.cost_basis = pos.cost_basis
            position.last_sale_price = pos.last_sale_price
            position.last_sale_date = pos.last_sale_date

        return positions

    def get_positions_list(self):
        positions = []
        for sid, pos in iteritems(self.positions):
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions

    def sync_last_sale_prices(self, dt):
        data_portal = self._data_portal
        for asset, position in iteritems(self.positions):
            last_sale_price = data_portal.get_spot_value(
                asset, 'price', dt, self.data_frequency
            )

            if not np.isnan(last_sale_price):
                position.last_sale_price = last_sale_price

    def stats(self):
        amounts = []
        last_sale_prices = []
        for pos in itervalues(self.positions):
            amounts.append(pos.amount)
            last_sale_prices.append(pos.last_sale_price)

        position_values = calc_position_values(
            amounts,
            last_sale_prices,
            self._position_value_multipliers
        )

        position_exposures = calc_position_exposures(
            amounts,
            last_sale_prices,
            self._position_exposure_multipliers
        )

        long_value = calc_long_value(position_values)
        short_value = calc_short_value(position_values)
        gross_value = calc_gross_value(long_value, short_value)
        long_exposure = calc_long_exposure(position_exposures)
        short_exposure = calc_short_exposure(position_exposures)
        gross_exposure = calc_gross_exposure(long_exposure, short_exposure)
        net_exposure = calc_net(position_exposures)
        longs_count = calc_longs_count(position_exposures)
        shorts_count = calc_shorts_count(position_exposures)
        net_value = calc_net(position_values)

        return PositionStats(
            long_value=long_value,
            gross_value=gross_value,
            short_value=short_value,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            longs_count=longs_count,
            shorts_count=shorts_count,
            net_value=net_value
        )
