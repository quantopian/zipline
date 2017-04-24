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

from six import iteritems, itervalues

from zipline.finance.performance.position import Position
from zipline.finance.transaction import Transaction
from zipline.utils.input_validation import expect_types
import zipline.protocol as zp
from zipline.assets import (
    Future,
    Asset
)
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


def calc_position_values(positions):
    values = []

    for position in positions:
        if isinstance(position.asset, Future):
            # Futures don't have an inherent position value.
            values.append(0.0)
        else:
            values.append(position.last_sale_price * position.amount)

    return values


def calc_net(values):
    # Returns 0.0 if there are no values.
    return sum(values, np.float64())


def calc_position_exposures(positions):
    exposures = []

    for position in positions:
        exposure = position.amount * position.last_sale_price

        if isinstance(position.asset, Future):
            exposure *= position.asset.multiplier

        exposures.append(exposure)

    return exposures


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

    def __init__(self, data_frequency):
        # asset => position object
        self.positions = positiondict()
        self._unpaid_dividends = {}
        self._unpaid_stock_dividends = {}
        self._positions_store = zp.Positions()

        self.data_frequency = data_frequency

    @expect_types(asset=Asset)
    def update_position(self, asset, amount=None, last_sale_price=None,
                        last_sale_date=None, cost_basis=None):
        if asset not in self.positions:
            position = Position(asset)
            self.positions[asset] = position
        else:
            position = self.positions[asset]

        if amount is not None:
            position.amount = amount
        if last_sale_price is not None:
            position.last_sale_price = last_sale_price
        if last_sale_date is not None:
            position.last_sale_date = last_sale_date
        if cost_basis is not None:
            position.cost_basis = cost_basis

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        asset = txn.asset

        if asset not in self.positions:
            position = Position(asset)
            self.positions[asset] = position
        else:
            position = self.positions[asset]

        position.update(txn)

        if position.amount == 0:
            del self.positions[asset]

            try:
                # if this position exists in our user-facing dictionary,
                # remove it as well.
                del self._positions_store[asset]
            except KeyError:
                pass

    @expect_types(asset=Asset)
    def handle_commission(self, asset, cost):
        # Adjust the cost basis of the stock if we own it
        if asset in self.positions:
            self.positions[asset].adjust_commission_cost_basis(asset, cost)

    def handle_splits(self, splits):
        """
        Processes a list of splits by modifying any positions as needed.

        Parameters
        ----------
        splits: list
            A list of splits.  Each split is a tuple of (asset, ratio).

        Returns
        -------
        int: The leftover cash from fractional sahres after modifying each
            position.
        """
        total_leftover_cash = 0

        for asset, ratio in splits:
            if asset in self.positions:
                # Make the position object handle the split. It returns the
                # leftover cash from a fractional share, if there is any.
                position = self.positions[asset]
                leftover_cash = position.handle_split(asset, ratio)
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

        return net_cash_payment

    def maybe_create_close_position_transaction(self, asset, dt, data_portal):
        if not self.positions.get(asset):
            return None

        amount = self.positions.get(asset).amount
        price = data_portal.get_spot_value(
            asset, 'price', dt, self.data_frequency)

        # Get the last traded price if price is no longer available
        if isnan(price):
            price = self.positions.get(asset).last_sale_price

        txn = Transaction(
            asset=asset,
            amount=(-1 * amount),
            dt=dt,
            price=price,
            commission=0,
            order_id=None,
        )
        return txn

    def get_positions(self):

        positions = self._positions_store

        for asset, pos in iteritems(self.positions):

            if pos.amount == 0:
                # Clear out the position if it has become empty since the last
                # time get_positions was called.  Catching the KeyError is
                # faster than checking `if asset in positions`, and this can be
                # potentially called in a tight inner loop.
                try:
                    del positions[asset]
                except KeyError:
                    pass
                continue

            position = zp.Position(asset)
            position.amount = pos.amount
            position.cost_basis = pos.cost_basis
            position.last_sale_price = pos.last_sale_price
            position.last_sale_date = pos.last_sale_date

            # Adds the new position if we didn't have one before, or overwrite
            # one we have currently
            positions[asset] = position

        return positions

    def get_positions_list(self):
        positions = []
        for asset, pos in iteritems(self.positions):
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions

    def sync_last_sale_prices(self, dt, handle_non_market_minutes,
                              data_portal):
        if not handle_non_market_minutes:
            for asset, position in iteritems(self.positions):
                last_sale_price = data_portal.get_spot_value(
                    asset, 'price', dt, self.data_frequency
                )

                if not np.isnan(last_sale_price):
                    position.last_sale_price = last_sale_price
        else:
            for asset, position in iteritems(self.positions):
                last_sale_price = data_portal.get_adjusted_value(
                    asset,
                    'price',
                    data_portal.trading_calendar.previous_minute(dt),
                    dt,
                    self.data_frequency
                )

                if not np.isnan(last_sale_price):
                    position.last_sale_price = last_sale_price

    def stats(self):
        amounts = []
        last_sale_prices = []
        for pos in itervalues(self.positions):
            amounts.append(pos.amount)
            last_sale_prices.append(pos.last_sale_price)

        position_values = calc_position_values(itervalues(self.positions))
        position_exposures = calc_position_exposures(
            itervalues(self.positions)
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
