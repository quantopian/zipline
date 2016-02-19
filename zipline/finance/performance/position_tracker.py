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
from zipline.finance.performance.position import Position
from zipline.finance.transaction import Transaction

try:
    # optional cython based OrderedDict
    from cyordereddict import OrderedDict
except ImportError:
    from collections import OrderedDict
from six import iteritems, itervalues

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

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

        # Dict, keyed on dates, that contains lists of close position events
        # for any Assets in this tracker's positions
        self._auto_close_position_sids = {}

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
                # Futures auto-close timing is controlled by the Future's
                # auto_close_date property
                self._insert_auto_close_position_date(
                    dt=asset.auto_close_date,
                    sid=sid
                )

    def _insert_auto_close_position_date(self, dt, sid):
        """
        Inserts the given SID in to the list of positions to be auto-closed by
        the given dt.

        Parameters
        ----------
        dt : pandas.Timestamp
            The date before-which the given SID will be auto-closed
        sid : int
            The SID of the Asset to be auto-closed
        """
        if dt is not None:
            self._auto_close_position_sids.setdefault(dt, set()).add(sid)

    def auto_close_position_events(self, next_trading_day):
        """
        Generates CLOSE_POSITION events for any SIDs whose auto-close date is
        before or equal to the given date.

        Parameters
        ----------
        next_trading_day : pandas.Timestamp
            The time before-which certain Assets need to be closed

        Yields
        ------
        Event
            A close position event for any sids that should be closed before
            the next_trading_day parameter
        """
        past_asset_end_dates = set()

        # Check the auto_close_position_dates dict for SIDs to close
        for date, sids in self._auto_close_position_sids.items():
            if date > next_trading_day:
                continue
            past_asset_end_dates.add(date)

            for sid in sids:
                # Yield a CLOSE_POSITION event
                event = Event({
                    'dt': date,
                    'type': DATASOURCE_TYPE.CLOSE_POSITION,
                    'sid': sid,
                })
                yield event

        # Clear out past dates
        while past_asset_end_dates:
            self._auto_close_position_sids.pop(past_asset_end_dates.pop())

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
        None
        """
        for split in splits:
            sid = split[0]
            if sid in self.positions:
                # Make the position object handle the split. It returns the
                # leftover cash from a fractional share, if there is any.
                position = self.positions[sid]
                leftover_cash = position.handle_split(sid, split[1])
                self._update_asset(split[0])
                return leftover_cash

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

    def maybe_create_close_position_transaction(self, event):
        if not self.positions.get(event.sid):
            return None

        amount = self.positions.get(event.sid).amount
        price = self._data_portal.get_spot_value(
            event.sid, 'close', event.dt, self.data_frequency)

        txn = Transaction(
            sid=event.sid,
            amount=(-1 * amount),
            dt=event.dt,
            price=price,
            commission=0,
            order_id=0
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
            if dt >= asset.end_date:
                # if the asset no longer exists, yet we somehow are being
                # asked for the last price, we want 0 instead of NaN.
                position.last_sale_price = 0
            else:
                position.last_sale_price = data_portal.get_spot_value(
                    asset, 'price', dt, self.data_frequency)

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

    def __getstate__(self):
        state_dict = {}

        state_dict['asset_finder'] = self.asset_finder
        state_dict['positions'] = dict(self.positions)
        state_dict['unpaid_dividends'] = self._unpaid_dividends
        state_dict['unpaid_stock_dividends'] = self._unpaid_stock_dividends
        state_dict['auto_close_position_sids'] = self._auto_close_position_sids
        state_dict['data_frequency'] = self.data_frequency

        STATE_VERSION = 3
        state_dict[VERSION_LABEL] = STATE_VERSION
        return state_dict

    def __setstate__(self, state):
        OLDEST_SUPPORTED_STATE = 3
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("PositionTracker saved state is too old.")

        self.asset_finder = state['asset_finder']
        self.positions = positiondict()
        self.data_frequency = state['data_frequency']
        # note that positions_store is temporary and gets regened from
        # .positions
        self._positions_store = zp.Positions()

        self._unpaid_dividends = state['unpaid_dividends']
        self._unpaid_stock_dividends = state['unpaid_stock_dividends']
        self._auto_close_position_sids = state['auto_close_position_sids']

        # Arrays for quick calculations of positions value
        self._position_value_multipliers = OrderedDict()
        self._position_exposure_multipliers = OrderedDict()

        # Update positions is called without a finder
        self.update_positions(state['positions'])

        # FIXME
        self._data_portal = None
