#
# Copyright 2015 Quantopian, Inc.
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
import pandas as pd
from pandas.lib import checknull
from collections import namedtuple
try:
    # optional cython based OrderedDict
    from cyordereddict import OrderedDict
except ImportError:
    from collections import OrderedDict
from six import iteritems, itervalues

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.finance.slippage import Transaction
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


def calc_position_stats(pt):
    amounts = []
    last_sale_prices = []
    for pos in itervalues(pt.positions):
        amounts.append(pos.amount)
        last_sale_prices.append(pos.last_sale_price)

    position_value_multipliers = pt._position_value_multipliers
    position_exposure_multipliers = pt._position_exposure_multipliers

    position_values = calc_position_values(
        amounts,
        last_sale_prices,
        position_value_multipliers
    )

    position_exposures = calc_position_exposures(
        amounts,
        last_sale_prices,
        position_exposure_multipliers
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


class PositionTracker(object):

    def __init__(self, asset_finder):
        self.asset_finder = asset_finder

        # sid => position object
        self.positions = positiondict()
        # Arrays for quick calculations of positions value
        self._position_value_multipliers = OrderedDict()
        self._position_exposure_multipliers = OrderedDict()
        self._position_payout_multipliers = OrderedDict()
        self._unpaid_dividends = pd.DataFrame(
            columns=zp.DIVIDEND_PAYMENT_FIELDS,
        )
        self._positions_store = zp.Positions()

        # Dict, keyed on dates, that contains lists of close position events
        # for any Assets in this tracker's positions
        self._auto_close_position_sids = {}

    def _update_asset(self, sid):
        try:
            self._position_value_multipliers[sid]
            self._position_exposure_multipliers[sid]
            self._position_payout_multipliers[sid]
        except KeyError:
            # Check if there is an AssetFinder
            if self.asset_finder is None:
                raise PositionTrackerMissingAssetFinder()

            # Collect the value multipliers from applicable sids
            asset = self.asset_finder.retrieve_asset(sid)
            if isinstance(asset, Equity):
                self._position_value_multipliers[sid] = 1
                self._position_exposure_multipliers[sid] = 1
                self._position_payout_multipliers[sid] = 0
            if isinstance(asset, Future):
                self._position_value_multipliers[sid] = 0
                self._position_exposure_multipliers[sid] = \
                    asset.contract_multiplier
                self._position_payout_multipliers[sid] = \
                    asset.contract_multiplier
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

    def update_last_sale(self, event):
        # NOTE, PerformanceTracker already vetted as TRADE type
        sid = event.sid
        if sid not in self.positions:
            return 0

        price = event.price

        if checknull(price):
            return 0

        pos = self.positions[sid]
        old_price = pos.last_sale_price
        pos.last_sale_date = event.dt
        pos.last_sale_price = price

        # Calculate cash adjustment on assets with multipliers
        return ((price - old_price) * self._position_payout_multipliers[sid]
                * pos.amount)

    def update_positions(self, positions):
        # update positions in batch
        self.positions.update(positions)
        for sid, pos in iteritems(positions):
            self._update_asset(sid)

    def update_position(self, sid, amount=None, last_sale_price=None,
                        last_sale_date=None, cost_basis=None):
        pos = self.positions[sid]

        if amount is not None:
            pos.amount = amount
            self._update_asset(sid=sid)
        if last_sale_price is not None:
            pos.last_sale_price = last_sale_price
        if last_sale_date is not None:
            pos.last_sale_date = last_sale_date
        if cost_basis is not None:
            pos.cost_basis = cost_basis

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        sid = txn.sid
        position = self.positions[sid]
        position.update(txn)
        self._update_asset(sid)

    def handle_commission(self, commission):
        # Adjust the cost basis of the stock if we own it
        if commission.sid in self.positions:
            self.positions[commission.sid].\
                adjust_commission_cost_basis(commission)

    def handle_split(self, split):
        if split.sid in self.positions:
            # Make the position object handle the split. It returns the
            # leftover cash from a fractional share, if there is any.
            position = self.positions[split.sid]
            leftover_cash = position.handle_split(split)
            self._update_asset(split.sid)
            return leftover_cash

    def _maybe_earn_dividend(self, dividend):
        """
        Take a historical dividend record and return a Series with fields in
        zipline.protocol.DIVIDEND_FIELDS (plus an 'id' field) representing
        the cash/stock amount we are owed when the dividend is paid.
        """
        if dividend['sid'] in self.positions:
            return self.positions[dividend['sid']].earn_dividend(dividend)
        else:
            return zp.dividend_payment()

    def earn_dividends(self, dividend_frame):
        """
        Given a frame of dividends whose ex_dates are all the next trading day,
        calculate and store the cash and/or stock payments to be paid on each
        dividend's pay date.
        """
        earned = dividend_frame.apply(self._maybe_earn_dividend, axis=1)\
                               .dropna(how='all')
        if len(earned) > 0:
            # Store the earned dividends so that they can be paid on the
            # dividends' pay_dates.
            self._unpaid_dividends = pd.concat(
                [self._unpaid_dividends, earned],
            )

    def _maybe_pay_dividend(self, dividend):
        """
        Take a historical dividend record, look up any stored record of
        cash/stock we are owed for that dividend, and return a Series
        with fields drawn from zipline.protocol.DIVIDEND_PAYMENT_FIELDS.
        """
        try:
            unpaid_dividend = self._unpaid_dividends.loc[dividend['id']]
            return unpaid_dividend
        except KeyError:
            return zp.dividend_payment()

    def pay_dividends(self, dividend_frame):
        """
        Given a frame of dividends whose pay_dates are all the next trading
        day, grant the cash and/or stock payments that were calculated on the
        given dividends' ex dates.
        """
        payments = dividend_frame.apply(self._maybe_pay_dividend, axis=1)\
                                 .dropna(how='all')

        # Mark these dividends as paid by dropping them from our unpaid
        # table.
        self._unpaid_dividends.drop(payments.index)

        # Add stock for any stock dividends paid.  Again, the values here may
        # be negative in the case of short positions.
        stock_payments = payments[payments['payment_sid'].notnull()]
        for _, row in stock_payments.iterrows():
            stock = row['payment_sid']
            share_count = row['share_count']
            # note we create a Position for stock dividend if we don't
            # already own the asset
            position = self.positions[stock]

            position.amount += share_count
            self._update_asset(stock)

        # Add cash equal to the net cash payed from all dividends.  Note that
        # "negative cash" is effectively paid if we're short an asset,
        # representing the fact that we're required to reimburse the owner of
        # the stock for any dividends paid while borrowing.
        net_cash_payment = payments['cash_amount'].fillna(0).sum()
        return net_cash_payment

    def maybe_create_close_position_transaction(self, event):
        try:
            pos = self.positions[event.sid]
            amount = pos.amount
            if amount == 0:
                return None
        except KeyError:
            return None
        if 'price' in event:
            price = event.price
        else:
            price = pos.last_sale_price
        txn = Transaction(
            sid=event.sid,
            amount=(-1 * pos.amount),
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
        return positions

    def get_positions_list(self):
        positions = []
        for sid, pos in iteritems(self.positions):
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions

    def __getstate__(self):
        state_dict = {}

        state_dict['asset_finder'] = self.asset_finder
        state_dict['positions'] = dict(self.positions)
        state_dict['unpaid_dividends'] = self._unpaid_dividends
        state_dict['auto_close_position_sids'] = self._auto_close_position_sids

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
        # note that positions_store is temporary and gets regened from
        # .positions
        self._positions_store = zp.Positions()

        self._unpaid_dividends = state['unpaid_dividends']
        self._auto_close_position_sids = state['auto_close_position_sids']

        # Arrays for quick calculations of positions value
        self._position_value_multipliers = OrderedDict()
        self._position_exposure_multipliers = OrderedDict()
        self._position_payout_multipliers = OrderedDict()

        # Update positions is called without a finder
        self.update_positions(state['positions'])
