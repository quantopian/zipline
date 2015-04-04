from __future__ import division
from operator import mul

import logbook
import numpy as np
import pandas as pd
from pandas.lib import checknull
try:
    # optional cython based OrderedDict
    from cyordereddict import OrderedDict
except ImportError:
    from collections import OrderedDict
from six import iteritems
from six.moves import map, filter

from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

import zipline.protocol as zp
from . position import positiondict

log = logbook.Logger('Performance')


class PositionTracker(object):

    def __init__(self):
        # sid => position object
        self.positions = positiondict()
        # Arrays for quick calculations of positions value
        self._position_amounts = OrderedDict()
        self._position_last_sale_prices = OrderedDict()
        self._unpaid_dividends = pd.DataFrame(
            columns=zp.DIVIDEND_PAYMENT_FIELDS,
        )
        self._positions_store = zp.Positions()

    def update_last_sale(self, event):
        # NOTE, PerformanceTracker already vetted as TRADE type
        sid = event.sid
        if sid not in self.positions:
            return

        price = event.price
        if not checknull(price):
            pos = self.positions[sid]
            pos.last_sale_date = event.dt
            pos.last_sale_price = price
            self._position_last_sale_prices[sid] = price
            self._position_values = None  # invalidate cache
        sid = event.sid
        price = event.price

    def update_positions(self, positions):
        # update positions in batch
        self.positions.update(positions)
        for sid, pos in iteritems(positions):
            self._position_amounts[sid] = pos.amount
            self._position_last_sale_prices[sid] = pos.last_sale_price
            # Invalidate cache.
            self._position_values = None  # invalidate cache

    def update_position(self, sid, amount=None, last_sale_price=None,
                        last_sale_date=None, cost_basis=None):
        pos = self.positions[sid]

        if amount is not None:
            pos.amount = amount
            self._position_amounts[sid] = amount
            self._position_values = None  # invalidate cache
        if last_sale_price is not None:
            pos.last_sale_price = last_sale_price
            self._position_last_sale_prices[sid] = last_sale_price
            self._position_values = None  # invalidate cache
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
        self._position_amounts[sid] = position.amount
        self._position_last_sale_prices[sid] = position.last_sale_price
        self._position_values = None  # invalidate cache

    def handle_commission(self, commission):
        # Adjust the cost basis of the stock if we own it
        if commission.sid in self.positions:
            self.positions[commission.sid].\
                adjust_commission_cost_basis(commission)

    _position_values = None

    @property
    def position_values(self):
        """
        Invalidate any time self._position_amounts or
        self._position_last_sale_prices is changed.
        """
        if self._position_values is None:
            vals = list(map(mul, self._position_amounts.values(),
                        self._position_last_sale_prices.values()))
            self._position_values = vals
        return self._position_values

    def calculate_positions_value(self):
        if len(self.position_values) == 0:
            return np.float64(0)

        return sum(self.position_values)

    def _longs_count(self):
        return sum(map(lambda x: x > 0, self.position_values))

    def _long_exposure(self):
        return sum(filter(lambda x: x > 0, self.position_values))

    def _shorts_count(self):
        return sum(map(lambda x: x < 0, self.position_values))

    def _short_exposure(self):
        return sum(filter(lambda x: x < 0, self.position_values))

    def _gross_exposure(self):
        return self._long_exposure() + abs(self._short_exposure())

    def _net_exposure(self):
        return self.calculate_positions_value()

    def handle_split(self, split):
        if split.sid in self.positions:
            # Make the position object handle the split. It returns the
            # leftover cash from a fractional share, if there is any.
            position = self.positions[split.sid]
            leftover_cash = position.handle_split(split)
            self._position_amounts[split.sid] = position.amount
            self._position_last_sale_prices[split.sid] = \
                position.last_sale_price
            self._position_values = None  # invalidate cache
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
            # already own the security
            position = self.positions[stock]

            position.amount += share_count
            self._position_amounts[stock] = position.amount
            self._position_last_sale_prices[stock] = position.last_sale_price

        # Add cash equal to the net cash payed from all dividends.  Note that
        # "negative cash" is effectively paid if we're short a security,
        # representing the fact that we're required to reimburse the owner of
        # the stock for any dividends paid while borrowing.
        net_cash_payment = payments['cash_amount'].fillna(0).sum()
        return net_cash_payment

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

        state_dict['positions'] = dict(self.positions)
        state_dict['unpaid_dividends'] = self._unpaid_dividends

        STATE_VERSION = 1
        state_dict[VERSION_LABEL] = STATE_VERSION
        return state_dict

    def __setstate__(self, state):
        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("PositionTracker saved state is too old.")

        self.positions = positiondict()
        # note that positions_store is temporary and gets regened from
        # .positions
        self._positions_store = zp.Positions()

        self._unpaid_dividends = state['unpaid_dividends']

        # Arrays for quick calculations of positions value
        self._position_amounts = OrderedDict()
        self._position_last_sale_prices = OrderedDict()

        self.update_positions(state['positions'])
