#
# Copyright 2014 Quantopian, Inc.
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

"""
Position Tracking
=================

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | sid             | the identifier for the security held in this       |
    |                 | position.                                          |
    +-----------------+----------------------------------------------------+
    | amount          | whole number of shares in the position             |
    +-----------------+----------------------------------------------------+
    | last_sale_price | price at last sale of the security on the exchange |
    +-----------------+----------------------------------------------------+
    | cost_basis      | the volume weighted average price paid per share   |
    +-----------------+----------------------------------------------------+

"""

from __future__ import division
from math import (
    copysign,
    floor,
)

import logbook
import pandas as pd
import zipline.protocol as zp
import zipline.finance.tax_lots as tax_lots

log = logbook.Logger('Performance')


class Position_OLD(object):

    def __init__(self, sid, amount=0, cost_basis=0.0,
                 last_sale_price=0.0, last_sale_date=None):

        self.sid = sid
        self.amount = amount
        self.cost_basis = cost_basis  # per share
        self.last_sale_price = last_sale_price
        self.last_sale_date = last_sale_date

    def earn_dividend(self, dividend):
        """
        Register the number of shares we held at this dividend's ex date so
        that we can pay out the correct amount on the dividend's pay date.
        """
        assert dividend['sid'] == self.sid
        out = {'id': dividend['id']}

        # stock dividend
        if dividend['payment_sid']:
            out['payment_sid'] = dividend['payment_sid']
            out['share_count'] = floor(self.amount * float(dividend['ratio']))

        # cash dividend
        if dividend['net_amount']:
            out['cash_amount'] = self.amount * dividend['net_amount']
        elif dividend['gross_amvount']:
            out['cash_amount'] = self.amount * dividend['gross_amount']

        payment_owed = zp.dividend_payment(out)
        return payment_owed

    def handle_split(self, split):
        """
        Update the position by the split ratio, and return the resulting
        fractional share that will be converted into cash.

        Returns the unused cash.
        """
        if self.sid != split.sid:
            raise Exception("updating split with the wrong sid!")

        ratio = split.ratio

        log.info("handling split for sid = " + str(split.sid) +
                 ", ratio = " + str(split.ratio))
        log.info("before split: " + str(self))

        # adjust the # of shares by the ratio
        # (if we had 100 shares, and the ratio is 3,
        #  we now have 33 shares)
        # (old_share_count / ratio = new_share_count)
        # (old_price * ratio = new_price)

        # e.g., 33.333
        raw_share_count = self.amount / float(ratio)

        # e.g., 33
        full_share_count = floor(raw_share_count)

        # e.g., 0.333
        fractional_share_count = raw_share_count - full_share_count

        # adjust the cost basis to the nearest cent, e.g., 60.0
        new_cost_basis = round(self.cost_basis * ratio, 2)

        # adjust the last sale price
        new_last_sale_price = round(self.last_sale_price * ratio, 2)

        self.cost_basis = new_cost_basis
        self.last_sale_price = new_last_sale_price
        self.amount = full_share_count

        return_cash = round(float(fractional_share_count * new_cost_basis), 2)

        log.info("after split: " + str(self))
        log.info("returning cash: " + str(return_cash))

        # return the leftover cash, which will be converted into cash
        # (rounded to the nearest cent)
        return return_cash

    def update(self, txn):
        if self.sid != txn.sid:
            raise Exception('updating position with txn for a '
                            'different sid')

        total_shares = self.amount + txn.amount

        if total_shares == 0:
            self.cost_basis = 0.0
        else:
            prev_direction = copysign(1, self.amount)
            txn_direction = copysign(1, txn.amount)

            if prev_direction != txn_direction:
                # we're covering a short or closing a position
                if abs(txn.amount) > abs(self.amount):
                    # we've closed the position and gone short
                    # or covered the short position and gone long
                    self.cost_basis = txn.price
            else:
                prev_cost = self.cost_basis * self.amount
                txn_cost = txn.amount * txn.price
                total_cost = prev_cost + txn_cost
                self.cost_basis = total_cost / total_shares

            # Update the last sale price if txn is
            # best data we have so far
            if self.last_sale_date is None or txn.dt > self.last_sale_date:
                self.last_sale_price = txn.price
                self.last_sale_date = txn.dt

        self.amount = total_shares

    def adjust_commission_cost_basis(self, commission):
        """
        A note about cost-basis in zipline: all positions are considered
        to share a cost basis, even if they were executed in different
        transactions with different commission costs, different prices, etc.

        Due to limitations about how zipline handles positions, zipline will
        currently spread an externally-delivered commission charge across
        all shares in a position.
        """

        if commission.sid != self.sid:
            raise Exception('Updating a commission for a different sid?')
        if commission.cost == 0.0:
            return

        # If we no longer hold this position, there is no cost basis to
        # adjust.
        if self.amount == 0:
            return

        prev_cost = self.cost_basis * self.amount
        new_cost = prev_cost + commission.cost
        self.cost_basis = new_cost / self.amount

    def __repr__(self):
        template = "sid: {sid}, amount: {amount}, cost_basis: {cost_basis}, \
last_sale_price: {last_sale_price}"
        return template.format(
            sid=self.sid,
            amount=self.amount,
            cost_basis=self.cost_basis,
            last_sale_price=self.last_sale_price
        )

    def to_dict(self):
        """
        Creates a dictionary representing the state of this position.
        Returns a dict object of the form:
        """
        return {
            'sid': self.sid,
            'amount': self.amount,
            'cost_basis': self.cost_basis,
            'last_sale_price': self.last_sale_price
        }


class cached_property(object):
    """
    Descriptor for caching property values in an instance's '_cache'.
    """
    def __init__(self, method):
        self._method_name = method.__name__
        self._method = method

    def __get__(self, instance, owner):
        if not hasattr(instance, '_cache'):
            instance._cache = {}
        if self._method_name not in instance._cache:
            instance._cache[self._method_name] = self._method(instance)
        return instance._cache[self._method_name]

    def __set__(self, instance, val):
        raise AttributeError(
            'Can not set property values: {}'.format(self._method_name))


class Position(object):
    def __init__(self, sid, amount=0, cost_basis=0.0,
                 last_sale_price=0.0, last_sale_date=None,
                 default_lot_method=None):

        self.sid = sid
        self.lots = set()
        if default_lot_method is None:
            default_lot_method = tax_lots.FIFO()
        assert not isinstance(default_lot_method, tax_lots.SpecificLots)
        self.default_lot_method = default_lot_method
        self._cache = {}

        if amount != 0:
            self.open(amount=amount, dt=last_sale_date, price=cost_basis)
            list(self.lots)[0].update_last_sale_price(last_sale_price)

    def clear_cache(self):
        self._cache.clear()

    @cached_property
    def closed_lots(self):
        return set(filter(lambda l: l.closed, self.lots))

    @cached_property
    def open_lots(self):
        return set(filter(lambda l: not l.closed, self.lots))

    @cached_property
    def amount(self):
        return sum(l.amount for l in self.open_lots)

    @cached_property
    def total_cost(self):
        return sum(l.total_cost for l in self.open_lots)

    @cached_property
    def cost_basis(self):
        return self.total_cost / self.amount

    @cached_property
    def market_value(self):
        return sum(l.market_value for l in self.open_lots)

    @cached_property
    def last_sale_price(self):
        return next(iter(self.open_lots)).last_sale_price

    def close(self, amount, dt, price, method=None, lots=None):

        if lots is None:
            lots = self.open_lots

        if method is None:
            method = self.default_lot_method

        closed_lots = method.close_lots(
            dt=dt, amount=amount, price=price, open_lots=lots)

        self.lots.update(closed_lots)
        self.clear_cache()

    def open(self, amount, dt, price):
        lot = tax_lots.Lot(
            sid=self.sid,
            dt=dt,
            amount=amount,
            cost_basis=price
        )
        self.lots.add(lot)
        self.clear_cache()

    def earn_dividend(self, dividend):
        return pd.concat([l.earn_dividend(dividend) for l in self.open_lots])

    def handle_split(self, split):
        return sum(l.handle_split(split) for l in self.open_lots)

    def update(self, txn):
        if self.sid != txn.sid:
            raise Exception('updating position with txn for a '
                            'different sid')
        self._update(txn.amount, txn.dt, txn.price)

    def _update(self, amount, dt, price):
        #FIXME Add commissions

        total_shares = self.amount + amount

        # close the entire position
        if total_shares == 0:
            self.close(amount=amount, dt=dt, price=price)

        else:
            prev_direction = copysign(1, self.amount)
            txn_direction = copysign(1, amount)

            # closing lots
            if prev_direction != txn_direction:

                # partial close
                if abs(amount) < abs(self.amount):
                    self.close(amount=amount, dt=dt, price=price)

                # full close and reopen in opposite direction
                else:
                    self.close(amount=self.amount, dt=txn.dt, price=price)
                    self.open(amount=self.amount - amount, dt=dt, price=price)

            # opening lots
            else:
                self.open(amount=amount, dt=dt, price=price)

        self.clear_cache()

    def adjust_commission_cost_basis(self, commission):
        raise NotImplementedError


class positiondict(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos
