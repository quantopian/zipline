
"""
Position Lots
=============

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | sid             | the identifier for the security held in this lot.  |
    +-----------------+----------------------------------------------------+
    | amount          | shares held in the lot                             |
    +-----------------+----------------------------------------------------+
    | dt              | date the lot was opened                            |
    +-----------------+----------------------------------------------------+
    | cost_basis      | cost (per share) at which the lot was opened       |
    +-----------------+----------------------------------------------------+
    | last_sale_price | price at last sale of the security                 |
    +-----------------+----------------------------------------------------+
    | closed          | whether the lot has been closed or remains open    |
    +-----------------+----------------------------------------------------+
    | close_dt        | the date on which the lot was closed               |
    +-----------------+----------------------------------------------------+
    | total_cost      | the total price paid to open the lot               |
    +-----------------+----------------------------------------------------+
    | market_value    | the current market value of the lot                |
    +-----------------+----------------------------------------------------+

"""

from __future__ import division
from math import (
    copysign,
    floor,
)

import logbook
import zipline.protocol as zp
log = logbook.Logger('Lot')


class Lot(object):
    def __init__(self, sid, dt, amount, cost_basis):
        """
        Note: no requirement that amount is a whole number. Fractional shares
        are OK.

        :param sid: sid
        :param dt: open date
        :param amount: lot shares
        :param cost: cost per share
        """
        self.sid = sid
        self.dt = dt
        self.amount = amount
        self.cost_basis = cost_basis
        self.last_sale_price = cost_basis
        self.closed = False
        self.close_dt = None

    @classmethod
    def from_transaction(cls, txn):
        lot = cls(
            sid=txn.sid,
            dt=txn.dt,
            amount=txn.amount,
            cost_basis=txn.price + txn.commission / txn.amount)
        return lot

    @property
    def market_value(self):
        return self.amount * self.last_sale_price

    @property
    def total_cost(self):
        return self.amount * self.cost_basis

    @property
    def pnl(self):
        return self.amount * (self.last_sale_price - self.cost_basis)

    def update_last_sale(self, dt, price):
        self.last_sale_price = price
        self.last_sale_date = dt

    def close(self, dt, amount, price):
        """
        Closes some or all of the lot. In addition to modifiying this lot
        inplace, a new Lot is returned representing the closed shares. If the
        lot is entirely closed (amount == self.amount) then the returned lot
        is this one.

        :param dt: close date
        :param amount: close amount (must have opposite sign of lot amount and
            must be smaller in absolute value)
        :param price: the close price
        """
        if copysign(1, amount) == copysign(1, self.amount):
            raise ValueError(
                'Close amount {new_amt} can not have the same sign as'
                'Lot amount {amt}.'.format(new_amt=amount, amt=self.amount))

        if abs(amount) > abs(self.amount):
            raise ValueError(
                'Tried to close {new_amt} shares but this lot only '
                'contains {amt}.'.format(new_amt=amount, amt=self.amount))

        self.amount -= amount

        if self.amount == 0:
            closed_lot = self
        else:
            closed_lot = Lot(
                sid=self.sid,
                dt=self.dt,
                amount=amount,
                cost_basis=self.cost_basis)

        closed_lot.last_sale_price = price
        closed_lot.closed = True
        closed_lot.close_dt = dt

        return closed_lot

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
        Update the lot by the split ratio. If the split creates fractional
        shares, return them to be turned into cash. If the Lot already contained
        fractional shares, then they are maintained.

        Returns the unused cash.

        NOTE removes rounding from old Position code
        """
        if self.sid != split.sid:
            raise Exception("updating split with the wrong sid!")

        ratio = split.ratio

        log.info("handling split for sid = " + str(split.sid) +
                 ", ratio = " + str(split.ratio))
        log.info("before split: " + str(self))

        if floor(self.amount) == self.amount:
            full_share_count = self.amount
            fractional_share_count = 0
        else:
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

        # adjust the cost_basis to the nearest cent, e.g., 60.0
        new_cost = self.cost_basis * ratio

        # adjust the last sale price
        new_last_sale_price = self.last_sale_price * ratio

        self.cost_basis = new_cost
        self.last_sale_price = new_last_sale_price
        self.amount = full_share_count

        return_cash = fractional_share_count * new_cost

        log.info("after split: " + str(self))
        log.info("returning cash: " + str(return_cash))

        # return the leftover cash, which will be converted into cash
        return return_cash

    def to_dict(self):
        return {
            'sid': self.sid,
            'dt': self.dt,
            'amount': self.amount,
            'cost_basis': self.cost_basis,
            'total_cost': self.total_cost,
            'last_sale_price': self.last_sale_price,
            'market_value': self.market_value,
            'closed': self.closed,
            'closed_dt': self.closed_dt
        }


class LotMethod(object):
    def sort_lots(self, lots):
        raise NotImplementedError()

    def close_lots(self, amount, dt, price, open_lots):
        # get lots with opposite signs from amount
        sgn = copysign(1, -amount)
        lots = (l for l in open_lots if copysign(1, l.amount) == sgn)

        # sort lots
        sorted_lots = iter(self.sort_lots(lots))
        closed_lots = []

        # close lots until shares are exhausted
        while amount != 0:
            try:
                lot = next(sorted_lots)
            except StopIteration:
                raise ValueError(
                    'Tried to close more lot shares than are available.')
            abs_lot_amt = min(abs(lot.amount), abs(amount))
            lot_amt = copysign(abs_lot_amt, amount)
            closed_lot = lot.close(dt=dt, price=price, amount=lot_amt)
            closed_lots.append(closed_lot)
            amount -= lot_amt

        return closed_lots


class FIFO(LotMethod):
    def sort_lots(self, lots):
        return sorted(lots, key=lambda l: l.dt)


class LIFO(LotMethod):
    def sort_lots(self, lots):
        return sorted(lots, key=lambda l: l.dt, reverse=True)


class HIFO(LotMethod):
    def sort_lots(self, lots):
        return sorted(lots, key=lambda l: l.cost_basis)


class LIFO(LotMethod):
    def sort_lots(self, lots):
        return sorted(lots, key=lambda l: l.cost_basis, reverse=True)


class SpecificLots(LotMethod):
    def __init__(self, lots):
        self.lots = lots

    def sort_lots(self, lots):
        return self.lots



