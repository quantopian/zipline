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
import logbook
import math

log = logbook.Logger('Performance')


class Position(object):

    def __init__(self, sid, amount=0, cost_basis=0.0,
                 last_sale_price=0.0, last_sale_date=None,
                 dividends=None):
        self.sid = sid
        self.amount = amount
        self.cost_basis = cost_basis  # per share
        self.last_sale_price = last_sale_price
        self.last_sale_date = last_sale_date
        self.dividends = dividends or []

    def update_dividends(self, midnight_utc):
        """
        midnight_utc is the 0 hour for the current (not yet open) trading day.
        This method will be invoked at the end of the market
        close handling, before the next market open.
        """
        payment = 0.0
        unpaid_dividends = []
        for dividend in self.dividends:
            if midnight_utc == dividend.ex_date:
                # if we own shares at midnight of the div_ex date
                # we are entitled to the dividend.
                dividend.amount_on_ex_date = self.amount
                if dividend.net_amount:
                    dividend.payment = self.amount * dividend.net_amount
                else:
                    dividend.payment = self.amount * dividend.gross_amount

            if midnight_utc == dividend.pay_date:
                # if it is the payment date, include this
                # dividend's actual payment (calculated on
                # ex_date)
                payment += dividend.payment
            else:
                unpaid_dividends.append(dividend)

        self.dividends = unpaid_dividends
        return payment

    def add_dividend(self, dividend):
        self.dividends.append(dividend)

    # Update the position by the split ratio, and return the
    # resulting fractional share that will be converted into cash.

    # Returns the unused cash.
    def handle_split(self, split):
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

        # ie, 33.333
        raw_share_count = self.amount / float(ratio)

        # ie, 33
        full_share_count = math.floor(raw_share_count)

        # ie, 0.333
        fractional_share_count = raw_share_count - full_share_count

        # adjust the cost basis to the nearest cent, ie, 60.0
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
            prev_direction = math.copysign(1, self.amount)
            txn_direction = math.copysign(1, txn.amount)

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


class positiondict(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos
