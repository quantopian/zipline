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

Performance Period
==================

Performance Periods are updated with every trade. When calling
code needs a portfolio object that fulfills the algorithm
protocol, use the PerformancePeriod.as_portfolio method. See that
method for comments on the specific fields provided (and
omitted).

    +---------------+------------------------------------------------------+
    | key           | value                                                |
    +===============+======================================================+
    | ending_value  | the total market value of the positions held at the  |
    |               | end of the period                                    |
    +---------------+------------------------------------------------------+
    | cash_flow     | the cash flow in the period (negative means spent)   |
    |               | from buying and selling securities in the period.    |
    |               | Includes dividend payments in the period as well.    |
    +---------------+------------------------------------------------------+
    | starting_value| the total market value of the positions held at the  |
    |               | start of the period                                  |
    +---------------+------------------------------------------------------+
    | starting_cash | cash on hand at the beginning of the period          |
    +---------------+------------------------------------------------------+
    | ending_cash   | cash on hand at the end of the period                |
    +---------------+------------------------------------------------------+
    | positions     | a list of dicts representing positions, see          |
    |               | :py:meth:`Position.to_dict()`                        |
    |               | for details on the contents of the dict              |
    +---------------+------------------------------------------------------+
    | pnl           | Dollar value profit and loss, for both realized and  |
    |               | unrealized gains.                                    |
    +---------------+------------------------------------------------------+
    | returns       | percentage returns for the entire portfolio over the |
    |               | period                                               |
    +---------------+------------------------------------------------------+
    | cumulative\   | The net capital used (positive is spent) during      |
    | _capital_used | the period                                           |
    +---------------+------------------------------------------------------+
    | max_capital\  | The maximum amount of capital deployed during the    |
    | _used         | period.                                              |
    +---------------+------------------------------------------------------+
    | period_close  | The last close of the market in period. datetime in  |
    |               | pytz.utc timezone.                                   |
    +---------------+------------------------------------------------------+
    | period_open   | The first open of the market in period. datetime in  |
    |               | pytz.utc timezone.                                   |
    +---------------+------------------------------------------------------+
    | transactions  | all the transactions that were acrued during this    |
    |               | period. Unset/missing for cumulative periods.        |
    +---------------+------------------------------------------------------+


"""

from __future__ import division
import logbook

import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

from six import iteritems, itervalues

import zipline.protocol as zp
from . position import positiondict

log = logbook.Logger('Performance')


class PerformancePeriod(object):

    def __init__(
            self,
            starting_cash,
            period_open=None,
            period_close=None,
            keep_transactions=True,
            keep_orders=False,
            serialize_positions=True):

        self.period_open = period_open
        self.period_close = period_close

        self.ending_value = 0.0
        self.period_cash_flow = 0.0
        self.pnl = 0.0
        # sid => position object
        self.positions = positiondict()
        self.ending_cash = starting_cash
        # rollover initializes a number of self's attributes:
        self.rollover()
        self.keep_transactions = keep_transactions
        self.keep_orders = keep_orders

        # Arrays for quick calculations of positions value
        self._position_amounts = pd.Series()
        self._position_last_sale_prices = pd.Series()

        self.calculate_performance()

        # An object to recycle via assigning new values
        # when returning portfolio information.
        # So as not to avoid creating a new object for each event
        self._portfolio_store = zp.Portfolio()
        self._positions_store = zp.Positions()
        self.serialize_positions = serialize_positions

    def rollover(self):
        self.starting_value = self.ending_value
        self.starting_cash = self.ending_cash
        self.period_cash_flow = 0.0
        self.pnl = 0.0
        self.processed_transactions = defaultdict(list)
        self.orders_by_modified = defaultdict(OrderedDict)
        self.orders_by_id = OrderedDict()

    def ensure_position_index(self, sid):
        try:
            self._position_amounts[sid]
            self._position_last_sale_prices[sid]
        except (KeyError, IndexError):
            self._position_amounts = \
                self._position_amounts.append(pd.Series({sid: 0.0}))
            self._position_last_sale_prices = \
                self._position_last_sale_prices.append(pd.Series({sid: 0.0}))

    def add_dividend(self, div):
        # The dividend is received on midnight of the dividend
        # declared date. We calculate the dividends based on the amount of
        # stock owned on midnight of the ex dividend date. However, the cash
        # is not dispersed until the payment date, which is
        # included in the event.
        self.positions[div.sid].add_dividend(div)

    def handle_split(self, split):
        if split.sid in self.positions:
            # Make the position object handle the split. It returns the
            # leftover cash from a fractional share, if there is any.
            position = self.positions[split.sid]
            leftover_cash = position.handle_split(split)
            self._position_amounts[split.sid] = position.amount
            self._position_last_sale_prices[split.sid] = \
                position.last_sale_price

            if leftover_cash > 0:
                self.handle_cash_payment(leftover_cash)

    def update_dividends(self, todays_date):
        """
        Check the payment date and ex date against today's date
        to determine if we are owed a dividend payment or if the
        payment has been disbursed.
        """
        cash_payments = 0.0
        for sid, pos in iteritems(self.positions):
            cash_payments += pos.update_dividends(todays_date)

        # credit our cash balance with the dividend payments, or
        # if we are short, debit our cash balance with the
        # payments.
        # debit our cumulative cash spent with the dividend
        # payments, or credit our cumulative cash spent if we are
        # short the stock.
        self.handle_cash_payment(cash_payments)

        # recalculate performance, including the dividend
        # payments
        self.calculate_performance()

    def handle_cash_payment(self, payment_amount):
        self.adjust_cash(payment_amount)

    def handle_commission(self, commission):
        # Deduct from our total cash pool.
        self.adjust_cash(-commission.cost)
        # Adjust the cost basis of the stock if we own it
        if commission.sid in self.positions:
            self.positions[commission.sid].\
                adjust_commission_cost_basis(commission)

    def adjust_cash(self, amount):
        self.period_cash_flow += amount

    def calculate_performance(self):
        self.ending_value = self.calculate_positions_value()

        total_at_start = self.starting_cash + self.starting_value
        self.ending_cash = self.starting_cash + self.period_cash_flow
        total_at_end = self.ending_cash + self.ending_value

        self.pnl = total_at_end - total_at_start
        if total_at_start != 0:
            self.returns = self.pnl / total_at_start
        else:
            self.returns = 0.0

    def record_order(self, order):
        if self.keep_orders:
            dt_orders = self.orders_by_modified[order.dt]
            if order.id in dt_orders:
                del dt_orders[order.id]
            dt_orders[order.id] = order
            # to preserve the order of the orders by modified date
            # we delete and add back. (ordered dictionary is sorted by
            # first insertion date).
            if order.id in self.orders_by_id:
                del self.orders_by_id[order.id]
            self.orders_by_id[order.id] = order

    def update_position(self, sid, amount=None, last_sale_price=None,
                        last_sale_date=None, cost_basis=None):
        pos = self.positions[sid]
        self.ensure_position_index(sid)

        if amount is not None:
            pos.amount = amount
            self._position_amounts[sid] = amount
        if last_sale_price is not None:
            pos.last_sale_price = last_sale_price
            self._position_last_sale_prices[sid] = last_sale_price
        if last_sale_date is not None:
            pos.last_sale_date = last_sale_date
        if cost_basis is not None:
            pos.cost_basis = cost_basis

    def execute_transaction(self, txn):
        # Update Position
        # ----------------
        position = self.positions[txn.sid]
        position.update(txn)
        self.ensure_position_index(txn.sid)
        self._position_amounts[txn.sid] = position.amount

        self.period_cash_flow -= txn.price * txn.amount

        if self.keep_transactions:
            self.processed_transactions[txn.dt].append(txn)

    def calculate_positions_value(self):
        return np.dot(self._position_amounts, self._position_last_sale_prices)

    def update_last_sale(self, event):
        if event.sid not in self.positions:
            return

        if event.type != zp.DATASOURCE_TYPE.TRADE:
            return

        if not pd.isnull(event.price):
            # isnan check will keep the last price if its not present
            self.update_position(event.sid, last_sale_price=event.price,
                                 last_sale_date=event.dt)

    def __core_dict(self):
        rval = {
            'ending_value': self.ending_value,
            # this field is renamed to capital_used for backward
            # compatibility.
            'capital_used': self.period_cash_flow,
            'starting_value': self.starting_value,
            'starting_cash': self.starting_cash,
            'ending_cash': self.ending_cash,
            'portfolio_value': self.ending_cash + self.ending_value,
            'pnl': self.pnl,
            'returns': self.returns,
            'period_open': self.period_open,
            'period_close': self.period_close
        }

        return rval

    def to_dict(self, dt=None):
        """
        Creates a dictionary representing the state of this performance
        period. See header comments for a detailed description.

        Kwargs:
            dt (datetime): If present, only return transactions for the dt.
        """
        rval = self.__core_dict()

        if self.serialize_positions:
            positions = self.get_positions_list()
            rval['positions'] = positions

        # we want the key to be absent, not just empty
        if self.keep_transactions:
            if dt:
                # Only include transactions for given dt
                transactions = [x.to_dict()
                                for x in self.processed_transactions[dt]]
            else:
                transactions = \
                    [y.to_dict()
                     for x in itervalues(self.processed_transactions)
                     for y in x]
            rval['transactions'] = transactions

        if self.keep_orders:
            if dt:
                # only include orders modified as of the given dt.
                orders = [x.to_dict()
                          for x in itervalues(self.orders_by_modified[dt])]
            else:
                orders = [x.to_dict() for x in itervalues(self.orders_by_id)]
            rval['orders'] = orders

        return rval

    def as_portfolio(self):
        """
        The purpose of this method is to provide a portfolio
        object to algorithms running inside the same trading
        client. The data needed is captured raw in a
        PerformancePeriod, and in this method we rename some
        fields for usability and remove extraneous fields.
        """
        # Recycles containing objects' Portfolio object
        # which is used for returning values.
        # as_portfolio is called in an inner loop,
        # so repeated object creation becomes too expensive
        portfolio = self._portfolio_store
        # maintaining the old name for the portfolio field for
        # backward compatibility
        portfolio.capital_used = self.period_cash_flow
        portfolio.starting_cash = self.starting_cash
        portfolio.portfolio_value = self.ending_cash + self.ending_value
        portfolio.pnl = self.pnl
        portfolio.returns = self.returns
        portfolio.cash = self.ending_cash
        portfolio.start_date = self.period_open
        portfolio.positions = self.get_positions()
        portfolio.positions_value = self.ending_value
        return portfolio

    def get_positions(self):

        positions = self._positions_store

        for sid, pos in iteritems(self.positions):
            if pos.amount != 0 and sid not in positions:
                positions[sid] = zp.Position(sid)
            position = positions[sid]
            position.amount = pos.amount
            position.cost_basis = pos.cost_basis
            position.last_sale_price = pos.last_sale_price

            # Remove positions with no amounts from portfolio container
            # that is passed to the algorithm.
            # These positions are still stored internally for use with
            # dividends etc.
            if pos.amount == 0 and sid in positions:
                del positions[sid]

        return positions

    def get_positions_list(self):
        positions = []
        for sid, pos in iteritems(self.positions):
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions
