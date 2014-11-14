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
from collections import (
    defaultdict,
    OrderedDict,
)
from six import iteritems, itervalues

import zipline.protocol as zp
from zipline.utils.state_methods import (
    _defaultdict_list_get_state,
    _defaultdict_ordered_get_state
)
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
        self._account_store = zp.Account()
        self._positions_store = zp.Positions()
        self.serialize_positions = serialize_positions

        self._unpaid_dividends = pd.DataFrame(
            columns=zp.DIVIDEND_PAYMENT_FIELDS,
        )

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

        # Add cash equal to the net cash payed from all dividends.  Note that
        # "negative cash" is effectively paid if we're short a security,
        # representing the fact that we're required to reimburse the owner of
        # the stock for any dividends paid while borrowing.
        net_cash_payment = payments['cash_amount'].fillna(0).sum()
        if net_cash_payment:
            self.handle_cash_payment(net_cash_payment)

        # Add stock for any stock dividends paid.  Again, the values here may
        # be negative in the case of short positions.
        stock_payments = payments[payments['payment_sid'].notnull()]
        for _, row in stock_payments.iterrows():
            stock = row['payment_sid']
            share_count = row['share_count']
            position = self.positions[stock]

            position.amount += share_count
            self.ensure_position_index(stock)
            self._position_amounts[stock] = position.amount
            self._position_last_sale_prices[stock] = \
                position.last_sale_price

        # Recalculate performance after applying dividend benefits.
        self.calculate_performance()

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

    def adjust_field(self, field, value):
        setattr(self, field, value)

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

        # NOTE: self.positions has defaultdict semantics, so this will create
        # an empty position if one does not already exist.
        position = self.positions[txn.sid]
        position.update(txn)
        self.ensure_position_index(txn.sid)
        self._position_amounts[txn.sid] = position.amount
        self._position_last_sale_prices[txn.sid] = position.last_sale_price

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

    def _get_state(self):
        """
        Return a serialized version of the performance period.
        """
        # Go through and call any custom serialization methods we've added
        state_dict = {}
        for k, v in self.__dict__.iteritems():
            if (not k.startswith('_')):
                state_dict[k] = v

        state_dict['_portfolio_store'] = self._portfolio_store
        state_dict['_account_store'] = self._account_store
        state_dict['_positions_store'] = self._positions_store
        state_dict['_position_amounts'] = self._position_amounts
        state_dict['_position_last_sale_prices'] =\
            self._position_last_sale_prices

        # We need to handle the defaultdict specially, otherwise
        # msgpack will unpack it as a dict, causing KeyError
        # nastiness.
        state_dict['processed_transactions'] =\
            _defaultdict_list_get_state(self.processed_transactions)
        state_dict['orders_by_modified'] =\
            _defaultdict_ordered_get_state(self.orders_by_modified)

        return 'PerformancePeriod', state_dict

    def _set_state(self, saved_state):
        """
        Reconstruct this performance period from saved_state.
        """
        self.__dict__.update(saved_state)

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

    def as_account(self):
        account = self._account_store

        # If no attribute is found on the PerformancePeriod resort to the
        # following default values. If an attribute is found use the existing
        # value. For instance, a broker may provide updates to these
        # attributes. In this case we do not want to over write the broker
        # values with the default values.
        account.settled_cash = \
            getattr(self, 'settled_cash', self.ending_cash)
        account.accrued_interest = \
            getattr(self, 'accrued_interest', 0.0)
        account.buying_power = \
            getattr(self, 'buying_power', float('inf'))
        account.equity_with_loan = \
            getattr(self, 'equity_with_loan',
                    self.ending_cash + self.ending_value)
        account.total_positions_value = \
            getattr(self, 'total_positions_value', self.ending_value)
        account.regt_equity = \
            getattr(self, 'regt_equity', self.ending_cash)
        account.regt_margin = \
            getattr(self, 'regt_margin', float('inf'))
        account.initial_margin_requirement = \
            getattr(self, 'initial_margin_requirement', 0.0)
        account.maintenance_margin_requirement = \
            getattr(self, 'maintenance_margin_requirement', 0.0)
        account.available_funds = \
            getattr(self, 'available_funds', self.ending_cash)
        account.excess_liquidity = \
            getattr(self, 'excess_liquidity', self.ending_cash)
        account.cushion = \
            getattr(self, 'cushion',
                    self.ending_cash / (self.ending_cash + self.ending_value))
        account.day_trades_remaining = \
            getattr(self, 'day_trades_remaining', float('inf'))
        account.leverage = \
            getattr(self, 'leverage',
                    self.ending_value / (self.ending_value + self.ending_cash))
        account.net_liquidation = \
            getattr(self, 'net_liquidation',
                    self.ending_cash + self.ending_value)
        return account

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
