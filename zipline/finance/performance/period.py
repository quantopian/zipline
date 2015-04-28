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

from zipline.finance.trading import with_environment
from zipline.assets import Future

try:
    # optional cython based OrderedDict
    from cyordereddict import OrderedDict
except ImportError:
    from collections import OrderedDict

from six import itervalues, iteritems

import zipline.protocol as zp

from zipline.utils.serialization_utils import (
    VERSION_LABEL
)

from .position_tracker import PositionTracker

log = logbook.Logger('Performance')
TRADE_TYPE = zp.DATASOURCE_TYPE.TRADE


def position_proxy(func):
    def _proxied(self, *args, **kwargs):
        meth_name = func.__name__
        meth = getattr(self.position_tracker, meth_name)
        return meth(*args, **kwargs)
    return _proxied


class ProxyError(Exception):
    def __init__(self):
        import inspect

        meth_name = inspect.stack()[1][3]
        TEMPLATE = "{meth_name} should have been proxied to position_tracker."
        msg = TEMPLATE.format(meth_name=meth_name)
        super(ProxyError, self).__init__(msg)


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
        self.ending_exposure = 0.0
        self.period_cash_flow = 0.0
        self.pnl = 0.0

        self.ending_cash = starting_cash
        # rollover initializes a number of self's attributes:
        self.rollover()
        self.keep_transactions = keep_transactions
        self.keep_orders = keep_orders

        # An object to recycle via assigning new values
        # when returning portfolio information.
        # So as not to avoid creating a new object for each event
        self._portfolio_store = zp.Portfolio()
        self._account_store = zp.Account()
        self.serialize_positions = serialize_positions

    _position_tracker = None

    @property
    def position_tracker(self):
        return self._position_tracker

    @position_tracker.setter
    def position_tracker(self, obj):
        if obj is None:
            raise ValueError("position_tracker can not be None")
        self._position_tracker = obj
        # we only calculate perf once we inject PositionTracker
        self.calculate_performance()

    def rollover(self):
        self.starting_value = self.ending_value
        self.starting_exposure = self.ending_exposure
        self.starting_cash = self.ending_cash
        self.period_cash_flow = 0.0
        self.pnl = 0.0
        self.processed_transactions = {}
        self.orders_by_modified = {}
        self.orders_by_id = OrderedDict()

    def handle_dividends_paid(self, net_cash_payment):
        if net_cash_payment:
            self.handle_cash_payment(net_cash_payment)
        self.calculate_performance()

    def handle_cash_payment(self, payment_amount):
        self.adjust_cash(payment_amount)

    def handle_commission(self, commission):
        # Deduct from our total cash pool.
        self.adjust_cash(-commission.cost)

    def adjust_cash(self, amount):
        self.period_cash_flow += amount

    def adjust_field(self, field, value):
        setattr(self, field, value)

    def calculate_performance(self):
        self.ending_value = self.calculate_positions_value()
        self.ending_exposure = self.calculate_positions_exposure()

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
            try:
                dt_orders = self.orders_by_modified[order.dt]
                if order.id in dt_orders:
                    del dt_orders[order.id]
            except KeyError:
                self.orders_by_modified[order.dt] = dt_orders = OrderedDict()
            dt_orders[order.id] = order
            # to preserve the order of the orders by modified date
            # we delete and add back. (ordered dictionary is sorted by
            # first insertion date).
            if order.id in self.orders_by_id:
                del self.orders_by_id[order.id]
            self.orders_by_id[order.id] = order

    @with_environment()
    def handle_execution(self, txn, env=None):
        asset = env.asset_finder.retrieve_asset(txn.sid)

        # Futures experience no cash flow on transactions
        if not isinstance(asset, Future):
            self.period_cash_flow -= txn.price * txn.amount

        if self.keep_transactions:
            try:
                self.processed_transactions[txn.dt].append(txn)
            except KeyError:
                self.processed_transactions[txn.dt] = [txn]

    # backwards compat. TODO: remove?
    @property
    def positions(self):
        return self.position_tracker.positions

    @property
    def position_amounts(self):
        return self.position_tracker.position_amounts

    @position_proxy
    def calculate_positions_exposure(self):
        raise ProxyError()

    @position_proxy
    def calculate_positions_value(self):
        raise ProxyError()

    @position_proxy
    def _longs_count(self):
        raise ProxyError()

    @position_proxy
    def _long_exposure(self):
        raise ProxyError()

    @position_proxy
    def _long_value(self):
        raise ProxyError()

    @position_proxy
    def _shorts_count(self):
        raise ProxyError()

    @position_proxy
    def _short_exposure(self):
        raise ProxyError()

    @position_proxy
    def _short_value(self):
        raise ProxyError()

    @position_proxy
    def _gross_exposure(self):
        raise ProxyError()

    @position_proxy
    def _gross_value(self):
        raise ProxyError()

    @position_proxy
    def _net_exposure(self):
        raise ProxyError()

    @position_proxy
    def _net_value(self):
        raise ProxyError()

    @property
    def _net_liquidation_value(self):
        return self.ending_cash + self._long_value() + self._short_value()

    def _gross_leverage(self):
        net_liq = self._net_liquidation_value
        if net_liq != 0:
            return self._gross_exposure() / net_liq

        return np.inf

    def _net_leverage(self):
        net_liq = self._net_liquidation_value
        if net_liq != 0:
            return self._net_exposure() / net_liq

        return np.inf

    def __core_dict(self):
        rval = {
            'ending_value': self.ending_value,
            'ending_exposure': self.ending_exposure,
            # this field is renamed to capital_used for backward
            # compatibility.
            'capital_used': self.period_cash_flow,
            'starting_value': self.starting_value,
            'starting_exposure': self.starting_exposure,
            'starting_cash': self.starting_cash,
            'ending_cash': self.ending_cash,
            'portfolio_value': self.ending_cash + self.ending_value,
            'pnl': self.pnl,
            'returns': self.returns,
            'period_open': self.period_open,
            'period_close': self.period_close,
            'gross_leverage': self._gross_leverage(),
            'net_leverage': self._net_leverage(),
            'short_exposure': self._short_exposure(),
            'long_exposure': self._long_exposure(),
            'short_value': self._short_value(),
            'long_value': self._long_value(),
            'longs_count': self._longs_count(),
            'shorts_count': self._shorts_count()
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
                try:
                    transactions = [x.to_dict()
                                    for x in self.processed_transactions[dt]]
                except KeyError:
                    transactions = []
            else:
                transactions = \
                    [y.to_dict()
                     for x in itervalues(self.processed_transactions)
                     for y in x]
            rval['transactions'] = transactions

        if self.keep_orders:
            if dt:
                # only include orders modified as of the given dt.
                try:
                    orders = [x.to_dict()
                              for x in itervalues(self.orders_by_modified[dt])]
                except KeyError:
                    orders = []
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
        portfolio.positions_exposure = self.ending_exposure
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
        account.total_positions_value = \
            getattr(self, 'total_positions_exposure', self.ending_exposure)
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
            getattr(self, 'leverage', self._gross_leverage())
        account.net_leverage = self._net_leverage()
        account.net_liquidation = \
            getattr(self, 'net_liquidation', self._net_liquidation_value)
        return account

    @position_proxy
    def get_positions(self):
        raise ProxyError()

    @position_proxy
    def get_positions_list(self):
        raise ProxyError()

    def __getstate__(self):
        state_dict = {k: v for k, v in iteritems(self.__dict__)
                      if not k.startswith('_')}

        state_dict['_portfolio_store'] = self._portfolio_store
        state_dict['_account_store'] = self._account_store

        state_dict['processed_transactions'] = \
            dict(self.processed_transactions)
        state_dict['orders_by_id'] = \
            dict(self.orders_by_id)
        state_dict['orders_by_modified'] = \
            dict(self.orders_by_modified)

        STATE_VERSION = 2
        state_dict[VERSION_LABEL] = STATE_VERSION
        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 1
        version = state.pop(VERSION_LABEL)

        if version < OLDEST_SUPPORTED_STATE:
            raise BaseException("PerformancePeriod saved state is too old.")

        processed_transactions = {}
        processed_transactions.update(state.pop('processed_transactions'))

        orders_by_id = OrderedDict()
        orders_by_id.update(state.pop('orders_by_id'))

        orders_by_modified = {}
        orders_by_modified.update(state.pop('orders_by_modified'))
        self.processed_transactions = processed_transactions
        self.orders_by_id = orders_by_id
        self.orders_by_modified = orders_by_modified

        # pop positions to use for v1
        positions = state.pop('positions', None)
        self.__dict__.update(state)

        if version == 1:
            # version 1 had PositionTracker logic inside of Period
            # we create the PositionTracker here.
            # Note: that in V2 it is assumed that the position_tracker
            # will be dependency injected and so is not reconstructed
            assert positions is not None, "positions should exist in v1"
            position_tracker = PositionTracker()
            position_tracker.update_positions(positions)
            self.position_tracker = position_tracker
