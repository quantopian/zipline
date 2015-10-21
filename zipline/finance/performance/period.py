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
    |               | from buying and selling assets in the period.        |
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

from collections import namedtuple
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

log = logbook.Logger('Performance')
TRADE_TYPE = zp.DATASOURCE_TYPE.TRADE


PeriodStats = namedtuple('PeriodStats',
                         ['net_liquidation',
                          'gross_leverage',
                          'net_leverage',
                          'ending_cash',
                          'pnl',
                          'returns',
                          'portfolio_value'])


def calc_net_liquidation(ending_cash, long_value, short_value):
    return ending_cash + long_value + short_value


def calc_leverage(exposure, net_liq):
    if net_liq != 0:
        return exposure / net_liq

    return np.inf


def calc_period_stats(pos_stats, starting_cash, starting_value,
                      period_cash_flow, payout):
    total_at_start = starting_cash + starting_value
    ending_cash = starting_cash + period_cash_flow + payout
    total_at_end = ending_cash + pos_stats.net_value

    pnl = total_at_end - total_at_start
    if total_at_start != 0:
        returns = pnl / total_at_start
    else:
        returns = 0.0

    portfolio_value = ending_cash + pos_stats.net_value + payout

    net_liq = calc_net_liquidation(ending_cash,
                                   pos_stats.long_value,
                                   pos_stats.short_value)
    gross_leverage = calc_leverage(pos_stats.gross_exposure, net_liq)
    net_leverage = calc_leverage(pos_stats.net_exposure, net_liq)

    return PeriodStats(
        net_liquidation=net_liq,
        gross_leverage=gross_leverage,
        net_leverage=net_leverage,
        ending_cash=ending_cash,
        pnl=pnl,
        returns=returns,
        portfolio_value=portfolio_value)


class PerformancePeriod(object):

    def __init__(
            self,
            starting_cash,
            asset_finder,
            data_portal,
            period_open=None,
            period_close=None,
            keep_transactions=True,
            keep_orders=False,
            serialize_positions=True):

        self.asset_finder = asset_finder
        self.data_portal = data_portal

        self.period_open = period_open
        self.period_close = period_close

        self.period_cash_flow = 0.0

        self.starting_cash = starting_cash
        self.starting_value = 0.0
        self.starting_exposure = 0.0

        self.keep_transactions = keep_transactions
        self.keep_orders = keep_orders

        self.processed_transactions = {}
        self.orders_by_modified = {}
        self.orders_by_id = OrderedDict()

        # An object to recycle via assigning new values
        # when returning portfolio information.
        # So as not to avoid creating a new object for each event
        self._portfolio_store = zp.Portfolio()
        self._account_store = zp.Account()
        self.serialize_positions = serialize_positions

        # This dict contains the known cash flow multipliers for sids and is
        # keyed on sid
        self._execution_cash_flow_multipliers = {}

    def rollover(self, pos_stats, prev_period_stats):
        self.starting_value = pos_stats.net_value
        self.starting_exposure = pos_stats.net_exposure
        self.starting_cash = prev_period_stats.ending_cash
        self.period_cash_flow = 0.0
        self.processed_transactions = {}
        self.orders_by_modified = {}
        self.orders_by_id = OrderedDict()

    def handle_dividends_paid(self, net_cash_payment):
        if net_cash_payment:
            self.handle_cash_payment(net_cash_payment)

    def handle_cash_payment(self, payment_amount):
        self.adjust_cash(payment_amount)

    def handle_commission(self, commission):
        # Deduct from our total cash pool.
        self.adjust_cash(-commission.cost)

    def adjust_cash(self, amount):
        self.period_cash_flow += amount

    def adjust_field(self, field, value):
        setattr(self, field, value)

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

    def handle_execution(self, txn):
        self.period_cash_flow += self._calculate_execution_cash_flow(txn)

        if self.keep_transactions:
            try:
                self.processed_transactions[txn.dt].append(txn)
            except KeyError:
                self.processed_transactions[txn.dt] = [txn]

    def _calculate_execution_cash_flow(self, txn):
        """
        Calculates the cash flow from executing the given transaction
        """
        # Check if the multiplier is cached. If it is not, look up the asset
        # and cache the multiplier.
        try:
            multiplier = self._execution_cash_flow_multipliers[txn.sid]
        except KeyError:
            asset = self.asset_finder.retrieve_asset(txn.sid)
            # Futures experience no cash flow on transactions
            if isinstance(asset, Future):
                multiplier = 0
            else:
                multiplier = 1
            self._execution_cash_flow_multipliers[txn.sid] = multiplier

        # Calculate and return the cash flow given the multiplier
        return -1 * txn.price * txn.amount * multiplier

    def stats(self, positions, pos_stats):
        # TODO: passing positions here seems off, since we have already
        # calculated pos_stats.
        futures_payouts = []
        for sid, pos in positions.iteritems():
            asset = self.asset_finder.retrieve_asset(sid)
            if isinstance(asset, Future):
                old_price_dt = max(pos.last_sale_date,
                                   self.period_open)
                if old_price_dt == pos.last_sale_date:
                    continue
                old_price = self.data_portal.get_previous_price(
                    sid,
                    'close',
                    dt=old_price_dt)
                price = self.data_portal.get_spot_value(
                    sid, 'close', dt=self.period_close)
                payout = (
                    (price - old_price)
                    *
                    asset.contract_multiplier
                    *
                    pos.amount
                )
                futures_payouts.append(payout)
        futures_payout = sum(futures_payouts)

        return calc_period_stats(pos_stats,
                                 self.starting_cash,
                                 self.starting_value,
                                 self.period_cash_flow,
                                 futures_payout)

    def __core_dict(self, pos_stats, period_stats):
        rval = {
            'ending_value': pos_stats.net_value,
            'ending_exposure': pos_stats.net_exposure,
            # this field is renamed to capital_used for backward
            # compatibility.
            'capital_used': self.period_cash_flow,
            'starting_value': self.starting_value,
            'starting_exposure': self.starting_exposure,
            'starting_cash': self.starting_cash,
            'ending_cash': period_stats.ending_cash,
            'portfolio_value': period_stats.portfolio_value,
            'pnl': period_stats.pnl,
            'returns': period_stats.returns,
            'period_open': self.period_open,
            'period_close': self.period_close,
            'gross_leverage': period_stats.gross_leverage,
            'net_leverage': period_stats.net_leverage,
            'short_exposure': pos_stats.short_exposure,
            'long_exposure': pos_stats.long_exposure,
            'short_value': pos_stats.short_value,
            'long_value': pos_stats.long_value,
            'longs_count': pos_stats.longs_count,
            'shorts_count': pos_stats.shorts_count,
        }

        return rval

    def to_dict(self, pos_stats, period_stats, position_tracker, dt=None):
        """
        Creates a dictionary representing the state of this performance
        period. See header comments for a detailed description.

        Kwargs:
            dt (datetime): If present, only return transactions for the dt.
        """
        rval = self.__core_dict(pos_stats, period_stats)

        if self.serialize_positions:
            positions = position_tracker.get_positions_list()
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

    def as_portfolio(self, pos_stats, period_stats, position_tracker, dt):
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
        portfolio.portfolio_value = period_stats.portfolio_value
        portfolio.pnl = period_stats.pnl
        portfolio.returns = period_stats.returns
        portfolio.cash = period_stats.ending_cash
        portfolio.start_date = self.period_open
        portfolio.positions = position_tracker.get_positions(dt)
        portfolio.positions_value = pos_stats.net_value
        portfolio.positions_exposure = pos_stats.net_exposure
        return portfolio

    def as_account(self, pos_stats, period_stats):
        account = self._account_store

        # If no attribute is found on the PerformancePeriod resort to the
        # following default values. If an attribute is found use the existing
        # value. For instance, a broker may provide updates to these
        # attributes. In this case we do not want to over write the broker
        # values with the default values.
        account.settled_cash = \
            getattr(self, 'settled_cash', period_stats.ending_cash)
        account.accrued_interest = \
            getattr(self, 'accrued_interest', 0.0)
        account.buying_power = \
            getattr(self, 'buying_power', float('inf'))
        account.equity_with_loan = \
            getattr(self, 'equity_with_loan', period_stats.portfolio_value)
        account.total_positions_value = \
            getattr(self, 'total_positions_value', pos_stats.net_value)
        account.total_positions_value = \
            getattr(self, 'total_positions_exposure', pos_stats.net_exposure)
        account.regt_equity = \
            getattr(self, 'regt_equity', period_stats.ending_cash)
        account.regt_margin = \
            getattr(self, 'regt_margin', float('inf'))
        account.initial_margin_requirement = \
            getattr(self, 'initial_margin_requirement', 0.0)
        account.maintenance_margin_requirement = \
            getattr(self, 'maintenance_margin_requirement', 0.0)
        account.available_funds = \
            getattr(self, 'available_funds', period_stats.ending_cash)
        account.excess_liquidity = \
            getattr(self, 'excess_liquidity', period_stats.ending_cash)
        account.cushion = \
            getattr(self, 'cushion',
                    period_stats.ending_cash / period_stats.portfolio_value)
        account.day_trades_remaining = \
            getattr(self, 'day_trades_remaining', float('inf'))
        account.leverage = getattr(self, 'leverage',
                                   period_stats.gross_leverage)
        account.net_leverage = period_stats.net_leverage

        account.net_liquidation = getattr(self, 'net_liquidation',
                                          period_stats.net_liquidation)
        return account

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

        STATE_VERSION = 3
        state_dict[VERSION_LABEL] = STATE_VERSION
        return state_dict

    def __setstate__(self, state):

        OLDEST_SUPPORTED_STATE = 3
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

        self._execution_cash_flow_multipliers = {}

        self.__dict__.update(state)


class TodaysPerformance(PerformancePeriod):
    pass


class CumulativePerformance(PerformancePeriod):
    pass
