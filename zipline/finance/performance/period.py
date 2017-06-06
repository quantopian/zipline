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

log = logbook.Logger('Performance')
TRADE_TYPE = zp.DATASOURCE_TYPE.TRADE


PeriodStats = namedtuple('PeriodStats',
                         ['net_liquidation',
                          'gross_leverage',
                          'net_leverage'])

PrevSubPeriodStats = namedtuple(
    'PrevSubPeriodStats', ['returns', 'pnl', 'cash_flow']
)

CurrSubPeriodStats = namedtuple(
    'CurrSubPeriodStats', ['starting_value', 'starting_cash']
)


def calc_net_liquidation(ending_cash, long_value, short_value):
    return ending_cash + long_value + short_value


def calc_leverage(exposure, net_liq):
    if net_liq != 0:
        return exposure / net_liq

    return np.inf


def calc_period_stats(pos_stats, ending_cash):
    net_liq = calc_net_liquidation(ending_cash,
                                   pos_stats.long_value,
                                   pos_stats.short_value)
    gross_leverage = calc_leverage(pos_stats.gross_exposure, net_liq)
    net_leverage = calc_leverage(pos_stats.net_exposure, net_liq)

    return PeriodStats(
        net_liquidation=net_liq,
        gross_leverage=gross_leverage,
        net_leverage=net_leverage)


def calc_payout(multiplier, amount, old_price, price):
    return (price - old_price) * multiplier * amount


class PerformancePeriod(object):

    def __init__(
            self,
            starting_cash,
            data_frequency,
            period_open=None,
            period_close=None,
            keep_transactions=True,
            keep_orders=False,
            serialize_positions=True,
            name=None):

        self.data_frequency = data_frequency

        # Start and end of the entire period
        self.period_open = period_open
        self.period_close = period_close

        self.initialize(starting_cash=starting_cash,
                        starting_value=0.0,
                        starting_exposure=0.0)

        self.ending_value = 0.0
        self.ending_exposure = 0.0
        self.ending_cash = starting_cash

        self.subperiod_divider = None

        # Keyed by asset, the previous last sale price of positions with
        # payouts on price differences, e.g. Futures.
        #
        # This dt is not the previous minute to the minute for which the
        # calculation is done, but the last sale price either before the period
        # start, or when the price at execution.
        self._payout_last_sale_prices = {}

        self.keep_transactions = keep_transactions
        self.keep_orders = keep_orders

        self.name = name

        # An object to recycle via assigning new values
        # when returning portfolio information.
        # So as not to avoid creating a new object for each event
        self._portfolio_store = zp.Portfolio()
        self._account_store = zp.Account()
        self.serialize_positions = serialize_positions

    _position_tracker = None

    def initialize(self, starting_cash, starting_value, starting_exposure):

        # Performance stats for the entire period, returned externally
        self.pnl = 0.0
        self.returns = 0.0
        self.cash_flow = 0.0
        self.starting_value = starting_value
        self.starting_exposure = starting_exposure
        self.starting_cash = starting_cash

        # The cumulative capital change occurred within the period
        self._total_intraperiod_capital_change = 0.0

        self.processed_transactions = {}
        self.orders_by_modified = {}
        self.orders_by_id = OrderedDict()

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

    def adjust_period_starting_capital(self, capital_change):
        self.ending_cash += capital_change
        self.starting_cash += capital_change

    def rollover(self):
        # We are starting a new period
        self.initialize(starting_cash=self.ending_cash,
                        starting_value=self.ending_value,
                        starting_exposure=self.ending_exposure)

        self.subperiod_divider = None

        payout_assets = self._payout_last_sale_prices.keys()

        for asset in payout_assets:
            if asset in self._payout_last_sale_prices:
                self._payout_last_sale_prices[asset] = \
                    self.position_tracker.positions[asset].last_sale_price
            else:
                del self._payout_last_sale_prices[asset]

    def initialize_subperiod_divider(self):
        self.calculate_performance()

        # Initialize a subperiod divider to stash the current performance
        # values. Current period starting values are set to equal ending values
        # of the previous subperiod
        self.subperiod_divider = SubPeriodDivider(
            prev_returns=self.returns,
            prev_pnl=self.pnl,
            prev_cash_flow=self.cash_flow,
            curr_starting_value=self.ending_value,
            curr_starting_cash=self.ending_cash
        )

    def set_current_subperiod_starting_values(self, capital_change):
        # Apply the capital change to the ending cash
        self.ending_cash += capital_change

        # Increment the total capital change occurred within the period
        self._total_intraperiod_capital_change += capital_change

        # Update the current subperiod starting cash to reflect the capital
        # change
        starting_value = self.subperiod_divider.curr_subperiod.starting_value
        self.subperiod_divider.curr_subperiod = CurrSubPeriodStats(
            starting_value=starting_value,
            starting_cash=self.ending_cash)

    def handle_dividends_paid(self, net_cash_payment):
        if net_cash_payment:
            self.handle_cash_payment(net_cash_payment)
        self.calculate_performance()

    def handle_cash_payment(self, payment_amount):
        self.adjust_cash(payment_amount)

    def handle_commission(self, cost):
        # Deduct from our total cash pool.
        self.adjust_cash(-cost)

    def adjust_cash(self, amount):
        self.cash_flow += amount

    def adjust_field(self, field, value):
        setattr(self, field, value)

    def _get_payout_total(self, positions):
        payouts = []
        for asset, old_price in iteritems(self._payout_last_sale_prices):
            pos = positions[asset]
            amount = pos.amount
            payout = calc_payout(
                asset.multiplier,
                amount,
                old_price,
                pos.last_sale_price)
            payouts.append(payout)

        return sum(payouts)

    def calculate_performance(self):
        pt = self.position_tracker
        pos_stats = pt.stats()
        self.ending_value = pos_stats.net_value
        self.ending_exposure = pos_stats.net_exposure

        payout = self._get_payout_total(pt.positions)

        self.ending_cash = self.starting_cash + self.cash_flow + \
            self._total_intraperiod_capital_change + payout

        total_at_end = self.ending_cash + self.ending_value

        # If there is a previous subperiod, the performance is calculated
        # from the previous and current subperiods. Otherwise, the performance
        # is calculated based on the start and end values of the whole period
        if self.subperiod_divider:
            starting_cash = self.subperiod_divider.curr_subperiod.starting_cash
            total_at_start = starting_cash + \
                self.subperiod_divider.curr_subperiod.starting_value

            # Performance for this subperiod
            pnl = total_at_end - total_at_start
            if total_at_start != 0:
                returns = pnl / total_at_start
            else:
                returns = 0.0

            # Performance for this whole period
            self.pnl = self.subperiod_divider.prev_subperiod.pnl + pnl
            self.returns = \
                (1 + self.subperiod_divider.prev_subperiod.returns) * \
                (1 + returns) - 1
        else:
            total_at_start = self.starting_cash + self.starting_value
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

    def handle_execution(self, txn):
        self.cash_flow += self._calculate_execution_cash_flow(txn)

        asset = txn.asset
        if isinstance(asset, Future):
            try:
                old_price = self._payout_last_sale_prices[asset]
                pos = self.position_tracker.positions[asset]
                amount = pos.amount
                price = txn.price
                cash_adj = calc_payout(
                    asset.multiplier, amount, old_price, price)
                self.adjust_cash(cash_adj)
                if amount + txn.amount == 0:
                    del self._payout_last_sale_prices[asset]
                else:
                    self._payout_last_sale_prices[asset] = price
            except KeyError:
                self._payout_last_sale_prices[asset] = txn.price

        if self.keep_transactions:
            try:
                self.processed_transactions[txn.dt].append(txn)
            except KeyError:
                self.processed_transactions[txn.dt] = [txn]

    @staticmethod
    def _calculate_execution_cash_flow(txn):
        """
        Calculates the cash flow from executing the given transaction
        """
        if isinstance(txn.asset, Future):
            return 0.0

        return -1 * txn.price * txn.amount

    # backwards compat. TODO: remove?
    @property
    def positions(self):
        return self.position_tracker.positions

    @property
    def position_amounts(self):
        return self.position_tracker.position_amounts

    def __core_dict(self):
        pos_stats = self.position_tracker.stats()
        period_stats = calc_period_stats(pos_stats, self.ending_cash)

        rval = {
            'ending_value': self.ending_value,
            'ending_exposure': self.ending_exposure,
            # this field is renamed to capital_used for backward
            # compatibility.
            'capital_used': self.cash_flow,
            'starting_value': self.starting_value,
            'starting_exposure': self.starting_exposure,
            'starting_cash': self.starting_cash,
            'ending_cash': self.ending_cash,
            'portfolio_value': self.ending_cash + self.ending_value,
            'pnl': self.pnl,
            'returns': self.returns,
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

    def to_dict(self, dt=None):
        """
        Creates a dictionary representing the state of this performance
        period. See header comments for a detailed description.

        Kwargs:
            dt (datetime): If present, only return transactions for the dt.
        """
        rval = self.__core_dict()

        if self.serialize_positions:
            positions = self.position_tracker.get_positions_list()
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
        portfolio.capital_used = self.cash_flow
        portfolio.starting_cash = self.starting_cash
        portfolio.portfolio_value = self.ending_cash + self.ending_value
        portfolio.pnl = self.pnl
        portfolio.returns = self.returns
        portfolio.cash = self.ending_cash
        portfolio.start_date = self.period_open
        portfolio.positions = self.position_tracker.get_positions()
        portfolio.positions_value = self.ending_value
        portfolio.positions_exposure = self.ending_exposure
        return portfolio

    def as_account(self):
        account = self._account_store

        pt = self.position_tracker
        pos_stats = pt.stats()
        period_stats = calc_period_stats(pos_stats, self.ending_cash)

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
        account.total_positions_exposure = \
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
        account.leverage = getattr(self, 'leverage',
                                   period_stats.gross_leverage)
        account.net_leverage = getattr(self, 'net_leverage',
                                       period_stats.net_leverage)
        account.net_liquidation = getattr(self, 'net_liquidation',
                                          period_stats.net_liquidation)
        return account


class SubPeriodDivider(object):
    """
    A marker for subdividing the period at the latest intraperiod capital
    change. prev_subperiod and curr_subperiod hold information respective to
    the previous and current subperiods.
    """

    def __init__(self, prev_returns, prev_pnl, prev_cash_flow,
                 curr_starting_value, curr_starting_cash):

        self.prev_subperiod = PrevSubPeriodStats(
            returns=prev_returns,
            pnl=prev_pnl,
            cash_flow=prev_cash_flow)

        self.curr_subperiod = CurrSubPeriodStats(
            starting_value=curr_starting_value,
            starting_cash=curr_starting_cash)
