#
# Copyright 2017 Quantopian, Inc.
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

from collections import namedtuple, OrderedDict
from math import isnan

import logbook
import numpy as np
import pandas as pd
from six import iteritems, itervalues, PY2

from zipline.assets import Future
from zipline.finance.transaction import Transaction
import zipline.protocol as zp
from zipline.utils.sentinel import sentinel
from .position import Position

log = logbook.Logger('Performance')


PositionStats = namedtuple(
    'PositionStats',
    [
        'net_exposure',
        'gross_value',
        'gross_exposure',
        'short_value',
        'short_exposure',
        'shorts_count',
        'long_value',
        'long_exposure',
        'longs_count',
        'net_value',
    ],
)


class PositionTracker(object):
    """The current state of the positions held.

    Parameters
    ----------
    data_frequency : {'daily', 'minute'}
        The data frequency of the simulation.
    """
    def __init__(self, data_frequency):
        # asset => position object
        self.positions = OrderedDict()
        self._unpaid_dividends = {}
        self._unpaid_stock_dividends = {}
        self._positions_store = zp.Positions()

        self.data_frequency = data_frequency

        # cache the stats until something alters our positions
        self._dirty_stats = True
        self._stats = None

    def update_position(self,
                        asset,
                        amount=None,
                        last_sale_price=None,
                        last_sale_date=None,
                        cost_basis=None):
        self._dirty_stats = True

        if asset not in self.positions:
            position = Position(asset)
            self.positions[asset] = position
        else:
            position = self.positions[asset]

        if amount is not None:
            position.amount = amount
        if last_sale_price is not None:
            position.last_sale_price = last_sale_price
        if last_sale_date is not None:
            position.last_sale_date = last_sale_date
        if cost_basis is not None:
            position.cost_basis = cost_basis

    def execute_transaction(self, txn):
        self._dirty_stats = True

        asset = txn.asset

        if asset not in self.positions:
            position = Position(asset)
            self.positions[asset] = position
        else:
            position = self.positions[asset]

        position.update(txn)

        if position.amount == 0:
            del self.positions[asset]

            try:
                # if this position exists in our user-facing dictionary,
                # remove it as well.
                del self._positions_store[asset]
            except KeyError:
                pass

    def handle_commission(self, asset, cost):
        # Adjust the cost basis of the stock if we own it
        if asset in self.positions:
            self._dirty_stats = True
            self.positions[asset].adjust_commission_cost_basis(asset, cost)

    def handle_splits(self, splits):
        """Processes a list of splits by modifying any positions as needed.

        Parameters
        ----------
        splits: list
            A list of splits.  Each split is a tuple of (asset, ratio).

        Returns
        -------
        int: The leftover cash from fractional shares after modifying each
            position.
        """
        total_leftover_cash = 0

        for asset, ratio in splits:
            if asset in self.positions:
                self._dirty_stats = True

                # Make the position object handle the split. It returns the
                # leftover cash from a fractional share, if there is any.
                position = self.positions[asset]
                leftover_cash = position.handle_split(asset, ratio)
                total_leftover_cash += leftover_cash

        return total_leftover_cash

    def earn_dividends(self, dividends, stock_dividends):
        """Given a list of dividends whose ex_dates are all the next trading
        day, calculate and store the cash and/or stock payments to be paid on
        each dividend's pay date.

        Parameters
        ----------
        dividends: iterable of (asset, amount, pay_date) namedtuples

        stock_dividends: iterable of (asset, payment_asset, ratio, pay_date)
            namedtuples.
        """
        for dividend in dividends:
            # Store the earned dividends so that they can be paid on the
            # dividends' pay_dates.
            div_owed = self.positions[dividend.asset].earn_dividend(dividend)
            try:
                self._unpaid_dividends[dividend.pay_date].append(div_owed)
            except KeyError:
                self._unpaid_dividends[dividend.pay_date] = [div_owed]

        for stock_dividend in stock_dividends:
            div_owed = self.positions[
                stock_dividend.asset
            ].earn_stock_dividend(stock_dividend)
            try:
                self._unpaid_stock_dividends[stock_dividend.pay_date].append(
                    div_owed,
                )
            except KeyError:
                self._unpaid_stock_dividends[stock_dividend.pay_date] = [
                    div_owed,
                ]

    def pay_dividends(self, next_trading_day):
        """
        Returns a cash payment based on the dividends that should be paid out
        according to the accumulated bookkeeping of earned, unpaid, and stock
        dividends.
        """
        net_cash_payment = 0.0

        try:
            payments = self._unpaid_dividends[next_trading_day]
            # Mark these dividends as paid by dropping them from our unpaid
            del self._unpaid_dividends[next_trading_day]
        except KeyError:
            payments = []

        # representing the fact that we're required to reimburse the owner of
        # the stock for any dividends paid while borrowing.
        for payment in payments:
            net_cash_payment += payment['amount']

        # Add stock for any stock dividends paid.  Again, the values here may
        # be negative in the case of short positions.
        try:
            stock_payments = self._unpaid_stock_dividends[next_trading_day]
        except KeyError:
            stock_payments = []

        for stock_payment in stock_payments:
            payment_asset = stock_payment['payment_asset']
            share_count = stock_payment['share_count']
            # note we create a Position for stock dividend if we don't
            # already own the asset
            if payment_asset in self.positions:
                position = self.positions[payment_asset]
            else:
                position = self.positions[payment_asset] = Position(
                    payment_asset,
                )

            position.amount += share_count

        return net_cash_payment

    def maybe_create_close_position_transaction(self, asset, dt, data_portal):
        if not self.positions.get(asset):
            return None

        amount = self.positions.get(asset).amount
        price = data_portal.get_spot_value(
            asset, 'price', dt, self.data_frequency)

        # Get the last traded price if price is no longer available
        if isnan(price):
            price = self.positions.get(asset).last_sale_price

        return Transaction(
            asset=asset,
            amount=-amount,
            dt=dt,
            price=price,
            order_id=None,
        )

    def get_positions(self):

        positions = self._positions_store

        for asset, pos in iteritems(self.positions):
            if pos.amount == 0:
                # Clear out the position if it has become empty since the last
                # time get_positions was called.  Catching the KeyError is
                # faster than checking `if asset in positions`, and this can be
                # potentially called in a tight inner loop.
                positions.pop(asset, None)
                continue

            # Adds the new position if we didn't have one before, or overwrite
            # one we have currently
            positions[asset] = pos.protocol_position

        return positions

    def get_position_list(self):
        positions = []
        for asset, pos in iteritems(self.positions):
            if pos.amount != 0:
                positions.append(pos.to_dict())
        return positions

    def _market_minute_get_price(self,
                                 asset,
                                 _,
                                 dt,
                                 data_portal,
                                 data_frequency):
        return data_portal.get_scalar_asset_spot_value(
            asset,
            'price',
            dt,
            data_frequency,
        )

    def _non_market_minute_get_price(self,
                                     asset,
                                     previous_minute,
                                     dt,
                                     data_portal,
                                     data_frequency):
        return data_portal.get_adjusted_value(
            asset,
            'price',
            previous_minute,
            dt,
            data_frequency,
        )

    def sync_last_sale_prices(self,
                              dt,
                              data_portal,
                              handle_non_market_minutes=False):
        self._dirty_stats = True

        if handle_non_market_minutes:
            previous_minute = data_portal.trading_calendar.previous_minute(dt)
            get_price = self._non_market_minute_get_price
        else:
            previous_minute = None
            get_price = self._market_minute_get_price

        data_frequency = self.data_frequency

        for asset, position in iteritems(self.positions):
            if False and position.last_sale_date == dt:
                # this position is already synced
                continue

            last_sale_price = get_price(
                position.asset,
                previous_minute,
                dt,
                data_portal,
                data_frequency,
            )

            if not np.isnan(last_sale_price):
                position.last_sale_price = last_sale_price
                position.last_sale_date = dt

    @property
    def stats(self):
        if not self._dirty_stats:
            return self._stats

        net_value = long_value = short_value = 0
        long_exposure = short_exposure = 0
        longs_count = shorts_count = 0
        for position in itervalues(self.positions):
            # NOTE: this loop does a lot of stuff!
            # we call this function every single minute of the simulations
            # so let's not iterate through every single position multiple
            # times.
            exposure = position.amount * position.last_sale_price

            if isinstance(position.asset, Future):
                # Futures don't have an inherent position value.
                value = 0
                exposure *= position.asset.multiplier
            else:
                value = exposure

            if exposure > 0:
                longs_count += 1
                long_value += value
                long_exposure += exposure
            elif exposure < 0:
                shorts_count += 1
                short_value += value
                short_exposure += exposure

        net_value = long_value + short_value
        gross_value = long_value - short_value
        gross_exposure = long_exposure - short_exposure
        net_exposure = long_exposure + short_exposure

        # TODO: investigate cnamedtuple here because instance creation speed
        # is much faster
        self._stats = stats = PositionStats(
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
        return stats


if PY2:
    def move_to_end(ordered_dict, key, last=False):
        if last:
            ordered_dict[key] = ordered_dict.pop(key)
        else:
            # please don't do this in python 2 ;_;
            new_first_element = ordered_dict.pop(key)

            # the items (without the given key) in the order they were inserted
            items = ordered_dict.items()

            # reset the ordered_dict to re-insert in the new order
            ordered_dict.clear()

            ordered_dict[key] = new_first_element

            # add the items back in their original order
            ordered_dict.update(items)
else:
    move_to_end = OrderedDict.move_to_end


PeriodStats = namedtuple(
    'PeriodStats',
    'net_liquidation gross_leverage net_leverage',
)


not_overridden = sentinel(
    'not_overridden',
    'Mark that an account field has not been overridden',
)


class Ledger(object):
    """The ledger tracks all orders and transactions as well as the current
    state of the portfolio and positions.

    Attributes
    ----------
    portfolio : zipline.protocol.Portfolio
        The updated portfolio being managed.
    account : zipline.protocol.Account
        The updated account being managed.
    position_tracker : PositionTracker
        The current set of positions.
    todays_returns : float
        The current day's returns. In minute emission mode, this is the partial
        day's returns. In daily emission mode, this is
        ``daily_returns[session]``.
    daily_returns : pd.Series
        The daily returns series. Days that have not yet finished will hold
        a value of ``np.nan``.
    """
    def __init__(self, trading_sessions, capital_base, data_frequency):
        if len(trading_sessions):
            start = trading_sessions[0]
        else:
            start = None

        # Have some fields of the portfolio changed? This should be accessed
        # through ``self._dirty_portfolio``
        self.__dirty_portfolio = False
        self._immutable_portfolio = zp.Portfolio(start, capital_base)
        self._portfolio = zp.MutableView(self._immutable_portfolio)

        self.daily_returns = pd.Series(
            np.nan,
            index=trading_sessions,
        )
        self._previous_total_returns = 0

        # this is a component of the cache key for the account
        self._position_stats = None

        # Have some fields of the account changed?
        self._dirty_account = True
        self._immutable_account = zp.Account(self._portfolio)
        self._account = zp.MutableView(self._immutable_account)

        # The broker blotter can override some fields on the account. This is
        # way to tangled up at the moment but we aren't fixing it today.
        self._account_overrides = {}

        self.position_tracker = PositionTracker(data_frequency)

        self._processed_transactions = {}

        self._orders_by_modified = {}
        self._orders_by_id = OrderedDict()

        # Keyed by asset, the previous last sale price of positions with
        # payouts on price differences, e.g. Futures.
        #
        # This dt is not the previous minute to the minute for which the
        # calculation is done, but the last sale price either before the period
        # start, or when the price at execution.
        self._payout_last_sale_prices = {}

    @property
    def todays_returns(self):
        # compute today's returns in returns space instead of portfolio-value
        # space to work even when we have capital changes
        return (
            (self.portfolio.returns + 1) /
            (self._previous_total_returns + 1) -
            1
        )

    @property
    def _dirty_portfolio(self):
        return self.__dirty_portfolio

    @_dirty_portfolio.setter
    def _dirty_portfolio(self, value):
        if value:
            # marking the portfolio as dirty also marks the account as dirty
            self.__dirty_portfolio = self._dirty_account = value
        else:
            self.__dirty_portfolio = value

    def start_of_session(self, session_label):
        self._processed_transactions.clear()
        self._orders_by_modified.clear()
        self._orders_by_id.clear()

        # Save the previous day's total returns so that ``todays_returns``
        # produces returns since yesterday. This does not happen in
        # ``end_of_session`` because we want ``todays_returns`` to produce the
        # correct value in metric ``end_of_session`` handlers.
        self._previous_total_returns = self.portfolio.returns

    def end_of_bar(self, dt):
        # make daily_returns hold the partial returns, this saves many
        # metrics from doing a concat and copying all of the previous
        # returns
        self.daily_returns[dt.normalize()] = self.todays_returns

    def end_of_session(self, session_label):
        # save the daily returns time-series
        self.daily_returns[session_label] = self.todays_returns

    def sync_last_sale_prices(self,
                              dt,
                              data_portal,
                              handle_non_market_minutes=False):
        self.position_tracker.sync_last_sale_prices(
            dt,
            data_portal,
            handle_non_market_minutes=handle_non_market_minutes,
        )
        self._dirty_portfolio = True

    @staticmethod
    def _calculate_payout(multiplier, amount, old_price, price):

        return (price - old_price) * multiplier * amount

    def _cash_flow(self, amount):
        self._dirty_portfolio = True
        p = self._portfolio
        p.cash_flow += amount
        p.cash += amount

    def process_transaction(self, transaction):
        """Add a transaction to ledger, updating the current state as needed.

        Parameters
        ----------
        transaction : zp.Transaction
            The transaction to execute.
        """
        asset = transaction.asset
        if isinstance(asset, Future):
            try:
                old_price = self._payout_last_sale_prices[asset]
            except KeyError:
                self._payout_last_sale_prices[asset] = transaction.price
            else:
                position = self.position_tracker.positions[asset]
                amount = position.amount
                price = transaction.price

                self._cash_flow(
                    self._calculate_payout(
                        asset.multiplier,
                        amount,
                        old_price,
                        price,
                    ),
                )

                if amount + transaction.amount == 0:
                    del self._payout_last_sale_prices[asset]
                else:
                    self._payout_last_sale_prices[asset] = price
        else:
            self._cash_flow(-(transaction.price * transaction.amount))

        self.position_tracker.execute_transaction(transaction)

        # we only ever want the dict form from now on
        transaction_dict = transaction.to_dict()
        try:
            self._processed_transactions[transaction.dt].append(
                transaction_dict,
            )
        except KeyError:
            self._processed_transactions[transaction.dt] = [transaction_dict]

    def process_splits(self, splits):
        """Processes a list of splits by modifying any positions as needed.

        Parameters
        ----------
        splits: list[(Asset, float)]
            A list of splits. Each split is a tuple of (asset, ratio).
        """
        leftover_cash = self.position_tracker.handle_splits(splits)
        if leftover_cash > 0:
            self._cash_flow(leftover_cash)

    def process_order(self, order):
        """Keep track of an order that was placed.

        Parameters
        ----------
        order : zp.Order
            The order to record.
        """
        try:
            dt_orders = self._orders_by_modified[order.dt]
        except KeyError:
            self._orders_by_modified[order.dt] = OrderedDict([
                (order.id, order),
            ])
            self._orders_by_id[order.id] = order
        else:
            self._orders_by_id[order.id] = dt_orders[order.id] = order
            # to preserve the order of the orders by modified date
            move_to_end(dt_orders, order.id, last=True)

        move_to_end(self._orders_by_id, order.id, last=True)

    def process_commission(self, commission):
        """Process the commission.

        Parameters
        ----------
        commission : zp.Event
            The commission being paid.
        """
        asset = commission['asset']
        cost = commission['cost']

        self.position_tracker.handle_commission(asset, cost)
        self._cash_flow(-cost)

    def close_position(self, asset, dt, data_portal):
        txn = self.position_tracker.maybe_create_close_position_transaction(
            asset,
            dt,
            data_portal,
        )
        if txn is not None:
            self.process_transaction(txn)

    def process_dividends(self, next_session, asset_finder, adjustment_reader):
        """Process dividends for the next session.

        This will earn us any dividends whose ex-date is the next session as
        well as paying out any dividends whose pay-date is the next session
        """
        position_tracker = self.position_tracker

        # Earn dividends whose ex_date is the next trading day. We need to
        # check if we own any of these stocks so we know to pay them out when
        # the pay date comes.
        held_sids = set(position_tracker.positions)
        if held_sids:
            cash_dividends = adjustment_reader.get_dividends_with_ex_date(
                held_sids,
                next_session,
                asset_finder
            )
            stock_dividends = (
                adjustment_reader.get_stock_dividends_with_ex_date(
                    held_sids,
                    next_session,
                    asset_finder
                )
            )

            # Earning a dividend just marks that we need to get paid out on
            # the dividend's pay-date. This does not affect our cash yet.
            position_tracker.earn_dividends(
                cash_dividends,
                stock_dividends,
            )

        # Pay out the dividends whose pay-date is the next session. This does
        # affect out cash.
        self._cash_flow(
            position_tracker.pay_dividends(
                next_session,
            ),
        )

    def capital_change(self, change_amount):
        self.update_portfolio()
        portfolio = self._portfolio

        # we update the cash and total value so this is not dirty
        portfolio.portfolio_value += change_amount
        portfolio.cash += change_amount

    def transactions(self, dt=None):
        """Retrieve the dict-form of all of the transactions in a given bar or
        for the whole simulation.

        Parameters
        ----------
        dt : pd.Timestamp or None, optional
            The particular datetime to look up transactions for. If not passed,
            or None is explicitly passed, all of the transactions will be
            returned.

        Returns
        -------
        transactions : list[dict]
            The transaction information.
        """
        if dt is None:
            # flatten the by-day transactions
            return [
                txn
                for by_day in itervalues(self._processed_transactions)
                for txn in by_day
            ]

        return self._processed_transactions.get(dt, [])

    def orders(self, dt=None):
        """Retrieve the dict-form of all of the orders in a given bar or for
        the whole simulation.

        Parameters
        ----------
        dt : pd.Timestamp or None, optional
            The particular datetime to look up order for. If not passed, or
            None is explicitly passed, all of the orders will be returned.

        Returns
        -------
        orders : list[dict]
            The order information.
        """
        if dt is None:
            # orders by id is already flattened
            return [o.to_dict() for o in itervalues(self._orders_by_id)]

        return [
            o.to_dict()
            for o in itervalues(self._orders_by_modified.get(dt, {}))
        ]

    @property
    def positions(self):
        return self.position_tracker.get_position_list()

    def _get_payout_total(self, positions):
        calculate_payout = self._calculate_payout
        payout_last_sale_prices = self._payout_last_sale_prices

        total = 0
        for asset, old_price in iteritems(payout_last_sale_prices):
            position = positions[asset]
            payout_last_sale_prices[asset] = price = position.last_sale_price
            amount = position.amount
            total += calculate_payout(
                asset.multiplier,
                amount,
                old_price,
                price,
            )

        return total

    def update_portfolio(self):
        """Force a computation of the current portfolio state.
        """
        if not self._dirty_portfolio:
            return

        portfolio = self._portfolio
        pt = self.position_tracker

        portfolio.positions = pt.get_positions()
        position_stats = pt.stats

        portfolio.positions_value = position_value = (
            position_stats.net_value
        )
        portfolio.positions_exposure = position_stats.net_exposure
        self._cash_flow(self._get_payout_total(pt.positions))

        start_value = portfolio.portfolio_value

        # update the new starting value
        portfolio.portfolio_value = end_value = portfolio.cash + position_value

        pnl = end_value - start_value
        if start_value != 0:
            returns = pnl / start_value
        else:
            returns = 0.0

        portfolio.pnl += pnl
        portfolio.returns = (
            (1 + portfolio.returns) *
            (1 + returns) -
            1
        )

        # the portfolio has been fully synced
        self._dirty_portfolio = False

    @property
    def portfolio(self):
        """Compute the current portfolio.

        Notes
        -----
        This is cached, repeated access will not recompute the portfolio until
        the portfolio has changed.
        """
        self.update_portfolio()
        return self._immutable_portfolio

    @staticmethod
    def _calculate_net_liquidation(ending_cash, long_value, short_value):
        return ending_cash + long_value + short_value

    @staticmethod
    def _calculate_leverage(exposure, net_liquidation):
        if net_liquidation != 0:
            return exposure / net_liquidation

        return np.inf

    def calculate_period_stats(self):
        position_stats = self.position_tracker.stats
        net_liquidation = self._calculate_net_liquidation(
            self._portfolio.cash,
            position_stats.long_value,
            position_stats.short_value,
        )
        gross_leverage = self._calculate_leverage(
            position_stats.gross_exposure,
            net_liquidation,
        )
        net_leverage = self._calculate_leverage(
            position_stats.net_exposure,
            net_liquidation,
        )

        return net_liquidation, gross_leverage, net_leverage

    def override_account_fields(self,
                                settled_cash=not_overridden,
                                accrued_interest=not_overridden,
                                buying_power=not_overridden,
                                equity_with_loan=not_overridden,
                                total_positions_value=not_overridden,
                                total_positions_exposure=not_overridden,
                                regt_equity=not_overridden,
                                regt_margin=not_overridden,
                                initial_margin_requirement=not_overridden,
                                maintenance_margin_requirement=not_overridden,
                                available_funds=not_overridden,
                                excess_liquidity=not_overridden,
                                cushion=not_overridden,
                                day_trades_remaining=not_overridden,
                                leverage=not_overridden,
                                net_leverage=not_overridden,
                                net_liquidation=not_overridden):
        """Override fields on ``self.account``.
        """
        # mark that the portfolio is dirty to override the fields again
        self._dirty_account = True
        self._account_overrides = kwargs = {
            k: v for k, v in locals().items() if v is not not_overridden
        }
        del kwargs['self']

    @property
    def account(self):
        if self._dirty_account:
            portfolio = self.portfolio

            account = self._account

            # If no attribute is found in the ``_account_overrides`` resort to
            # the following default values. If an attribute is found use the
            # existing value. For instance, a broker may provide updates to
            # these attributes. In this case we do not want to over write the
            # broker values with the default values.
            account.settled_cash = portfolio.cash
            account.accrued_interest = 0.0
            account.buying_power = np.inf
            account.equity_with_loan = portfolio.portfolio_value
            account.total_positions_value = (
                portfolio.portfolio_value - portfolio.cash
            )
            account.total_positions_exposure = (
                portfolio.positions_exposure
            )
            account.regt_equity = portfolio.cash
            account.regt_margin = np.inf
            account.initial_margin_requirement = 0.0
            account.maintenance_margin_requirement = 0.0
            account.available_funds = portfolio.cash
            account.excess_liquidity = portfolio.cash
            account.cushion = (
                (portfolio.cash / portfolio.portfolio_value)
                if portfolio.portfolio_value else
                np.nan
            )
            account.day_trades_remaining = np.inf
            (account.net_liquidation,
             account.gross_leverage,
             account.net_leverage) = self.calculate_period_stats()

            account.leverage = account.gross_leverage

            # apply the overrides
            for k, v in iteritems(self._account_overrides):
                setattr(account, k, v)

            # the account has been fully synced
            self._dirty_account = False

        return self._immutable_account
