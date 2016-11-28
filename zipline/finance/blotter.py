#
# Copyright 2015 Quantopian, Inc.
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
from logbook import Logger
from collections import defaultdict
from copy import copy

from six import iteritems

from zipline.finance.order import Order
from zipline.finance.slippage import VolumeShareSlippage
from zipline.finance.commission import PerShare
from zipline.finance.cancel_policy import NeverCancel

log = Logger('Blotter')
warning_logger = Logger('AlgoWarning')


class Blotter(object):
    def __init__(self, data_frequency, asset_finder, slippage_func=None,
                 commission=None, cancel_policy=None):
        # these orders are aggregated by sid
        self.open_orders = defaultdict(list)

        # keep a dict of orders by their own id
        self.orders = {}

        # all our legacy order management code works with integer sids.
        # this lets us convert those to assets when needed.  ideally, we'd just
        # revamp all the legacy code to work with assets.
        self.asset_finder = asset_finder

        # holding orders that have come in since the last event.
        self.new_orders = []
        self.current_dt = None

        self.max_shares = int(1e+11)

        self.slippage_func = slippage_func or VolumeShareSlippage()
        self.commission = commission or PerShare()

        self.data_frequency = data_frequency

        self.cancel_policy = cancel_policy if cancel_policy else NeverCancel()

    def __repr__(self):
        return """
{class_name}(
    slippage={slippage_func},
    commission={commission},
    open_orders={open_orders},
    orders={orders},
    new_orders={new_orders},
    current_dt={current_dt})
""".strip().format(class_name=self.__class__.__name__,
                   slippage_func=self.slippage_func,
                   commission=self.commission,
                   open_orders=self.open_orders,
                   orders=self.orders,
                   new_orders=self.new_orders,
                   current_dt=self.current_dt)

    def set_date(self, dt):
        self.current_dt = dt

    def order(self, sid, amount, style, order_id=None):
        """Place an order.

        Parameters
        ----------
        asset : zipline.assets.Asset
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        style : zipline.finance.execution.ExecutionStyle
            The execution style for the order.
        order_id : str, optional
            The unique identifier for this order.

        Returns
        -------
        order_id : str or None
            The unique identifier for this order, or None if no order was
            placed.

        Notes
        -----
        amount > 0 :: Buy/Cover
        amount < 0 :: Sell/Short
        Market order:    order(sid, amount)
        Limit order:     order(sid, amount, style=LimitOrder(limit_price))
        Stop order:      order(sid, amount, style=StopOrder(stop_price))
        StopLimit order: order(sid, amount, style=StopLimitOrder(limit_price,
                               stop_price))
        """
        # something could be done with amount to further divide
        # between buy by share count OR buy shares up to a dollar amount
        # numeric == share count  AND  "$dollar.cents" == cost amount

        if amount == 0:
            # Don't bother placing orders for 0 shares.
            return None
        elif amount > self.max_shares:
            # Arbitrary limit of 100 billion (US) shares will never be
            # exceeded except by a buggy algorithm.
            raise OverflowError("Can't order more than %d shares" %
                                self.max_shares)

        is_buy = (amount > 0)
        order = Order(
            dt=self.current_dt,
            sid=sid,
            amount=amount,
            stop=style.get_stop_price(is_buy),
            limit=style.get_limit_price(is_buy),
            id=order_id
        )

        self.open_orders[order.sid].append(order)
        self.orders[order.id] = order
        self.new_orders.append(order)

        return order.id

    def batch_order(self, order_arg_lists):
        """Place a batch of orders.

        Parameters
        ----------
        order_arg_lists : iterable[tuple]
            Tuples of args that `order` expects.

        Returns
        -------
        order_ids : list[str or None]
            The unique identifier (or None) for each of the orders placed
            (or not placed).

        Notes
        -----
        This is required for `Blotter` subclasses to be able to place a batch
        of orders, instead of being passed the order requests one at a time.
        """
        return [self.order(*order_args) for order_args in order_arg_lists]

    def cancel(self, order_id, relay_status=True):
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]

        if cur_order.open:
            order_list = self.open_orders[cur_order.sid]
            if cur_order in order_list:
                order_list.remove(cur_order)

            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.cancel()
            cur_order.dt = self.current_dt

            if relay_status:
                # we want this order's new status to be relayed out
                # along with newly placed orders.
                self.new_orders.append(cur_order)

    def cancel_all_orders_for_asset(self, asset, warn=False,
                                    relay_status=True):
        """
        Cancel all open orders for a given asset.
        """
        # (sadly) open_orders is a defaultdict, so this will always succeed.
        orders = self.open_orders[asset]

        # We're making a copy here because `cancel` mutates the list of open
        # orders in place.  The right thing to do here would be to make
        # self.open_orders no longer a defaultdict.  If we do that, then we
        # should just remove the orders once here and be done with the matter.
        for order in orders[:]:
            self.cancel(order.id, relay_status)
            if warn:
                # Message appropriately depending on whether there's
                # been a partial fill or not.
                if order.filled > 0:
                    warning_logger.warn(
                        'Your order for {order_amt} shares of '
                        '{order_sym} has been partially filled. '
                        '{order_filled} shares were successfully '
                        'purchased. {order_failed} shares were not '
                        'filled by the end of day and '
                        'were canceled.'.format(
                            order_amt=order.amount,
                            order_sym=order.sid.symbol,
                            order_filled=order.filled,
                            order_failed=order.amount - order.filled,
                        )
                    )
                elif order.filled < 0:
                    warning_logger.warn(
                        'Your order for {order_amt} shares of '
                        '{order_sym} has been partially filled. '
                        '{order_filled} shares were successfully '
                        'sold. {order_failed} shares were not '
                        'filled by the end of day and '
                        'were canceled.'.format(
                            order_amt=order.amount,
                            order_sym=order.sid.symbol,
                            order_filled=-1 * order.filled,
                            order_failed=-1 * (order.amount - order.filled),
                        )
                    )
                else:
                    warning_logger.warn(
                        'Your order for {order_amt} shares of '
                        '{order_sym} failed to fill by the end of day '
                        'and was canceled.'.format(
                            order_amt=order.amount,
                            order_sym=order.sid.symbol,
                        )
                    )

        assert not orders
        del self.open_orders[asset]

    def execute_cancel_policy(self, event):
        if self.cancel_policy.should_cancel(event):
            warn = self.cancel_policy.warn_on_cancel
            for asset in copy(self.open_orders):
                self.cancel_all_orders_for_asset(asset, warn,
                                                 relay_status=False)

    def reject(self, order_id, reason=''):
        """
        Mark the given order as 'rejected', which is functionally similar to
        cancelled. The distinction is that rejections are involuntary (and
        usually include a message from a broker indicating why the order was
        rejected) while cancels are typically user-driven.
        """
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]

        order_list = self.open_orders[cur_order.sid]
        if cur_order in order_list:
            order_list.remove(cur_order)

        if cur_order in self.new_orders:
            self.new_orders.remove(cur_order)
        cur_order.reject(reason=reason)
        cur_order.dt = self.current_dt
        # we want this order's new status to be relayed out
        # along with newly placed orders.
        self.new_orders.append(cur_order)

    def hold(self, order_id, reason=''):
        """
        Mark the order with order_id as 'held'. Held is functionally similar
        to 'open'. When a fill (full or partial) arrives, the status
        will automatically change back to open/filled as necessary.
        """
        if order_id not in self.orders:
            return

        cur_order = self.orders[order_id]
        if cur_order.open:
            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.hold(reason=reason)
            cur_order.dt = self.current_dt
            # we want this order's new status to be relayed out
            # along with newly placed orders.
            self.new_orders.append(cur_order)

    def process_splits(self, splits):
        """
        Processes a list of splits by modifying any open orders as needed.

        Parameters
        ----------
        splits: list
            A list of splits.  Each split is a tuple of (sid, ratio).

        Returns
        -------
        None
        """
        for split in splits:
            sid = split[0]
            if sid not in self.open_orders:
                return

            orders_to_modify = self.open_orders[sid]
            for order in orders_to_modify:
                order.handle_split(split[1])

    def get_transactions(self, bar_data):
        """
        Creates a list of transactions based on the current open orders,
        slippage model, and commission model.

        Parameters
        ----------
        bar_data: zipline._protocol.BarData

        Notes
        -----
        This method book-keeps the blotter's open_orders dictionary, so that
         it is accurate by the time we're done processing open orders.

        Returns
        -------
        transactions_list: List
            transactions_list: list of transactions resulting from the current
            open orders.  If there were no open orders, an empty list is
            returned.

        commissions_list: List
            commissions_list: list of commissions resulting from filling the
            open orders.  A commission is an object with "sid" and "cost"
            parameters.

        closed_orders: List
            closed_orders: list of all the orders that have filled.
        """

        closed_orders = []
        transactions = []
        commissions = []

        if self.open_orders:
            assets = self.asset_finder.retrieve_all(self.open_orders)
            asset_dict = {asset.sid: asset for asset in assets}

            for sid, asset_orders in iteritems(self.open_orders):
                asset = asset_dict[sid]

                for order, txn in \
                        self.slippage_func(bar_data, asset, asset_orders):
                    additional_commission = \
                        self.commission.calculate(order, txn)

                    if additional_commission > 0:
                        commissions.append({
                            "sid": order.sid,
                            "order": order,
                            "cost": additional_commission
                        })

                    order.filled += txn.amount
                    order.commission += additional_commission

                    order.dt = txn.dt

                    transactions.append(txn)

                    if not order.open:
                        closed_orders.append(order)

        return transactions, commissions, closed_orders

    def prune_orders(self, closed_orders):
        """
        Removes all given orders from the blotter's open_orders list.

        Parameters
        ----------
        closed_orders: iterable of orders that are closed.

        Returns
        -------
        None
        """
        # remove all closed orders from our open_orders dict
        for order in closed_orders:
            sid = order.sid
            sid_orders = self.open_orders[sid]
            try:
                sid_orders.remove(order)
            except ValueError:
                continue

        # now clear out the sids from our open_orders dict that have
        # zero open orders
        for sid in list(self.open_orders.keys()):
            if len(self.open_orders[sid]) == 0:
                del self.open_orders[sid]
