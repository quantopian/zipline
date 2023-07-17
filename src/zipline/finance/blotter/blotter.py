#
# Copyright 2018 Quantopian, Inc.
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
from abc import ABC, abstractmethod
from zipline.extensions import extensible
from zipline.finance.cancel_policy import NeverCancel


@extensible
class Blotter(ABC):
    def __init__(self, cancel_policy=None):
        self.cancel_policy = cancel_policy if cancel_policy else NeverCancel()
        self.current_dt = None

    def set_date(self, dt):
        self.current_dt = dt

    @abstractmethod
    def order(self, asset, amount, style, order_id=None):
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
        amount > 0 : Buy/Cover
        amount < 0 : Sell/Short
        Market order : order(asset, amount)
        Limit order : order(asset, amount, style=LimitOrder(limit_price))
        Stop order : order(asset, amount, style=StopOrder(stop_price))
        StopLimit order : order(asset, amount,
        style=StopLimitOrder(limit_price, stop_price))
        """

        raise NotImplementedError("order")

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

    @abstractmethod
    def cancel(self, order_id, relay_status=True):
        """Cancel a single order

        Parameters
        ----------
        order_id : int
            The id of the order

        relay_status : bool
            Whether or not to record the status of the order
        """
        raise NotImplementedError("cancel")

    @abstractmethod
    def cancel_all_orders_for_asset(self, asset, warn=False, relay_status=True):
        """
        Cancel all open orders for a given asset.
        """

        raise NotImplementedError("cancel_all_orders_for_asset")

    @abstractmethod
    def execute_cancel_policy(self, event):
        raise NotImplementedError("execute_cancel_policy")

    @abstractmethod
    def reject(self, order_id, reason=""):
        """
        Mark the given order as 'rejected', which is functionally similar to
        cancelled. The distinction is that rejections are involuntary (and
        usually include a message from a broker indicating why the order was
        rejected) while cancels are typically user-driven.
        """

        raise NotImplementedError("reject")

    @abstractmethod
    def hold(self, order_id, reason=""):
        """
        Mark the order with order_id as 'held'. Held is functionally similar
        to 'open'. When a fill (full or partial) arrives, the status
        will automatically change back to open/filled as necessary.
        """

        raise NotImplementedError("hold")

    @abstractmethod
    def process_splits(self, splits):
        """
        Processes a list of splits by modifying any open orders as needed.

        Parameters
        ----------
        splits: list
            A list of splits.  Each split is a tuple of (asset, ratio).

        Returns
        -------
        None
        """

        raise NotImplementedError("process_splits")

    @abstractmethod
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
            open orders.  A commission is an object with "asset" and "cost"
            parameters.

        closed_orders: List
            closed_orders: list of all the orders that have filled.
        """

        raise NotImplementedError("get_transactions")

    @abstractmethod
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

        raise NotImplementedError("prune_orders")
