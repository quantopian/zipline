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

from six import itervalues, iteritems

from zipline.assets import Equity, Future, Asset
from zipline.finance.blotter.blotter import Blotter
from zipline.extensions import register
from zipline.finance.order import Order
from zipline.finance.slippage import (
    DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT,
    VolatilityVolumeShare,
    FixedBasisPointsSlippage,
)
from zipline.finance.commission import (
    DEFAULT_PER_CONTRACT_COST,
    FUTURE_EXCHANGE_FEES_BY_SYMBOL,
    PerContract,
    PerShare,
)
from zipline.utils.input_validation import expect_types
import pandas as pd
log = Logger('Blotter Live')
warning_logger = Logger('AlgoWarning')

class BlotterLive(Blotter):
    def __init__(self, data_frequency, broker):
        self.broker = broker
        self._processed_closed_orders = []
        self._processed_transactions = []
        self.data_frequency = data_frequency
        self.new_orders = []
        self.max_shares = int(1e+11)

        self.slippage_models = {
            Equity: FixedBasisPointsSlippage(),
            Future: VolatilityVolumeShare(
                volume_limit=DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT,
            ),
        }
        self.commission_models = {
            Equity: PerShare(),
            Future: PerContract(
                cost=DEFAULT_PER_CONTRACT_COST,
                exchange_fee=FUTURE_EXCHANGE_FEES_BY_SYMBOL,
            ),
        }
        log.info('Initialized blotter_live')
    def __repr__(self):
        return """
    {class_name}(
        open_orders={open_orders},
        orders={orders},
        new_orders={new_orders},
    """.strip().format(class_name=self.__class__.__name__,
                       open_orders=self.open_orders,
                       orders=self.orders,
                       new_orders=self.new_orders)

    @property
    def orders(self):
        # IB returns orders from previous days too.
        # Need to filter for today to be in sync with zipline's behavior
        # TODO: This logic needs to be extended once GTC orders are supported
        today = pd.to_datetime('now', utc=True).date()
        return {order_id: order
                for order_id, order in iteritems(self.broker.orders)
                if order.dt.date() == today}

    @property
    def open_orders(self):
        assets = set([order.asset for order in itervalues(self.orders)
                      if order.open])
        return {
            asset: [order for order in itervalues(self.orders)
                    if order.asset == asset and order.open]
            for asset in assets
        }

    @expect_types(asset=Asset)
    def order(self, asset, amount, style, order_id=None):
        assert order_id is None
        if amount == 0:
            # it's a zipline fuck up.. we shouldn't get orders with amount 0. ignoring this order
            return ''
        order = self.broker.order(asset, amount, style)
        self.new_orders.append(order)

        return order.id

    def cancel(self, order_id, relay_status=True):
        return self.broker.cancel_order(order_id)

    def execute_cancel_policy(self, event):
        # Cancellation is handled at the broker
        pass

    def cancel_all_orders_for_asset(self, asset, warn=False, relay_status=True):
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
                            order_sym=order.asset.symbol,
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
                            order_sym=order.asset.symbol,
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
                            order_sym=order.asset.symbol,
                        )
                    )

        assert not orders
        del self.open_orders[asset]

    def reject(self, order_id, reason=''):
        log.warning("Unexpected reject request for {}: '{}'".format(
            order_id, reason))

    def hold(self, order_id, reason=''):
        log.warning("Unexpected hold request for {}: '{}'".format(
            order_id, reason))

    def get_transactions(self, bar_data):
        # All returned values from this function are delta between
        # the previous and actual call.
        def _list_delta(lst_a, lst_b):
            return [elem for elem in lst_a if elem not in set(lst_b)]

        today = pd.to_datetime('now', utc=True).date()
        all_transactions = [tx
                            for tx in itervalues(self.broker.transactions)
                            if tx.dt.date() == today]
        new_transactions = _list_delta(all_transactions,
                                       self._processed_transactions)
        self._processed_transactions = all_transactions

        new_commissions = [{'asset': tx.asset,
                            'cost': self.broker.orders[tx.order_id].commission,
                            'order': self.orders[tx.order_id]}
                           for tx in new_transactions]

        all_closed_orders = [order
                             for order in itervalues(self.orders)
                             if not order.open]
        new_closed_orders = _list_delta(all_closed_orders,
                                        self._processed_closed_orders)
        self._processed_closed_orders = all_closed_orders

        return new_transactions, new_commissions, new_closed_orders

    def prune_orders(self, closed_orders):
        # Orders are handled at the broker
        pass

    def process_splits(self, splits):
        # Splits are handled at the broker
        pass
