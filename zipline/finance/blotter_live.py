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
from six import itervalues

from zipline.finance.blotter import Blotter
from zipline.utils.input_validation import expect_types

from zipline.assets import Asset

log = Logger('Blotter Live')


class BlotterLive(Blotter):
    def __init__(self, data_frequency, broker):
        self.broker = broker
        self._processed_closed_orders = []
        self._processed_transactions = []
        self.data_frequency = data_frequency
        self.new_orders = []

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
        return self.broker.orders

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

        order = self.broker.order(asset, amount, style)
        self.new_orders.append(order)

        return order.id

    def cancel(self, order_id, relay_status=True):
        return self.broker.cancel_order(order_id)

    def execute_cancel_policy(self, event):
        # Cancellation is handled at the broker
        pass

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

        all_transactions = list(self.broker.transactions.values())
        new_transactions = _list_delta(all_transactions,
                                       self._processed_transactions)
        self._processed_transactions = all_transactions

        new_commissions = [{'asset': tx.asset,
                            'cost': tx.commission,
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
