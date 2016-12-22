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
from nose_parameterized import parameterized

import pandas as pd

from zipline.finance.blotter import Blotter
from zipline.finance.order import ORDER_STATUS, Order
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)

from zipline.gens.sim_engine import SESSION_END, BAR
from zipline.finance.cancel_policy import EODCancel, NeverCancel
from zipline.finance.slippage import (
    DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT,
    FixedSlippage,
)
from zipline.utils.classproperty import classproperty
from zipline.testing.fixtures import (
    WithCreateBarData,
    WithDataPortal,
    WithLogger,
    WithSimParams,
    ZiplineTestCase,
)


class BlotterTestCase(WithCreateBarData,
                      WithLogger,
                      WithDataPortal,
                      WithSimParams,
                      ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-05', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')
    ASSET_FINDER_EQUITY_SIDS = 24, 25

    @classmethod
    def init_class_fixtures(cls):
        super(BlotterTestCase, cls).init_class_fixtures()
        cls.asset_24 = cls.asset_finder.retrieve_asset(24)
        cls.asset_25 = cls.asset_finder.retrieve_asset(25)

    @classmethod
    def make_equity_daily_bar_data(cls):
        yield 24, pd.DataFrame(
            {
                'open': [50, 50],
                'high': [50, 50],
                'low': [50, 50],
                'close': [50, 50],
                'volume': [100, 400],
            },
            index=cls.sim_params.sessions,
        )
        yield 25, pd.DataFrame(
            {
                'open': [50, 50],
                'high': [50, 50],
                'low': [50, 50],
                'close': [50, 50],
                'volume': [100, 400],
            },
            index=cls.sim_params.sessions,
        )

    @classproperty
    def CREATE_BARDATA_DATA_FREQUENCY(cls):
        return cls.sim_params.data_frequency

    @parameterized.expand([(MarketOrder(), None, None),
                           (LimitOrder(10), 10, None),
                           (StopOrder(10), None, 10),
                           (StopLimitOrder(10, 20), 10, 20)])
    def test_blotter_order_types(self, style_obj, expected_lmt, expected_stp):

        blotter = Blotter('daily', self.asset_finder)

        blotter.order(self.asset_24, 100, style_obj)
        result = blotter.open_orders[self.asset_24][0]

        self.assertEqual(result.limit, expected_lmt)
        self.assertEqual(result.stop, expected_stp)

    def test_cancel(self):
        blotter = Blotter('daily', self.asset_finder)

        oid_1 = blotter.order(self.asset_24, 100, MarketOrder())
        oid_2 = blotter.order(self.asset_24, 200, MarketOrder())
        oid_3 = blotter.order(self.asset_24, 300, MarketOrder())

        # Create an order for another asset to verify that we don't remove it
        # when we do cancel_all on 24.
        blotter.order(self.asset_25, 150, MarketOrder())

        self.assertEqual(len(blotter.open_orders), 2)
        self.assertEqual(len(blotter.open_orders[self.asset_24]), 3)
        self.assertEqual(
            [o.amount for o in blotter.open_orders[self.asset_24]],
            [100, 200, 300],
        )

        blotter.cancel(oid_2)
        self.assertEqual(len(blotter.open_orders), 2)
        self.assertEqual(len(blotter.open_orders[self.asset_24]), 2)
        self.assertEqual(
            [o.amount for o in blotter.open_orders[self.asset_24]],
            [100, 300],
        )
        self.assertEqual(
            [o.id for o in blotter.open_orders[self.asset_24]],
            [oid_1, oid_3],
        )

        blotter.cancel_all_orders_for_asset(self.asset_24)
        self.assertEqual(len(blotter.open_orders), 1)
        self.assertEqual(list(blotter.open_orders), [self.asset_25])

    def test_blotter_eod_cancellation(self):
        blotter = Blotter('minute', self.asset_finder,
                          cancel_policy=EODCancel())

        # Make two orders for the same sid, so we can test that we are not
        # mutating the orders list as we are cancelling orders
        blotter.order(self.asset_24, 100, MarketOrder())
        blotter.order(self.asset_24, -100, MarketOrder())

        self.assertEqual(len(blotter.new_orders), 2)
        order_ids = [order.id for order in blotter.open_orders[self.asset_24]]

        self.assertEqual(blotter.new_orders[0].status, ORDER_STATUS.OPEN)
        self.assertEqual(blotter.new_orders[1].status, ORDER_STATUS.OPEN)

        blotter.execute_cancel_policy(BAR)
        self.assertEqual(blotter.new_orders[0].status, ORDER_STATUS.OPEN)
        self.assertEqual(blotter.new_orders[1].status, ORDER_STATUS.OPEN)

        blotter.execute_cancel_policy(SESSION_END)
        for order_id in order_ids:
            order = blotter.orders[order_id]
            self.assertEqual(order.status, ORDER_STATUS.CANCELLED)

    def test_blotter_never_cancel(self):
        blotter = Blotter('minute', self.asset_finder,
                          cancel_policy=NeverCancel())

        blotter.order(self.asset_24, 100, MarketOrder())

        self.assertEqual(len(blotter.new_orders), 1)
        self.assertEqual(blotter.new_orders[0].status, ORDER_STATUS.OPEN)

        blotter.execute_cancel_policy(BAR)
        self.assertEqual(blotter.new_orders[0].status, ORDER_STATUS.OPEN)

        blotter.execute_cancel_policy(SESSION_END)
        self.assertEqual(blotter.new_orders[0].status, ORDER_STATUS.OPEN)

    def test_order_rejection(self):
        blotter = Blotter(self.sim_params.data_frequency,
                          self.asset_finder)

        # Reject a nonexistent order -> no order appears in new_order,
        # no exceptions raised out
        blotter.reject(56)
        self.assertEqual(blotter.new_orders, [])

        # Basic tests of open order behavior
        open_order_id = blotter.order(self.asset_24, 100, MarketOrder())
        second_order_id = blotter.order(self.asset_24, 50, MarketOrder())
        self.assertEqual(len(blotter.open_orders[self.asset_24]), 2)
        open_order = blotter.open_orders[self.asset_24][0]
        self.assertEqual(open_order.status, ORDER_STATUS.OPEN)
        self.assertEqual(open_order.id, open_order_id)
        self.assertIn(open_order, blotter.new_orders)

        # Reject that order immediately (same bar, i.e. still in new_orders)
        blotter.reject(open_order_id)
        self.assertEqual(len(blotter.new_orders), 2)
        self.assertEqual(len(blotter.open_orders[self.asset_24]), 1)
        still_open_order = blotter.new_orders[0]
        self.assertEqual(still_open_order.id, second_order_id)
        self.assertEqual(still_open_order.status, ORDER_STATUS.OPEN)
        rejected_order = blotter.new_orders[1]
        self.assertEqual(rejected_order.status, ORDER_STATUS.REJECTED)
        self.assertEqual(rejected_order.reason, '')

        # Do it again, but reject it at a later time (after tradesimulation
        # pulls it from new_orders)
        blotter = Blotter(self.sim_params.data_frequency,
                          self.asset_finder)
        new_open_id = blotter.order(self.asset_24, 10, MarketOrder())
        new_open_order = blotter.open_orders[self.asset_24][0]
        self.assertEqual(new_open_id, new_open_order.id)
        # Pretend that the trade simulation did this.
        blotter.new_orders = []

        rejection_reason = "Not enough cash on hand."
        blotter.reject(new_open_id, reason=rejection_reason)
        rejected_order = blotter.new_orders[0]
        self.assertEqual(rejected_order.id, new_open_id)
        self.assertEqual(rejected_order.status, ORDER_STATUS.REJECTED)
        self.assertEqual(rejected_order.reason, rejection_reason)

        # You can't reject a filled order.
        # Reset for paranoia
        blotter = Blotter(self.sim_params.data_frequency,
                          self.asset_finder)
        blotter.slippage_func = FixedSlippage()
        filled_id = blotter.order(self.asset_24, 100, MarketOrder())
        filled_order = None
        blotter.current_dt = self.sim_params.sessions[-1]
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.sim_params.sessions[-1],
        )
        txns, _, closed_orders = blotter.get_transactions(bar_data)
        for txn in txns:
            filled_order = blotter.orders[txn.order_id]
        blotter.prune_orders(closed_orders)

        self.assertEqual(filled_order.id, filled_id)
        self.assertIn(filled_order, blotter.new_orders)
        self.assertEqual(filled_order.status, ORDER_STATUS.FILLED)
        self.assertNotIn(filled_order, blotter.open_orders[self.asset_24])

        blotter.reject(filled_id)
        updated_order = blotter.orders[filled_id]
        self.assertEqual(updated_order.status, ORDER_STATUS.FILLED)

    def test_order_hold(self):
        """
        Held orders act almost identically to open orders, except for the
        status indication. When a fill happens, the order should switch
        status to OPEN/FILLED as necessary
        """
        blotter = Blotter(self.sim_params.data_frequency,
                          self.asset_finder)
        # Nothing happens on held of a non-existent order
        blotter.hold(56)
        self.assertEqual(blotter.new_orders, [])

        open_id = blotter.order(self.asset_24, 100, MarketOrder())
        open_order = blotter.open_orders[self.asset_24][0]
        self.assertEqual(open_order.id, open_id)

        blotter.hold(open_id)
        self.assertEqual(len(blotter.new_orders), 1)
        self.assertEqual(len(blotter.open_orders[self.asset_24]), 1)
        held_order = blotter.new_orders[0]
        self.assertEqual(held_order.status, ORDER_STATUS.HELD)
        self.assertEqual(held_order.reason, '')

        blotter.cancel(held_order.id)
        self.assertEqual(len(blotter.new_orders), 1)
        self.assertEqual(len(blotter.open_orders[self.asset_24]), 0)
        cancelled_order = blotter.new_orders[0]
        self.assertEqual(cancelled_order.id, held_order.id)
        self.assertEqual(cancelled_order.status, ORDER_STATUS.CANCELLED)

        for data in ([100, self.sim_params.sessions[0]],
                     [400, self.sim_params.sessions[1]]):
            # Verify that incoming fills will change the order status.
            trade_amt = data[0]
            dt = data[1]

            order_size = 100
            expected_filled = int(trade_amt *
                                  DEFAULT_VOLUME_SLIPPAGE_BAR_LIMIT)
            expected_open = order_size - expected_filled
            expected_status = ORDER_STATUS.OPEN if expected_open else \
                ORDER_STATUS.FILLED

            blotter = Blotter(self.sim_params.data_frequency,
                              self.asset_finder)
            open_id = blotter.order(self.asset_24, order_size, MarketOrder())
            open_order = blotter.open_orders[self.asset_24][0]
            self.assertEqual(open_id, open_order.id)
            blotter.hold(open_id)
            held_order = blotter.new_orders[0]

            filled_order = None
            blotter.current_dt = dt
            bar_data = self.create_bardata(
                simulation_dt_func=lambda: dt,
            )
            txns, _, _ = blotter.get_transactions(bar_data)
            for txn in txns:
                filled_order = blotter.orders[txn.order_id]

            self.assertEqual(filled_order.id, held_order.id)
            self.assertEqual(filled_order.status, expected_status)
            self.assertEqual(filled_order.filled, expected_filled)
            self.assertEqual(filled_order.open_amount, expected_open)

    def test_prune_orders(self):
        blotter = Blotter(self.sim_params.data_frequency,
                          self.asset_finder)

        blotter.order(self.asset_24, 100, MarketOrder())
        open_order = blotter.open_orders[self.asset_24][0]

        blotter.prune_orders([])
        self.assertEqual(1, len(blotter.open_orders[self.asset_24]))

        blotter.prune_orders([open_order])
        self.assertEqual(0, len(blotter.open_orders[self.asset_24]))

        # prune an order that isn't in our our open orders list, make sure
        # nothing blows up

        other_order = Order(
            dt=blotter.current_dt,
            sid=self.asset_25,
            amount=1
        )

        blotter.prune_orders([other_order])

    def test_batch_order_matches_multiple_orders(self):
        """
        Ensure the effect of order_batch is the same as multiple calls to
        order.
        """
        blotter1 = Blotter(self.sim_params.data_frequency,
                           self.asset_finder)
        blotter2 = Blotter(self.sim_params.data_frequency,
                           self.asset_finder)
        for i in range(1, 4):
            order_arg_lists = [
                (self.asset_24, i * 100, MarketOrder()),
                (self.asset_25, i * 100, LimitOrder(i * 100 + 1)),
            ]

            order_batch_ids = blotter1.batch_order(order_arg_lists)
            order_ids = []
            for order_args in order_arg_lists:
                order_ids.append(blotter2.order(*order_args))
            self.assertEqual(len(order_batch_ids), len(order_ids))

            self.assertEqual(len(blotter1.open_orders),
                             len(blotter2.open_orders))

            for (asset, _, _), order_batch_id, order_id in zip(
                    order_arg_lists, order_batch_ids, order_ids
            ):
                self.assertEqual(len(blotter1.open_orders[asset]),
                                 len(blotter2.open_orders[asset]))
                self.assertEqual(order_batch_id,
                                 blotter1.open_orders[asset][i-1].id)
                self.assertEqual(order_id,
                                 blotter2.open_orders[asset][i-1].id)
