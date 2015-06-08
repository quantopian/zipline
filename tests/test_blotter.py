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

import datetime
from nose_parameterized import parameterized
from unittest import TestCase

from zipline.finance.blotter import Blotter, ORDER_STATUS
from zipline.finance.trading import with_environment
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.sources.test_source import create_trade

from zipline.utils.test_utils import(
    setup_logger,
    teardown_logger,
)


class BlotterTestCase(TestCase):

    @with_environment()
    def setUp(self, env=None):
        setup_logger(self)
        env.update_asset_finder(identifiers=[24])

    def tearDown(self):
        teardown_logger(self)

    @parameterized.expand([(MarketOrder(), None, None),
                           (LimitOrder(10), 10, None),
                           (StopOrder(10), None, 10),
                           (StopLimitOrder(10, 20), 10, 20)])
    def test_blotter_order_types(self, style_obj, expected_lmt, expected_stp):

        blotter = Blotter()

        blotter.order(24, 100, style_obj)
        result = blotter.open_orders[24][0]

        self.assertEqual(result.limit, expected_lmt)
        self.assertEqual(result.stop, expected_stp)

    def test_order_rejection(self):
        blotter = Blotter()
        # Reject a nonexistent order -> no order appears in new_order,
        # no exceptions raised out
        blotter.reject(56)
        self.assertEqual(blotter.new_orders, [])

        # Basic tests of open order behavior
        open_order_id = blotter.order(24, 100, MarketOrder())
        second_order_id = blotter.order(24, 50, MarketOrder())
        self.assertEqual(len(blotter.open_orders[24]), 2)
        open_order = blotter.open_orders[24][0]
        self.assertEqual(open_order.status, ORDER_STATUS.OPEN)
        self.assertEqual(open_order.id, open_order_id)
        self.assertIn(open_order, blotter.new_orders)

        # Reject that order immediately (same bar, i.e. still in new_orders)
        blotter.reject(open_order_id)
        self.assertEqual(len(blotter.new_orders), 2)
        self.assertEqual(len(blotter.open_orders[24]), 1)
        still_open_order = blotter.new_orders[0]
        self.assertEqual(still_open_order.id, second_order_id)
        self.assertEqual(still_open_order.status, ORDER_STATUS.OPEN)
        rejected_order = blotter.new_orders[1]
        self.assertEqual(rejected_order.status, ORDER_STATUS.REJECTED)
        self.assertEqual(rejected_order.reason, '')

        # Do it again, but reject it at a later time (after tradesimulation
        # pulls it from new_orders)
        blotter = Blotter()
        new_open_id = blotter.order(24, 10, MarketOrder())
        new_open_order = blotter.open_orders[24][0]
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
        blotter = Blotter()   # Reset for paranoia
        blotter.current_dt = datetime.datetime.now()
        filled_id = blotter.order(24, 100, MarketOrder())
        aapl_trade = create_trade(24, 50.0, 400, datetime.datetime.now())
        filled_order = None
        for txn, updated_order in blotter.process_trade(aapl_trade):
            filled_order = updated_order

        self.assertEqual(filled_order.id, filled_id)
        self.assertIn(filled_order, blotter.new_orders)
        self.assertEqual(filled_order.status, ORDER_STATUS.FILLED)
        self.assertNotIn(filled_order, blotter.open_orders[24])

        blotter.reject(filled_id)
        updated_order = blotter.orders[filled_id]
        self.assertEqual(updated_order.status, ORDER_STATUS.FILLED)

    def test_order_hold(self):
        """
        Held orders act almost identically to open orders, except for the
        status indication. When a fill happens, the order should switch
        status to OPEN/FILLED as necessary
        """
        blotter = Blotter()
        # Nothing happens on held of a non-existent order
        blotter.hold(56)
        self.assertEqual(blotter.new_orders, [])

        open_id = blotter.order(24, 100, MarketOrder())
        open_order = blotter.open_orders[24][0]
        self.assertEqual(open_order.id, open_id)

        blotter.hold(open_id)
        self.assertEqual(len(blotter.new_orders), 1)
        self.assertEqual(len(blotter.open_orders[24]), 1)
        held_order = blotter.new_orders[0]
        self.assertEqual(held_order.status, ORDER_STATUS.HELD)
        self.assertEqual(held_order.reason, '')

        blotter.cancel(held_order.id)
        self.assertEqual(len(blotter.new_orders), 1)
        self.assertEqual(len(blotter.open_orders[24]), 0)
        cancelled_order = blotter.new_orders[0]
        self.assertEqual(cancelled_order.id, held_order.id)
        self.assertEqual(cancelled_order.status, ORDER_STATUS.CANCELLED)

        for trade_amt in (100, 400):
            # Verify that incoming fills will change the order status.
            order_size = 100
            expected_filled = trade_amt * 0.25
            expected_open = order_size - expected_filled
            expected_status = ORDER_STATUS.OPEN if expected_open else \
                ORDER_STATUS.FILLED

            blotter = Blotter()
            blotter.current_dt = datetime.datetime.now()
            open_id = blotter.order(24, order_size, MarketOrder())
            open_order = blotter.open_orders[24][0]
            self.assertEqual(open_id, open_order.id)
            blotter.hold(open_id)
            held_order = blotter.new_orders[0]

            aapl_trade = create_trade(24, 50.0, trade_amt,
                                      datetime.datetime.now())
            filled_order = None
            for txn, updated_order in blotter.process_trade(aapl_trade):
                filled_order = updated_order
            self.assertEqual(filled_order.id, held_order.id)
            self.assertEqual(filled_order.status, expected_status)
            self.assertEqual(filled_order.filled, expected_filled)
            self.assertEqual(filled_order.open_amount, expected_open)
