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

from unittest import TestCase

from six.moves import range

from nose_parameterized import parameterized

from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)

from zipline.utils.test_utils import(
    setup_logger,
    teardown_logger,
)


class ExecutionStyleTestCase(TestCase):
    """
    Tests for zipline ExecutionStyle classes.
    """

    epsilon = .000001

    # Input, expected for buy, expected for sell.
    EXPECTED_PRICE_ROUNDING = [
        (0.00, 0.00, 0.00),
        (0.0005, 0.00, 0.00),
        (1.0005, 1.00, 1.00),  # Lowest value to round down on sell.
        (1.0005 + epsilon, 1.00, 1.01),
        (1.0095 - epsilon, 1.0, 1.01),
        (1.0095, 1.01, 1.01),  # Highest value to round up on buy.
        (0.01, 0.01, 0.01)
    ]

    # Test that the same rounding behavior is maintained if we add between 1
    # and 10 to all values, because floating point math is made of lies.
    EXPECTED_PRICE_ROUNDING += [
        (x + delta, y + delta, z + delta)
        for (x, y, z) in EXPECTED_PRICE_ROUNDING
        for delta in range(1, 10)
    ]

    INVALID_PRICES = [(-1,), (-1.0,), (0 - epsilon,)]

    def setUp(self):
        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

    @parameterized.expand(INVALID_PRICES)
    def test_invalid_prices(self, price):
        """
        Test that execution styles throw appropriate exceptions upon receipt
        of an invalid price field.
        """
        with self.assertRaises(ValueError):
            LimitOrder(price)

        with self.assertRaises(ValueError):
            StopOrder(price)

        for lmt, stp in [(price, 1), (1, price), (price, price)]:
            with self.assertRaises(ValueError):
                StopLimitOrder(lmt, stp)

    def test_market_order_prices(self):
        """
        Basic unit tests for the MarketOrder class.
        """
        style = MarketOrder()

        self.assertEqual(style.get_limit_price(True), None)
        self.assertEqual(style.get_limit_price(False), None)

        self.assertEqual(style.get_stop_price(True), None)
        self.assertEqual(style.get_stop_price(False), None)

    @parameterized.expand(EXPECTED_PRICE_ROUNDING)
    def test_limit_order_prices(self, price, expected_buy, expected_sell):
        """
        Test price getters for the LimitOrder class.
        """
        style = LimitOrder(price)

        self.assertEqual(expected_buy, style.get_limit_price(True))
        self.assertEqual(expected_sell, style.get_limit_price(False))

        self.assertEqual(None, style.get_stop_price(True))
        self.assertEqual(None, style.get_stop_price(False))

    @parameterized.expand(EXPECTED_PRICE_ROUNDING)
    def test_stop_order_prices(self, price, expected_buy, expected_sell):
        """
        Test price getters for StopOrder class.
        """
        style = StopOrder(price)

        self.assertEqual(None, style.get_limit_price(True))
        self.assertEqual(None, style.get_limit_price(False))

        self.assertEqual(expected_buy, style.get_stop_price(True))
        self.assertEqual(expected_sell, style.get_stop_price(False))

    @parameterized.expand(EXPECTED_PRICE_ROUNDING)
    def test_stop_limit_order_prices(self, price, expected_buy, expected_sell):
        """
        Test price getters for StopLimitOrder class.
        """

        style = StopLimitOrder(price, price + 1)

        self.assertEqual(expected_buy, style.get_limit_price(True))
        self.assertEqual(expected_sell, style.get_limit_price(False))

        self.assertEqual(expected_buy + 1, style.get_stop_price(True))
        self.assertEqual(expected_sell + 1, style.get_stop_price(False))
