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
from six.moves import range

from zipline.errors import(
    BadOrderParameters
)
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.testing.fixtures import (
    WithLogger,
    ZiplineTestCase,
)


class ExecutionStyleTestCase(WithLogger, ZiplineTestCase):
    """
    Tests for zipline ExecutionStyle classes.
    """

    epsilon = .000001

    # Input, expected on limit buy/stop sell, expected on limit sell/stop buy.
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

    class ArbitraryObject():
        def __str__(self):
            return """This should yield a bad order error when
            passed as a stop or limit price."""

    INVALID_PRICES = [
        (-1,),
        (-1.0,),
        (0 - epsilon,),
        (float('nan'),),
        (float('inf'),),
        (ArbitraryObject(),),
    ]

    @parameterized.expand(INVALID_PRICES)
    def test_invalid_prices(self, price):
        """
        Test that execution styles throw appropriate exceptions upon receipt
        of an invalid price field.
        """
        with self.assertRaises(BadOrderParameters):
            LimitOrder(price)

        with self.assertRaises(BadOrderParameters):
            StopOrder(price)

        for lmt, stp in [(price, 1), (1, price), (price, price)]:
            with self.assertRaises(BadOrderParameters):
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
    def test_limit_order_prices(self,
                                price,
                                expected_limit_buy_or_stop_sell,
                                expected_limit_sell_or_stop_buy):
        """
        Test price getters for the LimitOrder class.
        """
        style = LimitOrder(price)

        self.assertEqual(expected_limit_buy_or_stop_sell,
                         style.get_limit_price(True))
        self.assertEqual(expected_limit_sell_or_stop_buy,
                         style.get_limit_price(False))

        self.assertEqual(None, style.get_stop_price(True))
        self.assertEqual(None, style.get_stop_price(False))

    @parameterized.expand(EXPECTED_PRICE_ROUNDING)
    def test_stop_order_prices(self,
                               price,
                               expected_limit_buy_or_stop_sell,
                               expected_limit_sell_or_stop_buy):
        """
        Test price getters for StopOrder class. Note that the expected rounding
        direction for stop prices is the reverse of that for limit prices.
        """
        style = StopOrder(price)

        self.assertEqual(None, style.get_limit_price(False))
        self.assertEqual(None, style.get_limit_price(True))

        self.assertEqual(expected_limit_buy_or_stop_sell,
                         style.get_stop_price(False))
        self.assertEqual(expected_limit_sell_or_stop_buy,
                         style.get_stop_price(True))

    @parameterized.expand(EXPECTED_PRICE_ROUNDING)
    def test_stop_limit_order_prices(self,
                                     price,
                                     expected_limit_buy_or_stop_sell,
                                     expected_limit_sell_or_stop_buy):
        """
        Test price getters for StopLimitOrder class. Note that the expected
        rounding direction for stop prices is the reverse of that for limit
        prices.
        """

        style = StopLimitOrder(price, price + 1)

        self.assertEqual(expected_limit_buy_or_stop_sell,
                         style.get_limit_price(True))
        self.assertEqual(expected_limit_sell_or_stop_buy,
                         style.get_limit_price(False))

        self.assertEqual(expected_limit_buy_or_stop_sell + 1,
                         style.get_stop_price(False))
        self.assertEqual(expected_limit_sell_or_stop_buy + 1,
                         style.get_stop_price(True))
