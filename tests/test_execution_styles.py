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
from parameterized import parameterized
import pandas as pd

from zipline.errors import BadOrderParameters
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.testing.fixtures import (
    ZiplineTestCase,
    WithConstantFutureMinuteBarData,
)

from zipline.testing.predicates import assert_equal
import pytest


class ExecutionStyleTestCase(WithConstantFutureMinuteBarData, ZiplineTestCase):
    """
    Tests for zipline ExecutionStyle classes.
    """

    class ArbitraryObject:
        def __str__(self):
            return """This should yield a bad order error when
            passed as a stop or limit price."""

    epsilon = 0.000001

    INVALID_PRICES = [
        (-1,),
        (-1.0,),
        (0 - epsilon,),
        (float("nan"),),
        (float("inf"),),
        (ArbitraryObject(),),
    ]

    # Input, expected on limit buy/stop sell, expected on limit sell/stop buy.
    EXPECTED_PRICE_ROUNDING = [
        (0.00, 0.00, 0.00),
        (0.0005, 0.00, 0.00),
        (1.0005, 1.00, 1.00),  # Lowest value to round down on sell.
        (1.0005 + epsilon, 1.00, 1.01),
        (1.0095 - epsilon, 1.0, 1.01),
        (1.0095, 1.01, 1.01),  # Highest value to round up on buy.
        (0.01, 0.01, 0.01),
    ]

    # Testing for an asset with a tick_size of 0.0001
    smaller_epsilon = 0.00000001

    EXPECTED_PRECISION_ROUNDING = [
        (0.00, 0.00, 0.00),
        (0.0005, 0.0005, 0.0005),
        (0.00005, 0.00, 0.0001),
        (0.000005, 0.00, 0.00),
        (1.000005, 1.00, 1.00),  # Lowest value to round down on sell.
        (1.000005 + smaller_epsilon, 1.00, 1.0001),
        (1.000095 - smaller_epsilon, 1.0, 1.0001),
        (1.000095, 1.0001, 1.0001),  # Highest value to round up on buy.
        (0.01, 0.01, 0.01),
    ]

    # Testing for an asset with a tick_size of 0.05
    EXPECTED_CUSTOM_TICK_SIZE_ROUNDING = [
        (0.00, 0.00, 0.00),
        (0.0005, 0.00, 0.00),
        (1.0025, 1.00, 1.00),  # Lowest value to round down on sell.
        (1.0025 + epsilon, 1.00, 1.05),
        (1.0475 - epsilon, 1.0, 1.05),
        (1.0475, 1.05, 1.05),  # Highest value to round up on buy.
        (0.05, 0.05, 0.05),
    ]

    # Test that the same rounding behavior is maintained if we add between 1
    # and 10 to all values, because floating point math is made of lies.
    EXPECTED_PRICE_ROUNDING += [
        (x + delta, y + delta, z + delta)
        for (x, y, z) in EXPECTED_PRICE_ROUNDING
        for delta in range(1, 10)
    ]

    EXPECTED_PRECISION_ROUNDING += [
        (x + delta, y + delta, z + delta)
        for (x, y, z) in EXPECTED_PRECISION_ROUNDING
        for delta in range(1, 10)
    ]

    EXPECTED_CUSTOM_TICK_SIZE_ROUNDING += [
        (x + delta, y + delta, z + delta)
        for (x, y, z) in EXPECTED_CUSTOM_TICK_SIZE_ROUNDING
        for delta in range(1, 10)
    ]

    # Combine everything into one parameter set
    FINAL_PARAMETER_SET = (
        [(x, y, z, 1) for (x, y, z) in EXPECTED_PRICE_ROUNDING]
        + [(x, y, z, 2) for (x, y, z) in EXPECTED_PRECISION_ROUNDING]
        + [(x, y, z, 3) for (x, y, z) in EXPECTED_CUSTOM_TICK_SIZE_ROUNDING]
    )

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                1: {
                    "multiplier": 100,
                    "tick_size": 0.01,
                    "symbol": "F1",
                    "exchange": "TEST",
                },
                2: {
                    "multiplier": 100,
                    "tick_size": 0.0001,
                    "symbol": "F2",
                    "exchange": "TEST",
                },
                3: {
                    "multiplier": 100,
                    "tick_size": 0.05,
                    "symbol": "F3",
                    "exchange": "TEST",
                },
            },
            orient="index",
        )

    @classmethod
    def init_class_fixtures(cls):
        super(ExecutionStyleTestCase, cls).init_class_fixtures()

    @parameterized.expand(INVALID_PRICES)
    def test_invalid_prices(self, price):
        """
        Test that execution styles throw appropriate exceptions upon receipt
        of an invalid price field.
        """
        with pytest.raises(BadOrderParameters):
            LimitOrder(price)

        with pytest.raises(BadOrderParameters):
            StopOrder(price)

        for lmt, stp in [(price, 1), (1, price), (price, price)]:
            with pytest.raises(BadOrderParameters):
                StopLimitOrder(lmt, stp)

    def test_market_order_prices(self):
        """
        Basic unit tests for the MarketOrder class.
        """
        style = MarketOrder()

        assert_equal(style.get_limit_price(_is_buy=True), None)
        assert_equal(style.get_limit_price(_is_buy=False), None)

        assert_equal(style.get_stop_price(_is_buy=True), None)
        assert_equal(style.get_stop_price(_is_buy=False), None)

    @parameterized.expand(FINAL_PARAMETER_SET)
    def test_limit_order_prices(
        self,
        price,
        expected_limit_buy_or_stop_sell,
        expected_limit_sell_or_stop_buy,
        asset,
    ):
        """
        Test price getters for the LimitOrder class.
        """
        style = LimitOrder(price, asset=self.asset_finder.retrieve_asset(asset))

        assert_equal(
            expected_limit_buy_or_stop_sell, style.get_limit_price(is_buy=True)
        )
        assert_equal(
            expected_limit_sell_or_stop_buy, style.get_limit_price(is_buy=False)
        )

        assert_equal(None, style.get_stop_price(_is_buy=True))
        assert_equal(None, style.get_stop_price(_is_buy=False))

    @parameterized.expand(FINAL_PARAMETER_SET)
    def test_stop_order_prices(
        self,
        price,
        expected_limit_buy_or_stop_sell,
        expected_limit_sell_or_stop_buy,
        asset,
    ):
        """
        Test price getters for StopOrder class. Note that the expected rounding
        direction for stop prices is the reverse of that for limit prices.
        """
        style = StopOrder(price, asset=self.asset_finder.retrieve_asset(asset))

        assert_equal(None, style.get_limit_price(_is_buy=False))
        assert_equal(None, style.get_limit_price(_is_buy=True))

        assert_equal(
            expected_limit_buy_or_stop_sell, style.get_stop_price(is_buy=False)
        )
        assert_equal(expected_limit_sell_or_stop_buy, style.get_stop_price(is_buy=True))

    @parameterized.expand(FINAL_PARAMETER_SET)
    def test_stop_limit_order_prices(
        self,
        price,
        expected_limit_buy_or_stop_sell,
        expected_limit_sell_or_stop_buy,
        asset,
    ):
        """
        Test price getters for StopLimitOrder class. Note that the expected
        rounding direction for stop prices is the reverse of that for limit
        prices.
        """

        style = StopLimitOrder(
            price, price + 1, asset=self.asset_finder.retrieve_asset(asset)
        )

        assert_equal(
            expected_limit_buy_or_stop_sell, style.get_limit_price(is_buy=True)
        )
        assert_equal(
            expected_limit_sell_or_stop_buy, style.get_limit_price(is_buy=False)
        )

        assert_equal(
            expected_limit_buy_or_stop_sell + 1, style.get_stop_price(is_buy=False)
        )
        assert_equal(
            expected_limit_sell_or_stop_buy + 1, style.get_stop_price(is_buy=True)
        )
