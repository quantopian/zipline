import math

from nose_parameterized import parameterized
from unittest import TestCase

from zipline.finance.blotter import round_for_minimum_price_variation


class BlotterTestCase(TestCase):

    @parameterized.expand([(0.00, 0.00),
                           (0.01, 0.01),
                           (0.0005, 0.00),
                           (1.006, 1.00),
                           (1.0095, 1.01),
                           (1.00949, 1.00),
                           (1.0005, 1.00)])
    def test_round_for_minimum_price_variation_buy(self, price, expected):
        result = round_for_minimum_price_variation(price, is_buy=True)
        self.assertEqual(result, expected)
        self.assertEqual(math.copysign(1.0, result),
                         math.copysign(1.0, expected))

    @parameterized.expand([(0.00, 0.00),
                           (0.01, 0.01),
                           (0.0005, 0.00),
                           (1.006, 1.01),
                           (1.0005, 1.00),
                           (1.00051, 1.01),
                           (1.0095, 1.01)])
    def test_round_for_minimum_price_variation_sell(self, price, expected):
        result = round_for_minimum_price_variation(price, is_buy=False)
        self.assertEqual(result, expected)
        self.assertEqual(math.copysign(1.0, result),
                         math.copysign(1.0, expected))
