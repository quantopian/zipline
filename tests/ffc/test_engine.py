"""
Tests for SimpleFFCEngine
"""
from unittest import TestCase

from numpy import (
    full,
)
from numpy.testing import assert_array_equal
from pandas import (
    date_range,
)

from zipline.data.equities import USEquityPricing
from zipline.data.ffc.synthetic import (
    ConstantLoader,
)
from zipline.modelling.engine import SimpleFFCEngine
from zipline.modelling.factor import TestFactor


class RollingSumDifference(TestFactor):
    window_length = 3
    inputs = [USEquityPricing.open, USEquityPricing.close]

    @staticmethod
    def from_windows(open, close):
        return (open - close).sum(axis=0)


class ConstantInputTestCase(TestCase):

    def setUp(self):
        self.constants = {
            # Every day, assume every stock starts at 2, goes down to 1,
            # goes up to 4, and finishes at 3.
            USEquityPricing.low: 1,
            USEquityPricing.open: 2,
            USEquityPricing.close: 3,
            USEquityPricing.high: 4,
        }
        self.assets = [1, 2, 3]
        self.dates = date_range('2014-01-01', '2014-02-01', freq='D')

    def test_single_factor(self):

        loader = ConstantLoader(
            known_assets=self.assets,
            adjustments={},
            constants=self.constants,
        )
        engine = SimpleFFCEngine(loader, self.dates)
        result_shape = (num_dates, num_assets) = (5, len(self.assets))
        dates = self.dates[10:10 + num_dates]

        factor = RollingSumDifference()

        engine.add_factor(factor)
        engine.freeze()

        for input_ in factor.inputs:
            self.assertEqual(
                engine.extra_row_count(input_),
                factor.window_length - 1,
            )

        results = engine.compute_chunk(
            dates[0],
            dates[-1],
            self.assets,
        )

        self.assertEqual(
            set(results.keys()),
            {factor, USEquityPricing.open, USEquityPricing.close},
        )

        for window in results[USEquityPricing.open].traverse(num_dates):
            assert_array_equal(window, full(result_shape, 2))

        for window in results[USEquityPricing.close].traverse(num_dates):
            assert_array_equal(window, full(result_shape, 3))

        assert_array_equal(
            results[factor],
            full(result_shape, -factor.window_length),
        )

    def test_multiple_rolling_factors(self):

        loader = ConstantLoader(
            known_assets=self.assets,
            adjustments={},
            constants=self.constants,
        )
        engine = SimpleFFCEngine(loader, self.dates)
        shape = num_dates, num_assets = (5, len(self.assets))
        dates = self.dates[10:10 + num_dates]

        short_factor = RollingSumDifference(window_length=3)
        long_factor = RollingSumDifference(window_length=5)
        high_factor = RollingSumDifference(
            window_length=3,
            inputs=[USEquityPricing.open, USEquityPricing.high],
        )

        engine.add_factor(short_factor)
        engine.add_factor(long_factor)
        engine.add_factor(high_factor)
        engine.freeze()

        # open and close should get extended by long_factor
        self.assertEqual(
            engine.extra_row_count(USEquityPricing.open),
            long_factor.window_length - 1,
        )
        self.assertEqual(
            engine.extra_row_count(USEquityPricing.close),
            long_factor.window_length - 1,
        )
        # high should not get extended
        self.assertEqual(
            engine.extra_row_count(USEquityPricing.high),
            high_factor.window_length - 1,
        )

        results = engine.compute_chunk(
            dates[0],
            dates[-1],
            self.assets,
        )

        self.assertEqual(
            set(results.keys()),
            {
                short_factor,
                long_factor,
                high_factor,
                USEquityPricing.open,
                USEquityPricing.close,
                USEquityPricing.high,
            },
        )

        # row-wise sum over an array whose values are all (1 - 2)
        assert_array_equal(
            results[short_factor],
            full(shape, -short_factor.window_length),
        )
        assert_array_equal(
            results[long_factor],
            full(shape, -long_factor.window_length),
        )
        # row-wise sum over an array whose values are all (1 - 3)
        assert_array_equal(
            results[high_factor],
            full(shape, -2 * high_factor.window_length),
        )

    def test_numeric_factor(self):
        constants = self.constants
        loader = ConstantLoader(
            known_assets=self.assets,
            adjustments={},
            constants=constants,
        )
        engine = SimpleFFCEngine(loader, self.dates)
        shape = num_dates, num_assets = (5, len(self.assets))
        dates = self.dates[10:10 + num_dates]
        high, low = USEquityPricing.high, USEquityPricing.low
        open, close = USEquityPricing.open, USEquityPricing.close

        high_minus_low = RollingSumDifference(inputs=[high, low])
        open_minus_close = RollingSumDifference(inputs=[open, close])
        avg = (high_minus_low + open_minus_close) / 2

        engine.add_factor(avg)
        engine.freeze()

        results = engine.compute_chunk(
            dates[0],
            dates[-1],
            self.assets,
        )

        high_low_result = results[high_minus_low]
        expected_high_low = 3 * (constants[high] - constants[low])
        assert_array_equal(high_low_result, full(shape, expected_high_low))

        open_close_result = results[open_minus_close]
        expected_open_close = 3 * (constants[open] - constants[close])
        assert_array_equal(open_close_result, full(shape, expected_open_close))

        avg_result = results[avg]
        expected_avg = (expected_high_low + expected_open_close) / 2.0
        assert_array_equal(avg_result, full(shape, expected_avg))
