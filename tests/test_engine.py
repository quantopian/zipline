"""
Tests for SimpleFFCEngine
"""
from unittest import TestCase

from numpy import (
    float32,
    full,
    uint32,
    uint8,
)
from numpy.testing import assert_array_equal
from pandas import (
    date_range,
)

from zipline.data.dataset import (
    Column,
    DataSet,
)
from zipline.data.synthetic import (
    ConstantLoader,
)
from zipline.modelling.engine import SimpleFFCEngine
from zipline.modelling.factor import TestFactor


class SomeDataSet(DataSet):

    foo = Column(float32)
    bar = Column(uint32)
    buzz = Column(uint8)


def sum_all(foo, bar):
    return (foo + bar).sum()


class SumDifference(TestFactor):
    window_length = 3
    inputs = [SomeDataSet.foo, SomeDataSet.bar]

    @staticmethod
    def compute(foo, bar):
        return (foo - bar).sum(axis=0)


class ConstantInputTestCase(TestCase):

    def setUp(self):
        self.known_assets = [1, 2, 3]
        self.adjustments = {}
        self.loader = ConstantLoader(
            self.known_assets,
            self.adjustments,
            constants={
                SomeDataSet.foo: 1,
                SomeDataSet.bar: 2,
                SomeDataSet.buzz: 3,
            }
        )
        self.dates = date_range('2014-01-01', '2014-02-01', freq='D')
        self.engine = SimpleFFCEngine(self.loader, self.dates)

    def tearDown(self):
        pass

    def test_single_factor(self):

        engine = self.engine
        factor = SumDifference()
        shape = (num_dates, num_assets) = (2, len(self.known_assets))
        dates = self.dates[10:10 + num_dates]

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
            self.known_assets,
        )

        self.assertEqual(
            set(results.keys()),
            {factor, SomeDataSet.foo, SomeDataSet.bar},
        )

        for window in results[SomeDataSet.foo].traverse(num_dates):
            assert_array_equal(window, full(shape, 1))

        for window in results[SomeDataSet.bar].traverse(num_dates):
            assert_array_equal(window, full(shape, 2))

        assert_array_equal(
            results[factor],
            full(shape, -factor.window_length),
        )

    def test_multiple_factors(self):

        engine = self.engine
        shape = num_dates, num_assets = (2, len(self.known_assets))
        dates = self.dates[10:10 + num_dates]
        short_factor = SumDifference(window_length=3)
        long_factor = SumDifference(window_length=5)

        engine.add_factor(short_factor)
        engine.add_factor(long_factor)
        engine.freeze()

        for input_ in short_factor.inputs:
            self.assertEqual(
                engine.extra_row_count(input_),
                long_factor.window_length - 1,
            )

        results = engine.compute_chunk(
            dates[0],
            dates[-1],
            self.known_assets,
        )

        self.assertEqual(
            set(results.keys()),
            {short_factor, long_factor, SomeDataSet.foo, SomeDataSet.bar},
        )

        for factor in (short_factor, long_factor):
            assert_array_equal(
                results[factor],
                full(shape, -factor.window_length),
            )
