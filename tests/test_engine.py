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
    lookback = 3
    inputs = [SomeDataSet.foo, SomeDataSet.bar]

    @staticmethod
    def compute(foo, bar):
        return (foo - bar).sum(axis=1)


class SimpleFFCEngineTestCase(TestCase):

    def setUp(self):
        self.known_assets = [1, 2, 3]
        self.adjustments = {}
        self.loader = ConstantLoader(
            self.known_assets,
            self.adjustments,
            constants={
                SomeDataSet.foo: 1,
                SomeDataSet.bar: 2,
            }
        )
        # Create a disjoint set of dates to test that we correctly handle
        # lookbacks over incongruous ranges.
        self.jan = date_range('2014-01-01', '2014-01-07', tz='UTC')
        self.feb = date_range('2014-02-01', '2014-02-07', tz='UTC')
        self.all_dates = self.jan.union(self.feb)
        self.engine = SimpleFFCEngine(self.loader, self.all_dates)

    def tearDown(self):
        pass

    def test_single_factor(self):

        engine = self.engine

        factor = SumDifference()

        engine.add_factor(factor)
        engine.freeze()

        for input_ in factor.inputs:
            self.assertEqual(
                engine.extra_row_count(input_),
                factor.lookback - 1,
            )

        # Compute for february days so that lookback stretches into jan days.
        results = engine.compute_chunk(
            self.feb[0],
            self.feb[1],
            self.known_assets,
        )

        num_dates = 2
        num_assets = len(self.known_assets)

        assert_array_equal(
            results[factor],
            full((num_dates, num_assets), -3),
        )
