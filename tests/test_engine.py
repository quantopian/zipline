"""
Tests for SimpleFFCEngine
"""
from unittest import TestCase

from numpy import (
    float32,
    uint32,
    uint8,
)
from pandas import (
    date_range,
    DatetimeIndex,
)

from zipline.data.dataset import (
    Column,
    DataSet,
)
from zipline.data.synthetic import (
    ConstantLoader,
)
from zipline.modelling.engine import SimpleFFCEngine
from zipline.modelling.factor import Factor


class SomeDataSet(DataSet):

    foo = Column(float32)
    bar = Column(uint32)
    buzz = Column(uint8)


class SumLastThree(Factor):
    lookback = 3
    inputs = [SomeDataSet.foo, SomeDataSet.bar]

    def compute(self, assets, foo, bar):
        return (foo[-3:] + bar[-3:]).sum()


class SimpleFFCEngineTestCase(TestCase):

    def setUp(self):
        self.loader = ConstantLoader(1)
        # Create a disjoint set of dates to test that we correctly handle
        # lookbacks over incongruous ranges.
        self.feb = date_range('2014-01-01', '2014-01-07', tz='UTC')
        self.jan = date_range('2014-02-01', '2014-02-07', tz='UTC')
        self.all_dates = self.jan.union(self.feb)
        self.engine = SimpleFFCEngine(self.loader, self.all_dates)

    def tearDown(self):
        pass

    def test_single_factor(self):

        self.engine.add_factor(SumLastThree())
        self.engine.freeze()

        assets = [1, 2, 3, 4]
        # Compute for february days so that lookback stretches into jan days.
        results = self.engine.compute_chunk(self.feb, assets)
