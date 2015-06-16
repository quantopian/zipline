"""
Tests for filter terms.
"""
from unittest import TestCase

from numpy import (
    arange,
    eye,
    float64,
    percentile,
    uint8,
)
from numpy.testing import assert_array_equal

from pandas import (
    date_range,
    Int64Index,
)

from zipline.errors import BadPercentileBounds
from zipline.modelling.factor import TestFactor


class F(TestFactor):
    inputs = ()
    window_length = 0


class FilterTestCase(TestCase):

    def setUp(self):
        self.f = F()
        self.dates = date_range('2014-01-01', periods=5, freq='D')
        self.assets = Int64Index(range(5))

    def tearDown(self):
        pass

    def test_bad_input(self):
        f = self.f

        bad_percentiles = [
            (-.1, 10),
            (10, 100.1),
            (20, 10),
            (50, 50),
        ]
        for min_, max_ in bad_percentiles:
            with self.assertRaises(BadPercentileBounds):
                f.percentile_between(min_, max_)

    def test_rank_percentile_nice_partitions(self):
        # Test case with nicely-defined partitions.
        data = eye(5, dtype=float64)
        for quintile in range(5):
            factor = self.f.percentile_between(
                quintile * 20.0,
                (quintile + 1) * 20.0,
            )
            result = factor.compute_from_arrays(
                [data],
                uint8,
                self.dates,
                self.assets,
            )

            if quintile < 4:
                # There are 4 0s and one 1 in each row, so the first 4
                # quintiles should be all the locations with zeros in the input
                # array.
                assert_array_equal(result, ~eye(5, dtype=bool))
            else:
                # The last quintile should contain all the 1s.
                assert_array_equal(result, eye(5, dtype=bool))

    def test_rank_percentile_nasty_partitions(self):
        # Test case with nasty partitions: divide up 5 assets into quartiles.
        data = arange(25).reshape(5, 5) % 4
        for quartile in range(4):
            lower_bound = quartile * 25.0
            upper_bound = (quartile + 1) * 25.0
            factor = self.f.percentile_between(lower_bound, upper_bound)
            result = factor.compute_from_arrays(
                [data],
                uint8,
                self.dates,
                self.assets
            )

            # There isn't a nice definition of correct behavior here, so for
            # now we guarantee the behavior of numpy.percentile.
            min_value = percentile(data, lower_bound, axis=1, keepdims=True)
            max_value = percentile(data, upper_bound, axis=1, keepdims=True)
            assert_array_equal(
                result,
                (min_value <= data) & (data <= max_value),
            )
