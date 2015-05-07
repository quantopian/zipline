"""
Tests for Factor terms.
"""
from unittest import TestCase

from numpy import (
    array,
)
from numpy.testing import assert_array_equal
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
)
from six import iteritems

from zipline.errors import UnknownRankMethod
from zipline.modelling.factor import TestingFactor


class F(TestingFactor):
    inputs = ()
    window_length = 0


class FactorTestCase(TestCase):

    def setUp(self):
        self.f = F()
        self.dates = date_range('2014-01-01', periods=5, freq='D')
        self.assets = Int64Index(range(5))
        self.mask = DataFrame(True, index=self.dates, columns=self.assets)

    def tearDown(self):
        pass

    def test_bad_input(self):

        with self.assertRaises(UnknownRankMethod):
            self.f.rank("not a real rank method")

    def test_rank(self):

        # Generated with:
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]])
        expected_ranks = {
            'ordinal': array([[1., 3., 4., 5., 2.],
                              [2., 4., 5., 1., 3.],
                              [3., 5., 1., 2., 4.],
                              [4., 1., 2., 3., 5.],
                              [1., 3., 4., 5., 2.]]),
            'average': array([[1.5, 3., 4., 5., 1.5],
                              [2.5, 4., 5., 1., 2.5],
                              [3.5, 5., 1., 2., 3.5],
                              [4.5, 1., 2., 3., 4.5],
                              [1.5, 3., 4., 5., 1.5]]),
            'min': array([[1., 3., 4., 5., 1.],
                          [2., 4., 5., 1., 2.],
                          [3., 5., 1., 2., 3.],
                          [4., 1., 2., 3., 4.],
                          [1., 3., 4., 5., 1.]]),
            'max': array([[2., 3., 4., 5., 2.],
                          [3., 4., 5., 1., 3.],
                          [4., 5., 1., 2., 4.],
                          [5., 1., 2., 3., 5.],
                          [2., 3., 4., 5., 2.]]),
            'dense': array([[1., 2., 3., 4., 1.],
                            [2., 3., 4., 1., 2.],
                            [3., 4., 1., 2., 3.],
                            [4., 1., 2., 3., 4.],
                            [1., 2., 3., 4., 1.]]),
        }

        # Test with the default, which should be 'ordinal'.
        default_result = self.f.rank().compute_from_arrays([data], self.mask)
        assert_array_equal(default_result, expected_ranks['ordinal'])

        # Test with each method passed explicitly.
        for method, expected_result in iteritems(expected_ranks):
            result = self.f.rank(method=method).compute_from_arrays(
                [data],
                self.mask,
            )
            assert_array_equal(result, expected_ranks[method])
