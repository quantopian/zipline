"""
Tests for filter terms.
"""
from unittest import TestCase

from numpy import (
    arange,
    array,
    eye,
    float64,
    nan,
    nanpercentile,
    ones_like,
    putmask,
)
from numpy.testing import assert_array_equal

from pandas import (
    DataFrame,
    date_range,
    Int64Index,
)

from zipline.errors import BadPercentileBounds
from zipline.modelling.factor import TestingFactor


class SomeFactor(TestingFactor):
    inputs = ()
    window_length = 0


class FilterTestCase(TestCase):

    def setUp(self):
        self.f = SomeFactor()
        self.dates = date_range('2014-01-01', periods=5, freq='D')
        self.assets = Int64Index(range(5))
        self.mask = DataFrame(True, index=self.dates, columns=self.assets)

    def tearDown(self):
        pass

    def maskframe(self, array):
        return DataFrame(
            array,
            index=date_range('2014-01-01', periods=array.shape[0], freq='D'),
            columns=arange(array.shape[1]),
        )

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
        eye5 = eye(5, dtype=float64)
        eye6 = eye(6, dtype=float64)
        nanmask = array([[0, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0]], dtype=bool)
        nandata = eye6.copy()
        putmask(nandata, nanmask, nan)

        for quintile in range(5):
            factor = self.f.percentile_between(
                quintile * 20.0,
                (quintile + 1) * 20.0,
            )
            # Test w/o any NaNs
            result = factor.compute_from_arrays(
                [eye5],
                self.maskframe(ones_like(eye5, dtype=bool)),
            )
            # Test with NaNs in the data.
            nandata_result = factor.compute_from_arrays(
                [nandata],
                self.maskframe(ones_like(nandata, dtype=bool)),
            )
            # Test with Falses in the mask.
            nanmask_result = factor.compute_from_arrays(
                [eye6],
                self.maskframe(~nanmask),
            )

            assert_array_equal(nandata_result, nanmask_result)

            if quintile < 4:
                # There are 4 0s and one 1 in each row, so the first 4
                # quintiles should be all the locations with zeros in the input
                # array.
                assert_array_equal(result, ~eye5.astype(bool))
                # Should reject all the ones, plus the nans.
                assert_array_equal(
                    nandata_result,
                    ~(nanmask | eye6.astype(bool))
                )

            else:
                # The last quintile should contain all the 1s.
                assert_array_equal(result, eye(5, dtype=bool))
                # Should accept all the 1s.
                assert_array_equal(nandata_result, eye(6, dtype=bool))

    def test_rank_percentile_nasty_partitions(self):
        # Test case with nasty partitions: divide up 5 assets into quartiles.
        data = arange(25, dtype=float).reshape(5, 5) % 4
        nandata = data.copy()
        nandata[eye(5, dtype=bool)] = nan
        for quartile in range(4):
            lower_bound = quartile * 25.0
            upper_bound = (quartile + 1) * 25.0
            factor = self.f.percentile_between(lower_bound, upper_bound)

            # There isn't a nice definition of correct behavior here, so for
            # now we guarantee the behavior of numpy.nanpercentile.

            result = factor.compute_from_arrays([data], self.mask)
            min_value = nanpercentile(data, lower_bound, axis=1, keepdims=True)
            max_value = nanpercentile(data, upper_bound, axis=1, keepdims=True)
            assert_array_equal(
                result,
                (min_value <= data) & (data <= max_value),
            )

            nanresult = factor.compute_from_arrays([nandata], self.mask)
            min_value = nanpercentile(
                nandata,
                lower_bound,
                axis=1,
                keepdims=True,
            )
            max_value = nanpercentile(
                nandata,
                upper_bound,
                axis=1,
                keepdims=True,
            )
            assert_array_equal(
                nanresult,
                (min_value <= nandata) & (nandata <= max_value),
            )

    def test_sequenced_filter(self):
        first = SomeFactor() < 1
        first_input = eye(5)
        first_result = first.compute_from_arrays([first_input], self.mask)
        assert_array_equal(first_result, ~eye(5, dtype=bool))

        # Second should pick out the fourth column.
        second = SomeFactor().eq(3.0)
        second_input = arange(25, dtype=float).reshape(5, 5) % 5

        sequenced = first.then(second)

        result = sequenced.compute_from_arrays(
            [first_result, second_input],
            self.mask,
        )
        expected_result = (first_result & (second_input == 3.0))
        assert_array_equal(result, expected_result)

    def test_sequenced_filter_order_dependent(self):
        f = SomeFactor() < 1
        f_input = eye(5)
        f_result = f.compute_from_arrays([f_input], self.mask)
        assert_array_equal(f_result, ~eye(5, dtype=bool))

        g = SomeFactor().percentile_between(80, 100)
        g_input = arange(25, dtype=float).reshape(5, 5) % 5
        g_result = g.compute_from_arrays([g_input], self.mask)
        assert_array_equal(g_result, g_input == 4)

        result = f.then(g).compute_from_arrays(
            [f_result, g_input],
            self.mask,
        )
        # Input data is strictly increasing, so the result should be the top
        # value not filtered by first.
        expected_result = array(
            [[0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0]],
            dtype=bool,
        )
        assert_array_equal(result, expected_result)

        result = g.then(f).compute_from_arrays(
            [g_result, f_input],
            self.mask,
        )

        # Percentile calculated first, then diagonal is removed.
        expected_result = array(
            [[0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0]],
            dtype=bool,
        )
        assert_array_equal(result, expected_result)
