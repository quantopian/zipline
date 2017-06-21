"""
Tests for zipline/utils/pandas_utils.py
"""
import pandas as pd

from zipline.testing import parameter_space, ZiplineTestCase
from zipline.testing.predicates import assert_equal
from zipline.utils.pandas_utils import (
    categorical_df_concat,
    nearest_unequal_elements,
    sliding_apply,
)


class TestNearestUnequalElements(ZiplineTestCase):

    @parameter_space(tz=['UTC', 'US/Eastern'], __fail_fast=True)
    def test_nearest_unequal_elements(self, tz):

        dts = pd.to_datetime(
            ['2014-01-01', '2014-01-05', '2014-01-06', '2014-01-09'],
        ).tz_localize(tz)

        def t(s):
            return None if s is None else pd.Timestamp(s, tz=tz)

        for dt, before, after in (('2013-12-30', None, '2014-01-01'),
                                  ('2013-12-31', None, '2014-01-01'),
                                  ('2014-01-01', None, '2014-01-05'),
                                  ('2014-01-02', '2014-01-01', '2014-01-05'),
                                  ('2014-01-03', '2014-01-01', '2014-01-05'),
                                  ('2014-01-04', '2014-01-01', '2014-01-05'),
                                  ('2014-01-05', '2014-01-01', '2014-01-06'),
                                  ('2014-01-06', '2014-01-05', '2014-01-09'),
                                  ('2014-01-07', '2014-01-06', '2014-01-09'),
                                  ('2014-01-08', '2014-01-06', '2014-01-09'),
                                  ('2014-01-09', '2014-01-06', None),
                                  ('2014-01-10', '2014-01-09', None),
                                  ('2014-01-11', '2014-01-09', None)):
            computed = nearest_unequal_elements(dts, t(dt))
            expected = (t(before), t(after))
            self.assertEqual(computed, expected)

    @parameter_space(tz=['UTC', 'US/Eastern'], __fail_fast=True)
    def test_nearest_unequal_elements_short_dts(self, tz):

        # Length 1.
        dts = pd.to_datetime(['2014-01-01']).tz_localize(tz)

        def t(s):
            return None if s is None else pd.Timestamp(s, tz=tz)

        for dt, before, after in (('2013-12-31', None, '2014-01-01'),
                                  ('2014-01-01', None, None),
                                  ('2014-01-02', '2014-01-01', None)):
            computed = nearest_unequal_elements(dts, t(dt))
            expected = (t(before), t(after))
            self.assertEqual(computed, expected)

        # Length 0
        dts = pd.to_datetime([]).tz_localize(tz)
        for dt, before, after in (('2013-12-31', None, None),
                                  ('2014-01-01', None, None),
                                  ('2014-01-02', None, None)):
            computed = nearest_unequal_elements(dts, t(dt))
            expected = (t(before), t(after))
            self.assertEqual(computed, expected)

    def test_nearest_unequal_bad_input(self):
        with self.assertRaises(ValueError) as e:
            nearest_unequal_elements(
                pd.to_datetime(['2014', '2014']),
                pd.Timestamp('2014'),
            )

        self.assertEqual(str(e.exception), 'dts must be unique')

        with self.assertRaises(ValueError) as e:
            nearest_unequal_elements(
                pd.to_datetime(['2014', '2013']),
                pd.Timestamp('2014'),
            )

        self.assertEqual(
            str(e.exception),
            'dts must be sorted in increasing order',
        )


class TestCatDFConcat(ZiplineTestCase):

    def test_categorical_df_concat(self):

        inp = [
            pd.DataFrame(
                {
                    'A': pd.Series(['a', 'b', 'c'], dtype='category'),
                    'B': pd.Series([100, 102, 103], dtype='int64'),
                    'C': pd.Series(['x', 'x', 'x'], dtype='category'),
                }
            ),
            pd.DataFrame(
                {
                    'A': pd.Series(['c', 'b', 'd'], dtype='category'),
                    'B': pd.Series([103, 102, 104], dtype='int64'),
                    'C': pd.Series(['y', 'y', 'y'], dtype='category'),
                }
            ),
            pd.DataFrame(
                {
                    'A': pd.Series(['a', 'b', 'd'], dtype='category'),
                    'B': pd.Series([101, 102, 104], dtype='int64'),
                    'C': pd.Series(['z', 'z', 'z'], dtype='category'),
                }
            ),
        ]
        result = categorical_df_concat(inp)

        expected = pd.DataFrame(
            {
                'A': pd.Series(
                    ['a', 'b', 'c', 'c', 'b', 'd', 'a', 'b', 'd'],
                    dtype='category'
                ),
                'B': pd.Series(
                    [100, 102, 103, 103, 102, 104, 101, 102, 104],
                    dtype='int64'
                ),
                'C': pd.Series(
                    ['x', 'x', 'x', 'y', 'y', 'y', 'z', 'z', 'z'],
                    dtype='category'
                ),
            },
        )
        expected.index = pd.Int64Index([0, 1, 2, 0, 1, 2, 0, 1, 2])
        assert_equal(expected, result)
        assert_equal(
            expected['A'].cat.categories,
            result['A'].cat.categories
        )
        assert_equal(
            expected['C'].cat.categories,
            result['C'].cat.categories
        )

    def test_categorical_df_concat_value_error(self):

        mismatched_dtypes = [
            pd.DataFrame(
                {
                    'A': pd.Series(['a', 'b', 'c'], dtype='category'),
                    'B': pd.Series([100, 102, 103], dtype='int64'),
                }
            ),
            pd.DataFrame(
                {
                    'A': pd.Series(['c', 'b', 'd'], dtype='category'),
                    'B': pd.Series([103, 102, 104], dtype='float64'),
                }
            ),
        ]
        mismatched_column_names = [
            pd.DataFrame(
                {
                    'A': pd.Series(['a', 'b', 'c'], dtype='category'),
                    'B': pd.Series([100, 102, 103], dtype='int64'),
                }
            ),
            pd.DataFrame(
                {
                    'A': pd.Series(['c', 'b', 'd'], dtype='category'),
                    'X': pd.Series([103, 102, 104], dtype='int64'),
                }
            ),
        ]

        with self.assertRaises(ValueError) as cm:
            categorical_df_concat(mismatched_dtypes)
        self.assertEqual(
            str(cm.exception),
            "Input DataFrames must have the same columns/dtypes."
        )

        with self.assertRaises(ValueError) as cm:
            categorical_df_concat(mismatched_column_names)
        self.assertEqual(
            str(cm.exception),
            "Input DataFrames must have the same columns/dtypes."
        )


class TestSlidingApply(ZiplineTestCase):

    def test_simple_windows(self):
        df = pd.DataFrame(
            [
                [1,   2,  3],
                [4,   5,  6],
                [7,   8,  9],
                [10, 11, 12],
            ],
            index=range(4),
        )

        result = list(sliding_apply(df, window_length=2, f=pd.DataFrame.sum))
        self.assertEqual(len(result), 3)
        assert_equal(result[0], pd.Series([5, 7, 9]))
        assert_equal(result[1], pd.Series([11, 13, 15]))
        assert_equal(result[2], pd.Series([17, 19, 21]))

        # A window length greater than the length of the given data frame
        # should have empty results.
        result = list(sliding_apply(df, window_length=10, f=pd.DataFrame.sum))
        self.assertEqual(result, [])

        def custom_function(df):
            """
            Take the dot product of each dataframe window with a constant
            vector, then take the minimum value of the resulting array. This
            returns a scalar value.
            """
            return min(df.dot([0.5, 0.3, 0.2]))

        result = list(sliding_apply(df, 3, custom_function))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (1 * 0.5) + (2 * 0.3) + (3 * 0.2))
        self.assertEqual(result[1], (4 * 0.5) + (5 * 0.3) + (6 * 0.2))

    def test_min_periods(self):
        df = pd.DataFrame(
            [
                [1,   2,  3],
                [4,   5,  6],
                [7,   8,  9],
                [10, 11, 12],
            ],
            index=range(4),
        )

        # The 'min_periods' argument cannot be more than 'window_length'.
        with self.assertRaises(ValueError):
            list(
                sliding_apply(
                    df=df, window_length=2, f=pd.DataFrame.sum, min_periods=3,
                )
            )

        # With 'min_periods' being 1, all four rows of the dataframe should be
        # operated on.
        result = list(
            sliding_apply(
                df=df, window_length=3, f=pd.DataFrame.sum, min_periods=1,
            )
        )
        self.assertEqual(len(result), 4)

        # With only one row we just get back the row as-is.
        assert_equal(result[0], pd.Series([1, 2, 3]))

        # Sum of rows 0 to 1.
        assert_equal(result[1], pd.Series([5, 7, 9]))

        # Sum of rows 0 to 2.
        assert_equal(result[2], pd.Series([12, 15, 18]))

        # Sum of rows 1 to 3.
        assert_equal(result[3], pd.Series([21, 24, 27]))
