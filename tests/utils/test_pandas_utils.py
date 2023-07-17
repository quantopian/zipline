"""
Tests for zipline/utils/pandas_utils.py
"""
import pandas as pd

from zipline.testing.predicates import assert_equal
from zipline.utils.pandas_utils import (
    categorical_df_concat,
    nearest_unequal_elements,
    new_pandas,
    skip_pipeline_new_pandas,
)
import pytest


class TestNearestUnequalElements:
    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern"])
    def test_nearest_unequal_elements(self, tz):

        dts = pd.to_datetime(
            ["2014-01-01", "2014-01-05", "2014-01-06", "2014-01-09"],
        ).tz_localize(tz)

        def t(s):
            return None if s is None else pd.Timestamp(s, tz=tz)

        for dt, before, after in (
            ("2013-12-30", None, "2014-01-01"),
            ("2013-12-31", None, "2014-01-01"),
            ("2014-01-01", None, "2014-01-05"),
            ("2014-01-02", "2014-01-01", "2014-01-05"),
            ("2014-01-03", "2014-01-01", "2014-01-05"),
            ("2014-01-04", "2014-01-01", "2014-01-05"),
            ("2014-01-05", "2014-01-01", "2014-01-06"),
            ("2014-01-06", "2014-01-05", "2014-01-09"),
            ("2014-01-07", "2014-01-06", "2014-01-09"),
            ("2014-01-08", "2014-01-06", "2014-01-09"),
            ("2014-01-09", "2014-01-06", None),
            ("2014-01-10", "2014-01-09", None),
            ("2014-01-11", "2014-01-09", None),
        ):
            computed = nearest_unequal_elements(dts, t(dt))
            expected = (t(before), t(after))
            assert computed == expected

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern"])
    def test_nearest_unequal_elements_short_dts(self, tz):

        # Length 1.
        dts = pd.to_datetime(["2014-01-01"]).tz_localize(tz)

        def t(s):
            return None if s is None else pd.Timestamp(s, tz=tz)

        for dt, before, after in (
            ("2013-12-31", None, "2014-01-01"),
            ("2014-01-01", None, None),
            ("2014-01-02", "2014-01-01", None),
        ):
            computed = nearest_unequal_elements(dts, t(dt))
            expected = (t(before), t(after))
            assert computed == expected

        # Length 0
        dts = pd.to_datetime([]).tz_localize(tz)
        for dt, before, after in (
            ("2013-12-31", None, None),
            ("2014-01-01", None, None),
            ("2014-01-02", None, None),
        ):
            computed = nearest_unequal_elements(dts, t(dt))
            expected = (t(before), t(after))
            assert computed == expected

    def test_nearest_unequal_bad_input(self):
        with pytest.raises(ValueError, match="dts must be unique"):
            nearest_unequal_elements(
                pd.to_datetime(["2014", "2014"]),
                pd.Timestamp("2014"),
            )

        with pytest.raises(ValueError, match="dts must be sorted in increasing order"):
            nearest_unequal_elements(
                pd.to_datetime(["2014", "2013"]),
                pd.Timestamp("2014"),
            )


class TestCatDFConcat:
    @pytest.mark.skipif(new_pandas, reason=skip_pipeline_new_pandas)
    def test_categorical_df_concat(self):

        inp = [
            pd.DataFrame(
                {
                    "A": pd.Series(["a", "b", "c"], dtype="category"),
                    "B": pd.Series([100, 102, 103], dtype="int64"),
                    "C": pd.Series(["x", "x", "x"], dtype="category"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series(["c", "b", "d"], dtype="category"),
                    "B": pd.Series([103, 102, 104], dtype="int64"),
                    "C": pd.Series(["y", "y", "y"], dtype="category"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series(["a", "b", "d"], dtype="category"),
                    "B": pd.Series([101, 102, 104], dtype="int64"),
                    "C": pd.Series(["z", "z", "z"], dtype="category"),
                }
            ),
        ]
        result = categorical_df_concat(inp)

        expected = pd.DataFrame(
            {
                "A": pd.Series(
                    ["a", "b", "c", "c", "b", "d", "a", "b", "d"], dtype="category"
                ),
                "B": pd.Series(
                    [100, 102, 103, 103, 102, 104, 101, 102, 104], dtype="int64"
                ),
                "C": pd.Series(
                    ["x", "x", "x", "y", "y", "y", "z", "z", "z"], dtype="category"
                ),
            },
        )
        expected.index = pd.Index([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype="int64")
        assert_equal(expected, result)
        assert_equal(expected["A"].cat.categories, result["A"].cat.categories)
        assert_equal(expected["C"].cat.categories, result["C"].cat.categories)

    def test_categorical_df_concat_value_error(self):

        mismatched_dtypes = [
            pd.DataFrame(
                {
                    "A": pd.Series(["a", "b", "c"], dtype="category"),
                    "B": pd.Series([100, 102, 103], dtype="int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series(["c", "b", "d"], dtype="category"),
                    "B": pd.Series([103, 102, 104], dtype="float64"),
                }
            ),
        ]
        mismatched_column_names = [
            pd.DataFrame(
                {
                    "A": pd.Series(["a", "b", "c"], dtype="category"),
                    "B": pd.Series([100, 102, 103], dtype="int64"),
                }
            ),
            pd.DataFrame(
                {
                    "A": pd.Series(["c", "b", "d"], dtype="category"),
                    "X": pd.Series([103, 102, 104], dtype="int64"),
                }
            ),
        ]

        with pytest.raises(
            ValueError, match="Input DataFrames must have the same columns/dtypes."
        ):
            categorical_df_concat(mismatched_dtypes)

        with pytest.raises(
            ValueError, match="Input DataFrames must have the same columns/dtypes."
        ):
            categorical_df_concat(mismatched_column_names)
