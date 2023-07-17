"""Tests for common behaviors shared by all ComputableTerms.
"""
import numpy as np

from zipline.lib.labelarray import LabelArray
from zipline.pipeline import Classifier, Factor, Filter
from zipline.testing import parameter_space
from zipline.utils.numpy_utils import (
    categorical_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    NaTns,
)

from .base import BaseUSEquityPipelineTestCase
import pytest
import re


class Floats(Factor):
    inputs = ()
    window_length = 0
    dtype = float64_dtype


class AltFloats(Factor):
    inputs = ()
    window_length = 0
    dtype = float64_dtype


class Dates(Factor):
    inputs = ()
    window_length = 0
    dtype = datetime64ns_dtype


class AltDates(Factor):
    inputs = ()
    window_length = 0
    dtype = datetime64ns_dtype


class Bools(Filter):
    inputs = ()
    window_length = 0


class AltBools(Filter):
    inputs = ()
    window_length = 0


class Strs(Classifier):
    inputs = ()
    window_length = 0
    dtype = categorical_dtype
    missing_value = None


class AltStrs(Classifier):
    inputs = ()
    window_length = 0
    dtype = categorical_dtype
    missing_value = None


class Ints(Classifier):
    inputs = ()
    window_length = 0
    dtype = int64_dtype
    missing_value = -1


class AltInts(Classifier):
    inputs = ()
    window_length = 0
    dtype = int64_dtype
    missing_value = -1


class FillNATestCase(BaseUSEquityPipelineTestCase):
    @parameter_space(
        null_locs=[
            # No NaNs.
            np.zeros((4, 4), dtype=bool),
            # All NaNs.
            np.ones((4, 4), dtype=bool),
            # NaNs on Diagonal
            np.eye(4, dtype=bool),
            # Nans every third element.
            (np.arange(16).reshape(4, 4) % 3) == 0,
        ]
    )
    def test_fillna_with_scalar(self, null_locs):
        shape = (4, 4)
        num_cells = shape[0] * shape[1]

        floats = np.arange(num_cells, dtype=float).reshape(shape)
        floats[null_locs] = np.nan
        float_fillval = 999.0
        float_expected = np.where(null_locs, float_fillval, floats)
        float_expected_zero = np.where(null_locs, 0.0, floats)

        dates = (
            np.arange(num_cells, dtype="i8")
            .view("M8[D]")
            .astype("M8[ns]")
            .reshape(shape)
        )
        dates[null_locs] = NaTns
        date_fillval = np.datetime64("2014-01-02", "ns")
        date_expected = np.where(null_locs, date_fillval, dates)

        strs = np.arange(num_cells).astype(str).astype(object).reshape(shape)
        strs[null_locs] = None
        str_fillval = "filled"
        str_expected = np.where(null_locs, str_fillval, strs)

        ints = np.arange(num_cells, dtype="i8").reshape(shape)
        ints[null_locs] = -1
        int_fillval = 777
        int_expected = np.where(null_locs, int_fillval, ints)

        terms = {
            "floats": Floats().fillna(float_fillval),
            # Make sure we accept integer as a fill value on float-dtype
            # factors.
            "floats_fill_zero": Floats().fillna(0),
            "dates": Dates().fillna(date_fillval),
            "strs": Strs().fillna(str_fillval),
            "ints": Ints().fillna(int_fillval),
        }

        expected = {
            "floats": float_expected,
            "floats_fill_zero": float_expected_zero,
            "dates": date_expected,
            "strs": self.make_labelarray(str_expected),
            "ints": int_expected,
        }

        self.check_terms(
            terms,
            expected,
            initial_workspace={
                Floats(): floats,
                Dates(): dates,
                Strs(): self.make_labelarray(strs),
                Ints(): ints,
            },
            mask=self.build_mask(self.ones_mask(shape=(4, 4))),
        )

    @parameter_space(
        null_locs=[
            # No NaNs.
            np.zeros((4, 4), dtype=bool),
            # # All NaNs.
            np.ones((4, 4), dtype=bool),
            # NaNs on Diagonal
            np.eye(4, dtype=bool),
            # Nans every third element.
            (np.arange(16).reshape((4, 4)) % 3) == 0,
        ]
    )
    def test_fillna_with_expression(self, null_locs):
        shape = (4, 4)
        mask = self.build_mask(self.ones_mask(shape=(4, 4)))
        state = np.random.RandomState(4)
        assets = self.asset_finder.retrieve_all(mask.columns)

        def rand_vals(dtype):
            return state.randint(1, 100, shape).astype(dtype)

        floats = np.arange(16, dtype=float).reshape(shape)
        floats[null_locs] = np.nan
        float_fillval = rand_vals(float)
        float_expected = np.where(null_locs, float_fillval, floats)
        float_expected_1d = np.where(null_locs, float_fillval[:, [0]], floats)

        dates = np.arange(16, dtype="i8").view("M8[D]").astype("M8[ns]").reshape(shape)
        dates[null_locs] = NaTns
        date_fillval = rand_vals("M8[D]").astype("M8[ns]")
        date_expected = np.where(null_locs, date_fillval, dates)
        date_expected_1d = np.where(null_locs, date_fillval[:, [1]], dates)

        strs = np.arange(16).astype(str).astype(object).reshape(shape)
        strs[null_locs] = None
        str_fillval = rand_vals(str)
        str_expected = np.where(null_locs, str_fillval, strs)
        str_expected_1d = np.where(null_locs, str_fillval[:, [2]], strs)

        ints = np.arange(16).reshape(shape)
        ints[null_locs] = -1
        int_fillval = rand_vals(int64_dtype)
        int_expected = np.where(null_locs, int_fillval, ints)
        int_expected_1d = np.where(null_locs, int_fillval[:, [3]], ints)

        terms = {
            "floats": Floats().fillna(AltFloats()),
            "floats_1d": Floats().fillna(AltFloats()[assets[0]]),
            "dates": Dates().fillna(AltDates()),
            "dates_1d": Dates().fillna(AltDates()[assets[1]]),
            "strs": Strs().fillna(AltStrs()),
            "strs_1d": Strs().fillna(AltStrs()[assets[2]]),
            "ints": Ints().fillna(AltInts()),
            "ints_1d": Ints().fillna(AltInts()[assets[3]]),
        }

        expected = {
            "floats": float_expected,
            "floats_1d": float_expected_1d,
            "dates": date_expected,
            "dates_1d": date_expected_1d,
            "strs": self.make_labelarray(str_expected),
            "strs_1d": self.make_labelarray(str_expected_1d),
            "ints": int_expected,
            "ints_1d": int_expected_1d,
        }

        self.check_terms(
            terms,
            expected,
            initial_workspace={
                Floats(): floats,
                Dates(): dates,
                Strs(): self.make_labelarray(strs),
                Ints(): ints,
                AltFloats(): float_fillval,
                AltDates(): date_fillval,
                AltStrs(): self.make_labelarray(str_fillval),
                AltInts(): int_fillval,
            },
            mask=mask,
        )

    def should_error(self, f, exc_type, expected_message):
        with pytest.raises(exc_type, match=re.escape(expected_message)):
            f()

    def test_bad_inputs(self):
        def dtype_for(o):
            return np.array([o]).dtype

        self.should_error(
            lambda: Floats().fillna("3.0"),
            TypeError,
            " from {!r} to {!r} according to the rule 'same_kind'".format(
                dtype_for("3.0"), np.dtype(float)
            ),
        )

        self.should_error(
            lambda: Dates().fillna("2014-01-02"),
            TypeError,
            "from {!r} to {!r} according to the rule 'same_kind'".format(
                dtype_for("2014-01-02"), np.dtype("M8[ns]")
            ),
        )

        self.should_error(
            lambda: Ints().fillna("300"),
            TypeError,
            "from {!r} to {!r} according to the rule 'same_kind'".format(
                dtype_for("300"), np.dtype("i8")
            ),
        )

        self.should_error(
            lambda: Strs().fillna(10.0),
            TypeError,
            "Fill value 10.0 is not a valid choice for term Strs with dtype"
            " object.\n\n"
            "Coercion attempt failed with: "
            "String-dtype classifiers can only produce bytes or str or NoneType.",
        )

    def make_labelarray(self, strs):
        return LabelArray(strs, missing_value=None)
