from functools import reduce
import operator as op

import numpy as np
from numpy import nan
import pandas as pd

from zipline.lib.labelarray import LabelArray
from zipline.pipeline import Classifier
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.expression import methods_to_ops
from zipline.testing import parameter_space
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import (
    categorical_dtype,
    int64_dtype,
)

from .base import BaseUSEquityPipelineTestCase
import pytest
import re

bytes_dtype = np.dtype("S3")
unicode_dtype = np.dtype("U3")


class ClassifierTestCase(BaseUSEquityPipelineTestCase):
    @parameter_space(mv=[-1, 0, 1, 999])
    def test_integral_isnull(self, mv):
        class C(Classifier):
            dtype = int64_dtype
            missing_value = mv
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of missing and non-missing values.
        data = np.array(
            [[-1, 1, 0, 2], [3, 0, 1, 0], [-5, 0, -1, 0], [-3, 1, 2, 2]],
            dtype=int64_dtype,
        )

        self.check_terms(
            terms={"isnull": c.isnull(), "notnull": c.notnull()},
            expected={
                "isnull": data == mv,
                "notnull": data != mv,
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(mv=["0", None])
    def test_string_isnull(self, mv):
        class C(Classifier):
            dtype = categorical_dtype
            missing_value = mv
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of missing and non-missing values.
        raw = np.asarray(
            [
                ["", "a", "ab", "ba"],
                ["z", "ab", "a", "ab"],
                ["aa", "ab", "", "ab"],
                ["aa", "a", "ba", "ba"],
            ],
            dtype=categorical_dtype,
        )
        data = LabelArray(raw, missing_value=mv)

        self.check_terms(
            terms={"isnull": c.isnull(), "notnull": c.notnull()},
            expected={
                "isnull": np.equal(raw, mv),
                "notnull": np.not_equal(raw, mv),
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(compval=[0, 1, 999])
    def test_eq(self, compval):
        class C(Classifier):
            dtype = int64_dtype
            missing_value = -1
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = np.array(
            [[-1, 1, 0, 2], [3, 0, 1, 0], [-5, 0, -1, 0], [-3, 1, 2, 2]],
            dtype=int64_dtype,
        )

        self.check_terms(
            terms={
                "eq": c.eq(compval),
            },
            expected={
                "eq": (data == compval),
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(
        __fail_fast=True,
        compval=["a", "ab", "not in the array"],
        labelarray_dtype=(bytes_dtype, categorical_dtype, unicode_dtype),
    )
    def test_string_eq(self, compval, labelarray_dtype):

        compval = labelarray_dtype.type(compval)

        class C(Classifier):
            dtype = categorical_dtype
            missing_value = ""
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = LabelArray(
            np.asarray(
                [
                    ["", "a", "ab", "ba"],
                    ["z", "ab", "a", "ab"],
                    ["aa", "ab", "", "ab"],
                    ["aa", "a", "ba", "ba"],
                ],
                dtype=labelarray_dtype,
            ),
            missing_value="",
        )

        self.check_terms(
            terms={
                "eq": c.eq(compval),
            },
            expected={
                "eq": (data == compval),
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(
        missing=[-1, 0, 1],
        dtype_=[int64_dtype, categorical_dtype],
    )
    def test_disallow_comparison_to_missing_value(self, missing, dtype_):
        if dtype_ == categorical_dtype:
            missing = str(missing)

        class C(Classifier):
            dtype = dtype_
            missing_value = missing
            inputs = ()
            window_length = 0

        expected_msg = (
            "Comparison against self.missing_value ({v!r}) in C.eq().\n"
            "Missing values have NaN semantics, so the requested comparison "
            "would always produce False.\nUse the isnull() method to check "
            "for missing values.".format(
                v=missing,
            )
        )
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            C().eq(missing)

    @parameter_space(compval=[0, 1, 999], missing=[-1, 0, 999])
    def test_not_equal(self, compval, missing):
        class C(Classifier):
            dtype = int64_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = np.array(
            [[-1, 1, 0, 2], [3, 0, 1, 0], [-5, 0, -1, 0], [-3, 1, 2, 2]],
            dtype=int64_dtype,
        )

        self.check_terms(
            terms={
                "ne": c != compval,
            },
            expected={
                "ne": (data != compval) & (data != C.missing_value),
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(
        __fail_fast=True,
        compval=["a", "ab", "", "not in the array"],
        missing=["a", "ab", "", "not in the array"],
        labelarray_dtype=(bytes_dtype, unicode_dtype, categorical_dtype),
    )
    def test_string_not_equal(self, compval, missing, labelarray_dtype):

        compval = labelarray_dtype.type(compval)

        class C(Classifier):
            dtype = categorical_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = LabelArray(
            np.asarray(
                [
                    ["", "a", "ab", "ba"],
                    ["z", "ab", "a", "ab"],
                    ["aa", "ab", "", "ab"],
                    ["aa", "a", "ba", "ba"],
                ],
                dtype=labelarray_dtype,
            ),
            missing_value=missing,
        )

        expected = (data.as_int_array() != data.reverse_categories.get(compval, -1)) & (
            data.as_int_array() != data.reverse_categories[C.missing_value]
        )

        self.check_terms(
            terms={
                "ne": c != compval,
            },
            expected={
                "ne": expected,
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(
        __fail_fast=True,
        compval=["a", "b", "ab", "not in the array"],
        missing=["a", "ab", "", "not in the array"],
        labelarray_dtype=(categorical_dtype, bytes_dtype, unicode_dtype),
    )
    def test_string_elementwise_predicates(self, compval, missing, labelarray_dtype):
        if labelarray_dtype == bytes_dtype:
            compval = compval.encode("utf-8")
            missing = missing.encode("utf-8")

            startswith_re = b"^" + compval + b".*"
            endswith_re = b".*" + compval + b"$"
            substring_re = b".*" + compval + b".*"
        else:
            startswith_re = "^" + compval + ".*"
            endswith_re = ".*" + compval + "$"
            substring_re = ".*" + compval + ".*"

        class C(Classifier):
            dtype = categorical_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = LabelArray(
            np.asarray(
                [
                    ["", "a", "ab", "ba"],
                    ["z", "ab", "a", "ab"],
                    ["aa", "ab", "", "ab"],
                    ["aa", "a", "ba", "ba"],
                ],
                dtype=labelarray_dtype,
            ),
            missing_value=missing,
        )

        terms = {
            "startswith": c.startswith(compval),
            "endswith": c.endswith(compval),
            "has_substring": c.has_substring(compval),
            # Equivalent filters using regex matching.
            "startswith_re": c.matches(startswith_re),
            "endswith_re": c.matches(endswith_re),
            "has_substring_re": c.matches(substring_re),
        }

        expected = {
            "startswith": (data.startswith(compval) & (data != missing)),
            "endswith": (data.endswith(compval) & (data != missing)),
            "has_substring": (data.has_substring(compval) & (data != missing)),
        }
        for key in list(expected):
            expected[key + "_re"] = expected[key]

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(
        __fail_fast=True,
        container_type=(set, list, tuple, frozenset),
        labelarray_dtype=(categorical_dtype, bytes_dtype, unicode_dtype),
    )
    def test_element_of_strings(self, container_type, labelarray_dtype):

        missing = labelarray_dtype.type("not in the array")

        class C(Classifier):
            dtype = categorical_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        c = C()

        raw = np.asarray(
            [
                ["", "a", "ab", "ba"],
                ["z", "ab", "a", "ab"],
                ["aa", "ab", "", "ab"],
                ["aa", "a", "ba", "ba"],
            ],
            dtype=labelarray_dtype,
        )
        data = LabelArray(raw, missing_value=missing)

        choices = [
            container_type(choices)
            for choices in [
                [],
                ["a", ""],
                ["a", "a", "a", "ab", "a"],
                set(data.reverse_categories) - {missing},
                ["random value", "ab"],
                ["_" * i for i in range(30)],
            ]
        ]

        def make_expected(choice_set):
            return np.vectorize(choice_set.__contains__, otypes=[bool])(raw)

        terms = {str(i): c.element_of(s) for i, s in enumerate(choices)}
        expected = {str(i): make_expected(s) for i, s in enumerate(choices)}

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    def test_element_of_integral(self):
        """
        Element of is well-defined for integral classifiers.
        """

        class C(Classifier):
            dtype = int64_dtype
            missing_value = -1
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of missing and non-missing values.
        data = np.array(
            [[-1, 1, 0, 2], [3, 0, 1, 0], [-5, 0, -1, 0], [-3, 1, 2, 2]],
            dtype=int64_dtype,
        )

        terms = {}
        expected = {}
        for choices in [(0,), (0, 1), (0, 1, 2)]:
            terms[str(choices)] = c.element_of(choices)
            expected[str(choices)] = reduce(
                op.or_,
                (data == elem for elem in choices),
                np.zeros_like(data, dtype=bool),
            )

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    def test_element_of_rejects_missing_value(self):
        """
        Test that element_of raises a useful error if we attempt to pass it an
        array of choices that include the classifier's missing_value.
        """
        missing = "not in the array"

        class C(Classifier):
            dtype = categorical_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        c = C()

        for bad_elems in ([missing], [missing, "random other value"]):
            expected = (
                "Found self.missing_value ('not in the array') in choices"
                " supplied to C.element_of().\n"
                "Missing values have NaN semantics, so the requested"
                " comparison would always produce False.\n"
                "Use the isnull() method to check for missing values.\n"
                "Received choices were {}.".format(bad_elems)
            )
            with pytest.raises(ValueError, match=re.escape(expected)):
                c.element_of(bad_elems)

    @parameter_space(dtype_=Classifier.ALLOWED_DTYPES)
    def test_element_of_rejects_unhashable_type(self, dtype_):
        class C(Classifier):
            dtype = dtype_
            missing_value = dtype.type("1")
            inputs = ()
            window_length = 0

        c = C()

        expected = (
            """Expected `choices` to be an iterable of hashable values, but got [{'a': 1}] instead.\n"""
            """This caused the following error: TypeError("unhashable type: 'dict'")."""
        )
        with pytest.raises(TypeError, match=re.escape(expected)):
            c.element_of([{"a": 1}])

    @parameter_space(
        __fail_fast=True,
        labelarray_dtype=(categorical_dtype, bytes_dtype, unicode_dtype),
        relabel_func=[
            lambda s: str(s[0]),
            lambda s: str(len(s)),
            lambda s: str(len([c for c in s if c == "a"])),
            lambda s: None,
        ],
    )
    def test_relabel_strings(self, relabel_func, labelarray_dtype):
        class C(Classifier):
            inputs = ()
            dtype = categorical_dtype
            missing_value = None
            window_length = 0

        c = C()

        raw = np.asarray(
            [
                ["a", "aa", "aaa", "abab"],
                ["bab", "aba", "aa", "bb"],
                ["a", "aba", "abaa", "abaab"],
                ["a", "aa", "aaa", "aaaa"],
            ],
            dtype=labelarray_dtype,
        )
        raw_relabeled = np.vectorize(relabel_func, otypes=[object])(raw)

        data = LabelArray(raw, missing_value=None)

        terms = {
            "relabeled": c.relabel(relabel_func),
        }
        expected_results = {
            "relabeled": LabelArray(raw_relabeled, missing_value=None),
        }

        self.check_terms(
            terms,
            expected_results,
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(
        __fail_fast=True,
        missing_value=[None, "M"],
    )
    def test_relabel_missing_value_interactions(self, missing_value):

        mv = missing_value

        class C(Classifier):
            inputs = ()
            dtype = categorical_dtype
            missing_value = mv
            window_length = 0

        c = C()

        def relabel_func(s):
            if s == "B":
                return mv
            return "".join([s, s])

        raw = np.asarray(
            [
                ["A", "B", "C", mv],
                [mv, "A", "B", "C"],
                ["C", mv, "A", "B"],
                ["B", "C", mv, "A"],
            ],
            dtype=categorical_dtype,
        )
        data = LabelArray(raw, missing_value=mv)

        expected_relabeled_raw = np.asarray(
            [
                ["AA", mv, "CC", mv],
                [mv, "AA", mv, "CC"],
                ["CC", mv, "AA", mv],
                [mv, "CC", mv, "AA"],
            ],
            dtype=categorical_dtype,
        )

        terms = {
            "relabeled": c.relabel(relabel_func),
        }
        expected_results = {
            "relabeled": LabelArray(expected_relabeled_raw, missing_value=mv),
        }

        self.check_terms(
            terms,
            expected_results,
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    def test_relabel_int_classifier_not_yet_supported(self):
        class C(Classifier):
            inputs = ()
            dtype = int64_dtype
            missing_value = -1
            window_length = 0

        c = C()

        expected = (
            "relabel() is only defined on Classifiers producing strings "
            "but it was called on a Classifier of dtype int64."
        )
        with pytest.raises(TypeError, match=re.escape(expected)):
            c.relabel(lambda x: 0 / 0)  # Function should never be called.

    @parameter_space(
        compare_op=[op.gt, op.ge, op.le, op.lt],
        dtype_and_missing=[(int64_dtype, 0), (categorical_dtype, "")],
    )
    def test_bad_compare(self, compare_op, dtype_and_missing):
        class C(Classifier):
            inputs = ()
            window_length = 0
            dtype = dtype_and_missing[0]
            missing_value = dtype_and_missing[1]

        expected = "cannot compare classifiers with %s" % (
            methods_to_ops["__%s__" % compare_op.__name__],
        )
        with pytest.raises(TypeError, match=re.escape(expected)):
            compare_op(C(), object())

    @parameter_space(
        dtype_and_missing=[(int64_dtype, -1), (categorical_dtype, None)],
        use_mask=[True, False],
    )
    def test_peer_count(self, dtype_and_missing, use_mask):
        class C(Classifier):
            dtype = dtype_and_missing[0]
            missing_value = dtype_and_missing[1]
            inputs = ()
            window_length = 0

        c = C()

        if dtype_and_missing[0] == int64_dtype:
            data = np.array(
                [
                    [1, 1, -1, 2, 1, -1],
                    [2, 1, 3, 2, 2, 2],
                    [-1, 1, 10, 10, 10, -1],
                    [3, 3, 3, 3, 3, 3],
                ],
                dtype=int64_dtype,
            )
        else:
            data = LabelArray(
                [
                    ["a", "a", None, "b", "a", None],
                    ["b", "a", "c", "b", "b", "b"],
                    [None, "a", "aa", "aa", "aa", None],
                    ["c", "c", "c", "c", "c", "c"],
                ],
                missing_value=None,
            )

        if not use_mask:
            mask = self.build_mask(self.ones_mask(shape=data.shape))
            expected = np.array(
                [
                    [3, 3, nan, 1, 3, nan],
                    [4, 1, 1, 4, 4, 4],
                    [nan, 1, 3, 3, 3, nan],
                    [6, 6, 6, 6, 6, 6],
                ],
            )
        else:
            # Punch a couple holes in the mask to check that we handle the mask
            # correctly.
            mask = self.build_mask(
                np.array(
                    [
                        [1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1],
                    ],
                    dtype="bool",
                )
            )
            expected = np.array(
                [
                    [2, 2, nan, 1, nan, nan],
                    [3, 1, 1, 3, 3, nan],
                    [nan, 1, 3, 3, 3, nan],
                    [4, 4, nan, nan, 4, 4],
                ],
            )

        terms = {
            "peer_counts": c.peer_count(),
        }
        expected_results = {
            "peer_counts": expected,
        }

        self.check_terms(
            terms=terms,
            expected=expected_results,
            initial_workspace={c: data},
            mask=mask,
        )


class TestPostProcessAndToWorkSpaceValue:
    def test_reversability_categorical(self):
        class F(Classifier):
            inputs = ()
            window_length = 0
            dtype = categorical_dtype
            missing_value = "<missing>"

        f = F()
        column_data = LabelArray(
            np.array(
                [["a", f.missing_value], ["b", f.missing_value], ["c", "d"]],
            ),
            missing_value=f.missing_value,
        )

        assert_equal(
            f.postprocess(column_data.ravel()),
            pd.Categorical(
                ["a", f.missing_value, "b", f.missing_value, "c", "d"],
            ),
        )

        # only include the non-missing data
        pipeline_output = pd.Series(
            data=["a", "b", "c", "d"],
            index=pd.MultiIndex.from_arrays(
                [
                    [
                        pd.Timestamp("2014-01-01"),
                        pd.Timestamp("2014-01-02"),
                        pd.Timestamp("2014-01-03"),
                        pd.Timestamp("2014-01-03"),
                    ],
                    [0, 0, 0, 1],
                ]
            ),
            dtype="category",
        )

        assert_equal(
            f.to_workspace_value(pipeline_output, pd.Index([0, 1])),
            column_data,
        )

    def test_reversability_int64(self):
        class F(Classifier):
            inputs = ()
            window_length = 0
            dtype = int64_dtype
            missing_value = -1

        f = F()
        column_data = np.array(
            [[0, f.missing_value], [1, f.missing_value], [2, 3]],
        )

        assert_equal(f.postprocess(column_data.ravel()), column_data.ravel())

        # only include the non-missing data
        pipeline_output = pd.Series(
            data=[0, 1, 2, 3],
            index=pd.MultiIndex.from_arrays(
                [
                    [
                        pd.Timestamp("2014-01-01"),
                        pd.Timestamp("2014-01-02"),
                        pd.Timestamp("2014-01-03"),
                        pd.Timestamp("2014-01-03"),
                    ],
                    [0, 0, 0, 1],
                ]
            ),
            dtype=int64_dtype,
        )

        assert_equal(
            f.to_workspace_value(pipeline_output, pd.Index([0, 1])),
            column_data,
        )


class TestRepr:
    def test_quantiles_graph_repr(self):
        quantiles = TestingDataSet.float_col.latest.quantiles(5)
        assert quantiles.graph_repr() == "Quantiles(5)"
