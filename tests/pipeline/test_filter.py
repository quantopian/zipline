"""
Tests for filter terms.
"""
from functools import partial
from itertools import product
from operator import and_

from toolz import compose
import numpy as np
from numpy.random import RandomState
import pandas as pd

from zipline.errors import BadPercentileBounds
from zipline.lib.labelarray import labelarray_where
from zipline.pipeline import Filter, Factor, Pipeline
from zipline.pipeline.classifiers import Classifier
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.filters import (
    All,
    AllPresent,
    Any,
    AtLeastN,
    StaticAssets,
    StaticSids,
)
from zipline.testing import parameter_space, permute_rows, ZiplineTestCase
from zipline.testing.fixtures import WithSeededRandomPipelineEngine
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    object_dtype,
)
from .base import BaseUSEquityPipelineTestCase
import pytest


def rowwise_rank(array, mask=None):
    """
    Take a 2D array and return the 0-indexed sorted position of each element in
    the array for each row.

    Examples
    --------
    In [5]: data
    Out[5]:
    array([[-0.141, -1.103, -1.0171,  0.7812,  0.07  ],
           [ 0.926,  0.235, -0.7698,  1.4552,  0.2061],
           [ 1.579,  0.929, -0.557 ,  0.7896, -1.6279],
           [-1.362, -2.411, -1.4604,  1.4468, -0.1885],
           [ 1.272,  1.199, -3.2312, -0.5511, -1.9794]])

    In [7]: argsort(argsort(data))
    Out[7]:
    array([[2, 0, 1, 4, 3],
           [3, 2, 0, 4, 1],
           [4, 3, 1, 2, 0],
           [2, 0, 1, 4, 3],
           [4, 3, 0, 2, 1]])
    """
    # note that unlike scipy.stats.rankdata, the output here is 0-indexed, not
    # 1-indexed.
    return np.argsort(np.argsort(array))


class SomeFactor(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class SomeFilter(Filter):
    inputs = ()
    window_length = 0
    missing_value = False


class SomeDatetimeFactor(Factor):
    dtype = datetime64ns_dtype
    inputs = ()
    window_length = 0


class SomeOtherFactor(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class SomeClassifier(Classifier):
    dtype = int64_dtype
    inputs = ()
    window_length = 0
    missing_value = -1


class Mask(Filter):
    inputs = ()
    window_length = 0


class FilterTestCase(BaseUSEquityPipelineTestCase):
    def init_instance_fixtures(self):
        super(FilterTestCase, self).init_instance_fixtures()
        self.f = SomeFactor()
        self.g = SomeOtherFactor()
        self.c = SomeClassifier()
        self.datetime_f = SomeDatetimeFactor()

        self.factors_by_dtype_name = {
            "float64": self.f,
            "datetime64[ns]": self.datetime_f,
        }

    def test_bad_percentiles(self):
        f = self.f

        bad_percentiles = [
            (-0.1, 10),
            (10, 100.1),
            (20, 10),
            (50, 50),
        ]
        for min_, max_ in bad_percentiles:
            with pytest.raises(BadPercentileBounds):
                f.percentile_between(min_, max_)

    def test_top_and_bottom(self):
        data = self.randn_data(seed=5)  # Fix a seed for determinism.

        mask_data = np.ones_like(data, dtype=bool)
        mask_data[:, 0] = False

        nan_data = data.copy()
        nan_data[:, 0] = np.nan

        mask = Mask()

        methods = ["top", "bottom"]
        counts = 2, 3, 10
        term_combos = list(product(methods, counts, [True, False]))

        def termname(method, count, masked):
            return "_".join([method, str(count), "mask" if masked else ""])

        def expected_result(method, count, masked):
            # Ranking with a mask is equivalent to ranking with nans applied on
            # the masked values.
            to_rank = nan_data if masked else data

            if method == "top":
                return rowwise_rank(-to_rank) < count
            elif method == "bottom":
                return rowwise_rank(to_rank) < count

        # Add a term for each permutation of top/bottom, count, and
        # mask/no_mask.
        terms = {}
        expected = {}
        for method, count, masked in term_combos:
            kwargs = {"N": count}
            if masked:
                kwargs["mask"] = mask
            term = getattr(self.f, method)(**kwargs)
            name = termname(method, count, masked)
            terms[name] = term
            expected[name] = expected_result(method, count, masked)

        self.check_terms(
            terms,
            expected,
            initial_workspace={self.f: data, mask: mask_data},
            mask=self.build_mask(self.ones_mask()),
        )

    def test_percentile_between(self):

        quintiles = range(5)
        filter_names = ["pct_" + str(q) for q in quintiles]
        iter_quintiles = list(zip(filter_names, quintiles))
        terms = {
            name: self.f.percentile_between(q * 20.0, (q + 1) * 20.0)
            for name, q in iter_quintiles
        }

        # Test with 5 columns and no NaNs.
        eye5 = np.eye(5, dtype=np.float64)
        expected = {}
        for name, quintile in iter_quintiles:
            if quintile < 4:
                # There are four 0s and one 1 in each row, so the first 4
                # quintiles should be all the locations with zeros in the input
                # array.
                expected[name] = ~eye5.astype(bool)
            else:
                # The top quintile should match the sole 1 in each row.
                expected[name] = eye5.astype(bool)

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace={self.f: eye5},
            mask=self.build_mask(np.ones((5, 5))),
        )

        # Test with 6 columns, no NaNs, and one masked entry per day.
        eye6 = np.eye(6, dtype=np.float64)
        mask = np.array(
            [
                [1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 1],
            ],
            dtype=bool,
        )
        expected = {}
        for name, quintile in iter_quintiles:
            if quintile < 4:
                # Should keep all values that were 0 in the base data and were
                # 1 in the mask.
                expected[name] = mask & ~eye6.astype(bool)
            else:
                # The top quintile should match the sole 1 in each row.
                expected[name] = eye6.astype(bool)

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace={self.f: eye6},
            mask=self.build_mask(mask),
        )

        # Test with 6 columns, no mask, and one NaN per day.  Should have the
        # same outcome as if we had masked the NaNs.
        # In particular, the NaNs should never pass any filters.
        eye6_withnans = eye6.copy()
        np.putmask(eye6_withnans, ~mask, np.nan)
        expected = {}
        for name, quintile in iter_quintiles:
            if quintile < 4:
                # Should keep all values that were 0 in the base data and were
                # 1 in the mask.
                expected[name] = mask & (~eye6.astype(bool))
            else:
                # Should keep all the 1s in the base data.
                expected[name] = eye6.astype(bool)

        self.check_terms(
            terms,
            expected,
            initial_workspace={self.f: eye6},
            mask=self.build_mask(mask),
        )

    def test_percentile_nasty_partitions(self):
        # Test percentile with nasty partitions: divide up 5 assets into
        # quartiles.
        # There isn't a nice mathematical definition of correct behavior here,
        # so for now we guarantee the behavior of numpy.nanpercentile.  This is
        # mostly for regression testing in case we write our own specialized
        # percentile calculation at some point in the future.

        data = np.arange(25, dtype=float).reshape(5, 5) % 4
        quartiles = range(4)
        filter_names = ["pct_" + str(q) for q in quartiles]

        terms = {
            name: self.f.percentile_between(q * 25.0, (q + 1) * 25.0)
            for name, q in zip(filter_names, quartiles)
        }

        expected = {}
        for name, quartile in zip(filter_names, quartiles):
            lower = quartile * 25.0
            upper = (quartile + 1) * 25.0
            expected[name] = and_(
                np.nanpercentile(data, lower, axis=1, keepdims=True) <= data,
                data <= np.nanpercentile(data, upper, axis=1, keepdims=True),
            )

        self.check_terms(
            terms,
            expected,
            initial_workspace={self.f: data},
            mask=self.build_mask(np.ones((5, 5))),
        )

    def test_percentile_after_mask(self):
        f_input = np.eye(5)
        g_input = np.arange(25, dtype=float).reshape(5, 5)
        initial_mask = self.build_mask(np.ones((5, 5)))

        custom_mask = self.f < 1
        without_mask = self.g.percentile_between(80, 100)
        with_mask = self.g.percentile_between(80, 100, mask=custom_mask)

        terms = {
            "mask": custom_mask,
            "without_mask": without_mask,
            "with_mask": with_mask,
        }
        expected = {
            # Mask that accepts everything except the diagonal.
            "mask": ~np.eye(5, dtype=bool),
            # Second should pass the largest value each day.  Each row is
            # strictly increasing, so we always select the last value.
            "without_mask": np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                ],
                dtype=bool,
            ),
            # With a mask, we should remove the diagonal as an option before
            # computing percentiles.  On the last day, we should get the
            # second-largest value, rather than the largest.
            "with_mask": np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                ],  # Different from with!
                dtype=bool,
            ),
        }

        self.check_terms(
            terms,
            expected,
            initial_workspace={self.f: f_input, self.g: g_input},
            mask=initial_mask,
        )

    def test_isnan(self):
        data = self.randn_data(seed=10)
        diag = np.eye(*data.shape, dtype=bool)
        data[diag] = np.nan

        self.check_terms(
            terms={
                "isnan": self.f.isnan(),
                "isnull": self.f.isnull(),
            },
            expected={
                "isnan": diag,
                "isnull": diag,
            },
            initial_workspace={self.f: data},
            mask=self.build_mask(self.ones_mask()),
        )

    def test_notnan(self):
        data = self.randn_data(seed=10)
        diag = np.eye(*data.shape, dtype=bool)
        data[diag] = np.nan

        self.check_terms(
            terms={
                "notnan": self.f.notnan(),
                "notnull": self.f.notnull(),
            },
            expected={
                "notnan": ~diag,
                "notnull": ~diag,
            },
            initial_workspace={self.f: data},
            mask=self.build_mask(self.ones_mask()),
        )

    def test_isfinite(self):
        data = self.randn_data(seed=10)
        data[:, 0] = np.nan
        data[:, 2] = np.inf
        data[:, 4] = -np.inf

        self.check_terms(
            terms={"isfinite": self.f.isfinite()},
            expected={"isfinite": np.isfinite(data)},
            initial_workspace={self.f: data},
            mask=self.build_mask(self.ones_mask()),
        )

    def test_all_present_float_factor_input(self):
        """Test float factor input to `AllPresent`"""

        class SomeWindowSafeFactor(Factor):
            dtype = float64_dtype
            inputs = ()
            window_length = 0
            window_safe = True

        input_factor = SomeWindowSafeFactor()

        shape = (10, 6)
        data = self.randn_data(seed=10, shape=shape)
        data[np.eye(*shape, dtype=bool)] = input_factor.missing_value

        expected_3 = np.array(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )

        expected_4 = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        self.check_terms(
            terms={
                "3": AllPresent([input_factor], window_length=3),
                "4": AllPresent([input_factor], window_length=4),
            },
            expected={
                "3": expected_3,
                "4": expected_4,
            },
            initial_workspace={input_factor: data},
            mask=self.build_mask(np.ones(shape=shape)),
        )

    def test_all_present_int_factor_input(self):
        """Test int factor input to `AllPresent`"""

        class SomeWindowSafeIntFactor(Factor):
            dtype = int64_dtype
            inputs = ()
            window_length = 0
            window_safe = True
            missing_value = 0

        input_factor = SomeWindowSafeIntFactor()

        shape = (10, 6)
        data = RandomState(5).choice(range(1, 5), size=shape, replace=True)
        data[np.eye(*shape, dtype=bool)] = input_factor.missing_value

        expected_3 = np.array(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )

        expected_4 = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
        self.check_terms(
            terms={
                "3": AllPresent([input_factor], window_length=3),
                "4": AllPresent([input_factor], window_length=4),
            },
            expected={
                "3": expected_3,
                "4": expected_4,
            },
            initial_workspace={input_factor: data},
            mask=self.build_mask(np.ones(shape=shape)),
        )

    def test_all_present_classifier_input(self):
        """Test classifier factor input to `AllPresent`"""

        class SomeWindowSafeStringClassifier(Classifier):
            dtype = object_dtype
            inputs = ()
            window_length = 0
            missing_value = ""
            window_safe = True

        input_factor = SomeWindowSafeStringClassifier()

        shape = (10, 6)
        data = RandomState(6).choice(
            np.array(["a", "e", "i", "o", "u"], dtype=object_dtype),
            size=shape,
            replace=True,
        )
        data[np.eye(*shape, dtype=bool)] = input_factor.missing_value

        expected_3 = np.array(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )

        expected_4 = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )

        self.check_terms(
            terms={
                "3": AllPresent([input_factor], window_length=3),
                "4": AllPresent([input_factor], window_length=4),
            },
            expected={
                "3": expected_3,
                "4": expected_4,
            },
            initial_workspace={input_factor: data},
            mask=self.build_mask(np.ones(shape=shape)),
        )

    def test_all_present_filter_input(self):
        """Test error is raised when filter factor is input to `AllPresent`"""
        expected_msg = "Input to filter `AllPresent` cannot be a Filter."
        with pytest.raises(TypeError, match=expected_msg):
            AllPresent([Mask()], window_length=4)

    def test_all(self):

        data = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 0],
            ],
            dtype=bool,
        )

        # With a window_length of N, 0's should be "sticky" for the (N - 1)
        # days after the 0 in the base data.

        # Note that, the way ``self.run_graph`` works, we compute the same
        # number of output rows for all inputs, so we only get the last 4
        # outputs for expected_3 even though we have enought input data to
        # compute 5 rows.
        expected_3 = np.array(
            [
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=bool,
        )

        expected_4 = np.array(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=bool,
        )

        class Input(Filter):
            inputs = ()
            window_length = 0

        self.check_terms(
            terms={
                "3": All(inputs=[Input()], window_length=3),
                "4": All(inputs=[Input()], window_length=4),
            },
            expected={
                "3": expected_3,
                "4": expected_4,
            },
            initial_workspace={Input(): data},
            mask=self.build_mask(np.ones(shape=data.shape)),
        )

    def test_any(self):

        # FUN FACT: The inputs and outputs here are exactly the negation of
        # the inputs and outputs for test_all above. This isn't a coincidence.
        #
        # By de Morgan's Laws, we have::
        #
        #     ~(a & b) == (~a | ~b)
        #
        # negating both sides, we have::
        #
        #      (a & b) == ~(a | ~b)
        #
        # Since all(a, b) is isomorphic to (a & b), and any(a, b) is isomorphic
        # to (a | b), we have::
        #
        #     all(a, b) == ~(any(~a, ~b))
        #
        data = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=bool,
        )

        # With a window_length of N, 1's should be "sticky" for the (N - 1)
        # days after the 1 in the base data.

        # Note that, the way ``self.run_graph`` works, we compute the same
        # number of output rows for all inputs, so we only get the last 4
        # outputs for expected_3 even though we have enought input data to
        # compute 5 rows.
        expected_3 = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
            ],
            dtype=bool,
        )

        expected_4 = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1],
            ],
            dtype=bool,
        )

        class Input(Filter):
            inputs = ()
            window_length = 0

        self.check_terms(
            terms={
                "3": Any(inputs=[Input()], window_length=3),
                "4": Any(inputs=[Input()], window_length=4),
            },
            expected={
                "3": expected_3,
                "4": expected_4,
            },
            initial_workspace={Input(): data},
            mask=self.build_mask(np.ones(shape=data.shape)),
        )

    def test_at_least_N(self):

        # With a window_length of K, AtLeastN should return 1
        # if N or more 1's exist in the lookback window

        # This smoothing filter gives customizable "stickiness"

        data = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )

        expected_1 = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
            ],
            dtype=bool,
        )

        expected_2 = np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=bool,
        )

        expected_3 = np.array(
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=bool,
        )

        expected_4 = np.array(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )

        class Input(Filter):
            inputs = ()
            window_length = 0

        all_but_one = AtLeastN(inputs=[Input()], window_length=4, N=3)

        all_but_two = AtLeastN(inputs=[Input()], window_length=4, N=2)

        any_equiv = AtLeastN(inputs=[Input()], window_length=4, N=1)

        all_equiv = AtLeastN(inputs=[Input()], window_length=4, N=4)

        self.check_terms(
            terms={
                "AllButOne": all_but_one,
                "AllButTwo": all_but_two,
                "AnyEquiv": any_equiv,
                "AllEquiv": all_equiv,
                "Any": Any(inputs=[Input()], window_length=4),
                "All": All(inputs=[Input()], window_length=4),
            },
            expected={
                "Any": expected_1,
                "AnyEquiv": expected_1,
                "AllButTwo": expected_2,
                "AllButOne": expected_3,
                "All": expected_4,
                "AllEquiv": expected_4,
            },
            initial_workspace={Input(): data},
            mask=self.build_mask(np.ones(shape=data.shape)),
        )

    @parameter_space(factor_len=[2, 3, 4])
    def test_window_safe(self, factor_len):
        # all true data set of (days, securities)
        data = np.full(self.default_shape, True, dtype=bool)

        class InputFilter(Filter):
            inputs = ()
            window_length = 0

        class TestFactor(CustomFactor):
            dtype = float64_dtype
            inputs = (InputFilter(),)
            window_length = factor_len

            def compute(self, today, assets, out, filter_):
                # sum for each column
                out[:] = np.sum(filter_, axis=0)

        n = self.default_shape[0]
        output_shape = ((n - factor_len + 1), self.default_shape[1])
        np.full(output_shape, factor_len, dtype=np.float64)

        self.check_terms(
            terms={
                "windowsafe": TestFactor(),
            },
            expected={
                "windowsafe": np.full(output_shape, factor_len, dtype=np.float64),
            },
            initial_workspace={InputFilter(): data},
            mask=self.build_mask(self.ones_mask()),
        )

    def test_numerical_expression_filters_are_window_safe(self):
        class TestFactor(CustomFactor):
            inputs = ()
            window_length = 3

            def compute(self, today, assets, out):
                raise AssertionError("Never called")

        # Factors are not window safe by default.
        factor = TestFactor()
        assert not factor.window_safe

        filter_ = TestFactor() > 3
        assert filter_.window_safe

    @parameter_space(
        dtype=("float64", "datetime64[ns]"), seed=(1, 2, 3), __fail_fast=True
    )
    def test_top_with_groupby(self, dtype, seed):
        permute = partial(permute_rows, seed)
        permuted_array = compose(permute, partial(np.array, dtype=int64_dtype))

        shape = (8, 8)

        # Shuffle the input rows to verify that we correctly pick out the top
        # values independently of order.
        factor_data = permute(np.arange(0, 64, dtype=dtype).reshape(shape))

        classifier_data = permuted_array(
            [
                [0, 0, 1, 1, 2, 2, 0, 0],
                [0, 0, 1, 1, 2, 2, 0, 0],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        f = self.factors_by_dtype_name[dtype]
        c = self.c
        self.check_terms(
            terms={
                "1": f.top(1, groupby=c),
                "2": f.top(2, groupby=c),
                "3": f.top(3, groupby=c),
            },
            initial_workspace={
                f: factor_data,
                c: classifier_data,
            },
            expected={
                # Should be the rightmost location of each entry in
                # classifier_data.
                "1": permuted_array(
                    [
                        [0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=bool,
                ),
                # Should be the first and second-rightmost location of each
                # entry in classifier_data.
                "2": permuted_array(
                    [
                        [0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                    ],
                    dtype=bool,
                ),
                # Should be the first, second, and third-rightmost location of
                # each entry in classifier_data.
                "3": permuted_array(
                    [
                        [0, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1],
                    ],
                    dtype=bool,
                ),
            },
            mask=self.build_mask(self.ones_mask(shape=shape)),
        )

    @parameter_space(
        dtype=("float64", "datetime64[ns]"), seed=(1, 2, 3), __fail_fast=True
    )
    def test_top_and_bottom_with_groupby(self, dtype, seed):
        permute = partial(permute_rows, seed)
        permuted_array = compose(permute, partial(np.array, dtype=int64_dtype))

        shape = (8, 8)

        # Shuffle the input rows to verify that we correctly pick out the top
        # values independently of order.
        factor_data = permute(np.arange(0, 64, dtype=dtype).reshape(shape))
        classifier_data = permuted_array(
            [
                [0, 0, 1, 1, 2, 2, 0, 0],
                [0, 0, 1, 1, 2, 2, 0, 0],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        f = self.factors_by_dtype_name[dtype]
        c = self.c

        self.check_terms(
            terms={
                "top1": f.top(1, groupby=c),
                "top2": f.top(2, groupby=c),
                "top3": f.top(3, groupby=c),
                "bottom1": f.bottom(1, groupby=c),
                "bottom2": f.bottom(2, groupby=c),
                "bottom3": f.bottom(3, groupby=c),
            },
            initial_workspace={
                f: factor_data,
                c: classifier_data,
            },
            expected={
                # Should be the rightmost location of each entry in
                # classifier_data.
                "top1": permuted_array(
                    [
                        [0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ],
                    dtype=bool,
                ),
                # Should be the leftmost location of each entry in
                # classifier_data.
                "bottom1": permuted_array(
                    [
                        [1, 0, 1, 0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                # Should be the first and second-rightmost location of each
                # entry in classifier_data.
                "top2": permuted_array(
                    [
                        [0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                    ],
                    dtype=bool,
                ),
                # Should be the first and second leftmost location of each
                # entry in classifier_data.
                "bottom2": permuted_array(
                    [
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                # Should be the first, second, and third-rightmost location of
                # each entry in classifier_data.
                "top3": permuted_array(
                    [
                        [0, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 1, 1],
                    ],
                    dtype=bool,
                ),
                # Should be the first, second, and third-leftmost location of
                # each entry in classifier_data.
                "bottom3": permuted_array(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 1, 0, 1, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
            },
            mask=self.build_mask(self.ones_mask(shape=shape)),
        )

    @parameter_space(
        dtype=("float64", "datetime64[ns]"),
        seed=(1, 2, 3),
        __fail_fast=True,
    )
    def test_top_and_bottom_with_groupby_and_mask(self, dtype, seed):
        permute = partial(permute_rows, seed)
        permuted_array = compose(permute, partial(np.array, dtype=int64_dtype))

        shape = (8, 8)

        # Shuffle the input rows to verify that we correctly pick out the top
        # values independently of order.
        factor_data = permute(np.arange(0, 64, dtype=dtype).reshape(shape))
        classifier_data = permuted_array(
            [
                [0, 0, 1, 1, 2, 2, 0, 0],
                [0, 0, 1, 1, 2, 2, 0, 0],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        f = self.factors_by_dtype_name[dtype]
        c = self.c

        self.check_terms(
            terms={
                "top2": f.top(2, groupby=c),
                "bottom2": f.bottom(2, groupby=c),
            },
            initial_workspace={
                f: factor_data,
                c: classifier_data,
            },
            expected={
                # Should be the rightmost two entries in classifier_data,
                # ignoring the off-diagonal.
                "top2": permuted_array(
                    [
                        [0, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1],
                        [0, 1, 1, 0, 0, 0, 1, 1],
                        [0, 1, 0, 1, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                    ],
                    dtype=bool,
                ),
                # Should be the rightmost two entries in classifier_data,
                # ignoring the off-diagonal.
                "bottom2": permuted_array(
                    [
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
            },
            mask=self.build_mask(permute(np.rot90(self.eye_mask(shape=shape)))),
        )


class SidFactor(CustomFactor):
    """A factor that just returns each asset's sid."""

    inputs = ()
    window_length = 1

    def compute(self, today, sids, out):
        out[:] = sids


class SpecificAssetsTestCase(WithSeededRandomPipelineEngine, ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = tuple(range(10))
    ASSET_FINDER_COUNTRY_CODE = "US"
    SEEDED_RANDOM_PIPELINE_DEFAULT_DOMAIN = US_EQUITIES

    def _check_filters(self, evens, odds, first_five, last_three):
        pipe = Pipeline(
            columns={
                "sid": SidFactor(),
                "evens": evens,
                "odds": odds,
                "first_five": first_five,
                "last_three": last_three,
            },
        )

        start, end = self.trading_days[[-10, -1]]
        results = self.run_pipeline(pipe, start, end).unstack()

        sids = results.sid.astype(int64_dtype)

        assert_equal(results.evens, ~(sids % 2).astype(bool))
        assert_equal(results.odds, (sids % 2).astype(bool))
        assert_equal(results.first_five, sids < 5)
        assert_equal(results.last_three, sids >= 7)

    def test_specific_assets(self):
        assets = self.asset_finder.retrieve_all(self.ASSET_FINDER_EQUITY_SIDS)

        self._check_filters(
            evens=StaticAssets(assets[::2]),
            odds=StaticAssets(assets[1::2]),
            first_five=StaticAssets(assets[:5]),
            last_three=StaticAssets(assets[-3:]),
        )

    def test_specific_sids(self):
        sids = self.ASSET_FINDER_EQUITY_SIDS

        self._check_filters(
            evens=StaticSids(sids[::2]),
            odds=StaticSids(sids[1::2]),
            first_five=StaticSids(sids[:5]),
            last_three=StaticSids(sids[-3:]),
        )


class TestPostProcessAndToWorkSpaceValue:
    def test_reversability(self):
        class F(Filter):
            inputs = ()
            window_length = 0
            missing_value = False

        f = F()
        column_data = np.array(
            [[True, f.missing_value], [True, f.missing_value], [True, True]],
            dtype=bool,
        )

        assert_equal(f.postprocess(column_data.ravel()), column_data.ravel())

        # only include the non-missing data
        pipeline_output = pd.Series(
            data=True,
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
        )

        assert_equal(
            f.to_workspace_value(pipeline_output, pd.Index([0, 1])),
            column_data,
        )


class TestRepr:
    def test_maximum_repr(self):
        m = SomeFactor().top(1, groupby=SomeClassifier(), mask=SomeFilter())

        rep = repr(m)
        assert_equal(
            rep,
            "Maximum({}, groupby={}, mask={})".format(
                SomeFactor().recursive_repr(),
                SomeClassifier().recursive_repr(),
                SomeFilter().recursive_repr(),
            ),
        )

        short_rep = m.graph_repr()
        assert_equal(
            short_rep,
            "Maximum:\\l  "
            "groupby: SomeClassifier(...)\\l  "
            "mask: SomeFilter(...)\\l",
        )


class IfElseTestCase(BaseUSEquityPipelineTestCase, ZiplineTestCase):
    @classmethod
    def init_class_fixtures(cls):
        super(IfElseTestCase, cls).init_class_fixtures()
        cls.assets = cls.asset_finder.retrieve_all(
            cls.asset_finder.equities_sids,
        )

    @parameter_space(seed=[1, 2, 3])
    def test_if_then_else_factor(self, seed):
        f = SomeFactor()
        g = SomeOtherFactor()
        cond = SomeFilter()

        f_data = self.randn_data(seed=seed)
        g_data = self.randn_data(seed=seed + 1)
        cond_data = self.rand_mask(seed=seed + 2)

        workspace = {
            f: f_data,
            g: g_data,
            cond: cond_data,
        }
        terms = {
            "result": cond.if_else(f, g),
            "result_1d": cond.if_else(f, g[self.assets[0]]),
        }
        expected = {
            "result": np.where(cond_data, f_data, g_data),
            "result_1d": np.where(cond_data, f_data, g_data[:, [0]]),
        }

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace=workspace,
            mask=self.build_mask(self.ones_mask()),
        )

    @parameter_space(seed=[1000, 2000, 3000])
    def test_if_then_else_datetime_factor(self, seed):
        class SomeOtherDatetimeFactor(Factor):
            dtype = datetime64ns_dtype
            inputs = ()
            window_length = 0

        f = SomeDatetimeFactor()
        g = SomeOtherDatetimeFactor()
        cond = SomeFilter()

        f_data = self.randn_data(seed=seed)
        g_data = self.randn_data(seed=seed + 1)
        cond_data = self.rand_mask(seed=seed + 2)

        workspace = {
            f: f_data,
            g: g_data,
            cond: cond_data,
        }
        terms = {
            "result": cond.if_else(f, g),
            "result_1d": cond.if_else(f, g[self.assets[5]]),
        }
        expected = {
            "result": np.where(cond_data, f_data, g_data),
            "result_1d": np.where(cond_data, f_data, g_data[:, [5]]),
        }

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace=workspace,
            mask=self.build_mask(self.ones_mask()),
        )

    @parameter_space(seed=[10, 11, 12])
    def test_if_then_else_filter(self, seed):
        class Filter1(Filter):
            inputs = ()
            window_length = 0

        class Filter2(Filter):
            inputs = ()
            window_length = 0

        f = Filter1()
        g = Filter2()
        cond = SomeFilter()

        f_data = self.rand_mask(seed=seed)
        g_data = self.rand_mask(seed=seed + 1)
        cond_data = self.rand_mask(seed=seed + 2)

        workspace = {
            f: f_data,
            g: g_data,
            cond: cond_data,
        }
        terms = {
            "result": cond.if_else(f, g),
            "result_1d": cond.if_else(f, g[self.assets[1]]),
        }
        expected = {
            "result": np.where(cond_data, f_data, g_data),
            "result_1d": np.where(cond_data, f_data, g_data[:, [1]]),
        }

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace=workspace,
            mask=self.build_mask(self.ones_mask()),
        )

    @parameter_space(seed=[100, 101, 102])
    def test_if_then_else_string_classifier(self, seed):
        class Classifier1(Classifier):
            inputs = ()
            window_length = 0
            dtype = object

        class Classifier2(Classifier):
            inputs = ()
            window_length = 0
            dtype = object

        f = Classifier1()
        g = Classifier2()
        cond = SomeFilter()

        f_data = self.rand_categoricals(seed=seed, categories=["a", "b", "c"])
        g_data = self.rand_categoricals(
            seed=seed + 1,
            categories=["d", "e", "f"],
        )
        cond_data = self.rand_mask(seed=seed + 2)

        workspace = {
            f: f_data,
            g: g_data,
            cond: cond_data,
        }

        terms = {
            "result": cond.if_else(f, g),
            "result_1d": cond.if_else(f, g[self.assets[2]]),
        }
        expected = {
            "result": labelarray_where(cond_data, f_data, g_data),
            "result_1d": labelarray_where(cond_data, f_data, g_data[:, [2]]),
        }

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace=workspace,
            mask=self.build_mask(self.ones_mask()),
        )

    @parameter_space(seed=[200, 300, 400])
    def test_if_then_else_int_classifier(self, seed):
        class Classifier1(Classifier):
            inputs = ()
            window_length = 0
            dtype = int64_dtype
            missing_value = -1

        class Classifier2(Classifier):
            inputs = ()
            window_length = 0
            dtype = int64_dtype
            missing_value = -1

        f = Classifier1()
        g = Classifier2()
        cond = SomeFilter()

        f_data = self.rand_ints(seed=seed)
        g_data = self.rand_ints(seed=seed + 1)
        cond_data = self.rand_mask(seed=seed + 2)

        workspace = {
            f: f_data,
            g: g_data,
            cond: cond_data,
        }

        terms = {
            "result": cond.if_else(f, g),
            "result_1d": cond.if_else(f, g[self.assets[4]]),
        }
        expected = {
            "result": np.where(cond_data, f_data, g_data),
            "result_1d": np.where(cond_data, f_data, g_data[:, [4]]),
        }

        self.check_terms(
            terms=terms,
            expected=expected,
            initial_workspace=workspace,
            mask=self.build_mask(self.ones_mask()),
        )
