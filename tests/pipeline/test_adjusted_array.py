"""
Tests for chunked adjustments.
"""
from collections import namedtuple
from itertools import chain, product, zip_longest
from string import ascii_lowercase, ascii_uppercase
from textwrap import dedent

import numpy as np
from toolz import curry

from zipline.errors import WindowLengthNotPositive, WindowLengthTooLong
from zipline.lib.adjustment import (
    Boolean1DArrayOverwrite,
    BooleanOverwrite,
    Datetime641DArrayOverwrite,
    Datetime64Overwrite,
    Float641DArrayOverwrite,
    Float64Multiply,
    Float64Overwrite,
    Int64Overwrite,
    Object1DArrayOverwrite,
    ObjectOverwrite,
)
from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.labelarray import LabelArray
from zipline.testing import check_arrays
from zipline.testing.predicates import assert_equal
from zipline.utils.compat import unicode
from zipline.utils.numpy_utils import (
    coerce_to_dtype,
    datetime64ns_dtype,
    default_missing_value_for_dtype,
    bool_dtype,
    float64_dtype,
    int64_dtype,
    object_dtype,
)
import pytest


def moving_window(array, nrows):
    """
    Simple moving window generator over a 2D numpy array.
    """
    count = num_windows_of_length_M_on_buffers_of_length_N(nrows, len(array))
    for i in range(count):
        yield array[i : i + nrows]


def num_windows_of_length_M_on_buffers_of_length_N(M, N):
    """
    For a window of length M rolling over a buffer of length N,
    there are (N - M) + 1 legal windows.

    Example:
    If my array has N=4 rows, and I want windows of length M=2, there are
    3 legal windows: data[0:2], data[1:3], and data[2:4].
    """
    return N - M + 1


def valid_window_lengths(underlying_buffer_length):
    """
    An iterator of all legal window lengths on a buffer of a given length.

    Returns values from 1 to underlying_buffer_length.
    """
    return iter(range(1, underlying_buffer_length + 1))


@curry
def as_dtype(dtype, data):
    """
    Curried wrapper around array.astype for when you have the dtype before you
    have the data.
    """
    return np.asarray(data).astype(dtype)


@curry
def as_labelarray(initial_dtype, missing_value, array):
    """
    Curried wrapper around LabelArray, that round-trips the input data through
    `initial_dtype` first.
    """
    return LabelArray(
        array.astype(initial_dtype),
        missing_value=initial_dtype.type(missing_value),
    )


bytes_dtype = np.dtype("S3")
unicode_dtype = np.dtype("U3")

AdjustmentCase = namedtuple(
    "AdjustmentCase",
    [
        "name",
        "baseline",
        "window_length",
        "adjustments",
        "missing_value",
        "perspective_offset",
        "expected_result",
    ],
)


def _gen_unadjusted_cases(name, make_input, make_expected_output, missing_value):
    nrows = 6
    ncols = 3

    raw_data = np.arange(nrows * ncols).reshape(nrows, ncols)
    input_array = make_input(raw_data)
    expected_output_array = make_expected_output(raw_data)

    for windowlen in valid_window_lengths(nrows):
        num_legal_windows = num_windows_of_length_M_on_buffers_of_length_N(
            windowlen, nrows
        )

        yield AdjustmentCase(
            name="%s_length_%d" % (name, windowlen),
            baseline=input_array,
            window_length=windowlen,
            adjustments={},
            missing_value=missing_value,
            perspective_offset=0,
            expected_result=[
                expected_output_array[offset : offset + windowlen]
                for offset in range(num_legal_windows)
            ],
        )


def _gen_multiplicative_adjustment_cases(dtype):
    """
    Generate expected moving windows on a buffer with adjustments.

    We proceed by constructing, at each row, the view of the array we expect in
    in all windows anchored on that row.

    In general, if we have an adjustment to be applied once we process the row
    at index N, should see that adjustment applied to the underlying buffer for
    any window containing the row at index N.

    We then build all legal windows over these buffers.
    """
    adjustment_type = {
        float64_dtype: Float64Multiply,
    }[dtype]

    nrows, ncols = 6, 3
    adjustments = {}
    buffer_as_of = [None] * 6
    baseline = buffer_as_of[0] = np.full((nrows, ncols), 1, dtype=dtype)

    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(0, 0, 0, 0, coerce_to_dtype(dtype, 2)),
    ]
    buffer_as_of[1] = np.array(
        [[2, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=dtype
    )

    # No adjustment at index 2.
    buffer_as_of[2] = buffer_as_of[1]

    adjustments[3] = [
        adjustment_type(1, 2, 1, 1, coerce_to_dtype(dtype, 3)),
        adjustment_type(0, 1, 0, 0, coerce_to_dtype(dtype, 4)),
    ]
    buffer_as_of[3] = np.array(
        [[8, 1, 1], [4, 3, 1], [1, 3, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=dtype
    )

    adjustments[4] = [adjustment_type(0, 3, 2, 2, coerce_to_dtype(dtype, 5))]
    buffer_as_of[4] = np.array(
        [[8, 1, 5], [4, 3, 5], [1, 3, 5], [1, 1, 5], [1, 1, 1], [1, 1, 1]], dtype=dtype
    )

    adjustments[5] = [
        adjustment_type(0, 4, 1, 1, coerce_to_dtype(dtype, 6)),
        adjustment_type(2, 2, 2, 2, coerce_to_dtype(dtype, 7)),
    ]
    buffer_as_of[5] = np.array(
        [[8, 6, 5], [4, 18, 5], [1, 18, 35], [1, 6, 5], [1, 6, 1], [1, 1, 1]],
        dtype=dtype,
    )

    return _gen_expectations(
        baseline,
        default_missing_value_for_dtype(dtype),
        adjustments,
        buffer_as_of,
        nrows,
        perspective_offsets=(0, 1),
    )


def _gen_overwrite_adjustment_cases(dtype):
    """
    Generate test cases for overwrite adjustments.

    The algorithm used here is the same as the one used above for
    multiplicative adjustments.  The only difference is the semantics of how
    the adjustments are expected to modify the arrays.

    This is parameterized on `make_input` and `make_expected_output` functions,
    which take 2-D lists of values and transform them into desired input/output
    arrays. We do this so that we can easily test both vanilla numpy ndarrays
    and our own LabelArray class for strings.
    """
    adjustment_type = {
        float64_dtype: Float64Overwrite,
        datetime64ns_dtype: Datetime64Overwrite,
        int64_dtype: Int64Overwrite,
        bytes_dtype: ObjectOverwrite,
        unicode_dtype: ObjectOverwrite,
        object_dtype: ObjectOverwrite,
        bool_dtype: BooleanOverwrite,
    }[dtype]
    make_expected_dtype = as_dtype(dtype)
    missing_value = default_missing_value_for_dtype(datetime64ns_dtype)

    if dtype == object_dtype:
        # When we're testing object dtypes, we expect to have strings, but
        # coerce_to_dtype(object, 3) just gives 3 as a Python integer.
        def make_overwrite_value(dtype, value):
            return str(value)

    else:
        make_overwrite_value = coerce_to_dtype

    adjustments = {}
    buffer_as_of = [None] * 6
    baseline = make_expected_dtype(
        [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    buffer_as_of[0] = make_expected_dtype(
        [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(0, 0, 0, 0, make_overwrite_value(dtype, 1)),
    ]
    buffer_as_of[1] = make_expected_dtype(
        [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    # No adjustment at index 2.
    buffer_as_of[2] = buffer_as_of[1]

    adjustments[3] = [
        adjustment_type(1, 2, 1, 1, make_overwrite_value(dtype, 3)),
        adjustment_type(0, 1, 0, 0, make_overwrite_value(dtype, 4)),
    ]
    buffer_as_of[3] = make_expected_dtype(
        [[4, 2, 2], [4, 3, 2], [2, 3, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    adjustments[4] = [adjustment_type(0, 3, 2, 2, make_overwrite_value(dtype, 5))]
    buffer_as_of[4] = make_expected_dtype(
        [[4, 2, 5], [4, 3, 5], [2, 3, 5], [2, 2, 5], [2, 2, 2], [2, 2, 2]]
    )

    adjustments[5] = [
        adjustment_type(0, 4, 1, 1, make_overwrite_value(dtype, 6)),
        adjustment_type(2, 2, 2, 2, make_overwrite_value(dtype, 7)),
    ]
    buffer_as_of[5] = make_expected_dtype(
        [[4, 6, 5], [4, 6, 5], [2, 6, 7], [2, 6, 5], [2, 6, 2], [2, 2, 2]]
    )

    return _gen_expectations(
        baseline,
        missing_value,
        adjustments,
        buffer_as_of,
        nrows=6,
        perspective_offsets=(0, 1),
    )


def _gen_overwrite_1d_array_adjustment_case(dtype):
    """
    Generate test cases for overwrite adjustments.

    The algorithm used here is the same as the one used above for
    multiplicative adjustments.  The only difference is the semantics of how
    the adjustments are expected to modify the arrays.

    This is parameterized on `make_input` and `make_expected_output` functions,
    which take 1-D lists of values and transform them into desired input/output
    arrays. We do this so that we can easily test both vanilla numpy ndarrays
    and our own LabelArray class for strings.
    """
    adjustment_type = {
        bool_dtype: Boolean1DArrayOverwrite,
        float64_dtype: Float641DArrayOverwrite,
        datetime64ns_dtype: Datetime641DArrayOverwrite,
    }[dtype]
    make_expected_dtype = as_dtype(dtype)
    missing_value = default_missing_value_for_dtype(datetime64ns_dtype)

    adjustments = {}
    buffer_as_of = [None] * 6
    baseline = make_expected_dtype(
        [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    buffer_as_of[0] = make_expected_dtype(
        [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    vals1 = [1]
    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(
            0, 0, 0, 0, np.array([coerce_to_dtype(dtype, val) for val in vals1])
        )
    ]
    buffer_as_of[1] = make_expected_dtype(
        [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    # No adjustment at index 2.
    buffer_as_of[2] = buffer_as_of[1]

    vals3 = [4, 4, 1]
    adjustments[3] = [
        adjustment_type(
            0, 2, 0, 0, np.array([coerce_to_dtype(dtype, val) for val in vals3])
        )
    ]
    buffer_as_of[3] = make_expected_dtype(
        [[4, 2, 2], [4, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )

    vals4 = [5] * 4
    adjustments[4] = [
        adjustment_type(
            0, 3, 2, 2, np.array([coerce_to_dtype(dtype, val) for val in vals4])
        )
    ]
    buffer_as_of[4] = make_expected_dtype(
        [[4, 2, 5], [4, 2, 5], [1, 2, 5], [2, 2, 5], [2, 2, 2], [2, 2, 2]]
    )

    vals5 = range(1, 6)
    adjustments[5] = [
        adjustment_type(
            0, 4, 1, 1, np.array([coerce_to_dtype(dtype, val) for val in vals5])
        ),
    ]
    buffer_as_of[5] = make_expected_dtype(
        [[4, 1, 5], [4, 2, 5], [1, 3, 5], [2, 4, 5], [2, 5, 2], [2, 2, 2]]
    )
    return _gen_expectations(
        baseline,
        missing_value,
        adjustments,
        buffer_as_of,
        nrows=6,
        perspective_offsets=(0, 1),
    )


def _gen_expectations(
    baseline, missing_value, adjustments, buffer_as_of, nrows, perspective_offsets
):
    for windowlen, perspective_offset in product(
        valid_window_lengths(nrows), perspective_offsets
    ):
        # How long is an iterator of length-N windows on this buffer?
        # For example, for a window of length 3 on a buffer of length 6, there
        # are four valid windows.
        num_legal_windows = num_windows_of_length_M_on_buffers_of_length_N(
            windowlen, nrows
        )

        # Build the sequence of regions in the underlying buffer we expect to
        # see. For example, with a window length of 3 on a buffer of length 6,
        # we expect to see:
        #  (buffer[0:3], buffer[1:4], buffer[2:5], buffer[3:6])
        #
        slices = [slice(i, i + windowlen) for i in range(num_legal_windows)]

        # The sequence of perspectives we expect to take on the underlying
        # data. For example, with a window length of 3 and a perspective offset
        # of 1, we expect to see:
        #  (buffer_as_of[3], buffer_as_of[4], buffer_as_of[5], buffer_as_of[5])
        #
        initial_perspective = windowlen + perspective_offset - 1
        perspectives = range(
            initial_perspective, initial_perspective + num_legal_windows
        )

        def as_of(p):
            # perspective_offset can push us past the end of the underlying
            # buffer/adjustments. When it does, we should always see the latest
            # version of the buffer.
            if p >= len(buffer_as_of):
                return buffer_as_of[-1]
            return buffer_as_of[p]

        expected_iterator_results = [
            as_of(perspective)[slice_]
            for slice_, perspective in zip(slices, perspectives)
        ]

        test_name = "dtype_{}_length_{}_perpective_offset_{}".format(
            baseline.dtype,
            windowlen,
            perspective_offset,
        )

        yield AdjustmentCase(
            name=test_name,
            baseline=baseline,
            window_length=windowlen,
            adjustments=adjustments,
            missing_value=missing_value,
            perspective_offset=perspective_offset,
            expected_result=expected_iterator_results,
        )


class TestAdjustedArray:
    def test_traverse_invalidating(self):
        data = np.arange(5 * 3, dtype="f8").reshape(5, 3)
        original_data = data.copy()
        adjustments = {2: [Float64Multiply(0, 4, 0, 2, 2.0)]}
        adjusted_array = AdjustedArray(data, adjustments, float("nan"))

        for _ in adjusted_array.traverse(1, copy=False):
            pass

        assert_equal(data, original_data * 2)

        err_msg = "cannot traverse invalidated AdjustedArray"
        with pytest.raises(ValueError, match=err_msg):
            adjusted_array.traverse(1)

    def test_copy(self):
        data = np.arange(5 * 3, dtype="f8").reshape(5, 3)
        original_data = data.copy()
        adjustments = {2: [Float64Multiply(0, 4, 0, 2, 2.0)]}
        adjusted_array = AdjustedArray(data, adjustments, float("nan"))
        traverse_copy = adjusted_array.copy()
        clean_copy = adjusted_array.copy()

        a_it = adjusted_array.traverse(2, copy=False)
        b_it = traverse_copy.traverse(2, copy=False)
        for a, b in zip(a_it, b_it):
            assert_equal(a, b)

        err_msg = "cannot copy invalidated AdjustedArray"
        with pytest.raises(ValueError, match=err_msg):
            adjusted_array.copy()

        # the clean copy should have the original data even though the
        # original adjusted array has it's data mutated in place
        assert_equal(clean_copy.data, original_data)
        assert_equal(adjusted_array.data, original_data * 2)

    @pytest.mark.parametrize(
        """name, data, lookback, adjustments, missing_value,\
            perspective_offset, expected_output""",
        chain(
            _gen_unadjusted_cases(
                "float",
                make_input=as_dtype(float64_dtype),
                make_expected_output=as_dtype(float64_dtype),
                missing_value=default_missing_value_for_dtype(float64_dtype),
            ),
            _gen_unadjusted_cases(
                "datetime",
                make_input=as_dtype(datetime64ns_dtype),
                make_expected_output=as_dtype(datetime64ns_dtype),
                missing_value=default_missing_value_for_dtype(datetime64ns_dtype),
            ),
            # Test passing an array of strings to AdjustedArray.
            _gen_unadjusted_cases(
                "bytes_ndarray",
                make_input=as_dtype(bytes_dtype),
                make_expected_output=as_labelarray(bytes_dtype, b""),
                missing_value=b"",
            ),
            _gen_unadjusted_cases(
                "unicode_ndarray",
                make_input=as_dtype(unicode_dtype),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value="",
            ),
            _gen_unadjusted_cases(
                "object_ndarray",
                make_input=lambda a: a.astype(unicode).astype(object),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value="",
            ),
            # Test passing a LabelArray directly to AdjustedArray.
            _gen_unadjusted_cases(
                "bytes_labelarray",
                make_input=as_labelarray(bytes_dtype, b""),
                make_expected_output=as_labelarray(bytes_dtype, b""),
                missing_value=b"",
            ),
            _gen_unadjusted_cases(
                "unicode_labelarray",
                make_input=as_labelarray(unicode_dtype, None),
                make_expected_output=as_labelarray(unicode_dtype, None),
                missing_value="",
            ),
            _gen_unadjusted_cases(
                "object_labelarray",
                make_input=(lambda a: LabelArray(a.astype(unicode).astype(object), "")),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value="",
            ),
        ),
    )
    def test_no_adjustments(
        self,
        name,
        data,
        lookback,
        adjustments,
        missing_value,
        perspective_offset,
        expected_output,
    ):

        array = AdjustedArray(data, adjustments, missing_value)
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            in_out = zip(array.traverse(lookback), expected_output)
            for yielded, expected_yield in in_out:
                check_arrays(yielded, expected_yield)

    @pytest.mark.parametrize(
        "name, data, lookback, adjustments, missing_value,\
        perspective_offset, expected",
        _gen_multiplicative_adjustment_cases(float64_dtype),
    )
    def test_multiplicative_adjustments(
        self,
        name,
        data,
        lookback,
        adjustments,
        missing_value,
        perspective_offset,
        expected,
    ):

        array = AdjustedArray(data, adjustments, missing_value)
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(
                lookback,
                perspective_offset=perspective_offset,
            )
            for yielded, expected_yield in zip_longest(window_iter, expected):
                check_arrays(yielded, expected_yield)

    @pytest.mark.parametrize(
        "name, baseline, lookback, adjustments,\
        missing_value, perspective_offset, expected",
        chain(
            _gen_overwrite_adjustment_cases(bool_dtype),
            _gen_overwrite_adjustment_cases(int64_dtype),
            _gen_overwrite_adjustment_cases(float64_dtype),
            _gen_overwrite_adjustment_cases(datetime64ns_dtype),
            _gen_overwrite_1d_array_adjustment_case(float64_dtype),
            _gen_overwrite_1d_array_adjustment_case(datetime64ns_dtype),
            _gen_overwrite_1d_array_adjustment_case(bool_dtype),
            # There are six cases here:
            # Using np.bytes/np.unicode/object arrays as inputs.
            # Passing np.bytes/np.unicode/object arrays to LabelArray,
            # and using those as input.
            #
            # The outputs should always be LabelArrays.
            _gen_unadjusted_cases(
                "bytes_ndarray",
                make_input=as_dtype(bytes_dtype),
                make_expected_output=as_labelarray(bytes_dtype, b""),
                missing_value=b"",
            ),
            _gen_unadjusted_cases(
                "unicode_ndarray",
                make_input=as_dtype(unicode_dtype),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value="",
            ),
            _gen_unadjusted_cases(
                "object_ndarray",
                make_input=lambda a: a.astype(unicode).astype(object),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value="",
            ),
            _gen_unadjusted_cases(
                "bytes_labelarray",
                make_input=as_labelarray(bytes_dtype, b""),
                make_expected_output=as_labelarray(bytes_dtype, b""),
                missing_value=b"",
            ),
            _gen_unadjusted_cases(
                "unicode_labelarray",
                make_input=as_labelarray(unicode_dtype, ""),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value="",
            ),
            _gen_unadjusted_cases(
                "object_labelarray",
                make_input=(
                    lambda a: LabelArray(
                        a.astype(unicode).astype(object),
                        None,
                    )
                ),
                make_expected_output=as_labelarray(unicode_dtype, ""),
                missing_value=None,
            ),
        ),
    )
    def test_overwrite_adjustment_cases(
        self,
        name,
        baseline,
        lookback,
        adjustments,
        missing_value,
        perspective_offset,
        expected,
    ):
        array = AdjustedArray(baseline, adjustments, missing_value)

        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(
                lookback,
                perspective_offset=perspective_offset,
            )
            for yielded, expected_yield in zip_longest(window_iter, expected):
                check_arrays(yielded, expected_yield)

    def test_object1darrayoverwrite(self):
        pairs = [u + l for u, l in product(ascii_uppercase, ascii_lowercase)]
        categories = pairs + ["~" + c for c in pairs]
        baseline = LabelArray(
            np.array([["".join((r, c)) for c in "abc"] for r in ascii_uppercase]),
            None,
            categories,
        )
        full_expected = baseline.copy()

        def flip(cs):
            if cs is None:
                return None
            if cs[0] != "~":
                return "~" + cs
            return cs

        def make_overwrite(fr, lr, fc, lc):
            fr, lr, fc, lc = map(ord, (fr, lr, fc, lc))
            fr -= ord("A")
            lr -= ord("A")
            fc -= ord("a")
            lc -= ord("a")

            return Object1DArrayOverwrite(
                fr,
                lr,
                fc,
                lc,
                baseline[fr : lr + 1, fc].map(flip),
            )

        overwrites = {
            3: [make_overwrite("A", "B", "a", "a")],
            4: [make_overwrite("A", "C", "b", "c")],
            5: [make_overwrite("D", "D", "a", "b")],
        }

        it = AdjustedArray(baseline, overwrites, None).traverse(3)

        window = next(it)
        expected = full_expected[:3]
        check_arrays(window, expected)

        window = next(it)
        full_expected[0:2, 0] = LabelArray(["~Aa", "~Ba"], None)
        expected = full_expected[1:4]
        check_arrays(window, expected)

        window = next(it)
        full_expected[0:3, 1:3] = LabelArray(
            [["~Ab", "~Ac"], ["~Bb", "~Bc"], ["~Cb", "~Cb"]], None
        )
        expected = full_expected[2:5]
        check_arrays(window, expected)

        window = next(it)
        full_expected[3, :2] = "~Da"
        expected = full_expected[3:6]
        check_arrays(window, expected)

    def test_invalid_lookback(self):

        data = np.arange(30, dtype=float).reshape(6, 5)
        adj_array = AdjustedArray(data, {}, float("nan"))

        with pytest.raises(WindowLengthTooLong):
            adj_array.traverse(7)

        with pytest.raises(WindowLengthNotPositive):
            adj_array.traverse(0)

        with pytest.raises(WindowLengthNotPositive):
            adj_array.traverse(-1)

    def test_array_views_arent_writable(self):

        data = np.arange(30, dtype=float).reshape(6, 5)
        adj_array = AdjustedArray(data, {}, float("nan"))

        for frame in adj_array.traverse(3):
            with pytest.raises(ValueError):
                frame[0, 0] = 5.0

    def test_inspect(self):
        data = np.arange(15, dtype=float).reshape(5, 3)
        adj_array = AdjustedArray(
            data,
            {4: [Float64Multiply(2, 3, 0, 0, 4.0)]},
            float("nan"),
        )
        # TODO: CHECK WHY DO I NEED TO FIX THE INDENT IN THE EXPECTED?
        expected = dedent(
            """\
            Adjusted Array (float64):

            Data:
            array([[ 0.,  1.,  2.],
                   [ 3.,  4.,  5.],
                   [ 6.,  7.,  8.],
                   [ 9., 10., 11.],
                   [12., 13., 14.]])

            Adjustments:
            {4: [Float64Multiply(first_row=2, last_row=3, first_col=0, \
last_col=0, value=4.000000)]}
            """
        )
        got = adj_array.inspect()
        assert expected == got

    def test_update_labels(self):
        data = np.array(
            [
                ["aaa", "bbb", "ccc"],
                ["ddd", "eee", "fff"],
                ["ggg", "hhh", "iii"],
                ["jjj", "kkk", "lll"],
                ["mmm", "nnn", "ooo"],
            ]
        )
        label_array = LabelArray(data, missing_value="")

        adj_array = AdjustedArray(
            data=label_array,
            adjustments={4: [ObjectOverwrite(2, 3, 0, 0, "ppp")]},
            missing_value="",
        )

        expected_data = np.array(
            [
                ["aaa-foo", "bbb-foo", "ccc-foo"],
                ["ddd-foo", "eee-foo", "fff-foo"],
                ["ggg-foo", "hhh-foo", "iii-foo"],
                ["jjj-foo", "kkk-foo", "lll-foo"],
                ["mmm-foo", "nnn-foo", "ooo-foo"],
            ]
        )
        expected_label_array = LabelArray(expected_data, missing_value="")

        expected_adj_array = AdjustedArray(
            data=expected_label_array,
            adjustments={4: [ObjectOverwrite(2, 3, 0, 0, "ppp-foo")]},
            missing_value="",
        )

        adj_array.update_labels(lambda x: x + "-foo")

        # Check that the mapped AdjustedArray has the expected baseline
        # values and adjustment values.
        check_arrays(adj_array.data, expected_adj_array.data)
        assert adj_array.adjustments == expected_adj_array.adjustments

    A = Float64Multiply(0, 4, 1, 1, 0.5)
    B = Float64Overwrite(3, 3, 4, 4, 4.2)
    C = Float64Multiply(0, 2, 0, 0, 0.14)
    D = Float64Overwrite(0, 3, 0, 0, 4.0)
    E = Float64Overwrite(0, 0, 1, 1, 3.7)
    F = Float64Multiply(0, 4, 3, 3, 10.0)
    G = Float64Overwrite(5, 5, 4, 4, 1.7)
    H = Float64Multiply(0, 4, 2, 2, 0.99)
    S = Float64Multiply(0, 1, 4, 4, 5.06)

    @pytest.mark.parametrize(
        "initial_adjustments, adjustments_to_add,\
        expected_adjustments_with_append, expected_adjustments_with_prepend",
        [
            (
                # Initial adjustments
                {
                    1: [A, B],
                    2: [C],
                    4: [D],
                },
                # Adjustments to add
                {
                    1: [E],
                    2: [F, G],
                    3: [H, S],
                },
                # Expected adjustments with 'append'
                {
                    1: [A, B, E],
                    2: [C, F, G],
                    3: [H, S],
                    4: [D],
                },
                # Expected adjustments with 'prepend'
                {
                    1: [E, A, B],
                    2: [F, G, C],
                    3: [H, S],
                    4: [D],
                },
            )
        ],
    )
    def test_update_adjustments(
        self,
        initial_adjustments,
        adjustments_to_add,
        expected_adjustments_with_append,
        expected_adjustments_with_prepend,
    ):
        methods = ["append", "prepend"]
        expected_outputs = [
            expected_adjustments_with_append,
            expected_adjustments_with_prepend,
        ]

        for method, expected_output in zip(methods, expected_outputs):
            data = np.arange(30, dtype=float).reshape(6, 5)
            adjusted_array = AdjustedArray(data, initial_adjustments, float("nan"))

            adjusted_array.update_adjustments(adjustments_to_add, method)
            assert adjusted_array.adjustments == expected_output
