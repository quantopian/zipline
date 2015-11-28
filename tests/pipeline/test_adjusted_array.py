"""
Tests for chunked adjustments.
"""
from itertools import chain
from textwrap import dedent
from unittest import TestCase

from nose_parameterized import parameterized
from numpy import (
    arange,
    array,
    full,
)
from numpy.testing import assert_array_equal
from six.moves import zip_longest

from zipline.errors import WindowLengthNotPositive, WindowLengthTooLong
from zipline.lib.adjustment import (
    Datetime64Overwrite,
    Float64Multiply,
    Float64Overwrite,
)
from zipline.lib.adjusted_array import AdjustedArray, NOMASK
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    float64_dtype,
    make_datetime64ns,
)


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


def value_with_dtype(dtype, value):
    """
    Make a value with the specified numpy dtype.
    """
    name = dtype.name
    if name.startswith('datetime64'):
        if name != 'datetime64[ns]':
            raise TypeError("Expected datetime64[ns], but got %s." % name)
        return make_datetime64ns(value)
    return dtype.type(value)


def _gen_unadjusted_cases(dtype):

    nrows = 6
    ncols = 3
    data = arange(nrows * ncols).astype(dtype).reshape(nrows, ncols)

    for windowlen in valid_window_lengths(nrows):

        num_legal_windows = num_windows_of_length_M_on_buffers_of_length_N(
            windowlen, nrows
        )

        yield (
            "dtype_%s_length_%d" % (dtype, windowlen),
            data,
            windowlen,
            {},
            [
                data[offset:offset + windowlen]
                for offset in range(num_legal_windows)
            ],
        )


def _gen_multiplicative_adjustment_cases(dtype):
    """
    Generate expected moving windows on a buffer with adjustments.

    We proceed by constructing, at each row, the view of the array we expect in
    in all windows anchored on or after that row.

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
    baseline = buffer_as_of[0] = full((nrows, ncols), 1, dtype=dtype)

    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(0, 0, 0, 0, value_with_dtype(dtype, 2)),
    ]
    buffer_as_of[1] = array([[2, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=dtype)

    # No adjustment at index 2.
    buffer_as_of[2] = buffer_as_of[1]

    adjustments[3] = [
        adjustment_type(1, 2, 1, 1, value_with_dtype(dtype, 3)),
        adjustment_type(0, 1, 0, 0, value_with_dtype(dtype, 4)),
    ]
    buffer_as_of[3] = array([[8, 1, 1],
                             [4, 3, 1],
                             [1, 3, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=dtype)

    adjustments[4] = [
        adjustment_type(0, 3, 2, 2, value_with_dtype(dtype, 5))
    ]
    buffer_as_of[4] = array([[8, 1, 5],
                             [4, 3, 5],
                             [1, 3, 5],
                             [1, 1, 5],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=dtype)

    adjustments[5] = [
        adjustment_type(0, 4, 1, 1, value_with_dtype(dtype, 6)),
        adjustment_type(2, 2, 2, 2, value_with_dtype(dtype, 7)),
    ]
    buffer_as_of[5] = array([[8,  6,  5],
                             [4, 18,  5],
                             [1, 18, 35],
                             [1,  6,  5],
                             [1,  6,  1],
                             [1,  1,  1]], dtype=dtype)

    return _gen_expectations(baseline, adjustments, buffer_as_of, nrows)


def _gen_overwrite_adjustment_cases(dtype):
    """
    Generate test cases for overwrite adjustments.

    The algorithm used here is the same as the one used above for
    multiplicative adjustments.  The only difference is the semantics of how
    the adjustments are expected to modify the arrays.
    """
    adjustment_type = {
        float64_dtype: Float64Overwrite,
        datetime64ns_dtype: Datetime64Overwrite,
    }[dtype]

    nrows, ncols = 6, 3
    adjustments = {}
    buffer_as_of = [None] * 6
    baseline = buffer_as_of[0] = full((nrows, ncols), 2, dtype=dtype)

    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(0, 0, 0, 0, value_with_dtype(dtype, 1)),
    ]
    buffer_as_of[1] = array([[1, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2]], dtype=dtype)

    # No adjustment at index 2.
    buffer_as_of[2] = buffer_as_of[1]

    adjustments[3] = [
        adjustment_type(1, 2, 1, 1, value_with_dtype(dtype, 3)),
        adjustment_type(0, 1, 0, 0, value_with_dtype(dtype, 4)),
    ]
    buffer_as_of[3] = array([[4, 2, 2],
                             [4, 3, 2],
                             [2, 3, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2]], dtype=dtype)

    adjustments[4] = [
        adjustment_type(0, 3, 2, 2, value_with_dtype(dtype, 5))
    ]
    buffer_as_of[4] = array([[4, 2, 5],
                             [4, 3, 5],
                             [2, 3, 5],
                             [2, 2, 5],
                             [2, 2, 2],
                             [2, 2, 2]], dtype=dtype)

    adjustments[5] = [
        adjustment_type(0, 4, 1, 1, value_with_dtype(dtype, 6)),
        adjustment_type(2, 2, 2, 2, value_with_dtype(dtype, 7)),
    ]
    buffer_as_of[5] = array([[4,  6,  5],
                             [4,  6,  5],
                             [2,  6,  7],
                             [2,  6,  5],
                             [2,  6,  2],
                             [2,  2,  2]], dtype=dtype)

    return _gen_expectations(
        baseline,
        adjustments,
        buffer_as_of,
        nrows,
    )


def _gen_expectations(baseline, adjustments, buffer_as_of, nrows):

    for windowlen in valid_window_lengths(nrows):

        num_legal_windows = num_windows_of_length_M_on_buffers_of_length_N(
            windowlen, nrows
        )

        yield (
            "dtype_%s_length_%d" % (baseline.dtype, windowlen),
            baseline,
            windowlen,
            adjustments,
            [
                # This is a nasty expression...
                #
                # Reading from right to left: we want a slice of length
                # 'windowlen', starting at 'offset', from the buffer on which
                # we've applied all adjustments corresponding to the last row
                # of the data, which will be (offset + windowlen - 1).
                buffer_as_of[offset + windowlen - 1][offset:offset + windowlen]
                for offset in range(num_legal_windows)
            ],
        )


class AdjustedArrayTestCase(TestCase):

    @parameterized.expand(
        chain(
            _gen_unadjusted_cases(float64_dtype),
            _gen_unadjusted_cases(datetime64ns_dtype),
        )
    )
    def test_no_adjustments(self,
                            name,
                            data,
                            lookback,
                            adjustments,
                            expected):

        array = AdjustedArray(data, NOMASK, adjustments)
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(lookback)
            for yielded, expected_yield in zip_longest(window_iter, expected):
                self.assertEqual(yielded.dtype, data.dtype)
                assert_array_equal(yielded, expected_yield)

    @parameterized.expand(_gen_multiplicative_adjustment_cases(float64_dtype))
    def test_multiplicative_adjustments(self,
                                        name,
                                        data,
                                        lookback,
                                        adjustments,
                                        expected):

        array = AdjustedArray(data, NOMASK, adjustments)
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(lookback)
            for yielded, expected_yield in zip_longest(window_iter, expected):
                assert_array_equal(yielded, expected_yield)

    @parameterized.expand(
        chain(
            _gen_overwrite_adjustment_cases(float64_dtype),
            _gen_overwrite_adjustment_cases(datetime64ns_dtype),
        )
    )
    def test_overwrite_adjustment_cases(self,
                                        name,
                                        data,
                                        lookback,
                                        adjustments,
                                        expected):
        array = AdjustedArray(data, NOMASK, adjustments)
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(lookback)
            for yielded, expected_yield in zip_longest(window_iter, expected):
                self.assertEqual(yielded.dtype, data.dtype)
                assert_array_equal(yielded, expected_yield)

    def test_invalid_lookback(self):

        data = arange(30, dtype=float).reshape(6, 5)
        adj_array = AdjustedArray(data, NOMASK, {})

        with self.assertRaises(WindowLengthTooLong):
            adj_array.traverse(7)

        with self.assertRaises(WindowLengthNotPositive):
            adj_array.traverse(0)

        with self.assertRaises(WindowLengthNotPositive):
            adj_array.traverse(-1)

    def test_array_views_arent_writable(self):

        data = arange(30, dtype=float).reshape(6, 5)
        adj_array = AdjustedArray(data, NOMASK, {})

        for frame in adj_array.traverse(3):
            with self.assertRaises(ValueError):
                frame[0, 0] = 5.0

    def test_bad_input(self):
        msg = "Mask shape \(2, 3\) != data shape \(5, 5\)"
        data = arange(25).reshape(5, 5)
        bad_mask = array([[0, 1, 1], [0, 0, 1]], dtype=bool)

        with self.assertRaisesRegexp(ValueError, msg):
            AdjustedArray(data, bad_mask, {})

    def test_inspect(self):
        data = arange(15, dtype=float).reshape(5, 3)
        adj_array = AdjustedArray(
            data,
            NOMASK,
            {4: [Float64Multiply(2, 3, 0, 0, 4.0)]},
        )

        expected = dedent(
            """\
            Adjusted Array (float64):

            Data:
            array([[  0.,   1.,   2.],
                   [  3.,   4.,   5.],
                   [  6.,   7.,   8.],
                   [  9.,  10.,  11.],
                   [ 12.,  13.,  14.]])

            Adjustments:
            {4: [Float64Multiply(first_row=2, last_row=3, first_col=0, \
last_col=0, value=4.000000)]}
            """
        )
        got = adj_array.inspect()
        self.assertEqual(expected, got)
