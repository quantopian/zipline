"""
Tests for chunked adjustments.
"""
from itertools import (
    izip_longest,
)
from unittest import TestCase

from nose_parameterized import parameterized
from numpy import (
    arange,
    array,
    full,
)
from numpy.testing import assert_array_equal

from zipline.data.adjustment import (
    Float64Multiply,
    Float64Overwrite,
)
from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)
from zipline.errors import (
    WindowLengthNotPositive,
    WindowLengthTooLong,
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


def _gen_unadjusted_cases(dtype):

    nrows = 6
    ncols = 3
    data = arange(nrows * ncols, dtype=dtype).reshape(nrows, ncols)

    for windowlen in valid_window_lengths(nrows):

        num_legal_windows = num_windows_of_length_M_on_buffers_of_length_N(
            windowlen, nrows
        )

        yield (
            "length_%d" % windowlen,
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
        float: Float64Multiply,
    }[dtype]

    nrows, ncols = 6, 3
    adjustments = {}
    buffer_as_of = [None] * 6
    baseline = buffer_as_of[0] = full((nrows, ncols), 1, dtype=dtype)

    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(0, 0, 0, dtype(2)),
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
        adjustment_type(1, 2, 1, dtype(3)),
        adjustment_type(0, 1, 0, dtype(4)),
    ]
    buffer_as_of[3] = array([[8, 1, 1],
                             [4, 3, 1],
                             [1, 3, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=dtype)

    adjustments[4] = [
        adjustment_type(0, 3, 2, dtype(5))
    ]
    buffer_as_of[4] = array([[8, 1, 5],
                             [4, 3, 5],
                             [1, 3, 5],
                             [1, 1, 5],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=dtype)

    adjustments[5] = [
        adjustment_type(0, 4, 1, dtype(6)),
        adjustment_type(2, 2, 2, dtype(7)),
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
        float: Float64Overwrite,
    }[dtype]

    nrows, ncols = 6, 3
    adjustments = {}
    buffer_as_of = [None] * 6
    baseline = buffer_as_of[0] = full((nrows, ncols), 2, dtype=dtype)

    # Note that row indices are inclusive!
    adjustments[1] = [
        adjustment_type(0, 0, 0, dtype(1)),
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
        adjustment_type(1, 2, 1, dtype(3)),
        adjustment_type(0, 1, 0, dtype(4)),
    ]
    buffer_as_of[3] = array([[4, 2, 2],
                             [4, 3, 2],
                             [2, 3, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2]], dtype=dtype)

    adjustments[4] = [
        adjustment_type(0, 3, 2, dtype(5))
    ]
    buffer_as_of[4] = array([[4, 2, 5],
                             [4, 3, 5],
                             [2, 3, 5],
                             [2, 2, 5],
                             [2, 2, 2],
                             [2, 2, 2]], dtype=dtype)

    adjustments[5] = [
        adjustment_type(0, 4, 1, dtype(6)),
        adjustment_type(2, 2, 2, dtype(7)),
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
            "length_%d" % windowlen,
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

    @parameterized.expand(_gen_unadjusted_cases(float))
    def test_no_adjustments(self,
                            name,
                            data,
                            lookback,
                            adjustments,
                            expected):
        array = adjusted_array(
            data,
            NOMASK,
            adjustments,
        )
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(lookback)
            for yielded, expected_yield in izip_longest(window_iter, expected):
                assert_array_equal(yielded, expected_yield)

    @parameterized.expand(_gen_multiplicative_adjustment_cases(float))
    def test_multiplicative_adjustments(self,
                                        name,
                                        data,
                                        lookback,
                                        adjustments,
                                        expected):
        array = adjusted_array(
            data,
            NOMASK,
            adjustments,
        )
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(lookback)
            for yielded, expected_yield in izip_longest(window_iter, expected):
                assert_array_equal(yielded, expected_yield)

    @parameterized.expand(_gen_overwrite_adjustment_cases(float))
    def test_overwrite_adjustment_cases(self,
                                        name,
                                        data,
                                        lookback,
                                        adjustments,
                                        expected):
        array = adjusted_array(
            data,
            NOMASK,
            adjustments,
        )
        for _ in range(2):  # Iterate 2x ensure adjusted_arrays are re-usable.
            window_iter = array.traverse(lookback)
            for yielded, expected_yield in izip_longest(window_iter, expected):
                assert_array_equal(yielded, expected_yield)

    def test_invalid_lookback(self):

        data = arange(30, dtype=float).reshape(6, 5)
        adj_array = adjusted_array(data, NOMASK, {})

        with self.assertRaises(WindowLengthTooLong):
            adj_array.traverse(7)

        with self.assertRaises(WindowLengthNotPositive):
            adj_array.traverse(0)

        with self.assertRaises(WindowLengthNotPositive):
            adj_array.traverse(-1)

    def test_array_views_arent_writable(self):

        data = arange(30, dtype=float).reshape(6, 5)
        adj_array = adjusted_array(data, NOMASK, {})

        for frame in adj_array.traverse(3):
            with self.assertRaises(ValueError):
                frame[0, 0] = 5.0
