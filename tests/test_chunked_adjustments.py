"""
Tests for chunked adjustments.
"""
from itertools import (
    chain,
    izip_longest,
)
from unittest import TestCase

from nose_parameterized import parameterized
from numpy import (
    arange,
    float64,
)
from numpy.testing import assert_array_equal

from zipline.data.chunkerator import make_window_iterator


def _gen_float64_no_adjustments():

    data = arange(30, dtype=float).reshape(6, 5)
    for windowlen in range(2, 5):
        yield (
            "foobar",
            data,
            windowlen,
            [],
            # For an array with N rows, and a window length of M,
            # there are (N - M) + 1 legal windows.
            # Example:
            # If my array has 3 rows, and I want windows of length 2, there are
            # 2 legal windows: the first two rows, or the second two rows.
            [
                data[offset:offset + windowlen]
                for offset in (6 - windowlen) + 1
            ],
        )


class ChunkedDataTestCase(TestCase):

    @parameterized.expand(chain(
        _gen_float64_no_adjustments()
    ))
    def test_chunked_iterators(self,
                               name,
                               data,
                               lookback,
                               adjustments,
                               expected):
        pass
        # iterator = make_window_iterator(data, lookback, adjustments)
        # for yielded, expected_yield in izip_longest(iterator, expected):
        #     assert_array_equal(yielded, expected_yield)
