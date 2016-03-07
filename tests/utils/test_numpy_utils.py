"""
Tests for zipline.utils.numpy_utils.
"""
from datetime import datetime
from six import itervalues
from unittest import TestCase

from numpy import (
    array,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
)
from pandas import Timestamp
from toolz import concat, keyfilter
from toolz import curry
from toolz.curried.operator import ne

from zipline.utils.functional import mapall as lazy_mapall
from zipline.utils.numpy_utils import (
    is_float,
    is_int,
    is_datetime,
    make_datetime64D,
    make_datetime64ns,
    NaTns,
    NaTD,
)


def mapall(*args):
    "Strict version of mapall."
    return list(lazy_mapall(*args))


@curry
def make_array(dtype, value):
    return array([value], dtype=dtype)


CASES = {
    int: mapall(
        (int, int16, int32, int64, make_array(int)),
        [0, 1, -1]
    ),
    float: mapall(
        (float16, float32, float64, float, make_array(float)),
        [0., 1., -1., float('nan'), float('inf'), -float('inf')],
    ),
    datetime: mapall(
        (
            make_datetime64D,
            make_datetime64ns,
            Timestamp,
            make_array('datetime64[ns]'),
        ),
        [0, 1, 2],
    ) + [NaTD, NaTns],
}


def everything_but(k, d):
    """
    Return iterator of all values in d except the values in k.
    """
    assert k in d
    return concat(itervalues(keyfilter(ne(k), d)))


class TypeCheckTestCase(TestCase):

    def test_is_float(self):
        for good_value in CASES[float]:
            self.assertTrue(is_float(good_value))

        for bad_value in everything_but(float, CASES):
            self.assertFalse(is_float(bad_value))

    def test_is_int(self):
        for good_value in CASES[int]:
            self.assertTrue(is_int(good_value))

        for bad_value in everything_but(int, CASES):
            self.assertFalse(is_int(bad_value))

    def test_is_datetime(self):
        for good_value in CASES[datetime]:
            self.assertTrue(is_datetime(good_value))

        for bad_value in everything_but(datetime, CASES):
            self.assertFalse(is_datetime(bad_value))
