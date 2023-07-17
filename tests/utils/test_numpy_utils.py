"""
Tests for zipline.utils.numpy_utils.
"""
from datetime import datetime
import pytest

import numpy as np
from pandas import Timestamp
from toolz import concat, keyfilter
from toolz import curry

from zipline.utils.functional import mapall as lazy_mapall
from zipline.utils.numpy_utils import (
    bytes_array_to_native_str_object_array,
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
    return np.array([value], dtype=dtype)


CASES = {
    (int, is_int): mapall(
        (int, np.int16, np.int32, np.int64, make_array(int)), [0, 1, -1]
    ),
    (float, is_float): mapall(
        (np.float16, np.float32, np.float64, float, make_array(float)),
        [0.0, 1.0, -1.0, float("nan"), float("inf"), -float("inf")],
    ),
    (datetime, is_datetime): mapall(
        (
            make_datetime64D,
            make_datetime64ns,
            Timestamp,
            make_array("datetime64[ns]"),
        ),
        [0, 1, 2],
    )
    + [NaTD, NaTns],
}


def everything_but(k, d):
    """
    Return iterator of all values in d except the values in k.
    """
    assert k in d
    return concat(keyfilter(lambda x: x != k, d).values())


# TypeCheckTestCase
fixt = [(k, x) for k, v in CASES.items() for x in v]
not_fixt = [(k, x) for k in CASES.keys() for x in everything_but(k, CASES)]


@pytest.mark.parametrize(
    "data_type, value",
    fixt,
    ids=[f"{type(x[1])} {x[1]}" for x in fixt],
)
def test_check_data_type_is_true(data_type, value):
    is_data_type = data_type[1]
    assert is_data_type(value)


@pytest.mark.parametrize(
    "data_type, value",
    not_fixt,
    ids=[f"{str(k[0])} is not {x}" for k, x in not_fixt],
)
def test_check_is_not_data_type(data_type, value):
    is_data_type = data_type[1]
    assert not is_data_type(value)


def test_bytes_array_to_native_str_object_array():
    a = np.array([b"abc", b"def"], dtype="S3")
    result = bytes_array_to_native_str_object_array(a)
    expected = np.array(["abc", "def"], dtype=object)
    np.testing.assert_array_equal(result, expected)
