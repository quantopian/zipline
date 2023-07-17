""" Utilities for working with numpy arrays."""
from collections import OrderedDict
from datetime import datetime
from warnings import catch_warnings, filterwarnings

import numpy as np
from numpy.lib.stride_tricks import as_strided
from packaging.version import Version
from toolz import flip

numpy_version = Version(np.__version__)

uint8_dtype = np.dtype("uint8")
bool_dtype = np.dtype("bool")

uint32_dtype = np.dtype("uint32")
uint64_dtype = np.dtype("uint64")
int64_dtype = np.dtype("int64")

float32_dtype = np.dtype("float32")
float64_dtype = np.dtype("float64")

complex128_dtype = np.dtype("complex128")

datetime64D_dtype = np.dtype("datetime64[D]")
datetime64ns_dtype = np.dtype("datetime64[ns]")

object_dtype = np.dtype("O")
# We use object arrays for strings.
categorical_dtype = object_dtype

make_datetime64ns = flip(np.datetime64, "ns")
make_datetime64D = flip(np.datetime64, "D")

# Array compare that works across versions of numpy
try:
    assert_array_compare = np.testing.utils.assert_array_compare
except AttributeError:
    assert_array_compare = np.testing.assert_array_compare

NaTmap = {
    np.dtype(f"datetime64[{unit}]"): np.datetime64("NaT", unit)
    for unit in ("ns", "us", "ms", "s", "m", "D")
}


def NaT_for_dtype(dtype):
    """Retrieve NaT with the same units as ``dtype``.

    Parameters
    ----------
    dtype : dtype-coercable
        The dtype to lookup the NaT value for.

    Returns
    -------
    NaT : dtype
        The NaT value for the given dtype.
    """
    return NaTmap[np.dtype(dtype)]


NaTns = NaT_for_dtype(datetime64ns_dtype)
NaTD = NaT_for_dtype(datetime64D_dtype)

_FILLVALUE_DEFAULTS = {
    bool_dtype: False,
    float32_dtype: np.nan,
    float64_dtype: np.nan,
    datetime64ns_dtype: NaTns,
    object_dtype: None,
}

INT_DTYPES_BY_SIZE_BYTES = OrderedDict(
    [
        (1, np.dtype("int8")),
        (2, np.dtype("int16")),
        (4, np.dtype("int32")),
        (8, np.dtype("int64")),
    ]
)

UNSIGNED_INT_DTYPES_BY_SIZE_BYTES = OrderedDict(
    [
        (1, np.dtype("uint8")),
        (2, np.dtype("uint16")),
        (4, np.dtype("uint32")),
        (8, np.dtype("uint64")),
    ]
)


def int_dtype_with_size_in_bytes(size):
    try:
        return INT_DTYPES_BY_SIZE_BYTES[size]
    except KeyError as exc:
        raise ValueError("No integral dtype whose size is %d bytes." % size) from exc


def unsigned_int_dtype_with_size_in_bytes(size):
    try:
        return UNSIGNED_INT_DTYPES_BY_SIZE_BYTES[size]
    except KeyError as exc:
        raise ValueError(
            "No unsigned integral dtype whose size is %d bytes." % size
        ) from exc


class NoDefaultMissingValue(Exception):
    pass


def make_kind_check(python_types, numpy_kind):
    """
    Make a function that checks whether a scalar or array is of a given kind
    (e.g. float, int, datetime, timedelta).
    """

    def check(value):
        if hasattr(value, "dtype"):
            return value.dtype.kind == numpy_kind
        return isinstance(value, python_types)

    return check


is_float = make_kind_check(float, "f")
is_int = make_kind_check(int, "i")
is_datetime = make_kind_check(datetime, "M")
is_object = make_kind_check(object, "O")


def coerce_to_dtype(dtype, value):
    """
    Make a value with the specified numpy dtype.

    Only datetime64[ns] and datetime64[D] are supported for datetime dtypes.
    """
    name = dtype.name
    if name.startswith("datetime64"):
        if name == "datetime64[ns]":
            return make_datetime64ns(value)
        else:
            raise TypeError(f"Don't know how to coerce values of dtype {dtype}")
    return dtype.type(value)


def default_missing_value_for_dtype(dtype):
    """
    Get the default fill value for `dtype`.
    """
    try:
        return _FILLVALUE_DEFAULTS[dtype]
    except KeyError as exc:
        raise NoDefaultMissingValue(
            "No default value registered for dtype %s." % dtype
        ) from exc


def repeat_first_axis(array, count):
    """
    Restride `array` to repeat `count` times along the first axis.

    Parameters
    ----------
    array : np.array
        The array to restride.
    count : int
        Number of times to repeat `array`.

    Returns
    -------
    result : array
        Array of shape (count,) + array.shape, composed of `array` repeated
        `count` times along the first axis.

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(3); a
    array([0, 1, 2])
    >>> repeat_first_axis(a, 2)
    array([[0, 1, 2],
           [0, 1, 2]])
    >>> repeat_first_axis(a, 4)
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]])

    Notes
    ----
    The resulting array will share memory with `array`.  If you need to assign
    to the input or output, you should probably make a copy first.

    See Also
    --------
    repeat_last_axis
    """
    return as_strided(array, (count,) + array.shape, (0,) + array.strides)


def repeat_last_axis(array, count):
    """
    Restride `array` to repeat `count` times along the last axis.

    Parameters
    ----------
    array : np.array
        The array to restride.
    count : int
        Number of times to repeat `array`.

    Returns
    -------
    result : array
        Array of shape array.shape + (count,) composed of `array` repeated
        `count` times along the last axis.

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(3); a
    array([0, 1, 2])
    >>> repeat_last_axis(a, 2)
    array([[0, 0],
           [1, 1],
           [2, 2]])
    >>> repeat_last_axis(a, 4)
    array([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [2, 2, 2, 2]])

    Notes
    ----
    The resulting array will share memory with `array`.  If you need to assign
    to the input or output, you should probably make a copy first.

    See Also
    --------
    repeat_last_axis
    """
    return as_strided(array, array.shape + (count,), array.strides + (0,))


def rolling_window(array, length):
    """
    Restride an array of shape

        (X_0, ... X_N)

    into an array of shape

        (length, X_0 - length + 1, ... X_N)

    where each slice at index i along the first axis is equivalent to

        result[i] = array[length * i:length * (i + 1)]

    Parameters
    ----------
    array : np.ndarray
        The base array.
    length : int
        Length of the synthetic first axis to generate.

    Returns
    -------
    out : np.ndarray

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(25).reshape(5, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    >>> rolling_window(a, 2)
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    <BLANKLINE>
           [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]]])
    """
    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] <= length:
        raise IndexError(
            "Can't restride array of shape {shape} with"
            " a window length of {len}".format(
                shape=orig_shape,
                len=length,
            )
        )

    num_windows = orig_shape[0] - length + 1
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    return as_strided(array, new_shape, new_strides)


# Sentinel value that isn't NaT.
_notNaT = make_datetime64D(0)
iNaT = int(NaTns.view(int64_dtype))
assert iNaT == NaTD.view(int64_dtype), "iNaTns != iNaTD"


def isnat(obj):
    """
    Check if a value is np.NaT.
    """
    if obj.dtype.kind not in ("m", "M"):
        raise ValueError("%s is not a numpy datetime or timedelta")
    return obj.view(int64_dtype) == iNaT


def is_missing(data, missing_value):
    """
    Generic is_missing function that handles NaN and NaT.
    """
    if is_float(data) and np.isnan(missing_value):
        return np.isnan(data)
    elif is_datetime(data) and isnat(missing_value):
        return isnat(data)
    elif is_object(data) and missing_value is None:
        # XXX: Older versions of numpy returns True/False for array ==
        # None. Work around this by boxing None in a 1x1 array, which causes
        # numpy to do the broadcasted comparison we want.
        return data == np.array([missing_value])
    return data == missing_value


def same(x, y):
    """
    Check if two scalar values are "the same".

    Returns True if `x == y`, or if x and y are both NaN or both NaT.
    """
    if is_float(x) and np.isnan(x) and is_float(y) and np.isnan(y):
        return True
    elif is_datetime(x) and isnat(x) and is_datetime(y) and isnat(y):
        return True
    else:
        return x == y


def busday_count_mask_NaT(begindates, enddates, out=None):
    """
    Simple of numpy.busday_count that returns `float` arrays rather than int
    arrays, and handles `NaT`s by returning `NaN`s where the inputs were `NaT`.

    Doesn't support custom weekdays or calendars, but probably should in the
    future.

    See Also
    --------
    np.busday_count
    """
    if out is None:
        out = np.empty(np.broadcast(begindates, enddates).shape, dtype=float)

    beginmask = isnat(begindates)
    endmask = isnat(enddates)

    out = np.busday_count(
        # Temporarily fill in non-NaT values.
        np.where(beginmask, _notNaT, begindates),
        np.where(endmask, _notNaT, enddates),
        out=out,
    )

    # Fill in entries where either comparison was NaT with nan in the output.
    out[beginmask | endmask] = np.nan
    return out


class WarningContext:
    """
    Re-usable contextmanager for contextually managing warnings.
    """

    def __init__(self, *warning_specs):
        self._warning_specs = warning_specs
        self._catchers = []

    def __enter__(self):
        catcher = catch_warnings()
        catcher.__enter__()
        self._catchers.append(catcher)
        for args, kwargs in self._warning_specs:
            filterwarnings(*args, **kwargs)
        return self

    def __exit__(self, *exc_info):
        catcher = self._catchers.pop()
        return catcher.__exit__(*exc_info)


def ignore_nanwarnings():
    """
    Helper for building a WarningContext that ignores warnings from numpy's
    nanfunctions.
    """
    return WarningContext(
        (
            ("ignore",),
            {"category": RuntimeWarning, "module": "numpy.lib.nanfunctions"},
        )
    )


def vectorized_is_element(array, choices):
    """
    Check if each element of ``array`` is in choices.

    Parameters
    ----------
    array : np.ndarray
    choices : object
        Object implementing __contains__.

    Returns
    -------
    was_element : np.ndarray[bool]
        Array indicating whether each element of ``array`` was in ``choices``.
    """
    return np.vectorize(choices.__contains__, otypes=[bool])(array)


def as_column(a):
    """
    Convert an array of shape (N,) into an array of shape (N, 1).

    This is equivalent to `a[:, np.newaxis]`.

    Parameters
    ----------
    a : np.ndarray

    Example
    -------
    >>> import numpy as np
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> as_column(a)
    array([[0],
           [1],
           [2],
           [3],
           [4]])
    >>> as_column(a).shape
    (5, 1)
    """
    if a.ndim != 1:
        raise ValueError(
            "as_column expected an 1-dimensional array, "
            "but got an array of shape %s" % (a.shape,)
        )
    return a[:, None]


def changed_locations(a, include_first):
    """
    Compute indices of values in ``a`` that differ from the previous value.

    Parameters
    ----------
    a : np.ndarray
        The array on which to indices of change.
    include_first : bool
        Whether or not to consider the first index of the array as "changed".

    Example
    -------
    >>> import numpy as np
    >>> changed_locations(np.array([0, 0, 5, 5, 1, 1]), include_first=False)
    array([2, 4])

    >>> changed_locations(np.array([0, 0, 5, 5, 1, 1]), include_first=True)
    array([0, 2, 4])
    """
    if a.ndim > 1:
        raise ValueError("indices_of_changed_values only supports 1D arrays.")
    indices = np.flatnonzero(np.diff(a)) + 1

    if not include_first:
        return indices

    return np.hstack([[0], indices])


def compare_datetime_arrays(x, y):
    """
    Compare datetime64 ndarrays, treating NaT values as equal.
    """

    return np.array_equal(x.view("int64"), y.view("int64"))


def bytes_array_to_native_str_object_array(a):
    """Convert an array of dtype S to an object array containing `str`."""

    return a.astype(str).astype(object)
