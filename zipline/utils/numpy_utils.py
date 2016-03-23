"""
Utilities for working with numpy arrays.
"""
from datetime import datetime
from warnings import (
    catch_warnings,
    filterwarnings,
)

from numpy import (
    broadcast,
    busday_count,
    datetime64,
    dtype,
    empty,
    nan,
    where
)
from numpy.lib.stride_tricks import as_strided
from toolz import flip

uint8_dtype = dtype('uint8')
bool_dtype = dtype('bool')

int64_dtype = dtype('int64')

float32_dtype = dtype('float32')
float64_dtype = dtype('float64')

complex128_dtype = dtype('complex128')

datetime64D_dtype = dtype('datetime64[D]')
datetime64ns_dtype = dtype('datetime64[ns]')

make_datetime64ns = flip(datetime64, 'ns')
make_datetime64D = flip(datetime64, 'D')

NaTmap = {
    dtype('datetime64[%s]' % unit): datetime64('NaT', unit)
    for unit in ('ns', 'us', 'ms', 's', 'm', 'D')
}
NaT_for_dtype = NaTmap.__getitem__
NaTns = NaT_for_dtype(datetime64ns_dtype)
NaTD = NaT_for_dtype(datetime64D_dtype)


_FILLVALUE_DEFAULTS = {
    bool_dtype: False,
    float32_dtype: nan,
    float64_dtype: nan,
    datetime64ns_dtype: NaTns,
}


class NoDefaultMissingValue(Exception):
    pass


def make_kind_check(python_types, numpy_kind):
    """
    Make a function that checks whether a scalar or array is of a given kind
    (e.g. float, int, datetime, timedelta).
    """
    def check(value):
        if hasattr(value, 'dtype'):
            return value.dtype.kind == numpy_kind
        return isinstance(value, python_types)
    return check


is_float = make_kind_check(float, 'f')
is_int = make_kind_check(int, 'i')
is_datetime = make_kind_check(datetime, 'M')


def coerce_to_dtype(dtype, value):
    """
    Make a value with the specified numpy dtype.

    Only datetime64[ns] and datetime64[D] are supported for datetime dtypes.
    """
    name = dtype.name
    if name.startswith('datetime64'):
        if name == 'datetime64[D]':
            return make_datetime64D(value)
        elif name == 'datetime64[ns]':
            return make_datetime64ns(value)
        else:
            raise TypeError(
                "Don't know how to coerce values of dtype %s" % dtype
            )
    return dtype.type(value)


def default_missing_value_for_dtype(dtype):
    """
    Get the default fill value for `dtype`.
    """
    try:
        return _FILLVALUE_DEFAULTS[dtype]
    except KeyError:
        raise NoDefaultMissingValue(
            "No default value registered for dtype %s." % dtype
        )


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


# Sentinel value that isn't NaT.
_notNaT = make_datetime64D(0)


def busday_count_mask_NaT(begindates,
                          enddates,
                          out=None):
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
        out = empty(broadcast(begindates, enddates).shape, dtype=float)

    beginmask = (begindates == NaTD)
    endmask = (enddates == NaTD)

    out = busday_count(
        # Temporarily fill in non-NaT values.
        where(beginmask, _notNaT, begindates),
        where(endmask, _notNaT, enddates),
        out=out,
    )

    # Fill in entries where either comparison was NaT with nan in the output.
    out[beginmask | endmask] = nan
    return out


class WarningContext(object):
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
            ('ignore',),
            {'category': RuntimeWarning, 'module': 'numpy.lib.nanfunctions'},
        )
    )
