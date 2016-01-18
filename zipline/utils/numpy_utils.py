"""
Utilities for working with numpy arrays.
"""
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
float64_dtype = dtype('float64')
datetime64D_dtype = dtype('datetime64[D]')
datetime64ns_dtype = dtype('datetime64[ns]')

make_datetime64ns = flip(datetime64, 'ns')
make_datetime64D = flip(datetime64, 'D')
np_NaT = make_datetime64ns('NaT')

_FILLVALUE_DEFAULTS = {
    float64_dtype: nan,
    datetime64ns_dtype: np_NaT,
}


def default_fillvalue_for_dtype(dtype):
    """
    Get the default fill value for `dtype`.
    """
    return _FILLVALUE_DEFAULTS[dtype]


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
        out = empty(broadcast(begindates, enddates).shape, dtype=float)

    beginmask = (begindates == np_NaT)
    endmask = (enddates == np_NaT)

    out = busday_count(
        # Temporarily fill in non-NaT values.
        where(beginmask, _notNaT, begindates),
        where(endmask, _notNaT, enddates),
        out=out,
    )

    # Fill in entries where either comparison was NaT with nan in the output.
    out[beginmask | endmask] = nan
    return out
