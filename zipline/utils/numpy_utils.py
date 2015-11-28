"""
Utilities for working with numpy arrays.
"""
from numpy import dtype, datetime64
from numpy.lib.stride_tricks import as_strided

float64_dtype = dtype('float64')
datetime64ns_dtype = dtype('datetime64[ns]')


def make_datetime64ns(value):
    """
    Wrapper around the datetime64 constructor that uses 'ns' as a unit.
    """
    # This can't just be a partial because datetime64 is a C function and
    # doesn't accept kwargs.
    return datetime64(value, 'ns')


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
