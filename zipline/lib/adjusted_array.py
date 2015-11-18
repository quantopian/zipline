from numpy import bool_, datetime64, nan, ndarray
from zipline.errors import (
    WindowLengthNotPositive,
    WindowLengthTooLong,
)
from zipline.utils.memoize import lazyval
from zipline.utils.sentinel import sentinel

# These class names are all the same because of our bootleg templating system.
from ._floatwindow import AdjustedArrayWindow as FloatWindow
from ._datewindow import AdjustedArrayWindow as DateWindow
from ._boolwindow import AdjustedArrayWindow as BoolWindow

Infer = sentinel(
    'Infer',
    "Sentinel used to say 'infer missing_value from data type.'"
)
NOMASK = None
NUMERIC_DTYPES = frozenset(
    ['float32', 'float64', 'int32', 'int64', 'uint32', 'uint64'],
)
WINDOW_TYPES = {
    'float64': FloatWindow,
    'float32': FloatWindow,  # Cast float32 up to float64.
    'int64': FloatWindow,
    'datetime64': DateWindow,
    'bool': BoolWindow,
}
_FILLVALUE_DEFAULTS = {
    'float64': nan,
    'datetime64': datetime64('NaT'),
}


def default_fillvalue_for_dtype(dtype):
    """
    Get the default fill value for dtype `type_`.
    """
    return _FILLVALUE_DEFAULTS[dtype.name]


def _normalize_numeric(data):
    """
    Coerce numeric data into float64 so that we can represent missing values.

    If the input is of integral type or is a float32, we return the data after
    converting to float64.  Otherwise we return the data unchanged.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    coerced : ndarray
    """
    if data.dtype.name in NUMERIC_DTYPES:
        return data.astype('float64')
    return data


class AdjustedArray(object):
    """
    An array that can be iterated with a variable-length window, and which can
    provide different views on data from different perspectives.
    """
    __slots__ = ('_data', '_mask', '_fillvalue', '_adjustments', '__weakref__')

    def __init__(self, data, mask, adjustments, fillvalue=Infer):
        self._data = _normalize_numeric(data)
        self._adjustments = adjustments
        if fillvalue is Infer:
            fillvalue = default_fillvalue_for_dtype(self.data.dtype)
        self._fillvalue = fillvalue

        if mask is not NOMASK:
            if mask.dtype != bool_:
                raise ValueError("Mask must be a bool array.")
            if data.shape != mask.shape:
                raise ValueError(
                    "Mask shape %s != data shape %s." %
                    (mask.shape, data.shape),
                )
            self._mask = mask
            self._data[~self._mask] = self._fillvalue
        self._data.setflags(write=False)

    @lazyval
    def data(self):
        """
        The data stored by this Array.
        """
        return self._data

    @lazyval
    def dtype(self):
        """
        The dtype of this array.
        """
        return self._data.dtype

    @lazyval
    def _iterator_type(self):
        """
        The iterator type to produce when `traverse` is called on this Array.
        """
        return WINDOW_TYPES[self.dtype.name]

    def traverse(self, window_length, offset=0):
        """
        Produce an iterator rolling windows rows over our data.
        Each emitted window will have `window_length` rows.

        Parameters
        ----------
        window_length : int
            The number of rows in each emitted window.
        offset : int, optional
            Number of rows to skip before the first window.
        """
        data = self._data.copy()
        _check_window_params(data, window_length)
        return self._iterator_type(
            data,
            # Subtract offset from adjustment indices so that they're aligned
            # with the buffer we actually pass in.
            self._adjustments,
            offset,
            window_length,
        )


def ensure_ndarray(ndarray_or_adjusted_array):
    """
    Return the input as a numpy ndarray.

    This is a no-op if the input is already an ndarray.  If the input is an
    adjusted_array, this extracts a read-only view of its internal data buffer.

    Parameters
    ----------
    ndarray_or_adjusted_array : numpy.ndarray | zipline.data.adjusted_array

    Returns
    -------
    out : The input, converted to an ndarray.
    """
    if isinstance(ndarray_or_adjusted_array, ndarray):
        return ndarray_or_adjusted_array
    elif isinstance(ndarray_or_adjusted_array, AdjustedArray):
        return ndarray_or_adjusted_array.data
    else:
        raise TypeError(
            "Can't convert %s to ndarray" %
            type(ndarray_or_adjusted_array).__name__
        )


def _check_window_params(data, window_length):
    """
    Check that a window of length `window_length` is well-defined on `data`.

    Parameters
    ----------
    data : np.ndarray[ndim=2]
        The array of data to check.
    window_length : int
        Length of the desired window.

    Returns
    -------
    None

    Raises
    ------
    WindowLengthNotPositive
        If window_length < 1.
    WindowLengthTooLong
        If window_length is greater than the number of rows in `data`.
    """
    if window_length < 1:
        raise WindowLengthNotPositive(window_length=window_length)

    if window_length > data.shape[0]:
        raise WindowLengthTooLong(
            nrows=data.shape[0],
            window_length=window_length,
        )
