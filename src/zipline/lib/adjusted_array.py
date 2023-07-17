from textwrap import dedent
from functools import partial
import numpy as np

from toolz import merge_with
from zipline.errors import (
    WindowLengthNotPositive,
    WindowLengthTooLong,
)
from zipline.lib.labelarray import LabelArray
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    uint8_dtype,
)
from zipline.utils.memoize import lazyval

# These class names are all the same because of our bootleg templating system.
from ._float64window import AdjustedArrayWindow as Float64Window
from ._int64window import AdjustedArrayWindow as Int64Window
from ._labelwindow import AdjustedArrayWindow as LabelWindow
from ._uint8window import AdjustedArrayWindow as UInt8Window


BOOL_DTYPES = frozenset(
    map(np.dtype, [np.bool_, np.uint8]),
)
FLOAT_DTYPES = frozenset(
    map(np.dtype, [np.float32, np.float64]),
)
INT_DTYPES = frozenset(
    # NOTE: uint64 not supported because it can't be safely cast to int64.
    map(np.dtype, [np.int16, np.uint16, np.int32, np.int64, np.uint32]),
)
DATETIME_DTYPES = frozenset(map(np.dtype, ["datetime64[ns]"]))
# We use object arrays for strings.
OBJECT_DTYPES = frozenset(map(np.dtype, ["O"]))
STRING_KINDS = frozenset(["S", "U"])

REPRESENTABLE_DTYPES = BOOL_DTYPES.union(
    FLOAT_DTYPES,
    INT_DTYPES,
    DATETIME_DTYPES,
    OBJECT_DTYPES,
)


def can_represent_dtype(dtype):
    """Can we build an AdjustedArray for a baseline of `dtype``?"""
    return dtype in REPRESENTABLE_DTYPES or dtype.kind in STRING_KINDS


def is_categorical(dtype):
    """Do we represent this dtype with LabelArrays rather than ndarrays?"""
    return dtype in OBJECT_DTYPES or dtype.kind in STRING_KINDS


CONCRETE_WINDOW_TYPES = {
    float64_dtype: Float64Window,
    int64_dtype: Int64Window,
    uint8_dtype: UInt8Window,
}


def _normalize_array(data, missing_value):
    """
    Coerce buffer data for an AdjustedArray into a standard scalar
    representation, returning the coerced array and a dict of argument to pass
    to np.view to use when providing a user-facing view of the underlying data.

    - float* data is coerced to float64 with viewtype float64.
    - int32, int64, and uint32 are converted to int64 with viewtype int64.
    - datetime[*] data is coerced to int64 with a viewtype of datetime64[ns].
    - bool_ data is coerced to uint8 with a viewtype of bool_.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    coerced, view_kwargs : (np.ndarray, np.dtype)
        The input ``data`` array coerced to the appropriate pipeline type.
        This may return the original array or a view over the same data.
    """
    if isinstance(data, LabelArray):
        return data, {}

    data_dtype = data.dtype
    if data_dtype in BOOL_DTYPES:
        return data.astype(np.uint8, copy=False), {"dtype": np.dtype(np.bool_)}
    elif data_dtype in FLOAT_DTYPES:
        return data.astype(np.float64, copy=False), {"dtype": np.dtype(np.float64)}
    elif data_dtype in INT_DTYPES:
        return data.astype(np.int64, copy=False), {"dtype": np.dtype(np.int64)}
    elif is_categorical(data_dtype):
        if not isinstance(missing_value, LabelArray.SUPPORTED_SCALAR_TYPES):
            raise TypeError(
                "Invalid missing_value for categorical array.\n"
                "Expected None, bytes or unicode. Got %r." % missing_value,
            )
        return LabelArray(data, missing_value), {}
    elif data_dtype.kind == "M":
        try:
            outarray = data.astype("datetime64[ns]", copy=False).view("int64")
            return outarray, {"dtype": datetime64ns_dtype}
        except OverflowError as exc:
            raise ValueError(
                "AdjustedArray received a datetime array "
                "not representable as datetime64[ns].\n"
                "Min Date: %s\n"
                "Max Date: %s\n" % (data.min(), data.max())
            ) from exc
    else:
        raise TypeError(
            "Don't know how to construct AdjustedArray "
            "on data of type %s." % data_dtype
        )


def _merge_simple(adjustment_lists, front_idx, back_idx):
    """
    Merge lists of new and existing adjustments for a given index by appending
    or prepending new adjustments to existing adjustments.

    Notes
    -----
    This method is meant to be used with ``toolz.merge_with`` to merge
    adjustment mappings. In case of a collision ``adjustment_lists`` contains
    two lists, existing adjustments at index 0 and new adjustments at index 1.
    When there are no collisions, ``adjustment_lists`` contains a single list.

    Parameters
    ----------
    adjustment_lists : list[list[Adjustment]]
        List(s) of new and/or existing adjustments for a given index.
    front_idx : int
        Index of list in ``adjustment_lists`` that should be used as baseline
        in case of a collision.
    back_idx : int
        Index of list in ``adjustment_lists`` that should extend baseline list
        in case of a collision.

    Returns
    -------
    adjustments : list[Adjustment]
        List of merged adjustments for a given index.
    """
    if len(adjustment_lists) == 1:
        return list(adjustment_lists[0])
    else:
        return adjustment_lists[front_idx] + adjustment_lists[back_idx]


_merge_methods = {
    "append": partial(_merge_simple, front_idx=0, back_idx=1),
    "prepend": partial(_merge_simple, front_idx=1, back_idx=0),
}


class AdjustedArray:
    """
    An array that can be iterated with a variable-length window, and which can
    provide different views on data from different perspectives.

    Parameters
    ----------
    data : np.ndarray
        The baseline data values. This array may be mutated by
        ``traverse(..., copy=False)`` calls.
    adjustments : dict[int -> list[Adjustment]]
        A dict mapping row indices to lists of adjustments to apply when we
        reach that row.
    missing_value : object
        A value to use to fill missing data in yielded windows.
        Should be a value coercible to `data.dtype`.
    """

    __slots__ = (
        "_data",
        "_view_kwargs",
        "adjustments",
        "missing_value",
        "_invalidated",
        "__weakref__",
    )

    def __init__(self, data, adjustments, missing_value):
        self._data, self._view_kwargs = _normalize_array(data, missing_value)

        self.adjustments = adjustments
        self.missing_value = missing_value
        self._invalidated = False

    def copy(self):
        """Copy an adjusted array, deep-copying the ``data`` array."""
        if self._invalidated:
            raise ValueError("cannot copy invalidated AdjustedArray")

        return type(self)(
            self.data.copy(order="F"),
            self.adjustments,
            self.missing_value,
        )

    def update_adjustments(self, adjustments, method):
        """
        Merge ``adjustments`` with existing adjustments, handling index
        collisions according to ``method``.

        Parameters
        ----------
        adjustments : dict[int -> list[Adjustment]]
            The mapping of row indices to lists of adjustments that should be
            appended to existing adjustments.
        method : {'append', 'prepend'}
            How to handle index collisions. If 'append', new adjustments will
            be applied after previously-existing adjustments. If 'prepend', new
            adjustments will be applied before previously-existing adjustments.
        """
        try:
            merge_func = _merge_methods[method]
        except KeyError as exc:
            raise ValueError(
                "Invalid merge method %s\n"
                "Valid methods are: %s" % (method, ", ".join(_merge_methods))
            ) from exc

        self.adjustments = merge_with(
            merge_func,
            self.adjustments,
            adjustments,
        )

    @property
    def data(self):
        """
        The data stored in this array.
        """
        return self._data.view(**self._view_kwargs)

    @lazyval
    def dtype(self):
        """
        The dtype of the data stored in this array.
        """
        return self._view_kwargs.get("dtype") or self._data.dtype

    @lazyval
    def _iterator_type(self):
        """
        The iterator produced when `traverse` is called on this Array.
        """
        if isinstance(self._data, LabelArray):
            return LabelWindow
        return CONCRETE_WINDOW_TYPES[self._data.dtype]

    def traverse(self, window_length, offset=0, perspective_offset=0, copy=True):
        """
        Produce an iterator rolling windows rows over our data.
        Each emitted window will have `window_length` rows.

        Parameters
        ----------
        window_length : int
            The number of rows in each emitted window.
        offset : int, optional
            Number of rows to skip before the first window.  Default is 0.
        perspective_offset : int, optional
            Number of rows past the end of the current window from which to
            "view" the underlying data.
        copy : bool, optional
            Copy the underlying data. If ``copy=False``, the adjusted array
            will be invalidated and cannot be traversed again.
        """
        if self._invalidated:
            raise ValueError("cannot traverse invalidated AdjustedArray")

        data = self._data
        if copy:
            data = data.copy(order="F")
        else:
            self._invalidated = True

        _check_window_params(data, window_length)
        return self._iterator_type(
            data,
            self._view_kwargs,
            self.adjustments,
            offset,
            window_length,
            perspective_offset,
            rounding_places=None,
        )

    def inspect(self):
        """
        Return a string representation of the data stored in this array.
        """
        return dedent(
            """\
            Adjusted Array ({dtype}):

            Data:
            {data!r}

            Adjustments:
            {adjustments}
            """
        ).format(
            dtype=self.dtype.name,
            data=self.data,
            adjustments=self.adjustments,
        )

    def update_labels(self, func):
        """
        Map a function over baseline and adjustment values in place.

        Note that the baseline data values must be a LabelArray.
        """
        if not isinstance(self.data, LabelArray):
            raise TypeError(
                "update_labels only supported if data is of type LabelArray."
            )

        # Map the baseline values.
        self._data = self._data.map(func)

        # Map each of the adjustments.
        for _, row_adjustments in self.adjustments.items():
            for adjustment in row_adjustments:
                adjustment.value = func(adjustment.value)


def ensure_adjusted_array(ndarray_or_adjusted_array, missing_value):
    if isinstance(ndarray_or_adjusted_array, AdjustedArray):
        return ndarray_or_adjusted_array
    elif isinstance(ndarray_or_adjusted_array, np.ndarray):
        return AdjustedArray(
            ndarray_or_adjusted_array,
            {},
            missing_value,
        )
    else:
        raise TypeError(
            "Can't convert %s to AdjustedArray"
            % type(ndarray_or_adjusted_array).__name__
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
    if isinstance(ndarray_or_adjusted_array, np.ndarray):
        return ndarray_or_adjusted_array
    elif isinstance(ndarray_or_adjusted_array, AdjustedArray):
        return ndarray_or_adjusted_array.data
    else:
        raise TypeError(
            "Can't convert %s to ndarray" % type(ndarray_or_adjusted_array).__name__
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
