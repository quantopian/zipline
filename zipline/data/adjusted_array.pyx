"""
Class capable of yielding adjusted chunks of an ndarray.
"""
from numpy import (
    asarray,
    float64,
    full,
    uint8,
)
from numpy cimport (
    float64_t,
    ndarray,
    uint8_t,
)

from zipline.errors import (
    WindowLengthNotPositive,
    WindowLengthTooLong,
)

cdef extern from "math.h" nogil:
    float NAN


NOMASK = None


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


cpdef adjusted_array(ndarray data, uint8_t[:, :] mask, dict adjustments):
    """
    Factory function for producing adjusted arrays on inputs of different
    dtypes.

    If mask is None, the array is assumed to contain all valid data points.
    Otherwise mask should be an array of uint8 of the same shape
    as data, containing 0s for valid values and 1s for invalid values.
    """
    if data.dtype == float64:
        return Float64AdjustedArray(data, mask, adjustments)
    else:
        return Float64AdjustedArray(data.astype(float64), mask, adjustments)


cdef _check_window_length(object data, int window_length):

    if window_length < 1:
        raise WindowLengthNotPositive(window_length=window_length)

    if window_length > data.shape[0]:
        raise WindowLengthTooLong(
            nrows=data.shape[0],
            window_length=window_length,
        )


cdef class AdjustedArray:

    property data:
        def __get__(self):
            out = asarray(self._data, dtype=self.dtype)
            out.setflags(write=False)
            return out


cdef class Float64AdjustedArray(AdjustedArray):
    """
    Adjusted array of float64.
    """
    cdef:
        float64_t[:, :] _data
        dict adjustments

    def __cinit__(self,
                  float64_t[:, :] data not None,
                  uint8_t[:, :] mask,  # None is equivalent to all 0s.
                  dict adjustments):
        cdef Py_ssize_t row, col

        if mask is not NOMASK:
            assert mask.shape == data.shape
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    data[row, col] = NAN

        self._data = data
        self.adjustments = adjustments

    property dtype:
        def __get__(self):
            return float64

    cpdef traverse(self, Py_ssize_t window_length, Py_ssize_t offset=0):
        return _Float64AdjustedArrayWindow(
            self._data.copy(),
            self.adjustments,
            window_length,
            offset,
        )


cdef class _Float64AdjustedArrayWindow:
    """
    An iterator representing a moving view over an AdjustedArray.

    This object stores a copy of the data from the AdjustedArray over which
    it's iterating.  At each step in the iteration, it mutates its copy to
    allow us to show different data when looking back over the array.

    The arrays yielded by this iterator are always views over the underlying
    data.
    """

    cdef float64_t[:, :] data
    cdef readonly Py_ssize_t window_length
    cdef Py_ssize_t anchor, max_anchor, next_adj
    cdef dict adjustments
    cdef list adjustment_indices

    def __cinit__(self,
                  float64_t[:, :] data,
                  dict adjustments,
                  Py_ssize_t window_length,
                  Py_ssize_t offset):

        _check_window_length(data, window_length)

        self.data = data
        self.window_length = window_length

        # anchor is the index of the row **after** the row from which we're
        # looking back.
        self.anchor = window_length + offset
        self.max_anchor = data.shape[0]

        self.adjustments = adjustments
        self.adjustment_indices = sorted(adjustments, reverse=True)

        if len(self.adjustment_indices) > 0:
            self.next_adj = self.adjustment_indices.pop()
        else:
            self.next_adj = self.max_anchor

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            ndarray[float64_t, ndim=2] out
            object adjustment
            Py_ssize_t start, anchor

        anchor = self.anchor
        if anchor > self.max_anchor:
            raise StopIteration()

        # Apply any adjustments that occured before our current anchor.
        # Equivalently, apply any adjustments known on or before the date for
        # which we're calculating a window.
        while self.next_adj < anchor:

            for adjustment in self.adjustments[self.next_adj]:
                adjustment.mutate(self.data)

            if len(self.adjustment_indices) > 0:
                self.next_adj = self.adjustment_indices.pop()
            else:
                self.next_adj = self.max_anchor

        start = anchor - self.window_length
        out = asarray(self.data[start:self.anchor])
        out.setflags(write=False)

        self.anchor += 1
        return out

    def __repr__(self):
        return "%s(window_length=%d, anchor=%d, max_anchor=%d)" % (
            type(self).__name__,
            self.window_length,
            self.anchor,
            self.max_anchor,
        )
