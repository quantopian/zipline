"""
Class capable of yielding adjusted chunks of an ndarray.
"""
from numpy import (
    asarray,
    float64,
    full,
    uint8,
)
from numpy.ma import masked_array
from numpy cimport (
    float64_t,
    ndarray,
    uint8_t,
)

from zipline.errors import (
    LookbackNotPositive,
    LookbackTooLong,
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
        raise TypeError(
            "Can't generate window iterator for array of dtype %s" % data.dtype
        )


cdef _check_lookback(object data, int lookback):

    if lookback < 1:
        raise LookbackNotPositive(windowlen=lookback)

    if lookback > data.shape[0]:
        raise LookbackTooLong(nrows=data.shape[0], windowlen=lookback)


cdef class Float64AdjustedArray:
    """
    Adjusted array of float64.
    """
    cdef:
        float64_t[:, :] data
        uint8_t[:, :] mask
        dict adjustments

    def __cinit__(self,
                  float64_t[:, :] data not None,
                  uint8_t[:, :] mask,  # None is equivalent to all 1s.
                  dict adjustments):

        self.data = data
        self.adjustments = adjustments
        if mask is None:
            self.mask = full((data.shape[0], data.shape[1]), 1, dtype=uint8)
        else:
            assert mask.shape == data.shape
            self.mask = mask

    cpdef traverse(self, int lookback):
        return _Float64AdjustedArrayWindow(
            self.data.copy(),
            self.mask.copy(),
            lookback,
            self.adjustments,
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
    cdef uint8_t[:, :] mask
    cdef readonly int lookback
    cdef Py_ssize_t anchor, max_anchor, next_adj
    cdef dict adjustments
    cdef list adjustment_indices

    def __cinit__(self,
                  float64_t[:, :] data,
                  uint8_t[:, :] mask,
                  int lookback,
                  dict adjustments):

        _check_lookback(data, lookback)

        self.data = data
        self.mask = mask
        self.lookback = lookback

        # anchor is the index of the row **after** the row from which we're
        # looking back.
        self.anchor = lookback
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
                adjustment.mutate_mask(self.mask)

            if len(self.adjustment_indices) > 0:
                self.next_adj = self.adjustment_indices.pop()
            else:
                self.next_adj = self.max_anchor

        start = anchor - self.lookback
        out = asarray(self.data[start: self.anchor])
        mask_out = asarray(self.mask[start: self.anchor])

        self.anchor += 1
        return out, mask_out

    def __repr__(self):
        return "%s(lookback=%d, anchor=%d, max_anchor=%d)" % (
            type(self).__name__,
            self.lookback,
            self.anchor,
            self.max_anchor,
        )
