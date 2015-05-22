"""
Class capable of yielding adjusted chunks of an ndarray.
"""
from numpy import (
    asarray,
    float64,
)
from numpy cimport (
    float64_t,
    ndarray,
)

from zipline.errors import (
    LookbackNotPositive,
    LookbackTooLong,
)


cpdef adjusted_array(ndarray data, dict adjustments):
    """
    Factory function for producing adjusted arrays on inputs of different
    dtypes.
    """
    if data.dtype == float64:
        return Float64AdjustedArray(data, adjustments)
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
        readonly float64_t[:, ::1] data
        readonly dict adjustments

    def __cinit__(self,
                  float64_t[:, ::1] data not None,
                  dict adjustments):

        self.data = data
        self.adjustments = adjustments

    cpdef traverse(self, int lookback):
        return _Float64AdjustedArrayWindow(
            self.data.copy(),
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

    cdef float64_t[:, ::1] data
    cdef readonly int lookback
    cdef Py_ssize_t anchor, max_anchor, next_adj
    cdef dict adjustments
    cdef list adjustment_indices

    def __cinit__(self,
                  float64_t[:, ::1] data,
                  int lookback,
                  dict adjustments):

        _check_lookback(data, lookback)

        self.data = data
        self.lookback = lookback

        # anchor is the index of the row **after** the row from which we're
        # looking back.
        self.anchor = lookback
        self.max_anchor = data.shape[0]

        self.adjustments = adjustments

        if len(self.adjustments) > 0:
            self.adjustment_indices = sorted(adjustments, reverse=True)
            self.next_adj = self.adjustment_indices.pop()
        else:
            self.adjustment_indices = []
            self.next_adj = self.max_anchor

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            ndarray[float64_t, ndim=2] out
            object adjustment

        if self.anchor > self.max_anchor:
            raise StopIteration()

        # Apply any adjustments that occured before our current anchor.
        # Equivalently, apply any adjustments known on or before the date for
        # which we're calculating a window.
        while self.next_adj < self.anchor:
            print self.next_adj, self.anchor, self.adjustment_indices
            for adjustment in self.adjustments[self.next_adj]:
                adjustment.mutate(self.data)

            if len(self.adjustment_indices) > 0:
                self.next_adj = self.adjustment_indices.pop()
            else:
                # No more adjustments to apply.
                self.next_adj = self.max_anchor

        out = asarray(self.data[self.anchor - self.lookback: self.anchor])
        out.setflags(write=0)
        self.anchor += 1
        return out

    def __repr__(self):
        return "%s(lookback=%d, anchor=%d, max_anchor=%d)" % (
            type(self).__name__,
            self.lookback,
            self.anchor,
            self.max_anchor,
        )
