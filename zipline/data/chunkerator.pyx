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


cpdef adjusted_array(ndarray[:] data, dict adjustments):
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


cdef _check_lookback(float64_t[:, ::1] data, int lookback):
    if lookback > data.shape[0]:
        raise ValueError(
            "Can't construct a rolling window of length %d "
            "on buffer of shape %s" % (lookback, data.shape)
        )


cdef class Float64AdjustedArray:
    """
    Adjusted array of float64.
    """

    cdef readonly float64_t[:, ::1] data
    cdef readonly dict adjustments

    def __cinit__(self,
                  float64_t[:, ::1] data not None,
                  dict adjustments):

        self.data = data
        self.adjustments = adjustments

    cpdef window_iter(self, int lookback):
        return _Float64WindowIterator(
            self.data.copy(),
            lookback,
            self.adjustments,
        )


cdef class _Float64WindowIterator:

    cdef float64_t[:, ::1] data
    cdef readonly int lookback
    cdef int index, length
    cdef dict adjustments

    def __cinit__(self,
                  float64_t[:, ::1] data,
                  int lookback,
                  dict adjustments):

        self.data = data
        self.lookback = lookback
        self.index = 0
        self.length = data.shape[0] - lookback

    def __iter__(self):
        return self

    def __next__(self):
        cdef ndarray[float64_t, ndim=2] out

        if self.index == self.length:
            raise StopIteration

        # TODO: Apply any adjustments we have.

        out = asarray(self.data[self.index:self.index + self.lookback])
        self.index += 1
        return out
