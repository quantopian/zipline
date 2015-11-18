"""
Template for AdjustedArray windowed iterators.

This file is intended to be used by inserting it via a Cython include into a
file that's define a type symbol named `ctype` and string constant named
`dtype`.

See Also
--------
zipline.lib._floatwindow
zipline.lib._intwindow
zipline.lib._datewindow
"""
from numpy cimport ndarray
from numpy import asarray


cdef class AdjustedArrayWindow:
    """
    An iterator representing a moving view over an AdjustedArray.

    Concrete subtypes should subclass this and provide a `data` attribute for
    specific types.

    This object stores a copy of the data from the AdjustedArray over which
    it's iterating.  At each step in the iteration, it mutates its copy to
    allow us to show different data when looking back over the array.

    The arrays yielded by this iterator are always views over the underlying
    data.
    """
    cdef readonly Py_ssize_t window_length
    cdef Py_ssize_t anchor, max_anchor, next_adj
    cdef dict adjustments
    cdef list adjustment_indices
    # ctype must be defined by the file into which this is being copied.
    cdef ctype[:, :] data

    def __cinit__(self,
                  ctype[:, :] data not None,
                  dict adjustments,
                  Py_ssize_t offset,
                  Py_ssize_t window_length):

        self.data = data
        self.adjustments = adjustments
        self.adjustment_indices = sorted(adjustments, reverse=True)
        self.window_length = window_length
        self.anchor = window_length + offset
        self.max_anchor = data.shape[0]

        if len(self.adjustment_indices) > 0:
            self.next_adj = self.adjustment_indices.pop()
        else:
            self.next_adj = self.max_anchor

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            ndarray[ctype, ndim=2] out
            object adjustment
            Py_ssize_t start, anchor

        anchor = self.anchor
        if anchor > self.max_anchor:
            raise StopIteration()

        # Apply any adjustments that occured before our current anchor.
        # Equivalently, apply any adjustments known **on or before** the date
        # for which we're calculating a window.
        while self.next_adj < anchor:

            for adjustment in self.adjustments[self.next_adj]:
                adjustment.mutate(self.data)

            if len(self.adjustment_indices) > 0:
                self.next_adj = self.adjustment_indices.pop()
            else:
                self.next_adj = self.max_anchor

        start = anchor - self.window_length
        # dtype must be defined by the file into which this is being copied.
        out = asarray(self.data[start:self.anchor], dtype=dtype)
        out.setflags(write=False)

        self.anchor += 1
        return out

    def __repr__(self):
        return "%s(window_length=%d, anchor=%d, max_anchor=%d, dtype=%s)" % (
            type(self).__name__,
            self.window_length,
            self.anchor,
            self.max_anchor,
            dtype,
        )
