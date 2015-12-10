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

ctypedef ctype[:, :] databuffer


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
    cdef:
        # ctype must be defined by the file into which this is being copied.
        databuffer data
        object viewtype
        readonly Py_ssize_t window_length
        Py_ssize_t anchor, max_anchor, next_adj
        dict adjustments
        list adjustment_indices

    def __cinit__(self,
                  databuffer data not None,
                  object viewtype not None,
                  dict adjustments not None,
                  Py_ssize_t offset,
                  Py_ssize_t window_length):

        self.data = data
        self.viewtype = viewtype
        self.adjustments = adjustments
        self.adjustment_indices = sorted(adjustments, reverse=True)
        self.window_length = window_length
        self.anchor = window_length + offset
        self.max_anchor = data.shape[0]

        self.next_adj = self.pop_next_adj()

    cdef pop_next_adj(self):
        """
        Pop the index of the next adjustment to apply from self.adjustment_indices.
        """
        if len(self.adjustment_indices) > 0:
            return self.adjustment_indices.pop()
        else:
            return self.max_anchor

    def __iter__(self):
        return self

    def __next__(self):
        cdef:
            ndarray out
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

            self.next_adj = self.pop_next_adj()

        start = anchor - self.window_length
        out = asarray(self.data[start:self.anchor]).view(self.viewtype)
        out.setflags(write=False)

        self.anchor += 1
        return out

    def __repr__(self):
        return "<%s: window_length=%d, anchor=%d, max_anchor=%d, dtype=%r>" % (
            type(self).__name__,
            self.window_length,
            self.anchor,
            self.max_anchor,
            self.viewtype,
        )
