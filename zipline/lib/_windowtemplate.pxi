"""
Template for AdjustedArray windowed iterators.

This file is intended to be used by inserting it via a Cython include into a
file that's defined a type symbol named `databuffer` that can be used like a
2-D numpy array.

See Also
--------
zipline.lib._floatwindow
zipline.lib._intwindow
zipline.lib._datewindow
"""
from numpy cimport ndarray
from numpy import asanyarray


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
        readonly databuffer data
        readonly dict view_kwargs
        readonly Py_ssize_t window_length
        Py_ssize_t anchor, next_anchor, max_anchor, next_adj
        dict adjustments
        list adjustment_indices
        ndarray last_out

    def __cinit__(self,
                  databuffer data not None,
                  dict view_kwargs not None,
                  dict adjustments not None,
                  Py_ssize_t offset,
                  Py_ssize_t window_length):

        self.data = data
        self.view_kwargs = view_kwargs
        self.adjustments = adjustments
        self.adjustment_indices = sorted(adjustments, reverse=True)
        self.window_length = window_length
        self.anchor = window_length + offset
        self.next_anchor = self.anchor
        self.max_anchor = data.shape[0]

        self.next_adj = self.pop_next_adj()
        self.last_out = None

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
            object adjustment
            ndarray out
            Py_ssize_t start, anchor
            dict view_kwargs

        anchor = self.anchor = self.next_anchor
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

        # If our data is a custom subclass of ndarray, preserve that subclass
        # by using asanyarray instead of asarray.
        out = asanyarray(self.data[start:self.anchor])
        view_kwargs = self.view_kwargs
        if view_kwargs:
            out = out.view(**view_kwargs)
        out.setflags(write=False)

        self.next_anchor = self.anchor + 1
        self.last_out = out
        return out

    def seek(self, target_anchor):
        cdef ndarray out = None

        if target_anchor < self.anchor:
            raise Exception('Can not access data after window has passed.')

        if target_anchor == self.anchor:
            return self.last_out

        while self.anchor < target_anchor:
            out = next(self)

        self.last_out = out
        return out

    def __repr__(self):
        return "<%s: window_length=%d, anchor=%d, max_anchor=%d, dtype=%r>" % (
            type(self).__name__,
            self.window_length,
            self.anchor,
            self.max_anchor,
            self.view_kwargs.get('dtype'),
        )
