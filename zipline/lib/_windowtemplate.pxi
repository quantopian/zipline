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
from numpy import asanyarray, dtype, issubdtype


class Exhausted(Exception):
    pass


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

    The `rounding_places` attribute is an integer used to specify the number of
    decimal places to which the data should be rounded, given that the data is
    of dtype float. If `rounding_places` is None, no rounding occurs.
    """
    cdef:
        # ctype must be defined by the file into which this is being copied.
        readonly databuffer data
        readonly dict view_kwargs
        readonly Py_ssize_t window_length
        Py_ssize_t anchor, max_anchor, next_adj
        Py_ssize_t perspective_offset
        object rounding_places
        dict adjustments
        list adjustment_indices
        ndarray output

    def __cinit__(self,
                  databuffer data not None,
                  dict view_kwargs not None,
                  dict adjustments not None,
                  Py_ssize_t offset,
                  Py_ssize_t window_length,
                  Py_ssize_t perspective_offset,
                  object rounding_places):
        self.data = data
        self.view_kwargs = view_kwargs
        self.adjustments = adjustments
        self.adjustment_indices = sorted(adjustments, reverse=True)
        self.window_length = window_length
        self.anchor = window_length + offset - 1
        if perspective_offset > 1:
            # Limit perspective_offset to 1.
            # To support an offset greater than 1, work must be done to
            # ensure that adjustments are retrieved for the datetimes between
            # the end of the window and the vantage point defined by the
            # perspective offset.
            raise Exception("perspective_offset should not exceed 1, value "
                            "is perspective_offset={0}".format(
                                perspective_offset))
        self.perspective_offset = perspective_offset
        self.rounding_places = rounding_places
        self.max_anchor = data.shape[0]

        self.next_adj = self.pop_next_adj()
        self.output = None

    cdef pop_next_adj(self):
        """
        Pop the index of the next adjustment to apply from self.adjustment_indices.
        """
        if len(self.adjustment_indices) > 0:
            return self.adjustment_indices.pop()
        else:
            return self.max_anchor + self.perspective_offset

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._tick_forward(1)
        except Exhausted:
            raise StopIteration()

        self._update_output()
        return self.output

    def seek(self, Py_ssize_t target_anchor):
        cdef:
            Py_ssize_t anchor = self.anchor

        if target_anchor < anchor:
            raise Exception('Can not access data after window has passed.')

        if target_anchor == anchor:
            return self.output

        self._tick_forward(target_anchor - anchor)
        self._update_output()

        return self.output

    cdef inline _tick_forward(self, int N):
        cdef:
            object adjustment
            Py_ssize_t anchor = self.anchor
            Py_ssize_t target = anchor + N

        if target > self.max_anchor:
            raise Exhausted()

        # Apply any adjustments that occured before our current anchor.
        # Equivalently, apply any adjustments known **on or before** the date
        # for which we're calculating a window.
        while self.next_adj < target + self.perspective_offset:

            for adjustment in self.adjustments[self.next_adj]:
                adjustment.mutate(self.data)

            self.next_adj = self.pop_next_adj()

        self.anchor = target

    cdef inline _update_output(self):
        cdef:
            ndarray new_out
            Py_ssize_t anchor = self.anchor
            dict view_kwargs = self.view_kwargs

        new_out = asanyarray(self.data[anchor - self.window_length:anchor])
        if view_kwargs:
            new_out = new_out.view(**view_kwargs)
        if self.rounding_places is not None and \
                issubdtype(new_out.dtype, dtype('float64')):
            new_out = new_out.round(self.rounding_places)
        new_out.setflags(write=False)
        self.output = new_out

    def __repr__(self):
        return "<%s: window_length=%d, anchor=%d, max_anchor=%d, dtype=%r>" % (
            type(self).__name__,
            self.window_length,
            self.anchor,
            self.max_anchor,
            self.view_kwargs.get('dtype'),
        )
