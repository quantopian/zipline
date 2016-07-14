from numpy cimport ndarray, long_t
from numpy import searchsorted
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def next_divider_idx(ndarray[long_t, ndim=1] dividers, long_t minute_val):
    cdef int divider_idx
    cdef long target

    divider_idx = searchsorted(dividers, minute_val, side="right")
    target = dividers[divider_idx]

    if minute_val == target:
        # if dt is exactly on the divider, go to the next value
        return divider_idx + 1
    else:
        return divider_idx

@cython.boundscheck(False)
@cython.wraparound(False)
def previous_divider_idx(ndarray[long_t, ndim=1] dividers,
                      long_t minute_val):
    cdef int divider_idx

    divider_idx = searchsorted(dividers, minute_val)

    if divider_idx == 0:
        raise ValueError("Cannot go earlier in calendar!")

    return divider_idx - 1

def is_open(ndarray[long_t, ndim=1] opens,
            ndarray[long_t, ndim=1] closes,
            long_t minute_val):
    cdef open_idx, close_idx

    open_idx = searchsorted(opens, minute_val)
    close_idx = searchsorted(closes, minute_val)

    if open_idx != close_idx:
        # if the indices are not same, that means the market is open
        return True
    else:
        try:
            # if they are the same, it might be the first minute of a
            # session
            return minute_val == opens[open_idx]
        except IndexError:
            # this can happen if we're outside the schedule's range (like
            # after the last close)
            return False
