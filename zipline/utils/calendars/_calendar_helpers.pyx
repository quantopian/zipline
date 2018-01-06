cimport numpy as np
import numpy as np
from numpy cimport ndarray, int64_t
from numpy import empty, searchsorted, int64
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int next_divider_idx(ndarray[int64_t, ndim=1] dividers, int64_t minute_val):
    cdef int divider_idx
    cdef int64_t target

    divider_idx = searchsorted(dividers, minute_val, side="right")
    target = dividers[divider_idx]

    if minute_val == target:
        # if dt is exactly on the divider, go to the next value
        return divider_idx + 1
    else:
        return divider_idx

@cython.boundscheck(False)
@cython.wraparound(False)
def previous_divider_idx(ndarray[int64_t, ndim=1] dividers,
                      int64_t minute_val):
    cdef int divider_idx

    divider_idx = searchsorted(dividers, minute_val)

    if divider_idx == 0:
        raise ValueError("Cannot go earlier in calendar!")

    return divider_idx - 1

def is_open(ndarray[int64_t, ndim=1] opens,
            ndarray[int64_t, ndim=1] closes,
            int64_t minute_val):
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


cdef np.int64_t NANOS_IN_MINUTE = 60000000000


def compute_all_minutes(np.ndarray[np.int64_t] opens_in_ns,
                        np.ndarray[np.int64_t] closes_in_ns):
    cdef np.ndarray[np.int64_t] deltas = closes_in_ns - opens_in_ns

    # + 1 because we want 390 days per standard day, not 389
    cdef np.ndarray[np.int64_t] daily_sizes = (deltas // NANOS_IN_MINUTE) + 1
    cdef np.int64_t num_minutes = daily_sizes.sum()

    # One allocation for the entire thing. This assumes that each day
    # represents a contiguous block of minutes.
    cdef np.ndarray[np.int64_t] all_minutes = np.empty(
        num_minutes,
        dtype='int64',
    )

    cdef np.int64_t minute
    cdef np.uint64_t ix = 0

    cdef np.uint64_t day_ix
    cdef np.uint64_t minute_ix
    cdef np.int64_t size
    for day_ix in range(len(daily_sizes)):
        with cython.boundscheck(False), cython.wraparound(False):
            size = daily_sizes[day_ix]
            minute = opens_in_ns[day_ix]

        for minute_ix in range(size):
            with cython.boundscheck(False), cython.wraparound(False):
                all_minutes[ix + minute_ix] = minute

            minute += NANOS_IN_MINUTE

        ix += size

    return all_minutes.view('datetime64[ns]')
