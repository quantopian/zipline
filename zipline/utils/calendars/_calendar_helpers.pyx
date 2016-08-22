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

@cython.boundscheck(False)
@cython.wraparound(False)
def minutes_to_session_labels(ndarray[int64_t, ndim=1] minutes,
                              minute_to_session_label,
                              ndarray[int64_t, ndim=1] closes):
    cdef int current_idx, next_idx, close_idx
    current_idx = next_idx = close_idx = 0

    cdef ndarray[int64_t, ndim=1] results = empty(len(minutes), dtype=int64)

    while current_idx < len(minutes):
        close_idx += searchsorted(closes[close_idx:],
                                  minutes[current_idx], side="right")
        next_idx += next_divider_idx(minutes[next_idx:], closes[close_idx])
        results[current_idx:next_idx] = minute_to_session_label(
            minutes[current_idx]
        )
        current_idx = next_idx

    return results
