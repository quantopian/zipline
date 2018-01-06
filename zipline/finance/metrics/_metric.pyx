cimport cython
from libc.math cimport sqrt

cimport numpy as np
import numpy as np


cpdef minute_annual_volatility(np.ndarray[np.int64_t] date_labels,
                               np.ndarray[np.float64_t] minute_returns,
                               np.ndarray[np.float64_t] daily_returns):
    """Pre-compute the minute cumulative volatility field.
    """
    cdef np.ndarray out = np.empty_like(minute_returns)
    cdef np.int64_t previous_date = date_labels[0]
    cdef np.int64_t day_ix = 0
    cdef np.float64_t daily_sum = 0
    cdef np.float64_t todays_prod = 1
    cdef np.float64_t annualization_factor = sqrt(252.0)

    cdef np.float64_t tmp
    cdef np.float64_t intermediate_sum
    cdef np.float64_t mean
    cdef np.float64_t variance

    cdef Py_ssize_t ix
    cdef Py_ssize_t variance_ix
    cdef np.int64_t date_label
    cdef np.float64_t this_minute_returns


    for ix in range(len(minute_returns)):
        with cython.boundscheck(False), cython.wraparound(False):
            date = date_labels[ix]
            this_minute_returns = minute_returns[ix]

        if date != previous_date:
            previous_date = date
            daily_sum += daily_returns[day_ix]
            day_ix += 1
            todays_prod = 1

        if day_ix < 1:
            variance = np.nan
        else:
            todays_prod *= 1 + this_minute_returns

            intermediate_sum = daily_sum + todays_prod - 1
            mean = intermediate_sum / (day_ix + 1)

            variance = todays_prod - 1 - mean
            variance *= variance  # squared

            demeaned_old = daily_returns[:day_ix] - mean
            variance += demeaned_old.dot(demeaned_old)

            variance /= day_ix  # day_count - 1 for ddof=1

        with cython.boundscheck(False), cython.wraparound(False):
            out[ix] = sqrt(variance) * annualization_factor

    return out
