cimport cython
from libc.math cimport sqrt

cimport numpy as np
import numpy as np
from six import itervalues

from zipline._protocol cimport InnerPosition
from zipline.assets._assets cimport Future


cpdef update_position_last_sale_prices(positions, get_price, dt):
    """Update the positions' last sale prices.

    Parameters
    ----------
    positions : OrderedDict
        The positions to update.
    get_price : callable[Asset, float]
        The function to retrieve the price for the asset.
    dt : pd.Timestamp
        The dt to set as the last sale date if the price is not nan.
    """
    cdef InnerPosition inner_position
    cdef np.float64_t last_sale_price

    for outer_position in itervalues(positions):
        inner_position = outer_position.inner_position

        last_sale_price = get_price(inner_position.asset)

        # inline ~isnan because this gets called once per position per minute
        if last_sale_price == last_sale_price:
            inner_position.last_sale_price = last_sale_price
            inner_position.last_sale_date = dt


@cython.final
cdef class PositionStats:
    cdef readonly np.float64_t net_exposure
    cdef readonly np.float64_t gross_value
    cdef readonly np.float64_t gross_exposure
    cdef readonly np.float64_t short_value
    cdef readonly np.float64_t short_exposure
    cdef readonly np.uint64_t shorts_count
    cdef readonly np.float64_t long_value
    cdef readonly np.float64_t long_exposure
    cdef readonly np.uint64_t longs_count
    cdef readonly np.float64_t net_value


cpdef calculate_position_tracker_stats(positions):
    """Calculate various stats about the current positions.

    Parameters
    ----------
    positions : OrderedDict
        The ordered dictionary of positions.

    Returns
    -------
    position_stats : PositionStats
        The computed statistics.
    """
    cdef np.float64_t value
    cdef np.float64_t exposure

    cdef np.float64_t net_value
    cdef np.float64_t gross_value
    cdef np.float64_t long_value = 0.0
    cdef np.float64_t short_value = 0.0

    cdef np.float64_t net_exposure
    cdef np.float64_t gross_exposure
    cdef np.float64_t long_exposure = 0.0
    cdef np.float64_t short_exposure = 0.0

    cdef np.uint64_t longs_count = 0
    cdef np.uint64_t shorts_count = 0

    cdef InnerPosition position

    for outer_position in itervalues(positions):
        position = outer_position.inner_position

        # NOTE: this loop does a lot of stuff!
        # we call this function every single minute of the simulations
        # so let's not iterate through every single position multiple
        # times.
        exposure = position.amount * position.last_sale_price

        if type(position.asset) is Future:
            # Futures don't have an inherent position value.
            value = 0

            # unchecked cast, this is safe because we do a type check above
            exposure *= (<Future> position.asset).multiplier
        else:
            value = exposure

        if exposure > 0:
            longs_count += 1
            long_value += value
            long_exposure += exposure
        elif exposure < 0:
            shorts_count += 1
            short_value += value
            short_exposure += exposure

    net_value = long_value + short_value
    gross_value = long_value - short_value

    net_exposure = long_exposure + short_exposure
    gross_exposure = long_exposure - short_exposure

    # NOTE: I didn't define a ``__cinit__`` or whatever because I can't figure
    # out how to make Cython forward arguments without boxing/unboxing.
    # If we update Cython, try to replace this with a regular constructor and
    # check the compiler output.
    cdef PositionStats ret = PositionStats()
    ret.gross_exposure = gross_exposure
    ret.gross_value = gross_value
    ret.long_exposure = long_exposure
    ret.long_value = long_value
    ret.longs_count = longs_count
    ret.net_exposure = net_exposure
    ret.net_value = net_value
    ret.short_exposure = short_exposure
    ret.short_value = short_value
    ret.shorts_count = shorts_count

    return ret


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
