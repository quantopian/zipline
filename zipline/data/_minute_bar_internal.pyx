from numpy cimport ndarray, long_t
from numpy import searchsorted
from cpython cimport bool
cimport cython

cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.cdivision(True)
def minute_value(ndarray[long_t, ndim=1] market_opens,
                 Py_ssize_t pos,
                 short minutes_per_day):
    """
    Finds the value of the minute represented by `pos` in the given array of
    market opens.

    Parameters
    ----------
    market_opens: numpy array of ints
        Market opens, in minute epoch values.

    pos: int
        The index of the desired minute.

    minutes_per_day: int
        The number of minutes per day (e.g. 390 for NYSE).

    Returns
    -------
    int: The minute epoch value of the desired minute.
    """
    cdef short q, r

    q = cython.cdiv(pos, minutes_per_day)
    r = cython.cmod(pos, minutes_per_day)

    return market_opens[q] + r

def find_position_of_minute(ndarray[long_t, ndim=1] market_opens,
                            ndarray[long_t, ndim=1] market_closes,
                            long_t minute_val,
                            short minutes_per_day,
                            bool adjust_half_day_minutes):
    """
    Finds the position of a given minute in the given array of market opens.
    If not a market minute, adjusts to the last market minute.

    Parameters
    ----------
    market_opens: numpy array of ints
        Market opens, in minute epoch values.

    market_closes: numpy array of ints
        Market closes, in minute epoch values.

    minute_val: int
        The desired minute, as a minute epoch.

    minutes_per_day: int
        The number of minutes per day (e.g. 390 for NYSE).

    adjust_half_day_minutes: boolean
        Whether or not we want to adjust non trading minutes to early close on
        half days as opposed to normal close.

    Further explanation of the use adjust_half_day_minutes:
        adjust_half_day_minutes=True:
            We are using this method for the purpose finding a value for a
            minute, and therefore, all non market minutes must be adjusted to
            the last available (e.g. 9 pm EST -> 4 pm EST, 2 pm EST -> 1 pm EST
            on a half day)

        adjust_half_day_minutes=False:
            We are using this method for the purpose of finding the positions
            of minutes we want to ignore (1 pm to 4 pm EST on half days).
            The minute bar reader tape has 390 bars per day, with 0's filled in
            for the extra bars on half days. If we index a minute between
            1:01 pm and 4 pm on a half day, we want a position for that
            unadjusted time, not adjusted to 1 pm as in the above case
            (e.g. for all days: 9 pm EST -> 4 pm EST, 2 pm EST -> 2 pm EST)

    Returns
    -------
    int: The position of the given minute in the market opens array.
    """
    cdef Py_ssize_t market_open_loc, market_open, delta

    market_open_loc = \
        searchsorted(market_opens, minute_val, side='right') - 1
    market_open = market_opens[market_open_loc]
    market_close = market_closes[market_open_loc]

    if adjust_half_day_minutes:
        # The min of the distance to market open from minute_val and number
        # of trading minutes for that day
        delta = int_min(minute_val - market_open, market_close - market_open)
    else:
        # The min of the distance to market open from minute_val and number
        # of trading minutes for a normal day (390)
        delta = int_min(minute_val - market_open, minutes_per_day)

    return (market_open_loc * minutes_per_day) + delta

def find_last_traded_position_internal(
        ndarray[long_t, ndim=1] market_opens,
        ndarray[long_t, ndim=1] market_closes,
        long_t end_minute,
        long_t start_minute,
        volumes,
        short minutes_per_day):

    """
    Finds the position of the last traded minute for the given volumes array.

    Parameters
    ----------
    market_opens: numpy array of ints
        Market opens, in minute epoch values.

    market_closes: numpy array of ints
        Market closes, in minute epoch values.

    end_minute: int
        The minute from which to start looking backwards, as a minute epoch.

    start_minute: int
        The asset's start date, as a minute epoch.  Acts as the bottom limit of
        how far we can look backwards.

    volumes: bcolz carray
        The volume history for the given asset.

    minutes_per_day: int
        The number of minutes per day (e.g. 390 for NYSE).

    Returns
    -------
    int: The position of the last traded minute, starting from `minute_val`
    """
    cdef Py_ssize_t minute_pos, current_minute

    minute_pos = int_min(
        find_position_of_minute(market_opens, market_closes, end_minute,
                                minutes_per_day, True),
        len(volumes) - 1
    )

    while minute_pos >= 0:
        current_minute = minute_value(
            market_opens, minute_pos, minutes_per_day
        )

        if current_minute < start_minute:
            return -1

        if volumes[minute_pos] != 0:
            return minute_pos

        minute_pos -= 1

    # we've gone to the beginning of this asset's range, and still haven't
    # found a trade event
    return -1
