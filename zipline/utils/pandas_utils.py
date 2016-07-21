"""
Utilities for working with pandas objects.
"""
from itertools import product
import operator as op

import pandas as pd
from distutils.version import StrictVersion

pandas_version = StrictVersion(pd.__version__)


def explode(df):
    """
    Take a DataFrame and return a triple of

    (df.index, df.columns, df.values)
    """
    return df.index, df.columns, df.values


try:
    # This branch is hit in pandas 17
    sort_values = pd.DataFrame.sort_values
except AttributeError:
    # This branch is hit in pandas 16
    sort_values = pd.DataFrame.sort

if pandas_version >= StrictVersion('0.17.1'):
    july_5th_holiday_observance = lambda dtix: dtix[dtix.year != 2013]
else:
    july_5th_holiday_observance = lambda dt: None if dt.year == 2013 else dt


def _time_to_micros(time):
    """Convert a time into microseconds since midnight.
    Parameters
    ----------
    time : datetime.time
        The time to convert.
    Returns
    -------
    us : int
        The number of microseconds since midnight.
    Notes
    -----
    This does not account for leap seconds or daylight savings.
    """
    seconds = time.hour * 60 * 60 + time.minute * 60 + time.second
    return 1000000 * seconds + time.microsecond


_opmap = dict(zip(
    product((True, False), repeat=3),
    product((op.le, op.lt), (op.le, op.lt), (op.and_, op.or_)),
))


def mask_between_time(dts, start, end, include_start=True, include_end=True):
    """Return a mask of all of the datetimes in ``dts`` that are between
    ``start`` and ``end``.
    Parameters
    ----------
    dts : pd.DatetimeIndex
        The index to mask.
    start : time
        Mask away times less than the start.
    end : time
        Mask away times greater than the end.
    include_start : bool, optional
        Inclusive on ``start``.
    include_end : bool, optional
        Inclusive on ``end``.
    Returns
    -------
    mask : np.ndarray[bool]
        A bool array masking ``dts``.
    See Also
    --------
    :meth:`pandas.DatetimeIndex.indexer_between_time`
    """
    # This function is adapted from
    # `pandas.Datetime.Index.indexer_between_time` which was originally
    # written by Wes McKinney, Chang She, and Grant Roch.
    time_micros = dts._get_time_micros()
    start_micros = _time_to_micros(start)
    end_micros = _time_to_micros(end)

    left_op, right_op, join_op = _opmap[
        bool(include_start),
        bool(include_end),
        start_micros <= end_micros,
    ]

    return join_op(
        left_op(start_micros, time_micros),
        right_op(time_micros, end_micros),
    )


def nearest_unequal_elements(dts, dt):
    """
    Find values in ``dts`` closest but not equal to ``dt``.

    Returns a pair of (last_before, first_after).

    When ``dt`` is less than any element in ``dts``, ``last_before`` is None.
    When ``dt`` is greater any element in ``dts``, ``first_after`` is None.

    ``dts`` must be unique and sorted in increasing order.

    Parameters
    ----------
    dts : pd.DatetimeIndex
        Dates in which to search.
    dt : pd.Timestamp
        Date for which to find bounds.
    """
    if not dts.is_unique:
        raise ValueError("dts must be unique")

    if not dts.is_monotonic_increasing:
        raise ValueError("dts must be sorted in increasing order")

    if not len(dts):
        return None, None

    sortpos = dts.searchsorted(dt, side='left')
    try:
        sortval = dts[sortpos]
    except IndexError:
        # dt is greater than any value in the array.
        return dts[-1], None

    if dt < sortval:
        lower_ix = sortpos - 1
        upper_ix = sortpos
    elif dt == sortval:
        lower_ix = sortpos - 1
        upper_ix = sortpos + 1
    else:
        lower_ix = sortpos
        upper_ix = sortpos + 1

    lower_value = dts[lower_ix] if lower_ix >= 0 else None
    upper_value = dts[upper_ix] if upper_ix < len(dts) else None

    return lower_value, upper_value


def timedelta_to_integral_seconds(delta):
    """
    Convert a pd.Timedelta to a number of seconds as an int.
    """
    return int(delta.total_seconds())


def timedelta_to_integral_minutes(delta):
    """
    Convert a pd.Timedelta to a number of minutes as an int.
    """
    return timedelta_to_integral_seconds(delta) // 60


# Remove when we drop support for 0.17
if pandas_version >= StrictVersion('0.18'):
    def rolling_mean(arg,
                     window,
                     min_periods=None,
                     freq=None,
                     center=False,
                     **kwargs):
        return arg.rolling(
            window,
            min_periods=min_periods,
            freq=freq,
            center=center,
            **kwargs
        ).mean()

    def rolling_apply(arg,
                      window,
                      func,
                      min_periods=None,
                      freq=None,
                      center=False,
                      **kwargs):
        return arg.rolling(
            window,
            min_periods=min_periods,
            freq=freq,
            center=center,
            **kwargs
        ).apply(func)

    def ewma(arg,
             com=None,
             span=None,
             halflife=None,
             alpha=None,
             min_periods=0,
             freq=None,
             adjust=True,
             how=None,
             ignore_na=False):

        return arg.ewm(
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            freq=freq,
            adjust=adjust,
            ignore_na=ignore_na,
        ).mean()

    def ewmstd(arg,
               com=None,
               span=None,
               halflife=None,
               alpha=None,
               min_periods=0,
               freq=None,
               adjust=True,
               how=None,
               ignore_na=False):

        return arg.ewm(
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            freq=freq,
            adjust=adjust,
            ignore_na=ignore_na,
        ).std()

else:
    rolling_mean = pd.rolling_mean
    rolling_apply = pd.rolling_apply
    ewma = pd.ewma
    ewmstd = pd.ewmstd
