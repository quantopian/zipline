"""
Utilities for working with pandas objects.
"""
from contextlib import contextmanager
from itertools import product
import operator as op
import warnings

import pandas as pd
from distutils.version import StrictVersion

pandas_version = StrictVersion(pd.__version__)


def july_5th_holiday_observance(datetime_index):
    return datetime_index[datetime_index.year != 2013]


def explode(df):
    """
    Take a DataFrame and return a triple of

    (df.index, df.columns, df.values)
    """
    return df.index, df.columns, df.values


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


def find_in_sorted_index(dts, dt):
    """
    Find the index of ``dt`` in ``dts``.

    This function should be used instead of `dts.get_loc(dt)` if the index is
    large enough that we don't want to initialize a hash table in ``dts``. In
    particular, this should always be used on minutely trading calendars.

    Parameters
    ----------
    dts : pd.DatetimeIndex
        Index in which to look up ``dt``. **Must be sorted**.
    dt : pd.Timestamp
        ``dt`` to be looked up.

    Returns
    -------
    ix : int
        Integer index such that dts[ix] == dt.

    Raises
    ------
    KeyError
        If dt is not in ``dts``.
    """
    ix = dts.searchsorted(dt)
    if dts[ix] != dt:
        raise LookupError("{dt} is not in {dts}".format(dt=dt, dts=dts))
    return ix


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


@contextmanager
def ignore_pandas_nan_categorical_warning():
    with warnings.catch_warnings():
        # Pandas >= 0.18 doesn't like null-ish values in catgories, but
        # avoiding that requires a broader change to how missing values are
        # handled in pipeline, so for now just silence the warning.
        warnings.filterwarnings(
            'ignore',
            category=FutureWarning,
        )
        yield


_INDEXER_NAMES = [
    '_' + name for (name, _) in pd.core.indexing.get_indexers_list()
]


def clear_dataframe_indexer_caches(df):
    """
    Clear cached attributes from a pandas DataFrame.

    By default pandas memoizes indexers (`iloc`, `loc`, `ix`, etc.) objects on
    DataFrames, resulting in refcycles that can lead to unexpectedly long-lived
    DataFrames. This function attempts to clear those cycles by deleting the
    cached indexers from the frame.

    Parameters
    ----------
    df : pd.DataFrame
    """
    for attr in _INDEXER_NAMES:
        try:
            delattr(df, attr)
        except AttributeError:
            pass
