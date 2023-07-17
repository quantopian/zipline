"""
Utilities for working with pandas objects.
"""
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import operator as op
import warnings

import numpy as np
import pandas as pd
from packaging.version import Version
from zipline.utils.calendar_utils import days_at_time
from pandas.errors import PerformanceWarning

pandas_version = Version(pd.__version__)
new_pandas = pandas_version >= Version("0.19")
skip_pipeline_new_pandas = (
    "Pipeline categoricals are not yet compatible with pandas >=0.19"
)
skip_pipeline_blaze = "Blaze doesn't play nicely with Pandas >=1.0"


def july_5th_holiday_observance(datetime_index):
    return datetime_index[datetime_index.year != 2013]


def explode(df):
    """Take a DataFrame and return a triple of

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


_opmap = dict(
    zip(
        product((True, False), repeat=3),
        product((op.le, op.lt), (op.le, op.lt), (op.and_, op.or_)),
    )
)


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
    """Find the index of ``dt`` in ``dts``.

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
    if ix == len(dts) or dts[ix] != dt:
        raise LookupError("{dt} is not in {dts}".format(dt=dt, dts=dts))
    return ix


def nearest_unequal_elements(dts, dt):
    """Find values in ``dts`` closest but not equal to ``dt``.

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

    sortpos = dts.searchsorted(dt, side="left")
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
    """Convert a pd.Timedelta to a number of seconds as an int."""
    return int(delta.total_seconds())


def timedelta_to_integral_minutes(delta):
    """Convert a pd.Timedelta to a number of minutes as an int."""
    return timedelta_to_integral_seconds(delta) // 60


@contextmanager
def ignore_pandas_nan_categorical_warning():
    with warnings.catch_warnings():
        # Pandas >= 0.18 doesn't like null-ish values in categories, but
        # avoiding that requires a broader change to how missing values are
        # handled in pipeline, so for now just silence the warning.
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
        )
        yield


def categorical_df_concat(df_list, inplace=False):
    """Prepare list of pandas DataFrames to be used as input to pd.concat.
    Ensure any columns of type 'category' have the same categories across each
    dataframe.

    Parameters
    ----------
    df_list : list
        List of dataframes with same columns.
    inplace : bool
        True if input list can be modified. Default is False.

    Returns
    -------
    concatenated : df
        Dataframe of concatenated list.
    """

    if not inplace:
        df_list = deepcopy(df_list)

    # Assert each dataframe has the same columns/dtypes
    df = df_list[0]
    if not all([(df.dtypes.equals(df_i.dtypes)) for df_i in df_list[1:]]):
        raise ValueError("Input DataFrames must have the same columns/dtypes.")

    categorical_columns = df.columns[df.dtypes == "category"]

    for col in categorical_columns:
        new_categories = _sort_set_none_first(
            _union_all(frame[col].cat.categories for frame in df_list)
        )

        with ignore_pandas_nan_categorical_warning():
            for df in df_list:
                df[col].cat.set_categories(new_categories, inplace=True)

    return pd.concat(df_list)


def _union_all(iterables):
    """Union entries in ``iterables`` into a set."""
    return set().union(*iterables)


def _sort_set_none_first(set_):
    """Sort a set, sorting ``None`` before other elements, if present."""
    if None in set_:
        set_.remove(None)
        out = [None]
        out.extend(sorted(set_))
        set_.add(None)
        return out
    else:
        return sorted(set_)


def empty_dataframe(*columns):
    """Create an empty dataframe with columns of particular types.

    Parameters
    ----------
    *columns
        The (column_name, column_dtype) pairs.

    Returns
    -------
    typed_dataframe : pd.DataFrame
        The empty typed dataframe.

    Examples
    --------
    >>> df = empty_dataframe(
    ...     ('a', 'int64'),
    ...     ('b', 'float64'),
    ...     ('c', 'datetime64[ns]'),
    ... )

    >>> df
    Empty DataFrame
    Columns: [a, b, c]
    Index: []

    df.dtypes
    a             int64
    b           float64
    c    datetime64[ns]
    dtype: object
    """
    return pd.DataFrame(np.array([], dtype=list(columns)))


def check_indexes_all_same(indexes, message="Indexes are not equal."):
    """Check that a list of Index objects are all equal.

    Parameters
    ----------
    indexes : iterable[pd.Index]
        Iterable of indexes to check.

    Raises
    ------
    ValueError
        If the indexes are not all the same.
    """

    iterator = iter(indexes)
    first = next(iterator)
    for other in iterator:
        same = first == other
        if not same.all():
            bad_loc = np.flatnonzero(~same)[0]
            raise ValueError(
                "{}\nFirst difference is at index {}: "
                "{} != {}".format(message, bad_loc, first[bad_loc], other[bad_loc]),
            )
