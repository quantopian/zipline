"""
Utilities for working with pandas objects.
"""
from contextlib import contextmanager
from copy import deepcopy
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


def categorical_df_concat(df_list, inplace=False):
    """
    Prepare list of pandas DataFrames to be used as input to pd.concat.
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

    categorical_columns = df.columns[df.dtypes == 'category']

    for col in categorical_columns:
        new_categories = sorted(
            set().union(
                *(frame[col].cat.categories for frame in df_list)
            )
        )

        with ignore_pandas_nan_categorical_warning():
            for df in df_list:
                df[col].cat.set_categories(new_categories, inplace=True)

    return pd.concat(df_list)


def sliding_apply(df, window_length, f, min_periods=None):
    """
    Apply a function over rolling windows of a dataframe. This is different
    than pd.DataFrame.rolling().apply() because the function given to
    ``apply`` is not fed the entire dataframe view, only individual columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe over which to roll the given function.
    window_length : int
        The number of rows to view in each iteration.
    f : function[pd.DataFrame --> XXX]
        Function mapping each dataframe window to the desired output.
    min_periods : int, optional
        The minimum number of rows required to perform the operation done by
        the given function. If given, the number of rows in each iteration
        increments until reaching the ``window_length``, at which point the
        windows start to move along the dataframe as usual. If omitted, this
        value is set to the ``window_length``.

    Returns
    -------
    function_outputs : generator
        Iterable of the return values of ``f`` called on each window.

    Notes
    -----
    For some reason pandas errors when sliding over a dataframe with a
    RangeIndex, so this function currently fails for dataframes with a
    RangeIndex.
    """
    if min_periods is None:
        min_periods = window_length
    elif min_periods > window_length:
        raise ValueError(
            "'min_periods' argument ({0}) can't be more than the given window "
            "length ({1}).".format(min_periods, window_length)
        )

    slider = pd.lib.BlockSlider(df)
    num_rows = len(df)

    while min_periods < window_length:
        slider.move(0, min_periods)
        min_periods += 1
        yield f(slider.dummy)
        if min_periods > num_rows:
            return

    for start in range(num_rows - (window_length - 1)):
        slider.move(start, start + window_length)
        yield f(slider.dummy)
