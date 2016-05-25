"""
Utilities for working with pandas objects.
"""
import operator as op

import pandas as pd


def explode(df):
    """
    Take a DataFrame and return a triple of

    (df.index, df.columns, df.values)
    """
    return df.index, df.columns, df.values


try:
    # pandas 0.16 compat
    _df_sort_values = pd.DataFrame.sort_values
    _series_sort_values = pd.Series.sort_values
except AttributeError:
    _df_sort_values = pd.DataFrame.sort
    _series_sort_values = pd.Series.sort


def sort_values(ob, *args, **kwargs):
    if isinstance(ob, pd.DataFrame):
        return _df_sort_values(ob, *args, **kwargs)
    elif isinstance(ob, pd.Series):
        return _series_sort_values(ob, *args, **kwargs)
    raise ValueError(
        'sort_values expected a dataframe or series, not %s: %r' % (
            type(ob).__name__, ob,
        ),
    )


def _time_to_micros(time):
    """Convert a time into milliseconds since midnight.

    Parameters
    ----------
    time : datetime.time
        The time to convert.

    Returns
    -------
    ms : int
        The number of milliseconds since midnight.
    """
    seconds = time.hour * 60 * 60 + 60 * time.minute + time.second
    return 1000000 * seconds + time.microsecond


_opmap = {
    (True, True): (op.le, op.le),
    (True, False): (op.le, op.lt),
    (False, True): (op.lt, op.le),
    (False, False): (op.lt, op.lt),
}


def mask_between_time(dts, start, end, include_start=True, include_end=True):
    """Return a mask of all of the datetimes in ``dts`` that are between
    ``start`` and ``end``.

    Parameters
    ----------
    dts : pd.DatetimeIndex
        The index to mask.
    start : time
    end : time
        The start and end times.
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
    :meth:`pandas.DatetimeIndex.indexer_between_times`
    """
    time_micros = dts._get_time_micros()
    start_micros = _time_to_micros(start)
    end_micros = _time_to_micros(end)

    lop, rop = _opmap[include_start, include_end]
    if start_micros <= end_micros:
        join_op = op.and_
    else:
        join_op = op.or_

    return join_op(
        lop(start_micros, time_micros),
        rop(time_micros, end_micros),
    )
