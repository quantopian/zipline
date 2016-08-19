import datetime

import numpy as np
import pandas as pd
from six import iteritems
from six.moves import zip

from zipline.utils.numpy_utils import categorical_dtype, NaTns


def next_event_frame(events_by_sid,
                     dates,
                     missing_value,
                     field_dtype,
                     event_date_field_name,
                     return_field_name):
    """
    Make a DataFrame representing the simulated next known dates or values
    for an event.

    Parameters
    ----------
    dates : pd.DatetimeIndex.
        The index of the returned DataFrame.
    events_by_sid : dict[int -> pd.Series]
        Dict mapping sids to a series of dates. Each k:v pair of the series
        represents the date we learned of the event mapping to the date the
        event will occur.
    event_date_field_name : str
        The name of the date field that marks when the event occurred.

    Returns
    -------
    next_events: pd.DataFrame
        A DataFrame where each column is a security from `events_by_sid` where
        the values are the dates of the next known event with the knowledge we
        had on the date of the index. Entries falling after the last date will
        have `NaT` as the result in the output.


    See Also
    --------
    previous_date_frame
    """
    date_cols = {
        equity: np.full_like(dates, NaTns) for equity in events_by_sid
    }
    value_cols = {
        equity: np.full(len(dates), missing_value, dtype=field_dtype)
        for equity in events_by_sid
    }

    raw_dates = dates.values
    for equity, df in iteritems(events_by_sid):
        event_dates = df[event_date_field_name]
        values = df[return_field_name]
        data = date_cols[equity]
        if not event_dates.index.is_monotonic_increasing:
            event_dates = event_dates.sort_index()

        # Iterate over the raw Series values, since we're comparing against
        # numpy arrays anyway.
        iter_date_vals = zip(event_dates.index.values, event_dates.values,
                             values)
        for knowledge_date, event_date, value in iter_date_vals:
            date_mask = (
                (knowledge_date <= raw_dates) &
                (raw_dates <= event_date)
            )
            value_mask = (event_date <= data) | (data == NaTns)
            data_indices = np.where(date_mask & value_mask)
            data[data_indices] = event_date
            value_cols[equity][data_indices] = value
    return pd.DataFrame(index=dates, data=value_cols)


def previous_event_frame(events_by_sid,
                         date_index,
                         missing_value,
                         field_dtype,
                         event_date_field,
                         previous_return_field):
    """
    Make a DataFrame representing simulated previous dates or values for an
    event.

    Parameters
    ----------
    events_by_sid : dict[int -> DatetimeIndex]
        Dict mapping sids to a series of dates. Each k:v pair of the series
        represents the date we learned of the event mapping to the date the
        event will occur.
    date_index : DatetimeIndex.
        The index of the returned DataFrame.
    missing_value : any
        Data which missing values should be filled with.
    field_dtype: any
        The dtype of the field for which the previous values are being
        retrieved.
    event_date_field: str
        The name of the date field that marks when the event occurred.
    return_field: str
        The name of the field for which the previous values are being
        retrieved.

    Returns
    -------
    previous_events: pd.DataFrame
        A DataFrame where each column is a security from `events_by_sid` and
        the values are the values for the previous event that occurred on the
        date of the index. Entries falling before the first date will have
        `missing_value` filled in as the result in the output.

    See Also
    --------
    next_date_frame
    """
    sids = list(events_by_sid)
    populate_value = None if field_dtype == categorical_dtype else \
        missing_value
    out = np.full(
        (len(date_index), len(sids)),
        populate_value,
        dtype=field_dtype
    )
    d_n = date_index[-1].asm8
    for col_idx, sid in enumerate(sids):
        # events_by_sid[sid] is a DataFrame mapping knowledge_date to event
        # date and values.
        df = events_by_sid[sid]
        df = df[df[event_date_field] <= d_n]
        event_date_vals = df[event_date_field].values
        # Get knowledge dates corresponding to the values in which we are
        # interested
        kd_vals = df[df[event_date_field] <= d_n].index.values
        # The date at which a previous event is first known is the max of the
        #  kd and the event date.
        index_dates = np.maximum(kd_vals, event_date_vals)
        out[
            date_index.searchsorted(index_dates), col_idx
        ] = df[previous_return_field]

    frame = pd.DataFrame(out, index=date_index, columns=sids)
    frame.ffill(inplace=True)
    if field_dtype == categorical_dtype:
        frame[frame.isnull()] = missing_value
    return frame


def normalize_data_query_time(dt, time, tz):
    """Apply the correct time and timezone to a date.

    Parameters
    ----------
    dt : pd.Timestamp
        The original datetime that represents the date.
    time : datetime.time
        The time of day to use as the cutoff point for new data. Data points
        that you learn about after this time will become available to your
        algorithm on the next trading day.
    tz : tzinfo
        The timezone to normalize your dates to before comparing against
        `time`.

    Returns
    -------
    query_dt : pd.Timestamp
        The timestamp with the correct time and date in utc.
    """
    # merge the correct date with the time in the given timezone then convert
    # back to utc
    return pd.Timestamp(
        datetime.datetime.combine(dt.date(), time),
        tz=tz,
    ).tz_convert('utc')


def normalize_data_query_bounds(lower, upper, time, tz):
    """Adjust the first and last dates in the requested datetime index based on
    the provided query time and tz.

    lower : pd.Timestamp
        The lower date requested.
    upper : pd.Timestamp
        The upper date requested.
    time : datetime.time
        The time of day to use as the cutoff point for new data. Data points
        that you learn about after this time will become available to your
        algorithm on the next trading day.
    tz : tzinfo
        The timezone to normalize your dates to before comparing against
        `time`.
    """
    # Subtract one day to grab things that happened on the first day we are
    # requesting. This doesn't need to be a trading day, we are only adding
    # a lower bound to limit the amount of in memory filtering that needs
    # to happen.
    lower -= datetime.timedelta(days=1)
    if time is not None:
        return normalize_data_query_time(
            lower,
            time,
            tz,
        ), normalize_data_query_time(
            upper,
            time,
            tz,
        )
    return lower, upper


def normalize_timestamp_to_query_time(df,
                                      time,
                                      tz,
                                      inplace=False,
                                      ts_field='timestamp'):
    """Update the timestamp field of a dataframe to normalize dates around
    some data query time/timezone.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update. This needs a column named ``ts_field``.
    time : datetime.time
        The time of day to use as the cutoff point for new data. Data points
        that you learn about after this time will become available to your
        algorithm on the next trading day.
    tz : tzinfo
        The timezone to normalize your dates to before comparing against
        `time`.
    inplace : bool, optional
        Update the dataframe in place.
    ts_field : str, optional
        The name of the timestamp field in ``df``.

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the timestamp field normalized. If ``inplace`` is
        true, then this will be the same object as ``df`` otherwise this will
        be a copy.
    """
    if not inplace:
        # don't mutate the dataframe in place
        df = df.copy()

    dtidx = pd.DatetimeIndex(df.loc[:, ts_field], tz='utc')
    dtidx_local_time = dtidx.tz_convert(tz)
    to_roll_forward = dtidx_local_time.time >= time
    # for all of the times that are greater than our query time add 1
    # day and truncate to the date
    df.loc[to_roll_forward, ts_field] = (
        dtidx_local_time[to_roll_forward] + datetime.timedelta(days=1)
    ).normalize().tz_localize(None).tz_localize('utc')  # cast back to utc
    df.loc[~to_roll_forward, ts_field] = dtidx[~to_roll_forward].normalize()
    return df


def check_data_query_args(data_query_time, data_query_tz):
    """Checks the data_query_time and data_query_tz arguments for loaders
    and raises a standard exception if one is None and the other is not.

    Parameters
    ----------
    data_query_time : datetime.time or None
    data_query_tz : tzinfo or None

    Raises
    ------
    ValueError
        Raised when only one of the arguments is None.
    """
    if (data_query_time is None) ^ (data_query_tz is None):
        raise ValueError(
            "either 'data_query_time' and 'data_query_tz' must both be"
            " None or neither may be None (got %r, %r)" % (
                data_query_time,
                data_query_tz,
            ),
        )


def zip_with_floats(dates, flts):
        return pd.Series(flts, index=dates, dtype='float')


def zip_with_strs(dates, strs):
        return pd.Series(strs, index=dates, dtype='object')


def zip_with_dates(index_dates, dts):
    return pd.Series(pd.to_datetime(dts), index=index_dates)


def get_values_for_date_ranges(zip_date_index_with_vals,
                               vals_for_date_intervals,
                               starts,
                               ends,
                               date_index):
    """
    Returns a Series of values indexed by date based on the intervals defined
    by the start and end dates.

    Parameters
    ----------
    zip_date_index_with_vals : callable
        A function that takes in a list of dates and a list of values and
        returns a pd.Series with the values indexed by the dates.
    vals_for_date_intervals : list
        A list of values for each date interval in `date_intervals`.
    starts : DatetimeIndex
        A DatetimeIndex of start dates.
    ends : list
        A DatetimeIndex of end dates.
    date_index : DatetimeIndex
        The DatetimeIndex containing all dates for which values were requested.

    Returns
    -------
    date_index_with_vals : pd.Series
        A Series indexed by the given DatetimeIndex and with values assigned
        to dates based on the given date intervals.
    """
    # Fill in given values for given date ranges.
    end_indexes = date_index.values.searchsorted(ends)
    start_indexes = date_index.values.searchsorted(starts)
    num_days = (end_indexes - start_indexes) + 1

    # In case any of the end dates falls on days missing from the date_index,
    # searchsorted will have placed their index within `date_index` to the
    # index of the next start date, so we will have added 1 extra day for
    # each of these. Subtract those extra days, but ignore any cases where the
    # start and end dates are equal. Note: if any of the start dates is
    # missing, it won't affect calculations because searchsorted will advance
    # the index to the next date within the same range.
    num_days[np.where(~np.in1d(ends, date_index) & (num_days != 0))] -= 1
    return zip_date_index_with_vals(
        date_index,
        np.repeat(
            vals_for_date_intervals,
            num_days,
        )
    )
