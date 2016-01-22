import datetime

import numpy as np
import pandas as pd
from six import iteritems
from six.moves import zip

from zipline.utils.numpy_utils import NaTns


def next_date_frame(dates, events_by_sid):
    """
    Make a DataFrame representing the simulated next known date for an event.

    Parameters
    ----------
    dates : pd.DatetimeIndex.
        The index of the returned DataFrame.
    events_by_sid : dict[int -> pd.Series]
        Dict mapping sids to a series of dates. Each k:v pair of the series
        represents the date we learned of the event mapping to the date the
        event will occur.
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
    cols = {
        equity: np.full_like(dates, NaTns) for equity in events_by_sid
    }
    raw_dates = dates.values
    for equity, event_dates in iteritems(events_by_sid):
        data = cols[equity]
        if not event_dates.index.is_monotonic_increasing:
            event_dates = event_dates.sort_index()

        # Iterate over the raw Series values, since we're comparing against
        # numpy arrays anyway.
        iterkv = zip(event_dates.index.values, event_dates.values)
        for knowledge_date, event_date in iterkv:
            date_mask = (
                (knowledge_date <= raw_dates) &
                (raw_dates <= event_date)
            )
            value_mask = (event_date <= data) | (data == NaTns)
            data[date_mask & value_mask] = event_date

    return pd.DataFrame(index=dates, data=cols)


def previous_date_frame(date_index, events_by_sid):
    """
    Make a DataFrame representing simulated next earnings date_index.

    Parameters
    ----------
    date_index : DatetimeIndex.
        The index of the returned DataFrame.
    events_by_sid : dict[int -> DatetimeIndex]
        Dict mapping sids to a series of dates. Each k:v pair of the series
        represents the date we learned of the event mapping to the date the
        event will occur.

    Returns
    -------
    previous_events: pd.DataFrame
        A DataFrame where each column is a security from `events_by_sid` where
        the values are the dates of the previous event that occured on the date
        of the index. Entries falling before the first date will have `NaT` as
        the result in the output.

    See Also
    --------
    next_date_frame
    """
    sids = list(events_by_sid)
    out = np.full((len(date_index), len(sids)), NaTns, dtype='datetime64[ns]')
    dn = date_index[-1].asm8
    for col_idx, sid in enumerate(sids):
        # events_by_sid[sid] is Series mapping knowledge_date to actual
        # event_date.  We don't care about the knowledge date for
        # computing previous earnings.
        values = events_by_sid[sid].values
        values = values[values <= dn]
        out[date_index.searchsorted(values), col_idx] = values

    frame = pd.DataFrame(out, index=date_index, columns=sids)
    frame.ffill(inplace=True)
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
