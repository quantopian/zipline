import datetime

import numpy as np
import pandas as pd
from zipline.pipeline.common import TS_FIELD_NAME, SID_FIELD_NAME
from zipline.utils.numpy_utils import categorical_dtype
from zipline.utils.pandas_utils import mask_between_time


def is_sorted_ascending(a):
    """Check if a numpy array is sorted."""
    return (np.fmax.accumulate(a) <= a).all()


def validate_event_metadata(event_dates,
                            event_timestamps,
                            event_sids):
    assert is_sorted_ascending(event_dates), "event dates must be sorted"
    assert len(event_sids) == len(event_dates) == len(event_timestamps), \
        "mismatched arrays: %d != %d != %d" % (
            len(event_sids),
            len(event_dates),
            len(event_timestamps),
        )


def next_event_indexer(all_dates,
                       all_sids,
                       event_dates,
                       event_timestamps,
                       event_sids):
    """
    Construct an index array that, when applied to an array of values, produces
    a 2D array containing the values associated with the next event for each
    sid at each moment in time.

    Locations where no next event was known will be filled with -1.

    Parameters
    ----------
    all_dates : ndarray[datetime64[ns], ndim=1]
        Row labels for the target output.
    all_sids : ndarray[int, ndim=1]
        Column labels for the target output.
    event_dates : ndarray[datetime64[ns], ndim=1]
        Dates on which each input events occurred/will occur.  ``event_dates``
        must be in sorted order, and may not contain any NaT values.
    event_timestamps : ndarray[datetime64[ns], ndim=1]
        Dates on which we learned about each input event.
    event_sids : ndarray[int, ndim=1]
        Sids assocated with each input event.

    Returns
    -------
    indexer : ndarray[int, ndim=2]
        An array of shape (len(all_dates), len(all_sids)) of indices into
        ``event_{dates,timestamps,sids}``.
    """
    validate_event_metadata(event_dates, event_timestamps, event_sids)
    out = np.full((len(all_dates), len(all_sids)), -1, dtype=np.int64)

    sid_ixs = all_sids.searchsorted(event_sids)
    # side='right' here ensures that we include the event date itself
    # if it's in all_dates.
    dt_ixs = all_dates.searchsorted(event_dates, side='right')
    ts_ixs = all_dates.searchsorted(event_timestamps)

    # Walk backward through the events, writing the index of the event into
    # slots ranging from the event's timestamp to its asof.  This depends for
    # correctness on the fact that event_dates is sorted in ascending order,
    # because we need to overwrite later events with earlier ones if their
    # eligible windows overlap.
    for i in range(len(event_sids) - 1, -1, -1):
        start_ix = ts_ixs[i]
        end_ix = dt_ixs[i]
        out[start_ix:end_ix, sid_ixs[i]] = i

    return out


def previous_event_indexer(all_dates,
                           all_sids,
                           event_dates,
                           event_timestamps,
                           event_sids):
    """
    Construct an index array that, when applied to an array of values, produces
    a 2D array containing the values associated with the previous event for
    each sid at each moment in time.

    Locations where no previous event was known will be filled with -1.

    Parameters
    ----------
    all_dates : ndarray[datetime64[ns], ndim=1]
        Row labels for the target output.
    all_sids : ndarray[int, ndim=1]
        Column labels for the target output.
    event_dates : ndarray[datetime64[ns], ndim=1]
        Dates on which each input events occurred/will occur.  ``event_dates``
        must be in sorted order, and may not contain any NaT values.
    event_timestamps : ndarray[datetime64[ns], ndim=1]
        Dates on which we learned about each input event.
    event_sids : ndarray[int, ndim=1]
        Sids assocated with each input event.

    Returns
    -------
    indexer : ndarray[int, ndim=2]
        An array of shape (len(all_dates), len(all_sids)) of indices into
        ``event_{dates,timestamps,sids}``.
    """
    validate_event_metadata(event_dates, event_timestamps, event_sids)
    out = np.full((len(all_dates), len(all_sids)), -1, dtype=np.int64)

    eff_dts = np.maximum(event_dates, event_timestamps)
    sid_ixs = all_sids.searchsorted(event_sids)
    dt_ixs = all_dates.searchsorted(eff_dts)

    # Walk backwards through the events, writing the index of the event into
    # slots ranging from max(event_date, event_timestamp) to the start of the
    # previously-written event.  This depends for correctness on the fact that
    # event_dates is sorted in ascending order, because we need to have written
    # later events so we know where to stop forward-filling earlier events.
    last_written = {}
    for i in range(len(event_dates) - 1, -1, -1):
        sid_ix = sid_ixs[i]
        dt_ix = dt_ixs[i]
        out[dt_ix:last_written.get(sid_ix, None), sid_ix] = i
        last_written[sid_ix] = dt_ix
    return out


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


_midnight = datetime.time(0, 0)


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
    to_roll_forward = mask_between_time(
        dtidx_local_time,
        time,
        _midnight,
        include_end=False,
    )
    # For all of the times that are greater than our query time add 1
    # day and truncate to the date.
    # We normalize twice here because of a bug in pandas 0.16.1 that causes
    # tz_localize() to shift some timestamps by an hour if they are not grouped
    # together by DST/EST.
    df.loc[to_roll_forward, ts_field] = (
        dtidx_local_time[to_roll_forward] + datetime.timedelta(days=1)
    ).normalize().tz_localize(None).tz_localize('utc').normalize()

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


def last_in_date_group(df,
                       dates,
                       assets,
                       reindex=True,
                       have_sids=True,
                       extra_groupers=[]):
    """
    Determine the last piece of information known on each date in the date
    index for each group.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be grouped.
    dates : pd.DatetimeIndex
        The dates to use for grouping and reindexing.
    assets : pd.Int64Index
        The assets that should be included in the column multiindex.
    reindex : bool
        Whether or not the DataFrame should be reindexed against the date
        index. This will add back any dates to the index that were grouped
        away.
    have_sids : bool
        Whether or not the DataFrame has sids. If it does, they will be used
        in the groupby.
    extra_groupers : list of str
        Any extra field names that should be included in the groupby.

    Returns
    -------
    last_in_group : pd.DataFrame
        A DataFrame with dates as the index and fields used in the groupby as
        levels of a multiindex of columns.

    """
    idx = [dates[dates.searchsorted(
        df[TS_FIELD_NAME].values.astype('datetime64[D]')
    )]]
    if have_sids:
        idx += [SID_FIELD_NAME]
    idx += extra_groupers

    last_in_group = df.drop(TS_FIELD_NAME, axis=1).groupby(
        idx,
        sort=False,
    ).last()

    # For the number of things that we're grouping by (except TS), unstack
    # the df. Done this way because of an unresolved pandas bug whereby
    # passing a list of levels with mixed dtypes to unstack causes the
    # resulting DataFrame to have all object-type columns.
    for _ in range(len(idx) - 1):
        last_in_group = last_in_group.unstack(-1)

    if reindex:
        if have_sids:
            cols = last_in_group.columns
            last_in_group = last_in_group.reindex(
                index=dates,
                columns=pd.MultiIndex.from_product(
                    tuple(cols.levels[0:len(extra_groupers) + 1]) + (assets,),
                    names=cols.names,
                ),
            )
        else:
            last_in_group = last_in_group.reindex(dates)

    return last_in_group


def ffill_across_cols(df, columns, name_map):
    """
    Forward fill values in a DataFrame with special logic to handle cases
    that pd.DataFrame.ffill cannot and cast columns to appropriate types.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to do forward-filling on.
    columns : list of BoundColumn
        The BoundColumns that correspond to columns in the DataFrame to which
        special filling and/or casting logic should be applied.
    name_map: map of string -> string
        Mapping from the name of each BoundColumn to the associated column
        name in `df`.
    """
    df.ffill(inplace=True)

    # Fill in missing values specified by each column. This is made
    # significantly more complex by the fact that we need to work around
    # two pandas issues:

    # 1) When we have sids, if there are no records for a given sid for any
    #    dates, pandas will generate a column full of NaNs for that sid.
    #    This means that some of the columns in `dense_output` are now
    #    float instead of the intended dtype, so we have to coerce back to
    #    our expected type and convert NaNs into the desired missing value.

    # 2) DataFrame.ffill assumes that receiving None as a fill-value means
    #    that no value was passed.  Consequently, there's no way to tell
    #    pandas to replace NaNs in an object column with None using fillna,
    #    so we have to roll our own instead using df.where.
    for column in columns:
        column_name = name_map[column.name]
        # Special logic for strings since `fillna` doesn't work if the
        # missing value is `None`.
        if column.dtype == categorical_dtype:
            df[column_name] = df[
                column.name
            ].where(pd.notnull(df[column_name]),
                    column.missing_value)
        else:
            # We need to execute `fillna` before `astype` in case the
            # column contains NaNs and needs to be cast to bool or int.
            # This is so that the NaNs are replaced first, since pandas
            # can't convert NaNs for those types.
            df[column_name] = df[
                column_name
            ].fillna(column.missing_value).astype(column.dtype)
