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

def grouped_ffilled_reindex(df, index, group_columns, assets):
    """Perform a groupwise reindex(method='ffill') for a dataframe without
    altering the dtypes of columns. Any row which would have a `nan` written
    is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to reindex groupwise.
    index : pd.Index
        The new index per group.
    group_columns : str or list[str]
        The group_columns of column or columns to group by.

    Returns
    -------
    grouped_reindexed : pd.DataFrame
        The result of the grouping, reindexing, and forward filling.

    Examples
    --------
    >>> df = df = pd.DataFrame(
    ...     [[1, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 6], [3, 7]],
    ...     columns=['key', 'col'],
    ...     index=pd.Index([0, 1, 0, 1, 2, 1, 2])
    ... )
    >>> df
       key  col
    0    1    1
    1    1    2
    0    2    3
    1    2    4
    2    2    5
    1    3    6
    2    3    7
    >>> grouped_ffilled_reindex(df, [1, 2], 'key')
       key  col
    1    1    2
    2    1    2
    1    2    4
    2    2    5
    1    3    6
    2    3    7
    """
    groups = df.groupby(group_columns).indices
    # The output arrays and nan mask are preallocated to
    # ``len(index) * len(groups)`` because that is the maximum size possible
    # if we don't need to mask out any missing values. This also makes it
    # easy to fill by shifting our indices by ``len(index)`` per group.
    out_len = len(index) * len(groups)
    out_columns = {
        column: np.empty(out_len, dtype=df.dtypes[column])
        for column in df.columns
    }
    # In our reindex we will never write ``nan``, instead we will use a mask
    # array to filter out any missing rows before returning the final
    # dataframe.
    mask = np.empty(out_len, dtype=bool)
    # It is much faster to perform our operations on an ndarray per column
    # instead of the series so we expand our input dataframe into a dict
    # mapping string column name to the ndarray for that column.
    in_columns = {
        column: df[column].values
        for column in df.columns
    }

    for n, group_ix in enumerate(groups.values()):
        # ``group_ix`` is an array with all of the integer indices for the
        # elements of a single group.

        # The data for each group is written into shared column and mask arrays
        # so we adjust all the indices by group_number * len(index).
        offset = n * len(index)

        # Get the indices for the reindex.
        where = df.index[group_ix].get_indexer_for(index, method='ffill')

        # Any value which would have a ``nan`` written has an index of ``-1``
        # in ``where``. We mask out the ``nan`` values so that we can just
        # resize the output buffer once before creating the dataframe.
        group_mask = where != -1
        mask[offset:offset + len(index)] = group_mask

        for column, out_buf in out_columns.items():
            # For each column, select from the input array with the indices
            # computed for the reindex and write the result to a slice of our
            # preallocated output column array.
            in_columns[column][group_ix].take(
                where,
                out=out_buf[offset:offset + len(index)],
            )

    return pd.DataFrame(
        # Apply our mask to each of the output columns.
        {name: buf[mask] for name, buf in out_columns.items()},
        # The full output index is the new index tiled ``len(groups)`` times.
        # To get the actual output index we then slice based on our mask.
        index=np.tile(index, len(groups))[mask],
        # Ensure our output columns are in the same order as our input columns.
        columns=df.columns,
    )


def flat_last_in_date_group(df, dates, group_columns, assets):
    """Compute the last forward filled value in each date group.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe of non forward filled records.
    dates : pd.Index
        The date index to align the timestamps to.

    Returns
    -------
    ffilled : pd.DataFrame
        The correctly forward filled dataframe.
    """
    idx = [
        dates[dates.searchsorted(
            df[TS_FIELD_NAME].values.astype('datetime64[D]')
        )],
    ] + group_columns

    last_in_group = df.drop(TS_FIELD_NAME, axis=1).groupby(
        idx,
        sort=False,
    ).last()
    sids_to_add = pd.DataFrame(
        columns=[SID_FIELD_NAME], data=list(set(df.sid) - assets)
    )
    last_in_group.reset_index(group_columns, inplace=True)
    last_in_group = pd.concat([last_in_group, sids_to_add])
    last_in_group = grouped_ffilled_reindex(
        last_in_group,
        dates,
        group_columns,
        assets,
    )
    last_in_group.reset_index(inplace=True)
    last_in_group.rename(columns={'index': TS_FIELD_NAME}, inplace=True)
    return last_in_group


def last_in_date_group(df,
                       dates,
                       assets,
                       reindex=True,
                       have_sids=True,
                       extra_groupers=None):
    """
    Determine the last piece of information known on each date in the date
    index for each group. Input df MUST be sorted such that the correct last
    item is chosen from each group.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be grouped. Must be sorted so that
        the correct last item is chosen from each group.
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
    if extra_groupers is None:
        extra_groupers = []
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
