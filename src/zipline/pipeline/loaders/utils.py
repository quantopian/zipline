import numpy as np
import pandas as pd
from zipline.errors import NoFurtherDataError
from zipline.pipeline.common import TS_FIELD_NAME, SID_FIELD_NAME
from zipline.utils.date_utils import make_utc_aware
from zipline.utils.numpy_utils import categorical_dtype


def is_sorted_ascending(a):
    """Check if a numpy array is sorted."""
    return (np.fmax.accumulate(a) <= a).all()


def validate_event_metadata(event_dates, event_timestamps, event_sids):
    assert is_sorted_ascending(event_dates), "event dates must be sorted"
    assert (
        len(event_sids) == len(event_dates) == len(event_timestamps)
    ), "mismatched arrays: %d != %d != %d" % (
        len(event_sids),
        len(event_dates),
        len(event_timestamps),
    )


def next_event_indexer(
    all_dates, data_query_cutoff, all_sids, event_dates, event_timestamps, event_sids
):
    """
    Construct an index array that, when applied to an array of values, produces
    a 2D array containing the values associated with the next event for each
    sid at each moment in time.

    Locations where no next event was known will be filled with -1.

    Parameters
    ----------
    all_dates : ndarray[datetime64[ns], ndim=1]
        Row labels for the target output.
    data_query_cutoff : pd.DatetimeIndex
        The boundaries for the given trading sessions in ``all_dates``.
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
    dt_ixs = all_dates.searchsorted(pd.DatetimeIndex(event_dates), side="right")
    ts_ixs = data_query_cutoff.searchsorted(
        # pd.to_datetime(event_timestamps, utc=True), side="right"
        make_utc_aware(pd.DatetimeIndex(event_timestamps)),
        side="right",
    )

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


def previous_event_indexer(
    data_query_cutoff_times, all_sids, event_dates, event_timestamps, event_sids
):
    """
    Construct an index array that, when applied to an array of values, produces
    a 2D array containing the values associated with the previous event for
    each sid at each moment in time.

    Locations where no previous event was known will be filled with -1.

    Parameters
    ----------
    data_query_cutoff : pd.DatetimeIndex
        The boundaries for the given trading sessions.
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
    out = np.full(
        (len(data_query_cutoff_times), len(all_sids)),
        -1,
        dtype=np.int64,
    )

    eff_dts = np.maximum(event_dates, event_timestamps)
    sid_ixs = all_sids.searchsorted(event_sids)
    dt_ixs = data_query_cutoff_times.searchsorted(
        # pd.to_datetime(eff_dts, utc=True), side="right"
        make_utc_aware(pd.DatetimeIndex(eff_dts)),
        side="right",
    )

    # Walk backwards through the events, writing the index of the event into
    # slots ranging from max(event_date, event_timestamp) to the start of the
    # previously-written event.  This depends for correctness on the fact that
    # event_dates is sorted in ascending order, because we need to have written
    # later events so we know where to stop forward-filling earlier events.
    last_written = {}
    for i in range(len(event_dates) - 1, -1, -1):
        sid_ix = sid_ixs[i]
        dt_ix = dt_ixs[i]
        out[dt_ix : last_written.get(sid_ix, None), sid_ix] = i
        last_written[sid_ix] = dt_ix
    return out


def last_in_date_group(
    df,
    data_query_cutoff_times,
    assets,
    reindex=True,
    have_sids=True,
    extra_groupers=None,
):
    """
    Determine the last piece of information known on each date in the date
    index for each group. Input df MUST be sorted such that the correct last
    item is chosen from each group.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be grouped. Must be sorted so that
        the correct last item is chosen from each group.
    data_query_cutoff_times : pd.DatetimeIndex
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
    # get positions in `data_query_cutoff_times` just before `TS_FIELD_NAME` in `df`
    idx_before_ts = data_query_cutoff_times.searchsorted(
        make_utc_aware(pd.DatetimeIndex(df[TS_FIELD_NAME]))
    )
    idx = [data_query_cutoff_times[idx_before_ts]]

    if have_sids:
        idx += [SID_FIELD_NAME]
    if extra_groupers is None:
        extra_groupers = []
    idx += extra_groupers

    to_unstack = idx[-1 : -len(idx) : -1]
    last_in_group = (
        df.drop(TS_FIELD_NAME, axis=1)
        .groupby(idx, sort=False)
        .last()
        .unstack(level=to_unstack)
    )

    # For the number of things that we're grouping by (except TS), unstack
    # the df. Done this way because of an unresolved pandas bug whereby
    # passing a list of levels with mixed dtypes to unstack causes the
    # resulting DataFrame to have all object-type columns.
    # for _ in range(len(idx) - 1):
    #     last_in_group = last_in_group.unstack(-1)

    if reindex:
        if have_sids:
            cols = last_in_group.columns
            columns = pd.MultiIndex.from_product(
                tuple(cols.levels[0 : len(extra_groupers) + 1]) + (assets,),
                names=cols.names,
            )
            last_in_group = last_in_group.reindex(
                index=data_query_cutoff_times, columns=columns
            )
        else:
            last_in_group = last_in_group.reindex(data_query_cutoff_times)

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
            df[column_name] = df[column.name].where(
                pd.notnull(df[column_name]), column.missing_value
            )
        else:
            # We need to execute `fillna` before `astype` in case the
            # column contains NaNs and needs to be cast to bool or int.
            # This is so that the NaNs are replaced first, since pandas
            # can't convert NaNs for those types.
            df[column_name] = (
                df[column_name].fillna(column.missing_value).astype(column.dtype)
            )


def shift_dates(dates, start_date, end_date, shift):
    """Shift dates of a pipeline query back by ``shift`` days.

    Parameters
    ----------
    dates : DatetimeIndex
        All known dates.
    start_date : pd.Timestamp
        Start date of the pipeline query.
    end_date : pd.Timestamp
        End date of the pipeline query.
    shift : int
        The number of days to shift back the query dates.

    Returns
    -------
    shifted : pd.DatetimeIndex
        The range [start_date, end_date] from ``dates``, shifted backwards by
        ``shift`` days.

    Raises
    ------
    ValueError
        If ``start_date`` or ``end_date`` is not in ``dates``.
    NoFurtherDataError
        If shifting ``start_date`` back by ``shift`` days would push it off the
        end of ``dates``.
    """
    try:
        start = dates.get_loc(start_date)
    except KeyError as exc:
        if start_date < dates[0]:
            raise NoFurtherDataError(
                msg=(
                    "Pipeline Query requested data starting on {query_start}, "
                    "but first known date is {calendar_start}"
                ).format(
                    query_start=str(start_date),
                    calendar_start=str(dates[0]),
                )
            ) from exc
        else:
            raise ValueError(f"Query start {start_date} not in calendar") from exc

    # Make sure that shifting doesn't push us out of the calendar.
    if start < shift:
        raise NoFurtherDataError(
            msg=(
                "Pipeline Query requested data from {shift}"
                " days before {query_start}, but first known date is only "
                "{start} days earlier."
            ).format(shift=shift, query_start=start_date, start=start),
        )

    try:
        end = dates.get_loc(end_date)
    except KeyError as exc:
        if end_date > dates[-1]:
            raise NoFurtherDataError(
                msg=(
                    "Pipeline Query requesting data up to {query_end}, "
                    "but last known date is {calendar_end}"
                ).format(
                    query_end=end_date,
                    calendar_end=dates[-1],
                )
            ) from exc
        else:
            raise ValueError("Query end %s not in calendar" % end_date) from exc

    return dates[start - shift : end - shift + 1]  # +1 to be inclusive
