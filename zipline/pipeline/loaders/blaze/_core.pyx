from cpython cimport (
    PyDict_GetItem,
    PyObject,
    PyList_New,
    PyList_SET_ITEM,
)
from bisect import bisect_right, insort_left

cimport cython
cimport numpy as np
import numpy as np
import pandas as pd
from toolz import sliding_window
from trading_calendars.utils.pandas_utils import days_at_time

from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment cimport (
    AdjustmentKind,
    DatetimeIndex_t,
    make_adjustment_from_indices_fused,
    column_type,
)
from zipline.lib.labelarray import LabelArray
from zipline.pipeline.common import (
    AD_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)


cdef bint isnan(np.float64_t value):
    # this isn't defined in libc on windows...
    return value != value


ctypedef bint is_missing_function(column_type, column_type)


cdef bint is_missing_value(column_type value, column_type missing_value):
    if column_type is np.uint8_t:
        # we want is_missing_value(bool) to return false so that we ffill
        # both True and False values
        return False
    elif column_type is np.float64_t and isnan(missing_value):
        return isnan(value)
    else:
        return value == missing_value


cdef inline unsafe_setslice_column(column_type[:, ::1] array,
                                   Py_ssize_t start_row,
                                   Py_ssize_t stop_row,
                                   Py_ssize_t col_ix,
                                   column_type value):
    cdef Py_ssize_t row_ix
    for row_ix in range(start_row, stop_row):
        with cython.boundscheck(False), cython.wraparound(False):
            array[row_ix, col_ix] = value


cdef _ffill_missing_value_2d_inplace(np.ndarray[column_type, ndim=2] array,
                                     column_type missing_value,
                                     list non_null_ts_ixs_by_column_ix):
    """Inplace forward fill in a missing value aware way.

    Parameters
    ----------
    array : np.ndarray
        The array to forward fill with shape (len(dates), len(assets)).
    missing_value : any
        The missing value for this array.
    non_null_ts_ixs_by_column_ix : list[set[int]]
        ``non_null_ts_ixs_by_column_ix[n]`` holds a list of the non null
        timestamp indices for the asset at column ``n``.
    """
    cdef Py_ssize_t start_ix
    cdef Py_ssize_t end_ix
    cdef set non_null_ixs_set
    cdef list non_null_ixs_list
    cdef Py_ssize_t column_ix

    for column_ix, non_null_ixs_set in enumerate(non_null_ts_ixs_by_column_ix):
        if not non_null_ixs_set:
            # no data was seen for this asset, all the rows are missing
            unsafe_setslice_column[column_type](
                array,
                0,
                len(array),
                column_ix,
                missing_value,
            )
            continue

        non_null_ixs_list = sorted(non_null_ixs_set)

        # fill the missing value up the the first non null index
        unsafe_setslice_column[column_type](
            array,
            0,
            non_null_ixs_list[0],
            column_ix,
            missing_value,
        )

        for start_ix, end_ix in sliding_window(2, non_null_ixs_list):
            # for each non null index, fill the value forward up to the next
            # non null index right exclusive
            unsafe_setslice_column[column_type](
                array,
                start_ix + 1,
                end_ix,
                column_ix,
                array[start_ix, column_ix],
            )


        # fill through to the end of the array
        unsafe_setslice_column[column_type](
            array,
            non_null_ixs_list[-1] + 1,
            len(array),
            column_ix,
            array[non_null_ixs_list[-1], column_ix],
        )


@cython.final
cdef class AsAdjustedArray:
    """Marker type for array_for_column which enables the AdjustedArray
    path.
    """


@cython.final
cdef class AsBaselineArray:
    """Marker type for array_for_column which enables the baseline array
    path.
    """


ctypedef fused AsArrayKind:
    AsAdjustedArray
    AsBaselineArray


cdef inline insert_non_null_ad_index(list non_null_ad_ixs,
                                     Py_ssize_t ix,
                                     object asof_ix):
    """Insert an asof date index into the non_null_ad_ixs list after a
    ``bisect_right``.

    Parameters
    ----------
    non_null_ad_ixs : list[int]
        The list of unique asof_date indices.
    ix : Py_ssize_t
        The result of ``bisect_right(non_null_ad_ixs, asof_ix)``.
    asof_ix : int
        The asof date index.

    Notes
    -----
    This saves the work of searching the list a second time and ensures
    that ``non_null_ad_ixs`` remains unique.
    """
    if ix == 0:
        non_null_ad_ixs.insert(0, asof_ix)
    elif non_null_ad_ixs[ix - 1] != asof_ix:
        non_null_ad_ixs.insert(ix, asof_ix)


cdef _array_for_column_impl(object dtype,
                            np.ndarray[column_type, ndim=2] out_array,
                            Py_ssize_t size,
                            np.ndarray[np.int64_t] ts_ixs,
                            np.ndarray[np.int64_t] asof_dates,
                            np.ndarray[np.int64_t] asof_ixs,
                            np.ndarray[np.int64_t] sids,
                            dict column_ixs,
                            np.ndarray[column_type] input_array,
                            column_type missing_value,
                            bint is_missing(column_type, column_type),
                            AsArrayKind _array_kind):
    """This is the core algorithm for formatting raw blaze data into the
    baseline adjustments format required for consumption by the Pipeline API.

    For performance reasons, we represent input data with parallel arrays.
    Logically, however, we think of each row of the input data as representing
    a single event. Each event carries four pieces of information:

      asof_date - The date on which the event occurred.
      timestamp - The date on which we learned about the event.
      sid       - The sid to which the event pertains.
      value     - The value of the column being updated by the event.

    The essential idea of this algorithm is to process events in
    timestamp-sorted order, updating our worldview as each new event
    arrives. This models how we would actually process events in real time.

    When we process a new event, we first check if the event should be
    processed:

    - We skip events pertaining to sids that weren't requested.
    - We skip events for which timestamp is after all the dates we're
      interested in.
    - We skip events whose `value` field is missing.

    Once we've decided an event is relevant, there are two possible cases:

    1. The event is **novel**, meaning that its asof_date is greater than or
       equal to the asof_date of all events with the same sid that have been
       processed so far.

    2. The event is **stale**, meaning that we've already processed an event
       with the same sid and a later asof_date.

    Novel events appear in the baseline starting at their timestamp and
    continuing until the timestamp of the next novel event. We build the
    baseline by slotting novel events into the baseline as they're received and
    forward-filling as a final step.

    Stale events never appear in the baseline. There's always a newer event to
    show by the time we reach a stale event's timestamp.

    Every event also has the possibility of generating an adjustment that
    updates prior historical values:

    - If an event is novel, we emit an adjustment updating all days in the
      right-open interval: [event.asof_date, event.timestamp). This reflects
      the fact that the new event is now the best known value for all days on
      or after its event.asof_date. The upper bound of the adjustment is
      event.timestamp because we've already marked the event as best-known on
      or after its timestamp by writing the event into the baseline.

    - If the event is stale, we emit an adjustment updating all days in the
      right-open interval: [event.asof_date, next_event.asof_date), where
      "next_event" is the latest (by asof) already-processed event whose
      asof_date is after the new event. This reflects the fact that the new
      event is now the best-known value for the period ranging from its asof to
      the next-known asof. Note that, by the definition of staleness,
      next_event must exist.
    """
    cdef column_type value
    cdef np.int64_t ts_ix
    cdef np.int64_t asof_ix
    cdef np.int64_t sid
    cdef np.int64_t column_ix

    cdef PyObject* adjustments_list_ptr
    cdef list adjustments_list

    cdef list non_null_ad_ixs
    cdef list non_null_ad_ixs_by_column_ix = [
        [] for _ in range(out_array.shape[1])
    ]

    cdef set non_null_ts_ixs
    cdef list non_null_ts_ixs_by_column_ix = [
        set() for _ in range(out_array.shape[1])
    ]

    cdef np.ndarray[np.int64_t, ndim=2] most_recent_asof_date_for_ix = np.full(
        (<object> out_array).shape,
        pd.Timestamp.min.value,
        dtype='int64',
    )

    cdef dict adjustments

    cdef Py_ssize_t out_of_bounds_ix = len(out_array)

    if AsArrayKind is AsAdjustedArray:
        adjustments = {}

    cdef set categories
    if column_type is object:
        # for object columns we need to maintain the unique values for the
        # categories
        categories = {missing_value}

    cdef Py_ssize_t n
    for n in range(size if len(out_array) else 0):
        with cython.boundscheck(False), cython.wraparound(False):
            value = input_array[n]

        if is_missing(value, missing_value):
            # skip missing values
            continue

        if column_type is object:
            # maintain the categories for the label array if we have an object
            # column
            categories.add(value)

        with cython.boundscheck(False), cython.wraparound(False):
            ts_ix = ts_ixs[n]
            if ts_ix == out_of_bounds_ix:
                # this timestamp falls after the last date requested
                continue

            sid = sids[n]

            asof_ix = asof_ixs[n]
            if asof_ix == out_of_bounds_ix:
                raise ValueError(
                    'asof_date newer than timestamp: sid=%s, asof_date=%s' % (
                        sid,
                        np.datetime64(asof_dates[n], 'ns'),
                    ),
                )

        column_ix_ob = PyDict_GetItem(column_ixs, sid)
        if column_ix_ob is NULL:
            # ignore sids that are not requested
            continue

        column_ix = <object> column_ix_ob  # cast to np.int64_t

        with cython.boundscheck(False), cython.wraparound(False):
            asof_date = asof_dates[n]
            if asof_date >= most_recent_asof_date_for_ix[asof_ix, column_ix]:
                # The asof_date is the same or more recent than the
                # last recorded asof_date at the given index and we should
                # treat this value as the best known row. We use >=
                # because a more recent row with the same asof_date
                # should be treated as an adjustment and the new value
                # becomes the best-known.
                most_recent_asof_date_for_ix[asof_ix, column_ix] = asof_date
            else:
                # The asof_date is earlier than the asof_date written
                # at the given index. Ignore this row.
                continue

        if AsArrayKind is AsAdjustedArray:
            # Grab the list of adjustments for this timestamp. If this is the
            # first time we've seen this timestamp, PyDict_GetItem will return
            # NULL, in which case we need to insert a new empty list.
            adjustment_list_ptr = PyDict_GetItem(adjustments, ts_ix)
            if adjustment_list_ptr is NULL:
                adjustment_list = adjustments[ts_ix] = []
            else:
                adjustment_list = <list> adjustment_list_ptr

        non_null_ad_ixs = non_null_ad_ixs_by_column_ix[column_ix]
        ix = bisect_right(non_null_ad_ixs, asof_ix)
        if ix == len(non_null_ad_ixs):
            # The row we're currently processing has the latest as_of we've
            # seen so far. It should become the new baseline value for its
            # sid and timestamp.
            with cython.boundscheck(False), cython.wraparound(False):
                out_array[ts_ix, column_ix] = value

            if AsArrayKind is AsAdjustedArray:
                # We need to emit an adjustment if there's at least one output
                # day in the interval [event.asof_date, event.timestamp). The
                # upper bound doesn't include the timestamp because we've
                # already included the timestamp-date in the baseline.
                end = max(ts_ix - 1, 0)
                if end >= asof_ix:
                    # The above condition ensures that the adjustment spans at
                    # least one trading day, meaning it has an effect on the
                    # displayed data. We cannot construct adjustments where end
                    # < start so we can just skip these rows.
                    adjustment_list.append(
                        make_adjustment_from_indices_fused[column_type](
                            asof_ix,
                            end,
                            column_ix,
                            column_ix,
                            AdjustmentKind.OVERWRITE,
                            value,
                        ),
                    )

            # collect this information for forward filling
            non_null_ts_ixs = non_null_ts_ixs_by_column_ix[column_ix]
            non_null_ts_ixs.add(ts_ix)
        elif AsArrayKind is AsAdjustedArray:
            # The row we're currently processing has an asof_date earlier than
            # at least one row that we learned about before this row.
            #
            # This happens when the order that we received a sequence of events
            # doesn't match the order in which the events occurred.
            # For example:
            #
            # asof  sid timestamp  value
            #   t2    1        t5     v1
            #   t0    1        t6     v0
            #
            # On t5, we learn that value was v1 on t2.
            # On t6, we learn that value was v0 on t0.
            #
            # v0 should never appear in the baseline, because by the time we
            # learn about it, we'll have already learned about the newer value
            # of v1. However, if we look back from t6, we should see v0 for the
            # period from t0 to t1.
            end = max(non_null_ad_ixs[ix] - 1, 0)
            if end >= asof_ix:
                # see comment above about why we are not emitting some of
                # these adjustments
                adjustment_list.append(
                    make_adjustment_from_indices_fused[column_type](
                        asof_ix,
                        end,
                        column_ix,
                        column_ix,
                        AdjustmentKind.OVERWRITE,
                        value,
                    ),
                )

        # Remember that we've seen a data point for this sid on asof.
        insert_non_null_ad_index(non_null_ad_ixs, ix, asof_ix)

    _ffill_missing_value_2d_inplace(
        out_array,
        missing_value,
        non_null_ts_ixs_by_column_ix,
    )

    if column_type is object:
        baseline_array = LabelArray(
            out_array,
            missing_value,
            categories,
            sort=False,
        )
    else:
        # View the baseline array as the correct dtype. We work with
        # datetime64[ns] and bool as integers but need to return them as the
        # actual type.
        baseline_array = out_array.view(dtype)

    if AsArrayKind is AsAdjustedArray:
        return AdjustedArray(
            baseline_array,
            adjustments,
            missing_value,
        )
    else:
        return baseline_array


cdef array_for_column(object dtype,
                      tuple out_shape,
                      Py_ssize_t size,
                      np.ndarray[np.int64_t] ts_ixs,
                      np.ndarray[np.int64_t] asof_dates,
                      np.ndarray[np.int64_t] asof_ixs,
                      np.ndarray[np.int64_t] sids,
                      dict sid_column_ixs,
                      np.ndarray input_array,
                      object missing_value,
                      AsArrayKind array_kind):
    cdef np.ndarray out_array = np.full(
        out_shape,
        missing_value,
        dtype,
    )
    cdef str kind = input_array.dtype.kind

    if kind == 'i':
        return _array_for_column_impl[np.int64_t, AsArrayKind](
            dtype,
            out_array,
            size,
            ts_ixs,
            asof_dates,
            asof_ixs,
            sids,
            sid_column_ixs,
            input_array,
            missing_value,
            is_missing_value[np.int64_t],
            array_kind,
        )
    elif kind == 'M':
        return _array_for_column_impl[np.int64_t, AsArrayKind](
            dtype,
            out_array.view('int64'),
            size,
            ts_ixs,
            asof_dates,
            asof_ixs,
            sids,
            sid_column_ixs,
            input_array.view('int64'),
            missing_value.view('int64'),
            is_missing_value[np.int64_t],
            array_kind,
        )
    elif kind == 'f':
        return _array_for_column_impl[np.float64_t, AsArrayKind](
            dtype,
            out_array,
            size,
            ts_ixs,
            asof_dates,
            asof_ixs,
            sids,
            sid_column_ixs,
            input_array,
            missing_value,
            is_missing_value[np.float64_t],
            array_kind,
        )

    elif kind == 'O':
        return _array_for_column_impl[object, AsArrayKind](
            dtype,
            out_array,
            size,
            ts_ixs,
            asof_dates,
            asof_ixs,
            sids,
            sid_column_ixs,
            input_array,
            missing_value,
            is_missing_value[object],
            array_kind,
        )
    elif kind == 'b':
        return _array_for_column_impl[np.uint8_t, AsArrayKind](
            dtype,
            out_array.view('uint8'),
            size,
            ts_ixs,
            asof_dates,
            asof_ixs,
            sids,
            sid_column_ixs,
            input_array.view('uint8'),
            int(missing_value),
            is_missing_value[np.uint8_t],
            array_kind,
        )
    else:
        raise TypeError('unknown column dtype: %r' % input_array.dtype)


cpdef getname(object column):
    try:
        return column.metadata['blaze_column_name']
    except KeyError:
        return column.name


cdef arrays_from_rows(DatetimeIndex_t dates,
                      object data_query_time,
                      object data_query_tz,
                      object assets,
                      np.ndarray[np.int64_t] sids,
                      list columns,
                      object all_rows,
                      AsArrayKind array_kind):
    cdef dict column_ixs = dict(zip(assets, range(len(assets))))

    if data_query_time is not None:
        ts_dates = days_at_time(dates, data_query_time, data_query_tz)
    else:
        ts_dates = dates

    # We use searchsorted right here to be exclusive on the data query time.
    # This means that if a data_query_time = 8:45, and a timestamp is exactly
    # 8:45, we would mark that the data point became available the next day.
    cdef np.ndarray[np.int64_t] ts_ixs = ts_dates.searchsorted(
        all_rows[TS_FIELD_NAME].values,
        'right',
    )

    # We use searchsorted right here to align the asof_dates with what pipeline
    # expects. In a CustomFactor, when today = t_1, the last row of the input
    # array should be data whose asof_date is t_0.
    cdef np.ndarray[np.int64_t] asof_ixs = dates.searchsorted(
        all_rows[AD_FIELD_NAME].values,
        'right',
    )

    cdef tuple out_shape = (len(dates), len(assets))
    cdef dict out = {}
    cdef Py_ssize_t size = len(ts_ixs)

    for column in columns:
        values = all_rows[getname(column)].values
        if isinstance(values, pd.Categorical):
            # convert pandas categoricals into ndarray[object]
            values = values.get_values()

        out[column] = array_for_column[AsArrayKind](
            column.dtype,
            out_shape,
            size,
            ts_ixs,
            (
                all_rows[AD_FIELD_NAME].values.view('int64')
                if len(all_rows) else
                # workaround for empty data frames which often lost type
                # information; enforce than an empty column as an int64 type
                # instead of object type
                np.array([], dtype='int64')
            ),
            asof_ixs,
            sids,
            column_ixs,
            values.astype(column.dtype, copy=False),
            column.missing_value,
            array_kind,
        )

    return out


cdef arrays_from_rows_with_assets(DatetimeIndex_t dates,
                                  object data_query_time,
                                  object data_query_tz,
                                  object assets,
                                  list columns,
                                  object all_rows,
                                  AsArrayKind array_kind):
    return arrays_from_rows[AsArrayKind](
        dates,
        data_query_time,
        data_query_tz,
        assets,
        all_rows[SID_FIELD_NAME].values.astype('int64', copy=False),
        columns,
        all_rows,
        array_kind,
    )


cdef arrays_from_rows_without_assets(DatetimeIndex_t dates,
                                     object data_query_time,
                                     object data_query_tz,
                                     list columns,
                                     object all_rows,
                                     AsArrayKind array_kind):
    # The no assets case is implemented as a special case of the with assets
    # code where every row is tagged with a dummy sid of 0. This gives us the
    # desired shape of (len(dates), 1) without much cost.
    return arrays_from_rows[AsArrayKind](
        dates,
        data_query_time,
        data_query_tz,
        [0],  # pass just sid 0
        np.ndarray(
            (len(all_rows),),
            np.dtype('int64'),
            b'\0' * 8,  # one int64
            0,
            (0,),
            'C',
        ),
        columns,
        all_rows,
        array_kind,
    )


cpdef adjusted_arrays_from_rows_with_assets(DatetimeIndex_t dates,
                                            object data_query_time,
                                            object data_query_tz,
                                            object assets,
                                            list columns,
                                            object all_rows):
    """Construct the adjusted array objects from the input rows.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading days requested by the pipeline engine.
    data_query_time : datetime.time or None
        The time of day when the data is being queried. If None,
        midnight UTC will be used.
    data_query_tz : pytz.Timezone or None
        The timezone for the data_query_time.
    assets : iterable[int]
        The assets in the order requested.
    columns : list[BoundColumn]
        The columns being loaded.
    all_rows : pd.DataFrame
        The single dataframe of input rows. This **must** be sorted by the
        ``[TS_FIELD_NAME, AD_FIELD_NAME]`` columns.

    Returns
    -------
    adjusted_arrays : dict[BoundColumn, AdjustedArray]
        One AdjustedArray per loaded column.
    """
    return arrays_from_rows_with_assets[AsAdjustedArray](
        dates,
        data_query_time,
        data_query_tz,
        assets,
        columns,
        all_rows,
        AsAdjustedArray(),
    )


cpdef adjusted_arrays_from_rows_without_assets(DatetimeIndex_t dates,
                                               object data_query_time,
                                               object data_query_tz,
                                               list columns,
                                               object all_rows):
    """Construct the adjusted array objects from the input rows.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading days requested by the pipeline engine.
    data_query_time : datetime.time or None
        The time of day when the data is being queried. If None,
        midnight UTC will be used.
    data_query_tz : pytz.Timezone or None
        The timezone for the data_query_time.
    columns : list[BoundColumn]
        The columns being loaded.
    all_rows : pd.DataFrame
        The single dataframe of input rows. This **must** be sorted by the
        ``[TS_FIELD_NAME, AD_FIELD_NAME]`` columns.

    Returns
    -------
    adjusted_arrays : dict[BoundColumn, AdjustedArray]
        One AdjustedArray per loaded column.
    """
    return arrays_from_rows_without_assets[AsAdjustedArray](
        dates,
        data_query_time,
        data_query_tz,
        columns,
        all_rows,
        AsAdjustedArray(),
    )


cpdef baseline_arrays_from_rows_with_assets(DatetimeIndex_t dates,
                                            object data_query_time,
                                            object data_query_tz,
                                            object assets,
                                            list columns,
                                            object all_rows):
    """Construct the baseline arrays from the input rows.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading days requested by the pipeline engine.
    data_query_time : datetime.time or None
        The time of day when the data is being queried. If None,
        midnight UTC will be used.
    data_query_tz : pytz.Timezone or None
        The timezone for the data_query_time.
    assets : iterable[int]
        The assets in the order requested.
    columns : list[BoundColumn]
        The columns being loaded.
    all_rows : pd.DataFrame
        The single dataframe of input rows. This **must** be sorted by the
        ``[TS_FIELD_NAME, AD_FIELD_NAME]`` columns.

    Returns
    -------
    arrays : dict[BoundColumn, np.ndarray]
        One array per loaded column.
    """
    return arrays_from_rows_with_assets[AsBaselineArray](
        dates,
        data_query_time,
        data_query_tz,
        assets,
        columns,
        all_rows,
        AsBaselineArray(),
    )


cpdef baseline_arrays_from_rows_without_assets(DatetimeIndex_t dates,
                                               object data_query_time,
                                               object data_query_tz,
                                               list columns,
                                               object all_rows):
    """Construct the baseline arrays from the input rows.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading days requested by the pipeline engine.
    data_query_time : datetime.time or None
        The time of day when the data is being queried. If None,
        midnight UTC will be used.
    data_query_tz : pytz.Timezone or None
        The timezone for the data_query_time.
    columns : list[BoundColumn]
        The columns being loaded.
    all_rows : pd.DataFrame
        The single dataframe of input rows. This **must** be sorted by the
        ``[TS_FIELD_NAME, AD_FIELD_NAME]`` columns.

    Returns
    -------
    arrays : dict[BoundColumn, np.ndarray]
        One array per loaded column.
    """
    return arrays_from_rows_without_assets[AsBaselineArray](
        dates,
        data_query_time,
        data_query_tz,
        columns,
        all_rows,
        AsBaselineArray(),
    )
