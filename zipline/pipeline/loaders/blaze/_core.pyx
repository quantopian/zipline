from libc.math cimport isnan
from cpython cimport Py_INCREF, PyDict_GetItem, PyObject
from bisect import bisect_right, insort_left

cimport cython
cimport numpy as np
import numpy as np

from zipline.lib.adjusted_array import AdjustedArray
from zipline.lib.adjustment cimport (
    AdjustmentKind,
    DatetimeIndex_t,
    make_adjustment,
    column_type
)
from zipline.lib.labelarray import LabelArray
from zipline.pipeline.common import (
    AD_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME
)


ctypedef bint is_missing_function(column_type, column_type)


cdef bint is_missing_value(column_type value, column_type missing_value):
    return value == missing_value


cdef bint is_missing_nan(np.float64_t value, np.float64_t missing_value):
    return isnan(value)


cdef _ffill_missing_value_2d_inplace_impl(np.ndarray[column_type, ndim=2] array,
                                          column_type missing_value,
                                          is_missing_function is_missing):
    cdef np.ndarray[column_type] most_recent_row = np.full(
        array.shape[1],
        missing_value,
        dtype=array.dtype,
    )
    cdef column_type most_recent
    cdef column_type element
    cdef Py_ssize_t r
    cdef Py_ssize_t c
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            with cython.boundscheck(False), cython.wraparound(False):
                element = array[r, c]

            if is_missing(element, missing_value):
                with cython.boundscheck(False), cython.wraparound(False):
                    array[r, c] = most_recent_row[c]
            else:
                with cython.boundscheck(False), cython.wraparound(False):
                    most_recent_row[c] = element


cpdef _ffill_missing_value_2d_inplace(np.ndarray array, missing_value):
    cdef str kind = array.dtype.kind
    if kind == 'i':
        _ffill_missing_value_2d_inplace_impl[np.int64_t](
            array,
            missing_value,
            is_missing_value[np.int64_t],
        )
    elif kind == 'u':
        _ffill_missing_value_2d_inplace_impl[np.uint8_t](
            array,
            missing_value,
            is_missing_value[np.uint8_t],
        )
    elif kind == 'M':
        _ffill_missing_value_2d_inplace_impl[np.int64_t](
            array.view('int64'),
            missing_value.astype('int64'),
            is_missing_value[np.int64_t],
        )
    elif kind == 'f':
        if isnan(missing_value):
            _ffill_missing_value_2d_inplace_impl[np.float64_t](
                array,
                missing_value,
                is_missing_nan,
            )
        else:
            _ffill_missing_value_2d_inplace_impl[np.float64_t](
                array,
                missing_value,
                is_missing_value[np.float64_t],
            )
    elif kind == 'O':
        _ffill_missing_value_2d_inplace_impl[object](
            array,
            missing_value,
            is_missing_value[object],
        )
    else:
        raise TypeError('unknown column dtype: %r' % array.dtype)


cdef _adjusted_array_for_column_impl(object dtype,
                                     np.ndarray[column_type, ndim=2] out_array,
                                     Py_ssize_t size,
                                     np.ndarray[np.int64_t] ts_ixs,
                                     np.ndarray[np.int64_t] asof_ixs,
                                     np.ndarray[np.int64_t] sids,
                                     dict column_ixs,
                                     object mask,
                                     np.ndarray[column_type] input_array,
                                     column_type missing_value,
                                     bint is_missing(column_type, column_type)):
    cdef column_type value
    cdef np.int64_t ts_ix
    cdef np.int64_t asof_ix
    cdef np.int64_t sid
    cdef np.int64_t column_ix

    cdef PyObject* adjustments_list_ptr
    cdef list adjustments_list

    cdef list non_null_ixs
    cdef dict non_null_ixs_by_sid = {sid: [] for sid in sids}

    cdef dict adjustments = {}

    cdef set categories
    if column_type is object:
        # for object columns we need to maintain the unique values for the
        # categories
        categories = set()

    cdef Py_ssize_t n
    for n in range(size):
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
            asof_ix = asof_ixs[n]
            sid = sids[n]

        column_ix_ob = PyDict_GetItem(column_ixs, sid)
        if column_ix_ob is NULL:
            # ignore sids that are not requested
            continue

        column_ix = <object> column_ix_ob  # cast to np.int64_t

        adjustment_list_ptr = PyDict_GetItem(adjustments, ts_ix)
        if adjustment_list_ptr is NULL:
            adjustment_list = adjustments[ts_ix] = []
        else:
            adjustment_list = <list> adjustment_list_ptr
            Py_INCREF(adjustment_list)

        non_null_ixs = non_null_ixs_by_sid[sid]
        ix = bisect_right(non_null_ixs, asof_ix)
        if ix == len(non_null_ixs):
            # write the value into the baseline out array at ``ts_ix``
            # in the given sid column
            with cython.boundscheck(False), cython.wraparound(False):
                out_array[ts_ix, column_ix] = value

            # the value at ``ts_ix`` was set above
            end = max(ts_ix - 1, 0)
            if end >= asof_ix:
                adjustment_list.append(make_adjustment[column_type](
                    asof_ix,
                    end,
                    column_ix,
                    column_ix,
                    AdjustmentKind.OVERWRITE,
                    value,
                ))
        else:
            end = max(non_null_ixs[ix] - 1, 0)
            if end >= asof_ix:
                adjustment_list.append(make_adjustment[column_type](
                    asof_ix,
                    end,
                    column_ix,
                    column_ix,
                    AdjustmentKind.OVERWRITE,
                    value,
                ))

        insort_left(non_null_ixs, asof_ix)

    _ffill_missing_value_2d_inplace(out_array, missing_value)

    if column_type is object:
        baseline_array = LabelArray(
            out_array,
            missing_value,
            categories,
        )
    else:
        # this cast prevents a compiler crash
        baseline_array = <object> out_array

    return AdjustedArray(
        baseline_array,
        mask,
        adjustments,
        missing_value,
    )


cdef adjusted_array_for_column(object dtype,
                               tuple out_shape,
                               Py_ssize_t size,
                               np.ndarray[np.int64_t] ts_ixs,
                               np.ndarray[np.int64_t] asof_ixs,
                               np.ndarray[np.int64_t] sids,
                               dict sid_column_ixs,
                               object mask,
                               np.ndarray input_array,
                               object missing_value):
    cdef np.ndarray out_array = np.full(
        out_shape,
        missing_value,
        dtype,
    )
    cdef str kind = input_array.dtype.kind

    if kind == 'i':
        return _adjusted_array_for_column_impl[np.int64_t](
            dtype,
            out_array,
            size,
            ts_ixs,
            asof_ixs,
            sids,
            sid_column_ixs,
            mask,
            input_array,
            missing_value,
            is_missing_value[np.int64_t],
        )
    elif kind == 'M':
        return _adjusted_array_for_column_impl[np.int64_t](
            dtype,
            out_array.view('int64'),
            size,
            ts_ixs,
            asof_ixs,
            sids,
            sid_column_ixs,
            mask,
            input_array.view('int64'),
            missing_value.view('int64'),
            is_missing_value[np.int64_t],
        )
    elif kind == 'f':
        if isnan(missing_value):
            return _adjusted_array_for_column_impl[np.float64_t](
                dtype,
                out_array,
                size,
                ts_ixs,
                asof_ixs,
                sids,
                sid_column_ixs,
                mask,
                input_array,
                missing_value,
                is_missing_nan,
            )
        else:
            return _adjusted_array_for_column_impl[np.float64_t](
                dtype,
                out_array,
                size,
                ts_ixs,
                asof_ixs,
                sids,
                sid_column_ixs,
                mask,
                input_array,
                missing_value,
                is_missing_value[np.float64_t],
            )
    elif kind == 'O':
        return _adjusted_array_for_column_impl[object](
            dtype,
            out_array,
            size,
            ts_ixs,
            asof_ixs,
            sids,
            sid_column_ixs,
            mask,
            input_array,
            missing_value,
            is_missing_value[object],
        )
    elif kind == 'b':
        return _adjusted_array_for_column_impl[np.uint8_t](
            dtype,
            out_array.view('uint8'),
            size,
            ts_ixs,
            asof_ixs,
            sids,
            sid_column_ixs,
            mask,
            input_array.view('uint8'),
            int(missing_value),
            is_missing_value[np.uint8_t],
        )
    else:
        raise TypeError('unknown column dtype: %r' % input_array.dtype)


cdef adjusted_arrays_from_rows(DatetimeIndex_t dates,
                               DatetimeIndex_t ts_dates,
                               assets,
                               np.ndarray[np.int64_t] sids,
                               object mask,
                               list columns,
                               object all_rows):
    cdef dict column_ixs = dict(zip(assets, range(len(assets))))

    cdef np.ndarray[np.int64_t] ts_ixs = ts_dates.searchsorted(
        all_rows[TS_FIELD_NAME].values,
        'right',
    )
    ts_ixs[ts_ixs == len(ts_dates)] = len(ts_dates) - 1

    cdef np.ndarray[np.int64_t] asof_ixs = dates.searchsorted(
        all_rows[AD_FIELD_NAME].values,
        'right',
    )
    asof_ixs[ts_ixs == len(dates)] = len(dates) - 1

    cdef tuple out_shape = (len(dates), len(assets))
    cdef dict out = {}
    cdef Py_ssize_t size = len(ts_ixs)

    for column in columns:
        out[column] = adjusted_array_for_column(
            column.dtype,
            out_shape,
            size,
            ts_ixs,
            asof_ixs,
            sids,
            column_ixs,
            mask,
            all_rows[column.name].values.astype(column.dtype),
            column.missing_value,
        )

    return out


cpdef adjusted_arrays_from_rows_with_assets(DatetimeIndex_t dates,
                                            DatetimeIndex_t ts_dates,
                                            assets,
                                            object mask,
                                            list columns,
                                            object all_rows):
    """Construct the adjusted array objects from the input rows.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading days requested by the pipeline engine.
    ts_dates : pd.DatetimeIndex
        The trading days aligned to the data query time.
    assets : iterable[int]
        The assets in the order requested.
    mask : np.ndarray[bool]
        The mask provided by the pipeline engine.
    columns : list[BoundColumn]
        The columns being loaded.
    all_rows : pd.DataFrame
        The single dataframe of input rows. This **must** be sorted by the
        ``TS_FIELD_NAME`` column.

    Returns
    -------
    adjusted_arrays : dict[BoundColumn, AdjustedArray]
        One AdjustedArray per loaded column.
    """
    return adjusted_arrays_from_rows(
        dates,
        ts_dates,
        assets,
        all_rows[SID_FIELD_NAME].values.astype('int64'),
        mask,
        columns,
        all_rows,
    )


cpdef adjusted_arrays_from_rows_without_assets(DatetimeIndex_t dates,
                                               DatetimeIndex_t ts_dates,
                                               object mask,
                                               list columns,
                                               object all_rows):
    """Construct the adjusted array objects from the input rows.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The trading days requested by the pipeline engine.
    ts_dates : pd.DatetimeIndex
        The trading days aligned to the data query time.
    mask : np.ndarray[bool]
        The mask provided by the pipeline engine.
    columns : list[BoundColumn]
        The columns being loaded.
    all_rows : pd.DataFrame
        The single dataframe of input rows. This **must** be sorted by the
        ``TS_FIELD_NAME`` column.

    Returns
    -------
    adjusted_arrays : dict[BoundColumn, AdjustedArray]
        One AdjustedArray per loaded column.
    """
    return adjusted_arrays_from_rows(
        dates,
        ts_dates,
        [0],  # pass just sid 0
        np.ndarray(
            (len(all_rows),),
            np.dtype('int64'),
            b'\0' * 8,  # one int64
            0,
            (0,),
            'C',
        ),
        mask,
        columns,
        all_rows,
    )
