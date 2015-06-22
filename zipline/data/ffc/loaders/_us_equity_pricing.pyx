import shelve
import pandas as pd

import bcolz
cimport cython
import numpy as np
cimport numpy as np

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _load_adjusted_array_from_bcolz(daily_bar_table, daily_bar_index,
                                      trading_days,
                                      columns,
                                      assets,
                                      dates):
    """
    Load each column from bcolsz table, @daily_bar_table.

    @daily_bar_index is an index of the start position and dates of each
    asset from the table.

    @trading_days is the trading days allowed by the query, with the first
    date being the first date in the provided dataset.

    @columns, @assets, @dates are the same values as passed to
    load_adjusted_array
    """
    nrows = dates.shape[0]
    ncols = len(assets)

    cdef dict start_pos = daily_bar_index['start_pos']
    cdef dict start_day_offset = daily_bar_index['start_day_offset']
    cdef dict end_day_offset = daily_bar_index['end_day_offset']

    cdef np.intp_t query_start_offset = trading_days.searchsorted(dates[0])
    cdef np.intp_t date_len = dates.shape[0]
    cdef np.intp_t query_end_offset = query_start_offset + date_len

    cdef np.intp_t start, end
    cdef np.intp_t i

    cdef np.intp_t asset_start, asset_start_day_offset, asset_end_day_offset
    cdef np.intp_t start_ix, end_ix, offset_ix
    cdef np.ndarray[dtype=np.intp_t, ndim=1] asset_start_ix = np.zeros(
        ncols, dtype=np.intp)
    cdef np.ndarray[dtype=np.intp_t, ndim=1] asset_end_ix = np.zeros(
        ncols, dtype=np.intp)
    cdef np.ndarray[dtype=np.intp_t, ndim=1] asset_offset_ix = np.zeros(
        ncols, dtype=np.intp)

    for i, asset in enumerate(assets):
        # There are 8 cases to handle.
        # 1) The equity's trades cover all query dates.
        # 2) The equity's trades are all before the start of the query.
        # 3) The equity's trades start before the query start, but stop
        #    before the query end.
        # 4) The equity's trades start after query start but end before
        #    the query end.
        # 5) The equity's trades start after query start, but trade through or
        #    past the query end
        # 6) The equity's trades are start after query end.
        #
        # TODO: Build unit tests exercising each of these cases.
        #       Currently tested by comparing this functions output with
        #       data fetched using numexpr based indexing into the dataset.
        asset_start = start_pos[asset]
        asset_start_day_offset = start_day_offset[asset]
        asset_end_day_offset = end_day_offset[asset]

        if asset_start_day_offset > query_end_offset:
            # case 6
            # Leave values as 0, for empty set.
            continue
        if asset_end_day_offset < query_start_offset:
            # case 2
            # Leave values as 0, for empty set.
            continue
        if asset_start_day_offset <= query_start_offset:
            # case 1 or 3
            #
            # requires no offset in the container
            #
            # calculate start_ix based on distance between query start
            # and date offset
            #
            # requires no container offset
            offset_ix = 0
            start_ix = asset_start + (query_start_offset - \
                                      asset_start_day_offset)
        else:
            # case 4 or 5
            #
            # requires offset in the container, since the trading starts
            # after the container range
            #
            # calculate start_ix based on distance between query start
            # and date offset
            #
            # requires no container offset
            start_ix = asset_start
            offset_ix = asset_start_day_offset - query_start_offset

        if asset_end_day_offset >= query_end_offset:
            # case 1 or 5, just clip at the end of the query
            end_ix = asset_start + (query_end_offset - asset_start_day_offset)
        else:
            # case 3 or 4 , data ends before query end
            end_ix = asset_start + asset_end_day_offset + 1

        asset_offset_ix[i] = offset_ix
        asset_start_ix[i] = start_ix
        asset_end_ix[i] = end_ix

    cdef dict data_arrays = {}

    for col in columns:
        data_col = daily_bar_table[col.name][:]
        col_array = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
        for i in range(ncols):
            start_ix = asset_start_ix[i]
            end_ix = asset_end_ix[i]
            if start_ix == end_ix:
                continue
            asset_data = data_col[start_ix:end_ix]

            # Asset data may not necessarily be the same shape as the number
            # of dates if the asset has an earlier end date.
            start = asset_offset_ix[i]
            end = start + (end_ix - start_ix)
            col_array[start:end, i] = asset_data

        if col.dtype == np.float32:
            # Use int for nan check for better precision.
            where_nan = col_array == 0
            col_array = col_array.astype(np.float32) * 0.001
            col_array[where_nan] = np.nan

        data_arrays[col.name] = col_array
        del data_col

    return[
        adjusted_array(
            data_arrays[col.name],
            NOMASK,
            {})
        for col in columns]
