#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

import bcolz
cimport cython
import numpy as np
cimport numpy as np

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)


from zipline.data.adjustment import Float64Multiply

SID_QUERY_TEMPLATE = """
SELECT DISTINCT sid FROM {0}
WHERE effective_date >= {1} AND effective_date <= {2}
"""

ADJ_QUERY_TEMPLATE = """
SELECT sid, ratio, effective_date
FROM {0}
WHERE sid IN ({1}) AND effective_date >= {2} AND effective_date <= {3}
"""

SQLITE_MAX_IN_STATEMENT = 999
EPOCH = pd.Timestamp(0, tz='UTC')

cpdef _get_split_sids(adjustments_db, start_date, end_date):
    c = adjustments_db.cursor()
    query = SID_QUERY_TEMPLATE.format('splits', start_date, end_date)
    c.execute(query)
    return set([sid[0] for sid in c.fetchall()])

cpdef _get_merger_sids(adjustments_db, start_date, end_date):
    c = adjustments_db.cursor()
    query = SID_QUERY_TEMPLATE.format('mergers', start_date, end_date)
    c.execute(query)
    return set([sid[0] for sid in c.fetchall()])

cpdef _get_dividend_sids(adjustments_db, start_date, end_date):
    c = adjustments_db.cursor()
    query = SID_QUERY_TEMPLATE.format('dividends', start_date, end_date)
    c.execute(query)
    return set([sid[0] for sid in c.fetchall()])

cpdef _adjustments(adjustments_db, split_sids, merger_sids, dividends_sids,
                   assets, dates):
    start_date = (dates[0] - EPOCH).total_seconds()
    end_date = (dates[-1] - EPOCH).total_seconds()

    c = adjustments_db.cursor()

    splits_to_query = [str(a) for a in assets if a in split_sids]
    splits_results = []
    while splits_to_query:
        query_len = min(len(splits_to_query), SQLITE_MAX_IN_STATEMENT)
        query_assets = splits_to_query[:query_len]
        t= [str(a) for a in query_assets]
        statement = ADJ_QUERY_TEMPLATE.format('splits',
            ",".join(['?' for _ in query_assets]), start_date, end_date)
        c.execute(statement, t)
        splits_to_query = splits_to_query[query_len:]
        splits_results.extend(c.fetchall())

    mergers_to_query = [str(a) for a in assets if a in merger_sids]
    mergers_results = []
    while mergers_to_query:
        query_len = min(len(mergers_to_query), SQLITE_MAX_IN_STATEMENT)
        query_assets = mergers_to_query[:query_len]
        t= [str(a) for a in query_assets]
        statement = ADJ_QUERY_TEMPLATE.format('mergers',
            ",".join(['?' for _ in query_assets]), start_date, end_date)
        c.execute(statement, t)
        mergers_to_query = mergers_to_query[query_len:]
        mergers_results.extend(c.fetchall())

    dividends_to_query = [str(a) for a in assets if a in dividends_sids]
    dividends_results = []
    while dividends_to_query:
        query_len = min(len(dividends_to_query), SQLITE_MAX_IN_STATEMENT)
        query_assets = dividends_to_query[:query_len]
        t= [str(a) for a in query_assets]
        statement = ADJ_QUERY_TEMPLATE.format('dividends',
            ",".join(['?' for _ in query_assets]), start_date, end_date)
        c.execute(statement, t)
        dividends_to_query = dividends_to_query[query_len:]
        dividends_results.extend(c.fetchall())

    return splits_results, mergers_results, dividends_results


cpdef load_adjustments_from_sqlite(adjustments_db, columns, assets, dates):
    start_date = dates[0]
    end_date = dates[len(dates) - 1]
    start_date_str = start_date.strftime('%s')
    end_date_str = end_date.strftime('%s')

    split_sids = _get_split_sids(adjustments_db,
                                 start_date_str,
                                 end_date_str)
    merger_sids = _get_merger_sids(adjustments_db,
                                   start_date_str,
                                   end_date_str)
    dividend_sids = _get_dividend_sids(adjustments_db,
                                       start_date_str,
                                       end_date_str)

    splits, mergers, dividends = _adjustments(
        adjustments_db,
        split_sids,
        merger_sids,
        dividend_sids,
        assets,
        dates)

    cdef dict col_adjustments = {col.name: {} for col in columns}

    result = []

    cdef dict asset_ixs = {}

    for col in columns:
        result.append(col_adjustments[col.name])

    cdef int sid
    cdef int asset_ix

    for split in splits:
        # splits affect prices and volumes, volumes is the inverse
        effective_date = pd.Timestamp(split[2], unit='s', tz='UTC')
        date_loc = dates.searchsorted(effective_date)
        sid = split[0]
        try:
            asset_ix = asset_ixs[sid]
        except KeyError:
            asset_ixs[sid] = asset_ix = assets.searchsorted(sid)
        price_adj = Float64Multiply(0, date_loc, asset_ix, split[1])
        for col in columns:
            col_adj = col_adjustments[col.name]
            if col.name != 'volume':
                try:
                    col_adj[date_loc].append(price_adj)
                except KeyError:
                    col_adj[date_loc] = [price_adj]
            else:
                volume_adj = Float64Multiply(0, date_loc, asset_ix,
                                             1.0 / split[1])
                try:
                    col_adj[date_loc].append(volume_adj)
                except KeyError:
                    col_adj[date_loc] = [volume_adj]

    for merger in mergers:
        # mergers affect prices
        effective_date = pd.Timestamp(merger[2], unit='s', tz='UTC')
        date_loc = dates.searchsorted(effective_date)
        sid = merger[0]
        try:
            asset_ix = asset_ixs[sid]
        except KeyError:
            asset_ixs[sid] = asset_ix = assets.searchsorted(sid)
        adj = Float64Multiply(0, date_loc, asset_ix, merger[1])
        for col in columns:
            col_adj = col_adjustments[col.name]
            if col.name != 'volume':
                try:
                    col_adj[date_loc].append(adj)
                except KeyError:
                    col_adj[date_loc] = [adj]

    for dividend in dividends:
        # mergers affect prices
        effective_date = pd.Timestamp(dividend[2], unit='s', tz='UTC')
        date_loc = dates.searchsorted(effective_date)
        sid = dividend[0]
        try:
            asset_ix = asset_ixs[sid]
        except KeyError:
            asset_ixs[sid] = asset_ix = assets.searchsorted(sid)
        adj = Float64Multiply(0, date_loc, asset_ix, dividend[1])
        for col in columns:
            col_adj = col_adjustments[col.name]
            if col.name != 'volume':
                try:
                    col_adj[date_loc].append(adj)
                except KeyError:
                    col_adj[date_loc] = [adj]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef load_raw_arrays_from_bcolz(daily_bar_table,
                                 start_pos,
                                 start_day_offset,
                                 end_day_offset,
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
        # There are 6 cases to handle.
        # 1) The equity's trades cover all query dates.
        # 2) The equity's trades are all before the start of the query.
        # 3) The equity's trades start before the query start, but stop
        #    before the query end.
        # 4) The equity's trades start after query start and ends before
        #    the query end.
        # 5) The equity's trades start after query start, but trade through or
        #    past the query end
        # 6) The equity's trades are start after query end.
        #
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
            start_ix = asset_start
            offset_ix = asset_start_day_offset - query_start_offset

        if asset_end_day_offset >= query_end_offset:
            # case 1 or 5, just clip at the end of the query
            end_ix = asset_start + (query_end_offset - asset_start_day_offset)
        else:
            # case 3 or 4 , data ends before query end
            end_ix = asset_start + (
                asset_end_day_offset - asset_start_day_offset) + 1

        asset_offset_ix[i] = offset_ix
        asset_start_ix[i] = start_ix
        asset_end_ix[i] = end_ix

    cdef list data_arrays = []

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
            col_array = col_array.astype(np.float) * 0.001
            col_array[where_nan] = np.nan

        data_arrays.append(col_array)

        del data_col

    return data_arrays
