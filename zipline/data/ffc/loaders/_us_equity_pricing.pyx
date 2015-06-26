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
from numpy import (
    array,
    float64,
    intp,
    uint32,
    zeros,
)
from numpy cimport (
    float64_t,
    intp_t,
    ndarray,
    uint32_t,
    uint8_t,
)
from numpy.math cimport NAN

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)

ctypedef object ctable_t
ctypedef object DatetimeIndex_t
ctypedef object Int64Index_t

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

cpdef _adjustments(adjustments_db,
                   split_sids,
                   merger_sids,
                   dividends_sids,
                   dates, assets):
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


cpdef load_adjustments_from_sqlite(adjustments_db, columns, dates, assets):
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
        dates,
        assets,)

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
        # dividends affect prices
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
cpdef _compute_row_slices(dict asset_starts_absolute,
                          dict asset_ends_absolute,
                          dict asset_starts_calendar,
                          intp_t query_start,
                          intp_t query_end,
                          Int64Index_t requested_assets):
    """
    Core indexing functionality for loading raw data from bcolz.

    Parameters
    ----------
    asset_starts_absolute : dict
        Dictionary containing the index of the first row of each asset in the
        bcolz file from which we will query.

    asset_ends_absolute : dict
        Dictionary containing the index of the last row of each asset in the
        bcolz file from which we will query.

    asset_starts_calendar : dict
        Dictionary containing the index of in our calendar corresponding to the
        start date of each asset

    query_start : intp
    query_end : intp
        Start and end indices in our calendar of the dates for which we're
        querying.

    requested_assets : pandas.Int64Index
        The assets for which we want to load data.

    For each asset in requested assets, computes three values:
    1.) The index in the raw bcolz data of first row to load.
    2.) The index in the raw bcolz data of the last row to load.
    3.) The index in the dates of our query corresponding to the first row for
        each asset. This is non-zero iff the asset's lifetime begins partway
        through the requested query dates.

    Returns
    -------
    first_rows, last_rows, offsets : 3-tuple of ndarrays
    """
    cdef:
        intp_t nassets = len(requested_assets)

        # For each sid, we need to compute the following:
        ndarray[dtype=intp_t, ndim=1] first_row_a = zeros(nassets, dtype=intp)
        ndarray[dtype=intp_t, ndim=1] last_row_a = zeros(nassets, dtype=intp)
        ndarray[dtype=intp_t, ndim=1] offset_a = zeros(nassets, dtype=intp)

        # Loop variables.
        intp_t i
        intp_t asset
        intp_t asset_start_data
        intp_t asset_end_data
        intp_t asset_start_calendar
        intp_t asset_end_calendar

    for i, asset in enumerate(requested_assets):
        asset_start_data = asset_starts_absolute[asset]
        asset_end_data = asset_ends_absolute[asset]
        asset_start_calendar = asset_starts_calendar[asset]
        asset_end_calendar = (
            asset_start_calendar + (asset_end_data - asset_start_data)
        )

        # If the asset started during the query, then start with the asset's
        # first row.
        # Otherwise start with the asset's first row + the number of rows
        # before the query on which the asset existed.
        first_row_a[i] = (
            asset_start_data + max(0, (query_start - asset_start_calendar))
        )
        # If the asset ended during the query, the end with the asset's last
        # row.
        # Otherwise, end with the asset's last row minus the number of rows
        # after the query for which the asset
        last_row_a[i] = (
            asset_end_data - max(0, asset_end_calendar - query_end)
        )
        # If the asset existed on or before the query, no offset.
        # Otherwise, offset by the number of rows in the query in which the
        # asset did not yet exist.
        offset_a[i] = max(0, asset_start_calendar - query_start)

    return first_row_a, last_row_a, offset_a


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _read_bcolz_data(ctable_t table,
                       tuple shape,
                       list columns,
                       intp_t[:] first_rows,
                       intp_t[:] last_rows,
                       intp_t[:] offsets):
    """
    Load raw bcolz data for the given columns and indices.

    Parameters
    ----------
    table : bcolz.ctable
        The table from which to read.
    shape : tuple (length 2)
        The shape of the expected output arrays.
    columns : list[str]
        List of column names to read.

    first_rows : ndarray[intp]
    last_rows : ndarray[intp]
    offsets : ndarray[intp
        Arrays in the format returned by _compute_row_slices.

    Returns
    -------
    results : list of ndarray
        A 2D array of shape `shape` for each column in `columns`.
    """
    cdef:
        int nassets
        str column_name
        ndarray[dtype=uint32_t, ndim=1] raw_data
        ndarray[dtype=uint32_t, ndim=2] outbuf
        ndarray[dtype=uint8_t, ndim=2, cast=True] where_nan
        ndarray[dtype=float64_t, ndim=2] outbuf_as_float
        intp_t asset
        intp_t out_idx
        intp_t raw_idx
        intp_t first_row
        intp_t last_row
        intp_t offset
        list results = []

    nassets = shape[1]
    if not nassets== len(first_rows) == len(last_rows) == len(offsets):
        raise ValueError("Incompatible index arrays.")

    for column_name in columns:
        raw_data = table[column_name][:]
        outbuf = zeros(shape=shape, dtype=uint32)
        for asset in range(nassets):
            first_row = first_rows[asset]
            last_row = last_rows[asset]
            offset = offsets[asset]
            for out_idx, raw_idx in enumerate(range(first_row, last_row + 1)):
                outbuf[out_idx + offset, asset] = raw_data[raw_idx]

        if column_name in {'open', 'high', 'low', 'close'}:
            where_nan = (outbuf == 0)
            outbuf_as_float = outbuf.astype(float64) * .001
            outbuf_as_float[where_nan] = NAN
            results.append(outbuf_as_float)
        else:
            results.append(outbuf)
    return results
