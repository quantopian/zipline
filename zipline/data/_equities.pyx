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
import bcolz
cimport cython
from cpython cimport bool

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

ctypedef object carray_t
ctypedef object ctable_t
ctypedef object Timestamp_t
ctypedef object DatetimeIndex_t
ctypedef object Int64Index_t


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
                       intp_t[:] offsets,
                       bool read_all):
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
    read_all : bool
        Whether to read_all sid data at once, or to read a silce from the
        carray for each sid.

    Returns
    -------
    results : list of ndarray
        A 2D array of shape `shape` for each column in `columns`.
    """
    cdef:
        int nassets
        str column_name
        carray_t carray
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

    ndays = shape[0]
    nassets = shape[1]
    if not nassets== len(first_rows) == len(last_rows) == len(offsets):
        raise ValueError("Incompatible index arrays.")

    for column_name in columns:
        outbuf = zeros(shape=shape, dtype=uint32)
        if read_all:
            raw_data = table[column_name][:]

            for asset in range(nassets):
                first_row = first_rows[asset]
                last_row = last_rows[asset]
                offset = offsets[asset]
                if first_row <= last_row:
                    outbuf[offset:offset + (last_row + 1 - first_row), asset] =\
                        raw_data[first_row:last_row + 1]
                else:
                    continue
        else:
            carray = table[column_name]

            for asset in range(nassets):
                first_row = first_rows[asset]
                last_row = last_rows[asset]
                offset = offsets[asset]
                out_start = offset
                out_end = (last_row - first_row) + offset + 1
                if first_row <= last_row:
                    outbuf[offset:offset + (last_row + 1 - first_row), asset] =\
                        carray[first_row:last_row + 1]
                else:
                    continue

        if column_name in {'open', 'high', 'low', 'close'}:
            where_nan = (outbuf == 0)
            outbuf_as_float = outbuf.astype(float64) * .001
            outbuf_as_float[where_nan] = NAN
            results.append(outbuf_as_float)
        else:
            results.append(outbuf)
    return results
