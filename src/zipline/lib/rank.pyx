"""
Functions for ranking and sorting.
"""
cimport cython
from cpython cimport bool
from numpy cimport (
    float64_t,
    import_array,
    intp_t,
    int64_t,
    ndarray,
    NPY_DOUBLE,
    NPY_MERGESORT,
    PyArray_ArgSort,
    PyArray_DIMS,
    PyArray_EMPTY,
    uint8_t,
)
from numpy import apply_along_axis, float64, isnan, nan, zeros_like
from scipy.stats import rankdata

from zipline.utils.numpy_utils import (
    is_missing,
    float64_dtype,
    int64_dtype,
    datetime64ns_dtype,
)


import_array()


def rankdata_1d_descending(ndarray data, str method):
    """
    1D descending version of scipy.stats.rankdata.
    """
    return rankdata(-(data.view(float64)), method=method)


def masked_rankdata_2d(ndarray data,
                       ndarray mask,
                       object missing_value,
                       str method,
                       bool ascending):
    """
    Compute masked rankdata on data on float64, int64, or datetime64 data.
    """
    cdef str dtype_name = data.dtype.name
    if dtype_name not in ('float64', 'int64', 'datetime64[ns]'):
        raise TypeError(
            "Can't compute rankdata on array of dtype %r." % dtype_name
        )

    cdef ndarray missing_locations = (~mask | is_missing(data, missing_value))

    # Interpret the bytes of integral data as floats for sorting.
    data = data.copy().view(float64)
    data[missing_locations] = nan
    if not ascending:
        data = -data

    # OPTIMIZATION: Fast path the default case with our own specialized
    # Cython implementation.
    if method == 'ordinal':
        result = rankdata_2d_ordinal(data)
    else:
        # FUTURE OPTIMIZATION:
        # Write a less general "apply to rows" method that doesn't do all
        # the extra work that apply_along_axis does.
        result = apply_along_axis(rankdata, 1, data, method=method)

        # On SciPy >= 0.17, rankdata returns integers for any method except
        # average.
        if result.dtype.name != 'float64':
            result = result.astype('float64')

    # rankdata will sort missing values into last place, but we want our nans
    # to propagate, so explicitly re-apply.
    result[missing_locations] = nan
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
cpdef rankdata_2d_ordinal(ndarray[float64_t, ndim=2] array):
    """
    Equivalent to:

    numpy.apply_over_axis(scipy.stats.rankdata, 1, array, method='ordinal')
    """
    cdef:
        int nrows, ncols
        ndarray[Py_ssize_t, ndim=2] sort_idxs
        ndarray[float64_t, ndim=2] out

    nrows = array.shape[0]
    ncols = array.shape[1]

    # scipy.stats.rankdata explicitly uses MERGESORT instead of QUICKSORT for
    # the ordinal branch.  c.f. commit ab21d2fee2d27daca0b2c161bbb7dba7e73e70ba
    sort_idxs = PyArray_ArgSort(array, 1, NPY_MERGESORT)

    # Roughly, "out = np.empty_like(array)"
    out = PyArray_EMPTY(2, PyArray_DIMS(array), NPY_DOUBLE, False)

    cdef Py_ssize_t i
    cdef Py_ssize_t j

    for i in range(nrows):
        for j in range(ncols):
            out[i, sort_idxs[i, j]] = j + 1.0

    return out


@cython.embedsignature(True)
cpdef grouped_masked_is_maximal(ndarray[int64_t, ndim=2] data,
                                ndarray[int64_t, ndim=2] groupby,
                                ndarray[uint8_t, ndim=2] mask):
    """Build a mask of the top value for each row in ``data``, grouped by
    ``groupby`` and masked by ``mask``.

    Parameters
    ----------
    data : np.array[int64_t]
        Data on which we should find maximal values for each row.
    groupby : np.array[int64_t]
        Grouping labels for rows of ``data``. We choose one entry in each
        row for each unique grouping key in that row.
    mask : np.array[uint8_t]
        Boolean mask of locations to consider as possible maximal values.
        Locations with a 0 in ``mask`` are ignored.

    Returns
    -------
    maximal_locations : np.array[bool]
        Mask containing True for the maximal non-masked value in each row/group.
    """
    # Cython thinks ``.shape`` is an intp_t pointer on ndarrays, so we need to
    # cast to object to get the proper shape attribute.
    if not ((<object> data).shape
            == (<object> groupby).shape
            == (<object> data).shape):
        raise AssertionError(
            "Misaligned shapes in grouped_masked_is_maximal:"
            "data={}, groupby={}, mask={}".format(
                (<object> data).shape, (<object> groupby).shape, (<object> mask).shape,
            )
        )

    cdef:
        Py_ssize_t i
        Py_ssize_t j
        Py_ssize_t nrows
        Py_ssize_t ncols
        int64_t group
        int64_t value
        ndarray[uint8_t, ndim=2] out = zeros_like(mask)
        dict best_per_group = {}

    nrows = data.shape[0]
    ncols = data.shape[1]

    for i in range(nrows):
        best_per_group.clear()
        for j in range(ncols):

            # NOTE: Callers are responsible for masking out values that should
            # be treated as null here.
            if not mask[i, j]:
                continue

            value = data[i, j]
            group = groupby[i, j]

            if group not in best_per_group:
                best_per_group[group] = j
                continue

            if value > data[i, best_per_group[group]]:
                best_per_group[group] = j

        for j in best_per_group.values():
            out[i, j] = 1

    return out.view(bool)
