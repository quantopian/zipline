"""
Functions for ranking and sorting.
"""
cimport cython
from cpython cimport bool
from numpy cimport (
    float64_t,
    import_array,
    intp_t,
    ndarray,
    NPY_DOUBLE,
    NPY_MERGESORT,
    PyArray_ArgSort,
    PyArray_DIMS,
    PyArray_EMPTY,
)
from numpy import apply_along_axis, float64, isnan, nan
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
        ndarray[intp_t, ndim=2] sort_idxs
        ndarray[float64_t, ndim=2] out

    nrows = array.shape[0]
    ncols = array.shape[1]

    # scipy.stats.rankdata explicitly uses MERGESORT instead of QUICKSORT for
    # the ordinal branch.  c.f. commit ab21d2fee2d27daca0b2c161bbb7dba7e73e70ba
    sort_idxs = PyArray_ArgSort(array, 1, NPY_MERGESORT)

    # Roughly, "out = np.empty_like(array)"
    out = PyArray_EMPTY(2, PyArray_DIMS(array), NPY_DOUBLE, False)

    cdef intp_t i, j
    for i in range(nrows):
        for j in range(ncols):
            out[i, sort_idxs[i, j]] = j + 1.0

    return out
