"""
Functions for ranking and sorting.
"""
cimport cython
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
from numpy import nan

import_array()


cdef double NAN = nan


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def rankdata_2d_ordinal(ndarray[float64_t, ndim=2] array):
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
