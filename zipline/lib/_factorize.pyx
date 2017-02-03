"""
Factorization algorithms.
"""
from libc.math cimport log2, floor
cimport numpy as np
import numpy as np

from zipline.utils.numpy_utils import unsigned_int_dtype_with_size_in_bytes

np.import_array()


ctypedef fused unsigned_integral:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


cdef factorize_strings_known_impl(np.ndarray[object] values,
                                  Py_ssize_t nvalues,
                                  list categories,
                                  object missing_value,
                                  bint sort,
                                  np.ndarray[unsigned_integral] codes):
    if missing_value not in categories:
        categories.insert(0, missing_value)

    if sort:
        categories = sorted(categories)

    cdef dict reverse_categories = dict(
        zip(categories, range(len(categories)))
    )
    cdef Py_ssize_t i
    cdef Py_ssize_t missing_code = reverse_categories[missing_value]

    for i in range(nvalues):
        codes[i] = reverse_categories.get(values[i], missing_code)

    return codes, np.asarray(categories, dtype=object), reverse_categories


cpdef factorize_strings_known_categories(np.ndarray[object] values,
                                         list categories,
                                         object missing_value,
                                         bint sort):
    """
    Factorize an array whose categories are already known.

    Any entries not in the specified categories will be given the code for
    `missing_value`.
    """
    cdef Py_ssize_t ncategories = len(categories)
    cdef Py_ssize_t nvalues = len(values)
    if ncategories <= 2 ** 8:
        return factorize_strings_known_impl[np.uint8_t](
            values,
            nvalues,
            categories,
            missing_value,
            sort,
            np.empty(nvalues, dtype=np.uint8)
        )
    elif ncategories <= 2 ** 16:
        return factorize_strings_known_impl[np.uint16_t](
            values,
            nvalues,
            categories,
            missing_value,
            sort,
            np.empty(nvalues, np.uint16),
        )
    elif ncategories <= 2 ** 32:
        return factorize_strings_known_impl[np.uint32_t](
            values,
            nvalues,
            categories,
            missing_value,
            sort,
            np.empty(nvalues, np.uint32),
        )
    elif ncategories <= 2 ** 64:
        return factorize_strings_known_impl[np.uint64_t](
            values,
            nvalues,
            categories,
            missing_value,
            sort,
            np.empty(nvalues, np.uint64),
        )
    else:
        raise ValueError('ncategories larger than uint64')


cdef factorize_strings_impl(np.ndarray[object] values,
                            Py_ssize_t nvalues,
                            object missing_value,
                            bint sort,
                            np.ndarray[unsigned_integral] codes):
    cdef list categories = [missing_value]
    cdef dict reverse_categories = {missing_value: 0}

    cdef Py_ssize_t i, code
    cdef object key = None

    for i in range(nvalues):
        key = values[i]
        code = reverse_categories.get(key, -1)
        if code == -1:
            # Assign new code.
            code = len(reverse_categories)
            reverse_categories[key] = code
            categories.append(key)
        codes[i] = code

    cdef np.ndarray[np.int64_t, ndim=1] sorter
    cdef np.ndarray[unsigned_integral, ndim=1] reverse_indexer
    cdef int ncategories
    cdef np.ndarray[object] categories_array = np.asarray(
        categories,
        dtype=object,
    )

    if sort:
        # This is all adapted from pandas.core.algorithms.factorize.
        ncategories = len(categories_array)
        sorter = np.zeros(ncategories, dtype=np.int64)

        # Don't include missing_value in the argsort, because None is
        # unorderable with bytes/str in py3. Always just sort it to 0.
        sorter[1:] = categories_array[1:].argsort() + 1
        reverse_indexer = np.empty(ncategories, dtype=codes.dtype)
        reverse_indexer.put(sorter, np.arange(ncategories))

        codes = reverse_indexer.take(codes)
        categories_array = categories_array.take(sorter)
        reverse_categories = dict(zip(categories_array, range(ncategories)))

    return codes, categories_array, reverse_categories


cpdef factorize_strings(np.ndarray[object] values,
                        object missing_value,
                        int sort):
    """
    Factorize an array of (possibly duplicated) labels into an array of indices
    into a unique array of labels.

    This is ~30% faster than pandas.factorize, at the cost of not having
    special treatment for NaN, which we don't care about because we only
    support arrays of strings.

    (Though it's faster even if you throw in the nan checks that pandas does,
    because we're using dict and list instead of PyObjectHashTable and
    ObjectVector.  Python's builtin data structures are **really**
    well-optimized.)
    """
    cdef Py_ssize_t nvalues = len(values)
    cdef np.ndarray codes
    cdef np.ndarray categories_array
    cdef dict reverse_categories

    if nvalues <= 2 ** 8:
        # we won't try to shrink because the ``codes`` array cannot get any
        # smaller
        return factorize_strings_impl[np.uint8_t](
            values,
            nvalues,
            missing_value,
            sort,
            np.empty(nvalues, dtype=np.uint8)
        )
    elif nvalues <= 2 ** 16:
        (codes,
         categories_array,
         reverse_categories) = factorize_strings_impl[np.uint16_t](
            values,
            nvalues,
            missing_value,
            sort,
            np.empty(nvalues, np.uint16),
        )
    elif nvalues <= 2 ** 32:
        (codes,
         categories_array,
         reverse_categories) = factorize_strings_impl[np.uint32_t](
            values,
            nvalues,
            missing_value,
            sort,
            np.empty(nvalues, np.uint32),
        )
    elif nvalues <= 2 ** 64:
        (codes,
         categories_array,
         reverse_categories) = factorize_strings_impl[np.uint64_t](
            values,
            nvalues,
            missing_value,
            sort,
            np.empty(nvalues, np.uint64),
        )
    else:
        # unreachable
        raise ValueError('nvalues larger than uint64')

    if len(categories_array) < 2 ** codes.dtype.itemsize:
        # if there are a lot of duplicates in the values we may need to shrink
        # the width of the ``codes`` array
        codes = codes.astype(unsigned_int_dtype_with_size_in_bytes(
            floor(log2(len(categories_array))),
        ))

    return codes, categories_array, reverse_categories
