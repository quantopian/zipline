"""
Factorization algorithms.
"""
from cpython cimport Py_LT
from libc.math cimport log
cimport numpy as np
import numpy as np

from zipline.utils.numpy_utils import unsigned_int_dtype_with_size_in_bytes

np.import_array()


cdef inline double log2(double d):
    return log(d) / log(2);


cpdef inline smallest_uint_that_can_hold(Py_ssize_t maxval):
    """Choose the smallest numpy unsigned int dtype that can hold ``maxval``.
    """
    if maxval < 1:
        # lim x -> 0 log2(x) == -infinity so we floor at uint8
        return np.uint8
    else:
        # The number of bits required to hold the codes up to ``length`` is
        # log2(length). The number of bits per bytes is 8. We cannot have
        # fractional bytes so we need to round up. Finally, we can only have
        # integers with widths 1, 2, 4, or 8 so so we need to round up to the
        # next value by looking up the next largest size in ``_int_sizes``.
        return unsigned_int_dtype_with_size_in_bytes(
            _int_sizes[int(np.ceil(log2(maxval) / 8))]
        )


ctypedef fused unsigned_integral:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


cdef class _NoneFirstSortKey:
    """Box to sort ``None`` to the front of the categories list.
    """
    cdef object value

    def __cinit__(self, value):
        self.value = value

    def __richcmp__(_NoneFirstSortKey self, _NoneFirstSortKey other, int op):
        if op == Py_LT:
            return (
                self.value is None or
                (other.value is not None and self.value < other.value)
            )

        return NotImplemented


cdef factorize_strings_known_impl(np.ndarray[object] values,
                                  Py_ssize_t nvalues,
                                  list categories,
                                  object missing_value,
                                  bint sort,
                                  np.ndarray[unsigned_integral] codes):
    if sort:
        categories = sorted(categories, key=_NoneFirstSortKey)

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
    if missing_value not in categories:
        categories.insert(0, missing_value)

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
                            object missing_value,
                            bint sort,
                            np.ndarray[unsigned_integral] codes):
    cdef list categories = [missing_value]
    cdef dict reverse_categories = {missing_value: 0}

    cdef Py_ssize_t i, code
    cdef object key = None

    for i in range(len(values)):
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
        sorter = np.empty(ncategories, dtype=np.int64)

        # Don't include missing_value in the argsort, because None is
        # unorderable with bytes/str in py3. Always just sort it to 0.
        sorter[1:] = categories_array[1:].argsort() + 1
        sorter[0] = 0

        reverse_indexer = np.empty(ncategories, dtype=codes.dtype)
        reverse_indexer.put(sorter, np.arange(ncategories))

        codes = reverse_indexer.take(codes)
        categories_array = categories_array.take(sorter)
        reverse_categories = dict(zip(categories_array, range(ncategories)))

    return codes, categories_array, reverse_categories


cdef list _int_sizes = [1, 1, 2, 4, 4, 8, 8, 8, 8]


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

    # use exclusive less than because we need to account for the possibility
    # that the missing value is not in values
    if nvalues < 2 ** 8:
        # we won't try to shrink because the ``codes`` array cannot get any
        # smaller
        return factorize_strings_impl[np.uint8_t](
            values,
            missing_value,
            sort,
            np.empty(nvalues, dtype=np.uint8)
        )
    elif nvalues < 2 ** 16:
        (codes,
         categories_array,
         reverse_categories) = factorize_strings_impl[np.uint16_t](
            values,
            missing_value,
            sort,
            np.empty(nvalues, dtype=np.uint16),
        )
    elif nvalues < 2 ** 32:
        (codes,
         categories_array,
         reverse_categories) = factorize_strings_impl[np.uint32_t](
            values,
            missing_value,
            sort,
            np.empty(nvalues, dtype=np.uint32),
        )
    elif nvalues < 2 ** 64:
        (codes,
         categories_array,
         reverse_categories) = factorize_strings_impl[np.uint64_t](
            values,
            missing_value,
            sort,
            np.empty(nvalues, dtype=np.uint64),
        )
    else:
        # unreachable
        raise ValueError('nvalues larger than uint64')

    length = len(categories_array)
    narrowest_dtype = smallest_uint_that_can_hold(length)
    if codes.dtype != narrowest_dtype:
        # condense the codes down to the narrowest dtype possible
        codes = codes.astype(narrowest_dtype)

    return codes, categories_array, reverse_categories
