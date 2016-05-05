"""
Factorization algorithms.
"""
from numpy cimport ndarray, int64_t, PyArray_Check, import_array
from numpy import arange, asarray, empty, int64, isnan, ndarray, zeros

import_array()


cpdef factorize_strings_known_categories(ndarray[object] values,
                                         list categories,
                                         object missing_value,
                                         int sort):
    """
    Factorize an array whose categories are already known.

    Any entries not in the specified categories will be given the code for
    `missing_value`.
    """
    if missing_value not in categories:
        categories.insert(0, missing_value)

    if sort:
        categories = sorted(categories)

    cdef:
        Py_ssize_t      nvalues = len(values)
        dict reverse_categories = dict(
            zip(categories, range(len(categories)))
        )

    if not nvalues:
        return (
            asarray([], dtype=int64),
            asarray(categories, dtype=object),
            reverse_categories,
        )

    cdef:
        Py_ssize_t            i
        Py_ssize_t missing_code = reverse_categories[missing_value]
        ndarray[int64_t]  codes = empty(nvalues, dtype=int64)

    for i in range(nvalues):
        codes[i] = reverse_categories.get(values[i], missing_code)

    return codes, asarray(categories, dtype=object), reverse_categories


cpdef factorize_strings(ndarray[object] values,
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
    cdef:
        Py_ssize_t      nvalues = len(values)
        list         categories = [missing_value]
        dict reverse_categories = {missing_value: 0}

    # Short circuit on empty array.
    if not nvalues:
        return (
            asarray([], dtype=int64),
            asarray(categories, dtype=object),
            reverse_categories,
        )

    cdef:
        Py_ssize_t      i, code
        object              key = None
        ndarray[int64_t]  codes = empty(nvalues, dtype=int64)

    for i in range(nvalues):
        key = values[i]
        code = reverse_categories.get(key, -1)
        if code == -1:
            # Assign new code.
            code = len(reverse_categories)
            reverse_categories[key] = code
            categories.append(key)
        codes[i] = code

    cdef ndarray[int64_t, ndim=1] sorter
    cdef ndarray[int64_t, ndim=1] reverse_indexer
    cdef int ncategories
    cdef ndarray[object] categories_array = asarray(categories, dtype=object)

    if sort:
        # This is all adapted from pandas.core.algorithms.factorize.
        ncategories = len(categories_array)
        sorter = zeros(ncategories, dtype=int64)

        # Don't include missing_value in the argsort, because None is
        # unorderable with bytes/str in py3. Always just sort it to 0.
        sorter[1:] = categories_array[1:].argsort() + 1
        reverse_indexer = empty(ncategories, dtype=int64)
        reverse_indexer.put(sorter, arange(ncategories))

        codes = reverse_indexer.take(codes)
        categories_array = categories_array.take(sorter)
        reverse_categories = dict(zip(categories_array, range(ncategories)))

    return codes, categories_array, reverse_categories
