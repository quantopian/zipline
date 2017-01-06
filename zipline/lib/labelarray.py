"""
An ndarray subclass for working with arrays of strings.
"""
from functools import partial
from operator import eq, ne
import re

import numpy as np
from numpy import ndarray
import pandas as pd
from toolz import compose

from zipline.utils.compat import unicode
from zipline.utils.preprocess import preprocess
from zipline.utils.sentinel import sentinel
from zipline.utils.input_validation import (
    coerce,
    expect_kinds,
    expect_types,
    optional,
)
from zipline.utils.numpy_utils import (
    bool_dtype,
    int_dtype_with_size_in_bytes,
    is_object,
)
from zipline.utils.pandas_utils import ignore_pandas_nan_categorical_warning

from ._factorize import (
    factorize_strings,
    factorize_strings_known_categories,
)


def compare_arrays(left, right):
    "Eq check with a short-circuit for identical objects."
    return (
        left is right
        or ((left.shape == right.shape) and (left == right).all())
    )


def _make_unsupported_method(name):
    def method(*args, **kwargs):
        raise NotImplementedError(
            "Method %s is not supported on LabelArrays." % name
        )
    method.__name__ = name
    method.__doc__ = "Unsupported LabelArray Method: %s" % name
    return method


class MissingValueMismatch(ValueError):
    """
    Error raised on attempt to perform operations between LabelArrays with
    mismatched missing_values.
    """
    def __init__(self, left, right):
        super(MissingValueMismatch, self).__init__(
            "LabelArray missing_values don't match:"
            " left={}, right={}".format(left, right)
        )


class CategoryMismatch(ValueError):
    """
    Error raised on attempt to perform operations between LabelArrays with
    mismatched category arrays.
    """
    def __init__(self, left, right):
        (mismatches,) = np.where(left != right)
        assert len(mismatches), "Not actually a mismatch!"
        super(CategoryMismatch, self).__init__(
            "LabelArray categories don't match:\n"
            "Mismatched Indices: {mismatches}\n"
            "Left: {left}\n"
            "Right: {right}".format(
                mismatches=mismatches,
                left=left[mismatches],
                right=right[mismatches],
            )
        )

_NotPassed = sentinel('_NotPassed')


class LabelArray(ndarray):
    """
    An ndarray subclass for working with arrays of strings.

    Factorizes the input array into integers, but overloads equality on strings
    to check against the factor label.

    Parameters
    ----------
    values : array-like
        Array of values that can be passed to np.asarray with dtype=object.
    missing_value : str
        Scalar value to treat as 'missing' for operations on ``self``.
    categories : list[str], optional
        List of values to use as categories.  If not supplied, categories will
        be inferred as the unique set of entries in ``values``.
    sort : bool, optional
        Whether to sort categories.  If sort is False and categories is
        supplied, they are left in the order provided.  If sort is False and
        categories is None, categories will be constructed in a random order.

    Attributes
    ----------
    categories : ndarray[str]
        An array containing the unique labels of self.
    reverse_categories : dict[str -> int]
        Reverse lookup table for ``categories``. Stores the index in
        ``categories`` at which each entry each unique entry is found.
    missing_value : str or None
        A sentinel missing value with NaN semantics for comparisons.

    Notes
    -----
    Consumers should be cautious when passing instances of LabelArray to numpy
    functions.  We attempt to disallow as many meaningless operations as
    possible, but since a LabelArray is just an ndarray of ints with some
    additional metadata, many numpy functions (for example, trigonometric) will
    happily accept a LabelArray and treat its values as though they were
    integers.

    In a future change, we may be able to disallow more numerical operations by
    creating a wrapper dtype which doesn't register an implementation for most
    numpy ufuncs. Until that change is made, consumers of LabelArray should
    assume that it is undefined behavior to pass a LabelArray to any numpy
    ufunc that operates on semantically-numerical data.

    See Also
    --------
    http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html
    """
    SUPPORTED_SCALAR_TYPES = (bytes, unicode, type(None))

    @preprocess(
        values=coerce(list, partial(np.asarray, dtype=object)),
        categories=coerce(np.ndarray, list),
    )
    @expect_types(
        values=np.ndarray,
        missing_value=SUPPORTED_SCALAR_TYPES,
        categories=optional(list),
    )
    @expect_kinds(values=("O", "S", "U"))
    def __new__(cls,
                values,
                missing_value,
                categories=None,
                sort=True):

        # Numpy's fixed-width string types aren't very efficient. Working with
        # object arrays is faster than bytes or unicode arrays in almost all
        # cases.
        if not is_object(values):
            values = values.astype(object)

        if categories is None:
            codes, categories, reverse_categories = factorize_strings(
                values.ravel(),
                missing_value=missing_value,
                sort=sort,
            )
        else:
            codes, categories, reverse_categories = (
                factorize_strings_known_categories(
                    values.ravel(),
                    categories=categories,
                    missing_value=missing_value,
                    sort=sort,
                )
            )
        categories.setflags(write=False)

        return cls._from_codes_and_metadata(
            codes=codes.reshape(values.shape),
            categories=categories,
            reverse_categories=reverse_categories,
            missing_value=missing_value,
        )

    @classmethod
    def _from_codes_and_metadata(cls,
                                 codes,
                                 categories,
                                 reverse_categories,
                                 missing_value):
        """
        View codes as a LabelArray and set LabelArray metadata on the result.
        """
        ret = codes.view(type=cls, dtype=np.void)
        ret._categories = categories
        ret._reverse_categories = reverse_categories
        ret._missing_value = missing_value
        return ret

    @classmethod
    def from_categorical(cls, categorical, missing_value=None):
        """
        Create a LabelArray from a pandas categorical.

        Parameters
        ----------
        categorical : pd.Categorical
            The categorical object to convert.
        missing_value : bytes, unicode, or None, optional
            The missing value to use for this LabelArray.

        Returns
        -------
        la : LabelArray
            The LabelArray representation of this categorical.
        """
        return LabelArray(
            categorical,
            missing_value,
            categorical.categories,
        )

    @property
    def categories(self):
        # This is a property because it should be immutable.
        return self._categories

    @property
    def reverse_categories(self):
        # This is a property because it should be immutable.
        return self._reverse_categories

    @property
    def missing_value(self):
        # This is a property because it should be immutable.
        return self._missing_value

    @property
    def missing_value_code(self):
        return self.reverse_categories[self.missing_value]

    def has_label(self, value):
        return value in self.reverse_categories

    def __array_finalize__(self, obj):
        """
        Called by Numpy after array construction.

        There are three cases where this can happen:

        1. Someone tries to directly construct a new array by doing::

            >>> ndarray.__new__(LabelArray, ...)  # doctest: +SKIP

           In this case, obj will be None.  We treat this as an error case and
           fail.

        2. Someone (most likely our own __new__) does::

           >>> other_array.view(type=LabelArray)  # doctest: +SKIP

           In this case, `self` will be the new LabelArray instance, and
           ``obj` will be the array on which ``view`` is being called.

           The caller of ``obj.view`` is responsible for setting category
           metadata on ``self`` after we exit.

        3. Someone creates a new LabelArray by slicing an existing one.

           In this case, ``obj`` will be the original LabelArray.  We're
           responsible for copying over the parent array's category metadata.
        """
        if obj is None:
            raise TypeError(
                "Direct construction of LabelArrays is not supported."
            )

        # See docstring for an explanation of when these will or will not be
        # set.
        self._categories = getattr(obj, 'categories', None)
        self._reverse_categories = getattr(obj, 'reverse_categories', None)
        self._missing_value = getattr(obj, 'missing_value', None)

    def as_int_array(self):
        """
        Convert self into a regular ndarray of ints.

        This is an O(1) operation. It does not copy the underlying data.
        """
        return self.view(
            type=ndarray,
            dtype=int_dtype_with_size_in_bytes(self.itemsize),
        )

    def as_string_array(self):
        """
        Convert self back into an array of strings.

        This is an O(N) operation.
        """
        return self.categories[self.as_int_array()]

    def as_categorical(self, name=None):
        """
        Coerce self into a pandas categorical.

        This is only defined on 1D arrays, since that's all pandas supports.
        """
        if len(self.shape) > 1:
            raise ValueError("Can't convert a 2D array to a categorical.")

        with ignore_pandas_nan_categorical_warning():
            return pd.Categorical.from_codes(
                self.as_int_array(),
                # We need to make a copy because pandas >= 0.17 fails if this
                # buffer isn't writeable.
                self.categories.copy(),
                ordered=False,
                name=name,
            )

    def as_categorical_frame(self, index, columns, name=None):
        """
        Coerce self into a pandas DataFrame of Categoricals.
        """
        if len(self.shape) != 2:
            raise ValueError(
                "Can't convert a non-2D LabelArray into a DataFrame."
            )

        expected_shape = (len(index), len(columns))
        if expected_shape != self.shape:
            raise ValueError(
                "Can't construct a DataFrame with provided indices:\n\n"
                "LabelArray shape is {actual}, but index and columns imply "
                "that shape should be {expected}.".format(
                    actual=self.shape,
                    expected=expected_shape,
                )
            )

        return pd.Series(
            index=pd.MultiIndex.from_product([index, columns]),
            data=self.ravel().as_categorical(name=name),
        ).unstack()

    def __setitem__(self, indexer, value):
        self_categories = self.categories

        if isinstance(value, LabelArray):
            value_categories = value.categories
            if compare_arrays(self_categories, value_categories):
                return super(LabelArray, self).__setitem__(indexer, value)
            else:
                raise CategoryMismatch(self_categories, value_categories)

        elif isinstance(value, self.SUPPORTED_SCALAR_TYPES):
            value_code = self.reverse_categories.get(value, -1)
            if value_code < 0:
                raise ValueError("%r is not in LabelArray categories." % value)
            self.as_int_array()[indexer] = value_code
        else:
            raise NotImplementedError(
                "Setting into a LabelArray with a value of "
                "type {type} is not yet supported.".format(
                    type=type(value).__name__,
                ),
            )

    def __setslice__(self, i, j, sequence):
        """
        This method was deprecated in Python 2.0. It predates slice objects,
        but Python 2.7.11 still uses it if you implement it, which ndarray
        does.  In newer Pythons, __setitem__ is always called, but we need to
        manuallly forward in py2.
        """
        self.__setitem__(slice(i, j), sequence)

    def __getitem__(self, indexer):
        result = super(LabelArray, self).__getitem__(indexer)
        if result.ndim:
            # Result is still a LabelArray, so we can just return it.
            return result

        # Result is a scalar value, which will be an instance of np.void.
        # Map it back to one of our category entries.
        index = result.view(int_dtype_with_size_in_bytes(self.itemsize))
        return self.categories[index]

    def is_missing(self):
        """
        Like isnan, but checks for locations where we store missing values.
        """
        return (
            self.as_int_array() == self.reverse_categories[self.missing_value]
        )

    def not_missing(self):
        """
        Like ~isnan, but checks for locations where we store missing values.
        """
        return (
            self.as_int_array() != self.reverse_categories[self.missing_value]
        )

    def _equality_check(op):
        """
        Shared code for __eq__ and __ne__, parameterized on the actual
        comparison operator to use.
        """
        def method(self, other):

            if isinstance(other, LabelArray):
                self_mv = self.missing_value
                other_mv = other.missing_value
                if self_mv != other_mv:
                    raise MissingValueMismatch(self_mv, other_mv)

                self_categories = self.categories
                other_categories = other.categories
                if not compare_arrays(self_categories, other_categories):
                    raise CategoryMismatch(self_categories, other_categories)

                return (
                    op(self.as_int_array(), other.as_int_array())
                    & self.not_missing()
                    & other.not_missing()
                )

            elif isinstance(other, ndarray):
                # Compare to ndarrays as though we were an array of strings.
                # This is fairly expensive, and should generally be avoided.
                return op(self.as_string_array(), other) & self.not_missing()

            elif isinstance(other, self.SUPPORTED_SCALAR_TYPES):
                i = self._reverse_categories.get(other, -1)
                return op(self.as_int_array(), i) & self.not_missing()

            return op(super(LabelArray, self), other)
        return method

    __eq__ = _equality_check(eq)
    __ne__ = _equality_check(ne)
    del _equality_check

    def view(self, dtype=_NotPassed, type=_NotPassed):
        if type is _NotPassed and dtype not in (_NotPassed, self.dtype):
            raise TypeError("Can't view LabelArray as another dtype.")

        # The text signature on ndarray.view makes it look like the default
        # values for dtype and type are `None`, but passing None explicitly has
        # different semantics than not passing an arg at all, so we reconstruct
        # the kwargs dict here to simulate the args not being passed at all.
        kwargs = {}
        if dtype is not _NotPassed:
            kwargs['dtype'] = dtype
        if type is not _NotPassed:
            kwargs['type'] = type
        return super(LabelArray, self).view(**kwargs)

    # In general, we support resizing, slicing, and reshaping methods, but not
    # numeric methods.
    SUPPORTED_NDARRAY_METHODS = frozenset([
        'base',
        'compress',
        'copy',
        'data',
        'diagonal',
        'dtype',
        'flat',
        'flatten',
        'item',
        'itemset',
        'itemsize',
        'nbytes',
        'ndim',
        'ravel',
        'repeat',
        'reshape',
        'resize',
        'setflags',
        'shape',
        'size',
        'squeeze',
        'strides',
        'swapaxes',
        'take',
        'trace',
        'transpose',
        'view'
    ])
    PUBLIC_NDARRAY_METHODS = frozenset([
        s for s in dir(ndarray) if not s.startswith('_')
    ])

    # Generate failing wrappers for all unsupported methods.
    locals().update(
        {
            method: _make_unsupported_method(method)
            for method in PUBLIC_NDARRAY_METHODS - SUPPORTED_NDARRAY_METHODS
        }
    )

    def __repr__(self):
        repr_lines = repr(self.as_string_array()).splitlines()
        repr_lines[0] = repr_lines[0].replace('array(', 'LabelArray(', 1)
        repr_lines[-1] = repr_lines[-1].rsplit(',', 1)[0] + ')'
        # The extra spaces here account for the difference in length between
        # 'array(' and 'LabelArray('.
        return '\n     '.join(repr_lines)

    def empty_like(self, shape):
        """
        Make an empty LabelArray with the same categories as ``self``, filled
        with ``self.missing_value``.
        """
        return type(self)._from_codes_and_metadata(
            codes=np.full(
                shape,
                self.reverse_categories[self.missing_value],
                dtype=int_dtype_with_size_in_bytes(self.itemsize),
            ),
            categories=self.categories,
            reverse_categories=self.reverse_categories,
            missing_value=self.missing_value,
        )

    def map_predicate(self, f):
        """
        Map a function from str -> bool element-wise over ``self``.

        ``f`` will be applied exactly once to each non-missing unique value in
        ``self``. Missing values will always return False.
        """
        # Functions passed to this are of type str -> bool.  Don't ever call
        # them on None, which is the only non-str value we ever store in
        # categories.
        if self.missing_value is None:
            def f_to_use(x):
                return False if x is None else f(x)
        else:
            f_to_use = f

        # Call f on each unique value in our categories.
        results = np.vectorize(f_to_use, otypes=[bool_dtype])(self.categories)

        # missing_value should produce False no matter what
        results[self.reverse_categories[self.missing_value]] = False

        # unpack the results form each unique value into their corresponding
        # locations in our indices.
        return results[self.as_int_array()]

    def startswith(self, prefix):
        """
        Element-wise startswith.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        matches : np.ndarray[bool]
            An array with the same shape as self indicating whether each
            element of self started with ``prefix``.
        """
        return self.map_predicate(lambda elem: elem.startswith(prefix))

    def endswith(self, suffix):
        """
        Elementwise endswith.

        Parameters
        ----------
        suffix : str

        Returns
        -------
        matches : np.ndarray[bool]
            An array with the same shape as self indicating whether each
            element of self ended with ``suffix``
        """
        return self.map_predicate(lambda elem: elem.endswith(suffix))

    def has_substring(self, substring):
        """
        Elementwise contains.

        Parameters
        ----------
        substring : str

        Returns
        -------
        matches : np.ndarray[bool]
            An array with the same shape as self indicating whether each
            element of self ended with ``suffix``.
        """
        return self.map_predicate(lambda elem: substring in elem)

    @preprocess(pattern=coerce(from_=(bytes, unicode), to=re.compile))
    def matches(self, pattern):
        """
        Elementwise regex match.

        Parameters
        ----------
        pattern : str or compiled regex

        Returns
        -------
        matches : np.ndarray[bool]
            An array with the same shape as self indicating whether each
            element of self was matched by ``pattern``.
        """
        return self.map_predicate(compose(bool, pattern.match))

    # These types all implement an O(N) __contains__, so pre-emptively
    # coerce to `set`.
    @preprocess(container=coerce((list, tuple, np.ndarray), set))
    def element_of(self, container):
        """
        Check if each element of self is an of ``container``.

        Parameters
        ----------
        container : object
            An object implementing a __contains__ to call on each element of
            ``self``.

        Returns
        -------
        is_contained : np.ndarray[bool]
            An array with the same shape as self indicating whether each
            element of self was an element of ``container``.
        """
        return self.map_predicate(container.__contains__)
