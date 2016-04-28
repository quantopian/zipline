"""
An ndarray subclass for working with arrays of strings.
"""
from functools import partial
from numbers import Number
from operator import eq, ne
import re

import numpy as np
from numpy import ndarray
import pandas as pd
from toolz import compose

from zipline.utils.preprocess import preprocess
from zipline.utils.sentinel import sentinel
from zipline.utils.input_validation import (
    coerce,
    expect_kinds,
    expect_types,
    optional,
)
from zipline.utils.numpy_utils import is_object, int64_dtype

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

    See Also
    --------
    http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html
    """
    @preprocess(
        values=coerce(list, partial(np.asarray, dtype=object)),
    )
    @expect_types(
        values=np.ndarray,
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

        ret = codes.reshape(values.shape).view(type=cls)
        ret._categories = categories
        ret._reverse_categories = reverse_categories
        ret._missing_value = missing_value
        return ret

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

    def __array_finalize__(self, obj):
        """
        Called by Numpy after array construction.

        There are three cases where this can happen:

        1. Someone tries to directly construct a new array by doing::

            >>> ndarray.__new__(LabelArray, ...)

           In this case, obj will be None.  We treat this as an error case and
           fail.

        2. Someone (most likely our own __new__) calls
           other_array.view(type=LabelArray).

           In this case, `self` will be the new LabelArray instance, and
           ``obj` will be the array on which ``view`` is being called.

           The caller of ``obj.view`` is responsible for copying setting
           category metadata on ``self`` after we exit.

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

    def __array_wrap__(self, obj, context=None):
        """
        Called by numpy after completion of a ufunc.

        We coerce back into a vanilla ndarray if our dtype changed, since that
        indicates that our categories are no longer meaningful.
        """
        if obj.dtype != self.dtype:
            return obj.view(type=np.ndarray)
        return obj

    def as_int_array(self):
        """
        Convert self into a regular ndarray of ints.

        This is an O(1) operation. It does not copy the underlying data.
        """
        return self.view(type=ndarray)

    def as_string_array(self):
        """
        Convert self back into an array of strings.

        This is an O(N) operation.
        """
        return self.categories[self]

    def as_categorical(self, name=None):
        """
        Coerce self into a pandas categorical.

        This is only defined on 1D arrays, since that's all pandas supports.
        """
        if len(self.shape) > 1:
            raise ValueError("Can't convert a 2D array to a categorical.")
        return pd.Categorical.from_codes(
            self.as_int_array(),
            self.categories,
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

        elif isinstance(value, (bytes, unicode)):
            value_code = self.reverse_categories.get(value, None)
            if value_code is None:
                raise ValueError("%r is not in LabelArray categories." % value)
            return super(LabelArray, self).__setitem__(indexer, value_code)

        else:
            raise NotImplementedError(
                "Setting into a LabelArray with a value of "
                "type {type} is not yet supported.".format(
                    type=type(value).__name__,
                ),
            )

    def _equality_check(op):
        """
        Shared code for __eq__ and __ne__, parameterized on the actual
        comparison operator to use.
        """
        # What value should we return if we compare against a value not in our
        # categories?
        if op is eq:
            COMPARE_TO_UNKNOWN = False
        elif op is ne:
            COMPARE_TO_UNKNOWN = True
        else:
            raise AssertionError("_make_equality_check called with %s" % op)

        def method(self, other):
            self_categories = self.categories

            if isinstance(other, LabelArray):
                other_categories = other.categories
                if compare_arrays(self_categories, other_categories):
                    return op(self.as_int_array(), other.as_int_array())
                else:
                    raise CategoryMismatch(self_categories, other_categories)

            elif isinstance(other, ndarray):
                # Compare to ndarrays as though we were an array of strings.
                # This is fairly expensive, and should generally be avoided.
                return op(self.as_string_array(), other)

            elif isinstance(other, (bytes, unicode)):
                i = self._reverse_categories.get(other, None)
                if i is None:
                    # Requested string isn't in our categories.  Short circuit.
                    # This isn't full_like because that would try to return a
                    # LabelArray.
                    return np.full(self.shape, COMPARE_TO_UNKNOWN, dtype=bool)

                return op(self.as_int_array(), i)

            elif isinstance(other, Number):
                return NotImplemented

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
        'byteswap',
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
        'newbyteorder',
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
        # This happens if you call a ufunc on a LabelArray that changes the
        # dtype.  This is generally an indicator that the array has been used
        # incorrectly, and it means we're no longer valid for anything.
        if self.dtype != int64_dtype:
            return "Invalid LabelArray: dtype={}, shape={}".format(
                self.dtype, self.shape
            )
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
        out = np.full(
            shape,
            self.reverse_categories[self.missing_value],
            dtype=self.dtype
        ).view(
            type=type(self)
        )

        out._categories = self.categories
        out._reverse_categories = self.reverse_categories
        out._missing_value = self.missing_value

        return out

    def apply(self, f, dtype):
        """
        Map a function elementwise over entries in ``self``.

        ``f`` will be applied exactly once to each unique value in ``self``.
        """
        return np.vectorize(f, otypes=[dtype])(self.categories)[self]

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
        return self.apply(lambda elem: elem.startswith(prefix), dtype=bool)

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
            element of self ended with ``suffix``.w
        """
        return self.apply(lambda elem: elem.endswith(suffix), dtype=bool)

    def contains(self, substring):
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
        return self.apply(lambda elem: substring in elem, dtype=bool)

    @preprocess(pattern=re.compile)
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
        return self.apply(compose(bool, pattern.match))
