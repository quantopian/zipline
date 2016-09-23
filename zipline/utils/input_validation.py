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
from datetime import tzinfo
from functools import partial, wraps
from operator import attrgetter

from numpy import dtype
import pandas as pd
from pytz import timezone
from six import iteritems, string_types, PY3
from toolz import valmap, complement, compose
import toolz.curried.operator as op

from zipline.utils.functional import getattrs
from zipline.utils.preprocess import call, preprocess


def verify_indices_all_unique(obj):
    """
    Check that all axes of a pandas object are unique.

    Parameters
    ----------
    obj : pd.Series / pd.DataFrame / pd.Panel
        The object to validate.

    Returns
    -------
    obj : pd.Series / pd.DataFrame / pd.Panel
        The validated object, unchanged.

    Raises
    ------
    ValueError
        If any axis has duplicate entries.
    """
    axis_names = [
        ('index',),                            # Series
        ('index', 'columns'),                  # DataFrame
        ('items', 'major_axis', 'minor_axis')  # Panel
    ][obj.ndim - 1]  # ndim = 1 should go to entry 0,

    for axis_name, index in zip(axis_names, obj.axes):
        if index.is_unique:
            continue

        raise ValueError(
            "Duplicate entries in {type}.{axis}: {dupes}.".format(
                type=type(obj).__name__,
                axis=axis_name,
                dupes=sorted(index[index.duplicated()]),
            )
        )
    return obj


def optionally(preprocessor):
    """Modify a preprocessor to explicitly allow `None`.

    Parameters
    ----------
    preprocessor : callable[callable, str, any -> any]
        A preprocessor to delegate to when `arg is not None`.

    Returns
    -------
    optional_preprocessor : callable[callable, str, any -> any]
        A preprocessor that delegates to `preprocessor` when `arg is not None`.

    Usage
    -----
    >>> def preprocessor(func, argname, arg):
    ...     if not isinstance(arg, int):
    ...         raise TypeError('arg must be int')
    ...     return arg
    ...
    >>> @preprocess(a=optionally(preprocessor))
    ... def f(a):
    ...     return a
    ...
    >>> f(1)  # call with int
    1
    >>> f('a')  # call with not int
    Traceback (most recent call last):
       ...
    TypeError: arg must be int
    >>> f(None) is None  # call with explicit None
    True
    """
    @wraps(preprocessor)
    def wrapper(func, argname, arg):
        return arg if arg is None else preprocessor(func, argname, arg)

    return wrapper


def ensure_upper_case(func, argname, arg):
    if isinstance(arg, string_types):
        return arg.upper()
    else:
        raise TypeError(
            "{0}() expected argument '{1}' to"
            " be a string, but got {2} instead.".format(
                func.__name__,
                argname,
                arg,
            ),
        )


def ensure_dtype(func, argname, arg):
    """
    Argument preprocessor that converts the input into a numpy dtype.

    Usage
    -----
    >>> import numpy as np
    >>> from zipline.utils.preprocess import preprocess
    >>> @preprocess(dtype=ensure_dtype)
    ... def foo(dtype):
    ...     return dtype
    ...
    >>> foo(float)
    dtype('float64')
    """
    try:
        return dtype(arg)
    except TypeError:
        raise TypeError(
            "{func}() couldn't convert argument "
            "{argname}={arg!r} to a numpy dtype.".format(
                func=_qualified_name(func),
                argname=argname,
                arg=arg,
            ),
        )


def ensure_timezone(func, argname, arg):
    """Argument preprocessor that converts the input into a tzinfo object.

    Usage
    -----
    >>> from zipline.utils.preprocess import preprocess
    >>> @preprocess(tz=ensure_timezone)
    ... def foo(tz):
    ...     return tz
    >>> foo('utc')
    <UTC>
    """
    if isinstance(arg, tzinfo):
        return arg
    if isinstance(arg, string_types):
        return timezone(arg)

    raise TypeError(
        "{func}() couldn't convert argument "
        "{argname}={arg!r} to a timezone.".format(
            func=_qualified_name(func),
            argname=argname,
            arg=arg,
        ),
    )


def ensure_timestamp(func, argname, arg):
    """Argument preprocessor that converts the input into a pandas Timestamp
    object.

    Usage
    -----
    >>> from zipline.utils.preprocess import preprocess
    >>> @preprocess(ts=ensure_timestamp)
    ... def foo(ts):
    ...     return ts
    >>> foo('2014-01-01')
    Timestamp('2014-01-01 00:00:00')
    """
    try:
        return pd.Timestamp(arg)
    except ValueError as e:
        raise TypeError(
            "{func}() couldn't convert argument "
            "{argname}={arg!r} to a pandas Timestamp.\n"
            "Original error was: {t}: {e}".format(
                func=_qualified_name(func),
                argname=argname,
                arg=arg,
                t=_qualified_name(type(e)),
                e=e,
            ),
        )


def expect_dtypes(**named):
    """
    Preprocessing decorator that verifies inputs have expected numpy dtypes.

    Usage
    -----
    >>> from numpy import dtype, arange, int8, float64
    >>> @expect_dtypes(x=dtype(int8))
    ... def foo(x, y):
    ...    return x, y
    ...
    >>> foo(arange(3, dtype=int8), 'foo')
    (array([0, 1, 2], dtype=int8), 'foo')
    >>> foo(arange(3, dtype=float64), 'foo')  # doctest: +NORMALIZE_WHITESPACE
    ...                                       # doctest: +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...foo() expected a value with dtype 'int8' for argument 'x',
    but got 'float64' instead.
    """
    for name, type_ in iteritems(named):
        if not isinstance(type_, (dtype, tuple)):
            raise TypeError(
                "expect_dtypes() expected a numpy dtype or tuple of dtypes"
                " for argument {name!r}, but got {dtype} instead.".format(
                    name=name, dtype=dtype,
                )
            )

    @preprocess(dtypes=call(lambda x: x if isinstance(x, tuple) else (x,)))
    def _expect_dtype(dtypes):
        """
        Factory for dtype-checking functions that work with the @preprocess
        decorator.
        """
        def error_message(func, argname, value):
            # If the bad value has a dtype, but it's wrong, show the dtype
            # name.  Otherwise just show the value.
            try:
                value_to_show = value.dtype.name
            except AttributeError:
                value_to_show = value
            return (
                "{funcname}() expected a value with dtype {dtype_str} "
                "for argument {argname!r}, but got {value!r} instead."
            ).format(
                funcname=_qualified_name(func),
                dtype_str=' or '.join(repr(d.name) for d in dtypes),
                argname=argname,
                value=value_to_show,
            )

        def _actual_preprocessor(func, argname, argvalue):
            if getattr(argvalue, 'dtype', object()) not in dtypes:
                raise TypeError(error_message(func, argname, argvalue))
            return argvalue

        return _actual_preprocessor

    return preprocess(**valmap(_expect_dtype, named))


def expect_kinds(**named):
    """
    Preprocessing decorator that verifies inputs have expected dtype kinds.

    Usage
    -----
    >>> from numpy import int64, int32, float32
    >>> @expect_kinds(x='i')
    ... def foo(x):
    ...    return x
    ...
    >>> foo(int64(2))
    2
    >>> foo(int32(2))
    2
    >>> foo(float32(2))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...foo() expected a numpy object of kind 'i' for argument 'x',
    but got 'f' instead.
    """
    for name, kind in iteritems(named):
        if not isinstance(kind, (str, tuple)):
            raise TypeError(
                "expect_dtype_kinds() expected a string or tuple of strings"
                " for argument {name!r}, but got {kind} instead.".format(
                    name=name, kind=dtype,
                )
            )

    @preprocess(kinds=call(lambda x: x if isinstance(x, tuple) else (x,)))
    def _expect_kind(kinds):
        """
        Factory for kind-checking functions that work the @preprocess
        decorator.
        """
        def error_message(func, argname, value):
            # If the bad value has a dtype, but it's wrong, show the dtype
            # kind.  Otherwise just show the value.
            try:
                value_to_show = value.dtype.kind
            except AttributeError:
                value_to_show = value
            return (
                "{funcname}() expected a numpy object of kind {kinds} "
                "for argument {argname!r}, but got {value!r} instead."
            ).format(
                funcname=_qualified_name(func),
                kinds=' or '.join(map(repr, kinds)),
                argname=argname,
                value=value_to_show,
            )

        def _actual_preprocessor(func, argname, argvalue):
            if getattrs(argvalue, ('dtype', 'kind'), object()) not in kinds:
                raise TypeError(error_message(func, argname, argvalue))
            return argvalue

        return _actual_preprocessor

    return preprocess(**valmap(_expect_kind, named))


def expect_types(*_pos, **named):
    """
    Preprocessing decorator that verifies inputs have expected types.

    Usage
    -----
    >>> @expect_types(x=int, y=str)
    ... def foo(x, y):
    ...    return x, y
    ...
    >>> foo(2, '3')
    (2, '3')
    >>> foo(2.0, '3')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...foo() expected a value of type int for argument 'x',
    but got float instead.
    """
    if _pos:
        raise TypeError("expect_types() only takes keyword arguments.")

    for name, type_ in iteritems(named):
        if not isinstance(type_, (type, tuple)):
            raise TypeError(
                "expect_types() expected a type or tuple of types for "
                "argument '{name}', but got {type_} instead.".format(
                    name=name, type_=type_,
                )
            )

    def _expect_type(type_):
        # Slightly different messages for type and tuple of types.
        _template = (
            "%(funcname)s() expected a value of type {type_or_types} "
            "for argument '%(argname)s', but got %(actual)s instead."
        )
        if isinstance(type_, tuple):
            template = _template.format(
                type_or_types=' or '.join(map(_qualified_name, type_))
            )
        else:
            template = _template.format(type_or_types=_qualified_name(type_))

        return make_check(
            TypeError,
            template,
            lambda v: not isinstance(v, type_),
            compose(_qualified_name, type),
        )

    return preprocess(**valmap(_expect_type, named))


if PY3:
    _qualified_name = attrgetter('__qualname__')
else:
    def _qualified_name(obj):
        """
        Return the fully-qualified name (ignoring inner classes) of a type.
        """
        module = obj.__module__
        if module in ('__builtin__', '__main__', 'builtins'):
            return obj.__name__
        return '.'.join([module, obj.__name__])


def make_check(exc_type, template, pred, actual):
    """
    Factory for making preprocessing functions that check a predicate on the
    input value.

    Parameters
    ----------
    exc_type : Exception
        The exception type to raise if the predicate fails.
    template : str
        A template string to use to create error messages.
        Should have %-style named template parameters for 'funcname',
        'argname', and 'actual'.
    pred : function[object -> bool]
        A function to call on the argument being preprocessed.  If the
        predicate returns `True`, we raise an instance of `exc_type`.
    actual : function[object -> object]
        A function to call on bad values to produce the value to display in the
        error message.
    """

    def _check(func, argname, argvalue):
        if pred(argvalue):
            raise exc_type(
                template % {
                    'funcname': _qualified_name(func),
                    'argname': argname,
                    'actual': actual(argvalue),
                },
            )
        return argvalue
    return _check


def optional(type_):
    """
    Helper for use with `expect_types` when an input can be `type_` or `None`.

    Returns an object such that both `None` and instances of `type_` pass
    checks of the form `isinstance(obj, optional(type_))`.

    Parameters
    ----------
    type_ : type
       Type for which to produce an option.

    Examples
    --------
    >>> isinstance({}, optional(dict))
    True
    >>> isinstance(None, optional(dict))
    True
    >>> isinstance(1, optional(dict))
    False
    """
    return (type_, type(None))


def expect_element(*_pos, **named):
    """
    Preprocessing decorator that verifies inputs are elements of some
    expected collection.

    Usage
    -----
    >>> @expect_element(x=('a', 'b'))
    ... def foo(x):
    ...    return x.upper()
    ...
    >>> foo('a')
    'A'
    >>> foo('b')
    'B'
    >>> foo('c')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value in ('a', 'b') for argument 'x',
    but got 'c' instead.

    Notes
    -----
    This uses the `in` operator (__contains__) to make the containment check.
    This allows us to use any custom container as long as the object supports
    the container protocol.
    """
    if _pos:
        raise TypeError("expect_element() only takes keyword arguments.")

    def _expect_element(collection):
        if isinstance(collection, (set, frozenset)):
            # Special case the error message for set and frozen set to make it
            # less verbose.
            collection_for_error_message = tuple(sorted(collection))
        else:
            collection_for_error_message = collection

        template = (
            "%(funcname)s() expected a value in {collection} "
            "for argument '%(argname)s', but got %(actual)s instead."
        ).format(collection=collection_for_error_message)
        return make_check(
            ValueError,
            template,
            complement(op.contains(collection)),
            repr,
        )
    return preprocess(**valmap(_expect_element, named))


def expect_bounded(**named):
    """
    Preprocessing decorator verifying that inputs fall between bounds.

    Bounds should be passed as a pair of ``(min_value, max_value)``. Both
    bounds are checked inclusively.

    ``None`` may be passed as ``min_value`` or ``max_value`` to signify that
    the input is only bounded above or below.

    Usage
    -----
    >>> @expect_bounded(x=(1, 5))
    ... def foo(x):
    ...    return x + 1
    ...
    >>> foo(1)
    2
    >>> foo(5)
    6
    >>> foo(6)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value between 1 and 5 for argument 'x',
    but got 6 instead.

    >>> @expect_bounded(x=(2, None))
    ... def foo(x):
    ...    return x
    ...
    >>> foo(100000)
    100000
    >>> foo(1)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value greater than or equal to 2 for
    argument 'x', but got 1 instead.

    >>> @expect_bounded(x=(None, 5))
    ... def foo(x):
    ...    return x
    ...
    >>> foo(6)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value less than or equal to 5 for
    argument 'x', but got 6 instead.
    """
    def valid_bounds(t):
        return (
            isinstance(t, tuple)
            and len(t) == 2
            and t != (None, None)
        )

    for name, bounds in iteritems(named):
        if not valid_bounds(bounds):
            raise TypeError(
                "expect_bounded() expected a tuple of bounds for"
                " argument '{name}', but got {bounds} instead.".format(
                    name=name,
                    bounds=bounds,
                )
            )

    def _expect_bounded(bounds):
        (lower, upper) = bounds
        if lower is None:
            should_fail = lambda value: value > upper
            predicate_descr = "less than or equal to " + str(upper)
        elif upper is None:
            should_fail = lambda value: value < lower
            predicate_descr = "greater than or equal to " + str(lower)
        else:
            should_fail = lambda value: not (lower <= value <= upper)
            predicate_descr = "between %s and %s" % bounds

        template = (
            "%(funcname)s() expected a value {predicate}"
            " for argument '%(argname)s', but got %(actual)s instead."
        ).format(predicate=predicate_descr)

        return make_check(
            exc_type=ValueError,
            template=template,
            pred=should_fail,
            actual=repr,
        )
    return preprocess(**valmap(_expect_bounded, named))


def expect_dimensions(**dimensions):
    """
    Preprocessing decorator that verifies inputs are numpy arrays with a
    specific dimensionality.

    Usage
    -----
    >>> from numpy import array
    >>> @expect_dimensions(x=1, y=2)
    ... def foo(x, y):
    ...    return x[0] + y[0, 0]
    ...
    >>> foo(array([1, 1]), array([[1, 1], [2, 2]]))
    2
    >>> foo(array([1, 1]), array([1, 1]))  # doctest: +NORMALIZE_WHITESPACE
    ...                                    # doctest: +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a 2-D array for argument 'y',
    but got a 1-D array instead.
    """
    def _expect_dimension(expected_ndim):
        def _check(func, argname, argvalue):
            funcname = _qualified_name(func)
            actual_ndim = argvalue.ndim
            if actual_ndim != expected_ndim:
                if actual_ndim == 0:
                    actual_repr = 'scalar'
                else:
                    actual_repr = "%d-D array" % actual_ndim
                raise ValueError(
                    "{func}() expected a {expected:d}-D array"
                    " for argument {argname!r}, but got a {actual}"
                    " instead.".format(
                        func=funcname,
                        expected=expected_ndim,
                        argname=argname,
                        actual=actual_repr,
                    )
                )
            return argvalue
        return _check
    return preprocess(**valmap(_expect_dimension, dimensions))


def coerce(from_, to, **to_kwargs):
    """
    A preprocessing decorator that coerces inputs of a given type by passing
    them to a callable.

    Parameters
    ----------
    from : type or tuple or types
        Inputs types on which to call ``to``.
    to : function
        Coercion function to call on inputs.
    **to_kwargs
        Additional keywords to forward to every call to ``to``.

    Usage
    -----
    >>> @preprocess(x=coerce(float, int), y=coerce(float, int))
    ... def floordiff(x, y):
    ...     return x - y
    ...
    >>> floordiff(3.2, 2.5)
    1

    >>> @preprocess(x=coerce(str, int, base=2), y=coerce(str, int, base=2))
    ... def add_binary_strings(x, y):
    ...     return bin(x + y)[2:]
    ...
    >>> add_binary_strings('101', '001')
    '110'
    """
    def preprocessor(func, argname, arg):
        if isinstance(arg, from_):
            return to(arg, **to_kwargs)
        return arg
    return preprocessor


class error_keywords(object):

    def __init__(self, *args, **kwargs):
        self.messages = kwargs

    def __call__(self, func):
        def assert_keywords_and_call(*args, **kwargs):
            for field, message in iteritems(self.messages):
                if field in kwargs:
                    raise TypeError(message)
            return func(*args, **kwargs)
        return assert_keywords_and_call


coerce_string = partial(coerce, string_types)
