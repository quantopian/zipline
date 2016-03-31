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
from pytz import timezone
from six import iteritems, string_types, PY3
from toolz import valmap, complement, compose
import toolz.curried.operator as op

from zipline.utils.preprocess import preprocess


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


def expect_dtypes(*_pos, **named):
    """
    Preprocessing decorator that verifies inputs have expected numpy dtypes.

    Usage
    -----
    >>> from numpy import dtype, arange
    >>> @expect_dtypes(x=dtype(int))
    ... def foo(x, y):
    ...    return x, y
    ...
    >>> foo(arange(3), 'foo')
    (array([0, 1, 2]), 'foo')
    >>> foo(arange(3, dtype=float), 'foo')
    Traceback (most recent call last):
       ...
    TypeError: foo() expected an argument with dtype 'int64' for argument 'x', but got dtype 'float64' instead.  # noqa
    """
    if _pos:
        raise TypeError("expect_dtypes() only takes keyword arguments.")

    for name, type_ in iteritems(named):
        if not isinstance(type_, (dtype, tuple)):
            raise TypeError(
                "expect_dtypes() expected a numpy dtype or tuple of dtypes"
                " for argument {name!r}, but got {dtype} instead.".format(
                    name=name, dtype=dtype,
                )
            )

    def _expect_dtype(_dtype_or_dtype_tuple):
        """
        Factory for dtype-checking functions that work the @preprocess
        decorator.
        """
        # Slightly different messages for dtype and tuple of dtypes.
        if isinstance(_dtype_or_dtype_tuple, tuple):
            allowed_dtypes = _dtype_or_dtype_tuple
        else:
            allowed_dtypes = (_dtype_or_dtype_tuple,)
        template = (
            "%(funcname)s() expected a value with dtype {dtype_str} "
            "for argument '%(argname)s', but got %(actual)r instead."
        ).format(dtype_str=' or '.join(repr(d.name) for d in allowed_dtypes))

        def check_dtype(value):
            return getattr(value, 'dtype', None) not in allowed_dtypes

        def display_bad_value(value):
            # If the bad value has a dtype, but it's wrong, show the dtype
            # name.
            try:
                return value.dtype.name
            except AttributeError:
                return value

        return make_check(
            exc_type=TypeError,
            template=template,
            pred=check_dtype,
            actual=display_bad_value,
        )

    return preprocess(**valmap(_expect_dtype, named))


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
    >>> foo(2.0, '3')
    Traceback (most recent call last):
       ...
    TypeError: foo() expected an argument of type 'int' for argument 'x', but got float instead.  # noqa
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
    >>> foo('c')
    Traceback (most recent call last):
       ...
    ValueError: foo() expected a value in ('a', 'b') for argument 'x', but got 'c' instead.  # noqa

    Notes
    -----
    This uses the `in` operator (__contains__) to make the containment check.
    This allows us to use any custom container as long as the object supports
    the container protocol.
    """
    if _pos:
        raise TypeError("expect_element() only takes keyword arguments.")

    def _expect_element(collection):
        template = (
            "%(funcname)s() expected a value in {collection} "
            "for argument '%(argname)s', but got %(actual)s instead."
        ).format(collection=collection)
        return make_check(
            ValueError,
            template,
            complement(op.contains(collection)),
            repr,
        )
    return preprocess(**valmap(_expect_element, named))


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
    >>> foo(array([1, 1], array([1, 1])))
    Traceback (most recent call last):
       ...
    TypeError: foo() expected a 2-D array for argument 'y', but got a 1-D array instead.  # noqa
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


coerce_string = partial(coerce, string_types)
