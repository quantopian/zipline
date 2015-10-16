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

from six import iteritems, string_types
from toolz import valmap, complement, compose
import toolz.curried.operator as op

from zipline.utils.preprocess import preprocess


def ensure_upper_case(func, argname, arg):
    if isinstance(arg, string_types):
        return arg.upper()
    else:
        raise TypeError(
            "{0}() expected argument '{1}' to"
            " be a string, but got {2} instead.".format(
                func.__name__, argname, arg,)
        )


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

    return preprocess(**valmap(_expect_type, named))


def _qualified_name(obj):
    """
    Return the fully-qualified name (ignoring inner classes) of a type.
    """
    module = obj.__module__
    if module in ('__builtin__', '__main__', 'builtins'):
        return obj.__name__
    return '.'.join([module, obj.__name__])


def _mk_check(exc, template, pred, actual):
    def _check(func, argname, argvalue):
        if pred(argvalue):
            raise exc(
                template % {
                    'funcname': _qualified_name(func),
                    'argname': argname,
                    'actual': actual(argvalue),
                },
            )
        return argvalue
    return _check


def _expect_type(type_):
    """
    Factory for type-checking functions that work the @preprocess decorator.
    """
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

    return _mk_check(
        TypeError,
        template,
        lambda v: not isinstance(v, type_),
        compose(_qualified_name, type),
    )


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

    return preprocess(**valmap(_expect_element, named))


def _expect_element(collection):
    template = (
        "%(funcname)s() expected a value in {collection} "
        "for argument '%(argname)s', but got %(actual)s instead."
    ).format(collection=collection)
    return _mk_check(
        ValueError,
        template,
        complement(op.contains(collection)),
        repr,
    )
