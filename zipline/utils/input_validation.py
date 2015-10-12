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
from toolz import valmap

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


def _expect_type(type_):
    """
    Factory for type-checking functions that work the @preprocess decorator.
    """
    # Slightly different messages for type and tuple of types.
    _template = (
        "{{funcname}}() expected a value of type {type_or_types} "
        "for argument '{{argname}}', but got {{actual}} instead."
    )
    if isinstance(type_, tuple):
        template = _template.format(
            type_or_types=' or '.join(map(_qualified_name, type_))
        )
    else:
        template = _template.format(type_or_types=_qualified_name(type_))

    def _check_type(func, argname, argvalue):
        if not isinstance(argvalue, type_):
            raise TypeError(
                template.format(
                    funcname=_qualified_name(func),
                    argname=argname,
                    actual=_qualified_name(type(argvalue)),
                )
            )
        return argvalue
    return _check_type


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
