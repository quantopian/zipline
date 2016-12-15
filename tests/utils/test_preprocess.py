"""
Tests for zipline.utils.validate.
"""
from operator import attrgetter
from types import FunctionType
from unittest import TestCase

from nose_parameterized import parameterized
from numpy import arange, array, dtype
import pytz
from six import PY3

from zipline.utils.preprocess import call, preprocess
from zipline.utils.input_validation import (
    expect_dimensions,
    ensure_timezone,
    expect_element,
    expect_dtypes,
    expect_types,
    optional,
    optionally,
)


def noop(func, argname, argvalue):
    assert isinstance(func, FunctionType)
    assert isinstance(argname, str)
    return argvalue


if PY3:
    qualname = attrgetter('__qualname__')
else:
    def qualname(ob):
        return '.'.join((__name__, ob.__name__))


class PreprocessTestCase(TestCase):

    @parameterized.expand([
        ('too_many', (1, 2, 3), {}),
        ('too_few', (1,), {}),
        ('collision', (1,), {'a': 1}),
        ('unexpected', (1,), {'q': 1}),
    ])
    def test_preprocess_doesnt_change_TypeErrors(self, name, args, kwargs):
        """
        Verify that the validate decorator doesn't swallow typeerrors that
        would be raised when calling a function with invalid arguments
        """
        def undecorated(x, y):
            return x, y

        decorated = preprocess(x=noop, y=noop)(undecorated)

        with self.assertRaises(TypeError) as e:
            undecorated(*args, **kwargs)
        undecorated_errargs = e.exception.args

        with self.assertRaises(TypeError) as e:
            decorated(*args, **kwargs)
        decorated_errargs = e.exception.args

        self.assertEqual(len(decorated_errargs), 1)
        self.assertEqual(len(undecorated_errargs), 1)

        self.assertEqual(decorated_errargs[0], undecorated_errargs[0])

    def test_preprocess_co_filename(self):

        def undecorated():
            pass

        decorated = preprocess()(undecorated)

        self.assertEqual(
            undecorated.__code__.co_filename,
            decorated.__code__.co_filename,
        )

    def test_preprocess_preserves_docstring(self):

        @preprocess()
        def func():
            "My awesome docstring"

        self.assertEqual(func.__doc__, "My awesome docstring")

    def test_preprocess_preserves_function_name(self):

        @preprocess()
        def arglebargle():
            pass

        self.assertEqual(arglebargle.__name__, 'arglebargle')

    @parameterized.expand([
        ((1, 2), {}),
        ((1, 2), {'c': 3}),
        ((1,), {'b': 2}),
        ((), {'a': 1, 'b': 2}),
        ((), {'a': 1, 'b': 2, 'c': 3}),
    ])
    def test_preprocess_no_processors(self, args, kwargs):

        @preprocess()
        def func(a, b, c=3):
            return a, b, c

        self.assertEqual(func(*args, **kwargs), (1, 2, 3))

    def test_preprocess_bad_processor_name(self):
        a_processor = preprocess(a=int)

        # Should work fine.
        @a_processor
        def func_with_arg_named_a(a):
            pass

        @a_processor
        def func_with_default_arg_named_a(a=1):
            pass

        message = "Got processors for unknown arguments: %s." % {'a'}
        with self.assertRaises(TypeError) as e:
            @a_processor
            def func_with_no_args():
                pass
        self.assertEqual(e.exception.args[0], message)

        with self.assertRaises(TypeError) as e:
            @a_processor
            def func_with_arg_named_b(b):
                pass
        self.assertEqual(e.exception.args[0], message)

    @parameterized.expand([
        ((1, 2), {}),
        ((1, 2), {'c': 3}),
        ((1,), {'b': 2}),
        ((), {'a': 1, 'b': 2}),
        ((), {'a': 1, 'b': 2, 'c': 3}),
    ])
    def test_preprocess_on_function(self, args, kwargs):

        decorators = [
            preprocess(a=call(str), b=call(float), c=call(lambda x: x + 1)),
        ]

        for decorator in decorators:
            @decorator
            def func(a, b, c=3):
                return a, b, c
            self.assertEqual(func(*args, **kwargs), ('1', 2.0, 4))

    @parameterized.expand([
        ((1, 2), {}),
        ((1, 2), {'c': 3}),
        ((1,), {'b': 2}),
        ((), {'a': 1, 'b': 2}),
        ((), {'a': 1, 'b': 2, 'c': 3}),
    ])
    def test_preprocess_on_method(self, args, kwargs):
        decorators = [
            preprocess(a=call(str), b=call(float), c=call(lambda x: x + 1)),
        ]

        for decorator in decorators:
            class Foo(object):

                @decorator
                def method(self, a, b, c=3):
                    return a, b, c

                @classmethod
                @decorator
                def clsmeth(cls, a, b, c=3):
                    return a, b, c

            self.assertEqual(Foo.clsmeth(*args, **kwargs), ('1', 2.0, 4))
            self.assertEqual(Foo().method(*args, **kwargs), ('1', 2.0, 4))

    def test_expect_types(self):

        @expect_types(a=int, b=int)
        def foo(a, b, c):
            return a, b, c

        self.assertEqual(foo(1, 2, 3), (1, 2, 3))
        self.assertEqual(foo(1, 2, c=3), (1, 2, 3))
        self.assertEqual(foo(1, b=2, c=3), (1, 2, 3))
        self.assertEqual(foo(1, 2, c='3'), (1, 2, '3'))

        for not_int in (str, float):
            with self.assertRaises(TypeError) as e:
                foo(not_int(1), 2, 3)
            self.assertEqual(
                e.exception.args[0],
                "{qualname}() expected a value of type "
                "int for argument 'a', but got {t} instead.".format(
                    qualname=qualname(foo),
                    t=not_int.__name__,
                )
            )
            with self.assertRaises(TypeError):
                foo(1, not_int(2), 3)
            with self.assertRaises(TypeError):
                foo(not_int(1), not_int(2), 3)

    def test_expect_types_custom_funcname(self):

        class Foo(object):
            @expect_types(__funcname='ArgleBargle', a=int)
            def __init__(self, a):
                self.a = a

        foo = Foo(1)
        self.assertEqual(foo.a, 1)

        for not_int in (str, float):
            with self.assertRaises(TypeError) as e:
                Foo(not_int(1))
            self.assertEqual(
                e.exception.args[0],
                "ArgleBargle() expected a value of type "
                "int for argument 'a', but got {t} instead.".format(
                    t=not_int.__name__,
                )
            )

    def test_expect_types_with_tuple(self):
        @expect_types(a=(int, float))
        def foo(a):
            return a

        self.assertEqual(foo(1), 1)
        self.assertEqual(foo(1.0), 1.0)

        with self.assertRaises(TypeError) as e:
            foo('1')

        expected_message = (
            "{qualname}() expected a value of "
            "type int or float for argument 'a', but got str instead."
        ).format(qualname=qualname(foo))
        self.assertEqual(e.exception.args[0], expected_message)

    def test_expect_optional_types(self):

        @expect_types(a=optional(int))
        def foo(a=None):
            return a

        self.assertIs(foo(), None)
        self.assertIs(foo(None), None)
        self.assertIs(foo(a=None), None)

        self.assertEqual(foo(1), 1)
        self.assertEqual(foo(a=1), 1)

        with self.assertRaises(TypeError) as e:
            foo('1')

        expected_message = (
            "{qualname}() expected a value of "
            "type int or NoneType for argument 'a', but got str instead."
        ).format(qualname=qualname(foo))
        self.assertEqual(e.exception.args[0], expected_message)

    def test_expect_element(self):
        set_ = {'a', 'b'}

        @expect_element(a=set_)
        def f(a):
            return a

        self.assertEqual(f('a'), 'a')
        self.assertEqual(f('b'), 'b')

        with self.assertRaises(ValueError) as e:
            f('c')

        expected_message = (
            "{qualname}() expected a value in {set_!r}"
            " for argument 'a', but got 'c' instead."
        ).format(
            # We special-case set to show a tuple instead of the set repr.
            set_=tuple(sorted(set_)),
            qualname=qualname(f),
        )
        self.assertEqual(e.exception.args[0], expected_message)

    def test_expect_element_custom_funcname(self):

        set_ = {'a', 'b'}

        class Foo(object):
            @expect_element(__funcname='ArgleBargle', a=set_)
            def __init__(self, a):
                self.a = a

        with self.assertRaises(ValueError) as e:
            Foo('c')

        expected_message = (
            "ArgleBargle() expected a value in {set_!r}"
            " for argument 'a', but got 'c' instead."
        ).format(
            # We special-case set to show a tuple instead of the set repr.
            set_=tuple(sorted(set_)),
        )
        self.assertEqual(e.exception.args[0], expected_message)

    def test_expect_dtypes(self):

        @expect_dtypes(a=dtype(float), b=dtype('datetime64[ns]'))
        def foo(a, b, c):
            return a, b, c

        good_a = arange(3, dtype=float)
        good_b = arange(3).astype('datetime64[ns]')
        good_c = object()

        a_ret, b_ret, c_ret = foo(good_a, good_b, good_c)
        self.assertIs(a_ret, good_a)
        self.assertIs(b_ret, good_b)
        self.assertIs(c_ret, good_c)

        with self.assertRaises(TypeError) as e:
            foo(good_a, arange(3, dtype='int64'), good_c)

        expected_message = (
            "{qualname}() expected a value with dtype 'datetime64[ns]'"
            " for argument 'b', but got 'int64' instead."
        ).format(qualname=qualname(foo))
        self.assertEqual(e.exception.args[0], expected_message)

        with self.assertRaises(TypeError) as e:
            foo(arange(3, dtype='uint32'), good_c, good_c)

        expected_message = (
            "{qualname}() expected a value with dtype 'float64'"
            " for argument 'a', but got 'uint32' instead."
        ).format(qualname=qualname(foo))
        self.assertEqual(e.exception.args[0], expected_message)

    def test_expect_dtypes_with_tuple(self):

        allowed_dtypes = (dtype('datetime64[ns]'), dtype('float'))

        @expect_dtypes(a=allowed_dtypes)
        def foo(a, b):
            return a, b

        for d in allowed_dtypes:
            good_a = arange(3).astype(d)
            good_b = object()
            ret_a, ret_b = foo(good_a, good_b)
            self.assertIs(good_a, ret_a)
            self.assertIs(good_b, ret_b)

        with self.assertRaises(TypeError) as e:
            foo(arange(3, dtype='uint32'), object())

        expected_message = (
            "{qualname}() expected a value with dtype 'datetime64[ns]' "
            "or 'float64' for argument 'a', but got 'uint32' instead."
        ).format(qualname=qualname(foo))
        self.assertEqual(e.exception.args[0], expected_message)

    def test_expect_dtypes_custom_funcname(self):

        allowed_dtypes = (dtype('datetime64[ns]'), dtype('float'))

        class Foo(object):
            @expect_dtypes(__funcname='Foo', a=allowed_dtypes)
            def __init__(self, a):
                self.a = a

        with self.assertRaises(TypeError) as e:
            Foo(arange(3, dtype='uint32'))

        expected_message = (
            "Foo() expected a value with dtype 'datetime64[ns]' "
            "or 'float64' for argument 'a', but got 'uint32' instead."
        )
        self.assertEqual(e.exception.args[0], expected_message)

    def test_ensure_timezone(self):
        @preprocess(tz=ensure_timezone)
        def f(tz):
            return tz

        valid = {
            'utc',
            'EST',
            'US/Eastern',
        }
        invalid = {
            # unfortunatly, these are not actually timezones (yet)
            'ayy',
            'lmao',
        }

        # test coercing from string
        for tz in valid:
            self.assertEqual(f(tz), pytz.timezone(tz))

        # test pass through of tzinfo objects
        for tz in map(pytz.timezone, valid):
            self.assertEqual(f(tz), tz)

        # test invalid timezone strings
        for tz in invalid:
            self.assertRaises(pytz.UnknownTimeZoneError, f, tz)

    def test_optionally(self):
        error = TypeError('arg must be int')

        def preprocessor(func, argname, arg):
            if not isinstance(arg, int):
                raise error
            return arg

        @preprocess(a=optionally(preprocessor))
        def f(a):
            return a

        self.assertIs(f(1), 1)
        self.assertIsNone(f(None))

        with self.assertRaises(TypeError) as e:
            f('a')
        self.assertIs(e.exception, error)

    def test_expect_dimensions(self):

        @expect_dimensions(x=2)
        def foo(x, y):
            return x[0, 0]

        self.assertEqual(foo(arange(1).reshape(1, 1), 10), 0)

        with self.assertRaises(ValueError) as e:
            foo(arange(1), 1)
        errmsg = str(e.exception)
        expected = (
            "{qualname}() expected a 2-D array for argument 'x', but got"
            " a 1-D array instead.".format(qualname=qualname(foo))
        )
        self.assertEqual(errmsg, expected)

        with self.assertRaises(ValueError) as e:
            foo(arange(1).reshape(1, 1, 1), 1)
        errmsg = str(e.exception)
        expected = (
            "{qualname}() expected a 2-D array for argument 'x', but got"
            " a 3-D array instead.".format(qualname=qualname(foo))
        )
        self.assertEqual(errmsg, expected)

        with self.assertRaises(ValueError) as e:
            foo(array(0), 1)
        errmsg = str(e.exception)
        expected = (
            "{qualname}() expected a 2-D array for argument 'x', but got"
            " a scalar instead.".format(qualname=qualname(foo))
        )
        self.assertEqual(errmsg, expected)

    def test_expect_dimensions_custom_name(self):

        @expect_dimensions(__funcname='fizzbuzz', x=2)
        def foo(x, y):
            return x[0, 0]

        with self.assertRaises(ValueError) as e:
            foo(arange(1), 1)
        errmsg = str(e.exception)
        expected = (
            "fizzbuzz() expected a 2-D array for argument 'x', but got"
            " a 1-D array instead.".format(qualname=qualname(foo))
        )
        self.assertEqual(errmsg, expected)
