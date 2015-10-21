"""
Tests for zipline.utils.validate.
"""
from types import FunctionType
from unittest import TestCase
from nose_parameterized import parameterized

from zipline.utils.preprocess import call, preprocess
from zipline.utils.input_validation import (
    expect_element,
    expect_types,
    optional,
)


def noop(func, argname, argvalue):
    assert isinstance(func, FunctionType)
    assert isinstance(argname, str)
    return argvalue


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
                "{modname}.foo() expected a value of type "
                "int for argument 'a', but got {t} instead.".format(
                    modname=foo.__module__,
                    t=not_int.__name__,
                )
            )
            with self.assertRaises(TypeError):
                foo(1, not_int(2), 3)
            with self.assertRaises(TypeError):
                foo(not_int(1), not_int(2), 3)

    def test_expect_types_with_tuple(self):
        @expect_types(a=(int, float))
        def foo(a):
            return a

        self.assertEqual(foo(1), 1)
        self.assertEqual(foo(1.0), 1.0)

        with self.assertRaises(TypeError) as e:
            foo('1')

        expected_message = (
            "{modname}.foo() expected a value of "
            "type int or float for argument 'a', but got str instead."
        ).format(modname=foo.__module__)
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
            "{modname}.foo() expected a value of "
            "type int or NoneType for argument 'a', but got str instead."
        ).format(modname=foo.__module__)
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
            "{modname}.f() expected a value in {set_!r}"
            " for argument 'a', but got 'c' instead."
        ).format(set_=set_, modname=f.__module__)
        self.assertEqual(e.exception.args[0], expected_message)
