"""
Tests for zipline.utils.validate.
"""
from operator import attrgetter
from types import FunctionType

import numpy as np
import pytz
import pytest
import re

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


qualname = attrgetter("__qualname__")


class TestPreprocess:
    @pytest.mark.parametrize(
        "name, args, kwargs",
        [
            ("too_many", (1, 2, 3), {}),
            ("too_few", (1,), {}),
            ("collision", (1,), {"a": 1}),
            ("unexpected", (1,), {"q": 1}),
        ],
    )
    def test_preprocess_doesnt_change_TypeErrors(self, name, args, kwargs):
        """
        Verify that the validate decorator doesn't swallow typeerrors that
        would be raised when calling a function with invalid arguments
        """

        def undecorated(x, y):
            return x, y

        decorated = preprocess(x=noop, y=noop)(undecorated)

        with pytest.raises(TypeError) as excinfo:
            undecorated(*args, **kwargs)
        undecorated_errargs = excinfo.value.args

        with pytest.raises(TypeError) as excinfo:
            decorated(*args, **kwargs)
        decorated_errargs = excinfo.value.args

        assert len(decorated_errargs) == 1
        assert len(undecorated_errargs) == 1

        assert decorated_errargs[0] == undecorated_errargs[0]

    def test_preprocess_co_filename(self):
        def undecorated():
            pass

        decorated = preprocess()(undecorated)

        assert undecorated.__code__.co_filename == decorated.__code__.co_filename

    def test_preprocess_preserves_docstring(self):
        @preprocess()
        def func():
            "My awesome docstring"

        assert func.__doc__ == "My awesome docstring"

    def test_preprocess_preserves_function_name(self):
        @preprocess()
        def arglebargle():
            pass

        assert arglebargle.__name__ == "arglebargle"

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, 2), {}),
            ((1, 2), {"c": 3}),
            ((1,), {"b": 2}),
            ((), {"a": 1, "b": 2}),
            ((), {"a": 1, "b": 2, "c": 3}),
        ],
    )
    def test_preprocess_no_processors(self, args, kwargs):
        @preprocess()
        def func(a, b, c=3):
            return a, b, c

        assert func(*args, **kwargs) == (1, 2, 3)

    def test_preprocess_bad_processor_name(self):
        a_processor = preprocess(a=int)

        # Should work fine.
        @a_processor
        def func_with_arg_named_a(a):
            pass

        @a_processor
        def func_with_default_arg_named_a(a=1):
            pass

        message = "Got processors for unknown arguments: %s." % {"a"}
        with pytest.raises(TypeError, match=message):

            @a_processor
            def func_with_no_args():
                pass

        with pytest.raises(TypeError, match=message):

            @a_processor
            def func_with_arg_named_b(b):
                pass

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, 2), {}),
            ((1, 2), {"c": 3}),
            ((1,), {"b": 2}),
            ((), {"a": 1, "b": 2}),
            ((), {"a": 1, "b": 2, "c": 3}),
        ],
    )
    def test_preprocess_on_function(self, args, kwargs):

        decorators = [
            preprocess(a=call(str), b=call(float), c=call(lambda x: x + 1)),
        ]

        for decorator in decorators:

            @decorator
            def func(a, b, c=3):
                return a, b, c

            assert func(*args, **kwargs), ("1", 2.0, 4)

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((1, 2), {}),
            ((1, 2), {"c": 3}),
            ((1,), {"b": 2}),
            ((), {"a": 1, "b": 2}),
            ((), {"a": 1, "b": 2, "c": 3}),
        ],
    )
    def test_preprocess_on_method(self, args, kwargs):
        decorators = [
            preprocess(a=call(str), b=call(float), c=call(lambda x: x + 1)),
        ]

        for decorator in decorators:

            class Foo:
                @decorator
                def method(self, a, b, c=3):
                    return a, b, c

                @classmethod
                @decorator
                def clsmeth(cls, a, b, c=3):
                    return a, b, c

            assert Foo.clsmeth(*args, **kwargs) == ("1", 2.0, 4)
            assert Foo().method(*args, **kwargs) == ("1", 2.0, 4)

    def test_expect_types(self):
        @expect_types(a=int, b=int)
        def foo(a, b, c):
            return a, b, c

        assert foo(1, 2, 3) == (1, 2, 3)
        assert foo(1, 2, c=3) == (1, 2, 3)
        assert foo(1, b=2, c=3) == (1, 2, 3)
        assert foo(1, 2, c="3") == (1, 2, "3")

        for not_int in (str, float):
            msg = (
                "{qualname}() expected a value of type int for argument 'a', "
                "but got {t} instead.".format(
                    qualname=qualname(foo),
                    t=not_int.__name__,
                )
            )
            with pytest.raises(TypeError, match=re.escape(msg)):
                foo(not_int(1), 2, 3)

            with pytest.raises(TypeError):
                foo(1, not_int(2), 3)

            with pytest.raises(TypeError):
                foo(not_int(1), not_int(2), 3)

    def test_expect_types_custom_funcname(self):
        class Foo:
            @expect_types(__funcname="ArgleBargle", a=int)
            def __init__(self, a):
                self.a = a

        foo = Foo(1)
        assert foo.a == 1

        for not_int in (str, float):
            msg = (
                "ArgleBargle() expected a value of type int for argument 'a', "
                "but got {t} instead.".format(
                    t=not_int.__name__,
                )
            )
            with pytest.raises(TypeError, match=re.escape(msg)):
                Foo(not_int(1))

    def test_expect_types_with_tuple(self):
        @expect_types(a=(int, float))
        def foo(a):
            return a

        assert foo(1) == 1
        assert foo(1.0) == 1.0

        expected_message = (
            "{qualname}() expected a value of "
            "type int or float for argument 'a', but got str instead."
        ).format(qualname=qualname(foo))
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            foo("1")

    def test_expect_optional_types(self):
        @expect_types(a=optional(int))
        def foo(a=None):
            return a

        assert foo() is None
        assert foo(None) is None
        assert foo(a=None) is None

        assert foo(1) == 1
        assert foo(a=1) == 1

        expected_message = (
            "{qualname}() expected a value of "
            "type int or NoneType for argument 'a', but got str instead."
        ).format(qualname=qualname(foo))
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            foo("1")

    def test_expect_element(self):
        set_ = {"a", "b"}

        @expect_element(a=set_)
        def f(a):
            return a

        assert f("a") == "a"
        assert f("b") == "b"

        expected_message = (
            "{qualname}() expected a value in {set_!r}"
            " for argument 'a', but got 'c' instead."
        ).format(
            # We special-case set to show a tuple instead of the set repr.
            set_=tuple(sorted(set_)),
            qualname=qualname(f),
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            f("c")

    def test_expect_element_custom_funcname(self):

        set_ = {"a", "b"}

        class Foo:
            @expect_element(__funcname="ArgleBargle", a=set_)
            def __init__(self, a):
                self.a = a

        expected_message = (
            "ArgleBargle() expected a value in {set_!r}"
            " for argument 'a', but got 'c' instead."
        ).format(
            # We special-case set to show a tuple instead of the set repr.
            set_=tuple(sorted(set_)),
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            Foo("c")

    def test_expect_dtypes(self):
        @expect_dtypes(a=np.dtype(float), b=np.dtype("datetime64[ns]"))
        def foo(a, b, c):
            return a, b, c

        good_a = np.arange(3, dtype=float)
        good_b = np.arange(3).astype("datetime64[ns]")
        good_c = object()

        a_ret, b_ret, c_ret = foo(good_a, good_b, good_c)
        assert a_ret is good_a
        assert b_ret is good_b
        assert c_ret is good_c

        expected_message = (
            "{qualname}() expected a value with dtype 'datetime64[ns]'"
            " for argument 'b', but got 'int64' instead."
        ).format(qualname=qualname(foo))
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            foo(good_a, np.arange(3, dtype="int64"), good_c)

        expected_message = (
            "{qualname}() expected a value with dtype 'float64'"
            " for argument 'a', but got 'uint32' instead."
        ).format(qualname=qualname(foo))
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            foo(np.arange(3, dtype="uint32"), good_c, good_c)

    def test_expect_dtypes_with_tuple(self):

        allowed_dtypes = (np.dtype("datetime64[ns]"), np.dtype("float"))

        @expect_dtypes(a=allowed_dtypes)
        def foo(a, b):
            return a, b

        for d in allowed_dtypes:
            good_a = np.arange(3).astype(d)
            good_b = object()
            ret_a, ret_b = foo(good_a, good_b)
            assert good_a is ret_a
            assert good_b is ret_b

        expected_message = (
            "{qualname}() expected a value with dtype 'datetime64[ns]' "
            "or 'float64' for argument 'a', but got 'uint32' instead."
        ).format(qualname=qualname(foo))
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            foo(np.arange(3, dtype="uint32"), object())

    def test_expect_dtypes_custom_funcname(self):

        allowed_dtypes = (np.dtype("datetime64[ns]"), np.dtype("float"))

        class Foo:
            @expect_dtypes(__funcname="Foo", a=allowed_dtypes)
            def __init__(self, a):
                self.a = a

        expected_message = (
            "Foo() expected a value with dtype 'datetime64[ns]' "
            "or 'float64' for argument 'a', but got 'uint32' instead."
        )
        with pytest.raises(TypeError, match=re.escape(expected_message)):
            Foo(np.arange(3, dtype="uint32"))

    def test_ensure_timezone(self):
        @preprocess(tz=ensure_timezone)
        def f(tz):
            return tz

        valid = {
            "utc",
            "EST",
            "US/Eastern",
        }
        invalid = {
            # unfortunately, these are not actually timezones (yet)
            "ayy",
            "lmao",
        }

        # test coercing from string
        for tz in valid:
            assert f(tz) == pytz.timezone(tz)

        # test pass through of tzinfo objects
        for tz in map(pytz.timezone, valid):
            assert f(tz) == tz

        # test invalid timezone strings
        for tz in invalid:
            pytest.raises(pytz.UnknownTimeZoneError, f, tz)

    def test_optionally(self):
        error = TypeError("arg must be int")

        def preprocessor(func, argname, arg):
            if not isinstance(arg, int):
                raise error
            return arg

        @preprocess(a=optionally(preprocessor))
        def f(a):
            return a

        assert f(1) == 1
        assert f(None) is None

        with pytest.raises(TypeError, match=str(error)):
            f("a")

    def test_expect_dimensions(self):
        @expect_dimensions(x=2)
        def foo(x, y):
            return x[0, 0]

        assert foo(np.arange(1).reshape(1, 1), 10) == 0

        expected = (
            "{qualname}() expected a 2-D array for argument 'x', but got"
            " a 1-D array instead.".format(qualname=qualname(foo))
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            foo(np.arange(1), 1)

        expected = (
            "{qualname}() expected a 2-D array for argument 'x', but got"
            " a 3-D array instead.".format(qualname=qualname(foo))
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            foo(np.arange(1).reshape(1, 1, 1), 1)

        expected = (
            "{qualname}() expected a 2-D array for argument 'x', but got"
            " a scalar instead.".format(qualname=qualname(foo))
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            foo(np.array(0), 1)

    def test_expect_dimensions_custom_name(self):
        @expect_dimensions(__funcname="fizzbuzz", x=2)
        def foo(x, y):
            return x[0, 0]

        expected = (
            "fizzbuzz() expected a 2-D array for argument 'x', but got"
            " a 1-D array instead."
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            foo(np.arange(1), 1)
