#
# Copyright 2014 Quantopian, Inc.
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
from unittest import TestCase

from zipline.utils.argcheck import (
    verify_callable_argspec,
    Argument,
    NoStarargs,
    UnexpectedStarargs,
    NoKwargs,
    UnexpectedKwargs,
    NotCallable,
    NotEnoughArguments,
    TooManyArguments,
    MismatchedArguments,
)


class TestArgCheck(TestCase):
    def test_not_callable(self):
        """
        Check the results of a non-callable object.
        """
        not_callable = 'a'

        with self.assertRaises(NotCallable):
            verify_callable_argspec(not_callable)

    def test_no_starargs(self):
        """
        Tests when a function does not have *args and it was expected.
        """
        def f(a):
            pass

        with self.assertRaises(NoStarargs):
            verify_callable_argspec(f, expect_starargs=True)

    def test_starargs(self):
        """
        Tests when a function has *args and it was expected.
        """
        def f(*args):
            pass

        verify_callable_argspec(f, expect_starargs=True)

    def test_unexcpected_starargs(self):
        """
        Tests a function that unexpectedly accepts *args.
        """
        def f(*args):
            pass

        with self.assertRaises(UnexpectedStarargs):
            verify_callable_argspec(f, expect_starargs=False)

    def test_ignore_starargs(self):
        """
        Tests checking a function ignoring the presence of *args.
        """
        def f(*args):
            pass

        def g():
            pass

        verify_callable_argspec(f, expect_starargs=Argument.ignore)
        verify_callable_argspec(g, expect_starargs=Argument.ignore)

    def test_no_kwargs(self):
        """
        Tests when a function does not have **kwargs and it was expected.
        """
        def f():
            pass

        with self.assertRaises(NoKwargs):
            verify_callable_argspec(f, expect_kwargs=True)

    def test_kwargs(self):
        """
        Tests when a function has **kwargs and it was expected.
        """
        def f(**kwargs):
            pass

        verify_callable_argspec(f, expect_kwargs=True)

    def test_unexpected_kwargs(self):
        """
        Tests a function that unexpectedly accepts **kwargs.
        """
        def f(**kwargs):
            pass

        with self.assertRaises(UnexpectedKwargs):
            verify_callable_argspec(f, expect_kwargs=False)

    def test_ignore_kwargs(self):
        """
        Tests checking a function ignoring the presence of **kwargs.
        """
        def f(**kwargs):
            pass

        def g():
            pass

        verify_callable_argspec(f, expect_kwargs=Argument.ignore)
        verify_callable_argspec(g, expect_kwargs=Argument.ignore)

    def test_arg_subset(self):
        """
        Tests when the args are a subset of the expectations.
        """
        def f(a, b):
            pass

        with self.assertRaises(NotEnoughArguments):
            verify_callable_argspec(
                f, [Argument('a'), Argument('b'), Argument('c')]
            )

    def test_arg_superset(self):
        def f(a, b, c):
            pass

        with self.assertRaises(TooManyArguments):
            verify_callable_argspec(f, [Argument('a'), Argument('b')])

    def test_no_default(self):
        """
        Tests when an argument expects a default and it is not present.
        """
        def f(a):
            pass

        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('a', 1)])

    def test_default(self):
        """
        Tests when an argument expects a default and it is present.
        """
        def f(a=1):
            pass

        verify_callable_argspec(f, [Argument('a', 1)])

    def test_ignore_default(self):
        """
        Tests that ignoring defaults works as intended.
        """
        def f(a=1):
            pass

        verify_callable_argspec(f, [Argument('a')])

    def test_mismatched_args(self):
        def f(a, b):
            pass

        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('c'), Argument('d')])

    def test_ignore_args(self):
        """
        Tests the ignore argument list feature.
        """
        def f(a):
            pass

        def g():
            pass

        h = 'not_callable'

        verify_callable_argspec(f)
        verify_callable_argspec(g)
        with self.assertRaises(NotCallable):
            verify_callable_argspec(h)

    def test_out_of_order(self):
        """
        Tests the case where arguments are not in the correct order.
        """
        def f(a, b):
            pass

        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('b'), Argument('a')])

    def test_wrong_default(self):
        """
        Tests the case where a default is expected, but the default provided
        does not match the one expected.
        """
        def f(a=1):
            pass

        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(f, [Argument('a', 2)])

    def test_any_default(self):
        """
        Tests the any_default option.
        """
        def f(a=1):
            pass

        def g(a=2):
            pass

        def h(a):
            pass

        expected_args = [Argument('a', Argument.any_default)]
        verify_callable_argspec(f, expected_args)
        verify_callable_argspec(g, expected_args)
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(h, expected_args)

    def test_ignore_name(self):
        """
        Tests ignoring a param name.
        """
        def f(a):
            pass

        def g(b):
            pass

        def h(c=1):
            pass

        expected_args = [Argument(Argument.ignore, Argument.no_default)]
        verify_callable_argspec(f, expected_args)
        verify_callable_argspec(f, expected_args)
        with self.assertRaises(MismatchedArguments):
            verify_callable_argspec(h, expected_args)

    def test_bound_method(self):
        class C(object):
            def f(self, a, b):
                pass

        method = C().f

        verify_callable_argspec(method, [Argument('a'), Argument('b')])
        with self.assertRaises(NotEnoughArguments):
            # Assert that we don't count self.
            verify_callable_argspec(
                method,
                [Argument('self'), Argument('a'), Argument('b')],
            )
