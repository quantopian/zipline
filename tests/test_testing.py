"""
Tests for our testing utilities.
"""
from itertools import product
from unittest import TestCase

from numpy import array, empty

from zipline.testing import (
    check_arrays,
    make_alternating_boolean_array,
    make_cascading_boolean_array,
    parameter_space,
)
from zipline.utils.numpy_utils import bool_dtype


class TestParameterSpace(TestCase):

    x_args = [1, 2]
    y_args = [3, 4]

    @classmethod
    def setUpClass(cls):
        cls.xy_invocations = []
        cls.yx_invocations = []

    @classmethod
    def tearDownClass(cls):
        # This is the only actual test here.
        assert cls.xy_invocations == list(product(cls.x_args, cls.y_args))
        assert cls.yx_invocations == list(product(cls.y_args, cls.x_args))

    @parameter_space(x=x_args, y=y_args)
    def test_xy(self, x, y):
        self.xy_invocations.append((x, y))

    @parameter_space(x=x_args, y=y_args)
    def test_yx(self, y, x):
        # Ensure that product is called with args in the order that they appear
        # in the function's parameter list.
        self.yx_invocations.append((y, x))

    def test_nothing(self):
        # Ensure that there's at least one "real" test in the class, or else
        # our {setUp,tearDown}Class won't be called if, for example,
        # `parameter_space` returns None.
        pass


class TestMakeBooleanArray(TestCase):

    def test_make_alternating_boolean_array(self):
        check_arrays(
            make_alternating_boolean_array((3, 3)),
            array(
                [[True,  False,  True],
                 [False,  True, False],
                 [True,  False,  True]]
            ),
        )
        check_arrays(
            make_alternating_boolean_array((3, 3), first_value=False),
            array(
                [[False,  True, False],
                 [True,  False,  True],
                 [False,  True, False]]
            ),
        )
        check_arrays(
            make_alternating_boolean_array((1, 3)),
            array([[True, False, True]]),
        )
        check_arrays(
            make_alternating_boolean_array((3, 1)),
            array([[True], [False], [True]]),
        )
        check_arrays(
            make_alternating_boolean_array((3, 0)),
            empty((3, 0), dtype=bool_dtype),
        )

    def test_make_cascading_boolean_array(self):
        check_arrays(
            make_cascading_boolean_array((3, 3)),
            array(
                [[True,   True, False],
                 [True,  False, False],
                 [False, False, False]]
            ),
        )
        check_arrays(
            make_cascading_boolean_array((3, 3), first_value=False),
            array(
                [[False, False, True],
                 [False,  True, True],
                 [True,   True, True]]
            ),
        )
        check_arrays(
            make_cascading_boolean_array((1, 3)),
            array([[True, True, False]]),
        )
        check_arrays(
            make_cascading_boolean_array((3, 1)),
            array([[False], [False], [False]]),
        )
        check_arrays(
            make_cascading_boolean_array((3, 0)),
            empty((3, 0), dtype=bool_dtype),
        )
