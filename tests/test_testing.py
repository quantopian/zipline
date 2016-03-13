"""
Tests for our testing utilities.
"""
from itertools import product
from unittest import TestCase

from zipline.testing import parameter_space


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
