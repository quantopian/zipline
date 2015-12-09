"""
Tests for zipline.lib.adjustment
"""
from unittest import TestCase
from nose_parameterized import parameterized

from zipline.lib import adjustment as adj
from zipline.utils.numpy_utils import make_datetime64ns


class AdjustmentTestCase(TestCase):

    @parameterized.expand([
        ('add', adj.ADD),
        ('multiply', adj.MULTIPLY),
        ('overwrite', adj.OVERWRITE),
    ])
    def test_make_float_adjustment(self, name, adj_type):
        expected_types = {
            'add': adj.Float64Add,
            'multiply': adj.Float64Multiply,
            'overwrite': adj.Float64Overwrite,
        }
        result = adj.make_adjustment_from_indices(
            1, 2, 3, 4,
            adjustment_kind=adj_type,
            value=0.5,
        )
        expected = expected_types[name](
            first_row=1,
            last_row=2,
            first_col=3,
            last_col=4,
            value=0.5,
        )
        self.assertEqual(result, expected)

    def test_make_datetime_adjustment(self):
        overwrite_dt = make_datetime64ns(0)
        result = adj.make_adjustment_from_indices(
            1, 2, 3, 4,
            adjustment_kind=adj.OVERWRITE,
            value=overwrite_dt,
        )
        expected = adj.Datetime64Overwrite(
            first_row=1,
            last_row=2,
            first_col=3,
            last_col=4,
            value=overwrite_dt,
        )
        self.assertEqual(result, expected)

    def test_unsupported_type(self):
        class SomeClass(object):
            pass

        with self.assertRaises(TypeError) as e:
            adj.make_adjustment_from_indices(
                1, 2, 3, 4,
                adjustment_kind=adj.OVERWRITE,
                value=SomeClass(),
            )

        exc = e.exception
        expected_msg = (
            "Don't know how to make overwrite adjustments for values of type "
            "%r." % SomeClass
        )
        self.assertEqual(str(exc), expected_msg)
