"""
Tests for zipline.lib.adjustment
"""
from zipline.lib import adjustment as adj
from zipline.utils.numpy_utils import make_datetime64ns
import pytest


class TestAdjustment:
    @pytest.mark.parametrize(
        "name, adj_type",
        [
            ("add", adj.ADD),
            ("multiply", adj.MULTIPLY),
            ("overwrite", adj.OVERWRITE),
        ],
    )
    def test_make_float_adjustment(self, name, adj_type):
        expected_types = {
            "add": adj.Float64Add,
            "multiply": adj.Float64Multiply,
            "overwrite": adj.Float64Overwrite,
        }
        result = adj.make_adjustment_from_indices(
            1,
            2,
            3,
            4,
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
        assert result == expected

    def test_make_int_adjustment(self):
        result = adj.make_adjustment_from_indices(
            1,
            2,
            3,
            4,
            adjustment_kind=adj.OVERWRITE,
            value=1,
        )
        expected = adj.Int64Overwrite(
            first_row=1,
            last_row=2,
            first_col=3,
            last_col=4,
            value=1,
        )
        assert result == expected

    def test_make_datetime_adjustment(self):
        overwrite_dt = make_datetime64ns(0)
        result = adj.make_adjustment_from_indices(
            1,
            2,
            3,
            4,
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
        assert result == expected

    @pytest.mark.parametrize(
        "value",
        [
            "some text",
            "some text".encode(),
            None,
        ],
    )
    def test_make_object_adjustment(self, value):
        result = adj.make_adjustment_from_indices(
            1,
            2,
            3,
            4,
            adjustment_kind=adj.OVERWRITE,
            value=value,
        )

        expected = adj.ObjectOverwrite(
            first_row=1,
            last_row=2,
            first_col=3,
            last_col=4,
            value=value,
        )
        assert result == expected

    def test_unsupported_type(self):
        class SomeClass:
            pass

        expected_msg = (
            "Don't know how to make overwrite adjustments for values of type "
            "%r." % SomeClass
        )
        with pytest.raises(TypeError, match=expected_msg):
            adj.make_adjustment_from_indices(
                1,
                2,
                3,
                4,
                adjustment_kind=adj.OVERWRITE,
                value=SomeClass(),
            )
