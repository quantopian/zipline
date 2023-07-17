import pytest
from zipline.utils.math_utils import number_of_decimal_places

fixt = [(1, 0), (3.14, 2), ("3.14", 2), (-3.14, 2)]


@pytest.mark.parametrize("value, expected", fixt)
def test_number_of_decimal_places(value, expected):
    assert number_of_decimal_places(value) == expected
