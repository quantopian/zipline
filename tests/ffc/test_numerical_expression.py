from unittest import TestCase

from numpy import (
    empty,
    full,
)
from numpy.testing import assert_array_equal
from pandas import (
    date_range,
    Int64Index,
)

from zipline.modelling.factor import (
    NumericalExpression,
    TestFactor,
)


class F(TestFactor):
    inputs = ()
    window_length = 0


class G(TestFactor):
    inputs = ()
    window_length = 0


class H(TestFactor):
    inputs = ()
    window_length = 0


class NumericalExpressionTestCase(TestCase):

    def setUp(self):
        self.dates = date_range('2014-01-01', periods=5, freq='D')
        self.assets = Int64Index(range(5))
        self.f = F()
        self.g = G()
        self.fake_raw_data = {
            self.f: full((5, 5), 3),
            self.g: full((5, 5), 2),
        }

    def check_constant_output(self, expr, expected):
        outbuf = empty(shape=(5, 5), dtype=float)
        expr.compute_from_arrays(
            [self.fake_raw_data[input_] for input_ in expr.inputs],
            outbuf,
            self.dates,
            self.assets,
        )
        assert_array_equal(outbuf, full((5, 5), expected))

    def test_validate_good(self):
        f = self.f
        g = self.g

        NumericalExpression("x_0", (f,))
        NumericalExpression("x_0 ", (f,))
        NumericalExpression("x_0 + x_0", (f,))
        NumericalExpression("x_0 + 2", (f,))
        NumericalExpression("2 * x_0", (f,))
        NumericalExpression("x_0 + x_1", (f, g))
        NumericalExpression("x_0 + x_1 + x_0", (f, g))
        NumericalExpression("x_0 + 1 + x_1", (f, g))

    def test_validate_bad(self):
        f, g, h = F(), G(), H()

        # Too few inputs.
        with self.assertRaises(ValueError):
            NumericalExpression("x_0", ())
        with self.assertRaises(ValueError):
            NumericalExpression("x_0 + x_1", (f,))

        # Too many inputs.
        with self.assertRaises(ValueError):
            NumericalExpression("x_0", (f, g))
        with self.assertRaises(ValueError):
            NumericalExpression("x_0 + x_1", (f, g, h))

        # Invalid variable name.
        with self.assertRaises(ValueError):
            NumericalExpression("x_0x_1", (f,))
        with self.assertRaises(ValueError):
            NumericalExpression("x_0x_1", (f, g))

        # Variable index must start at 0.
        with self.assertRaises(ValueError):
            NumericalExpression("x_1", (f,))

        # Scalar operands must be numeric.
        with self.assertRaises(TypeError):
            "2" + f
        with self.assertRaises(TypeError):
            f + "2"

    def test_add(self):
        f, g = self.f, self.g

        self.check_constant_output(f + g, 5.0)

        self.check_constant_output((1 + f) + g, 6.0)
        self.check_constant_output(1 + (f + g), 6.0)
        self.check_constant_output((f + 1) + g, 6.0)
        self.check_constant_output(f + (1 + g), 6.0)
        self.check_constant_output((f + g) + 1, 6.0)
        self.check_constant_output(f + (g + 1), 6.0)

        self.check_constant_output((f + f) + f, 9.0)
        self.check_constant_output(f + (f + f), 9.0)

        self.check_constant_output((f + g) + f, 8.0)
        self.check_constant_output(f + (g + f), 8.0)

        self.check_constant_output((f + g) + (f + g), 10.0)
        self.check_constant_output((f + g) + (g + f), 10.0)
        self.check_constant_output((g + f) + (f + g), 10.0)
        self.check_constant_output((g + f) + (g + f), 10.0)

    def test_subtract(self):
        f, g = self.f, self.g

        self.check_constant_output(f - g, 1.0)  # 3 - 2

        self.check_constant_output((1 - f) - g, -4.)   # (1 - 3) - 2
        self.check_constant_output(1 - (f - g), 0.0)   # 1 - (3 - 2)
        self.check_constant_output((f - 1) - g, 0.0)   # (3 - 1) - 2
        self.check_constant_output(f - (1 - g), 4.0)   # 3 - (1 - 2)
        self.check_constant_output((f - g) - 1, 0.0)   # (3 - 2) - 1
        self.check_constant_output(f - (g - 1), 2.0)   # 3 - (2 - 1)

        self.check_constant_output((f - f) - f, -3.)   # (3 - 3) - 3
        self.check_constant_output(f - (f - f), 3.0)   # 3 - (3 - 3)

        self.check_constant_output((f - g) - f, -2.)   # (3 - 2) - 3
        self.check_constant_output(f - (g - f), 4.0)   # 3 - (2 - 3)

        self.check_constant_output((f - g) - (f - g), 0.0)  # (3 - 2) - (3 - 2)
        self.check_constant_output((f - g) - (g - f), 2.0)  # (3 - 2) - (2 - 3)
        self.check_constant_output((g - f) - (f - g), -2.)  # (2 - 3) - (3 - 2)
        self.check_constant_output((g - f) - (g - f), 0.0)  # (2 - 3) - (2 - 3)

    def test_multiply(self):
        f, g = self.f, self.g

        self.check_constant_output(f * g, 6.0)

        self.check_constant_output((2 * f) * g, 12.0)
        self.check_constant_output(2 * (f * g), 12.0)
        self.check_constant_output((f * 2) * g, 12.0)
        self.check_constant_output(f * (2 * g), 12.0)
        self.check_constant_output((f * g) * 2, 12.0)
        self.check_constant_output(f * (g * 2), 12.0)

        self.check_constant_output((f * f) * f, 27.0)
        self.check_constant_output(f * (f * f), 27.0)

        self.check_constant_output((f * g) * f, 18.0)
        self.check_constant_output(f * (g * f), 18.0)

        self.check_constant_output((f * g) * (f * g), 36.0)
        self.check_constant_output((f * g) * (g * f), 36.0)
        self.check_constant_output((g * f) * (f * g), 36.0)
        self.check_constant_output((g * f) * (g * f), 36.0)

        self.check_constant_output(f * f * f * 0 * f * f, 0.0)

    def test_divide(self):
        f, g = self.f, self.g

        self.check_constant_output(f / g, 3.0 / 2.0)

        self.check_constant_output(
            (2 / f) / g,
            (2 / 3.0) / 2.0
        )
        self.check_constant_output(
            2 / (f / g),
            2 / (3.0 / 2.0),
        )
        self.check_constant_output(
            (f / 2) / g,
            (3.0 / 2) / 2.0,
        )
        self.check_constant_output(
            f / (2 / g),
            3.0 / (2 / 2.0),
        )
        self.check_constant_output(
            (f / g) / 2,
            (3.0 / 2.0) / 2,
        )
        self.check_constant_output(
            f / (g / 2),
            3.0 / (2.0 / 2),
        )
        self.check_constant_output(
            (f / f) / f,
            (3.0 / 3.0) / 3.0
        )
        self.check_constant_output(
            f / (f / f),
            3.0 / (3.0 / 3.0),
        )
        self.check_constant_output(
            (f / g) / f,
            (3.0 / 2.0) / 3.0,
        )
        self.check_constant_output(
            f / (g / f),
            3.0 / (2.0 / 3.0),
        )

        self.check_constant_output(
            (f / g) / (f / g),
            (3.0 / 2.0) / (3.0 / 2.0),
        )
        self.check_constant_output(
            (f / g) / (g / f),
            (3.0 / 2.0) / (2.0 / 3.0),
        )
        self.check_constant_output(
            (g / f) / (f / g),
            (2.0 / 3.0) / (3.0 / 2.0),
        )
        self.check_constant_output(
            (g / f) / (g / f),
            (2.0 / 3.0) / (2.0 / 3.0),
        )
