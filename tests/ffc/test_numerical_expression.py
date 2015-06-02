import math
from operator import methodcaller
from unittest import TestCase

import numpy
from numpy import (
    empty,
    full,
    isnan,
)
from numpy.testing import assert_array_equal
from pandas import (
    date_range,
    Int64Index,
)

from zipline.modelling.factor import (
    NumericalExpression,
    NUMEXPR_MATH_FUNCS,
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
        self.assertFalse(isnan(expected))
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

    def test_negate(self):
        f, g = self.f, self.g

        self.check_constant_output(-f, -3.0)
        self.check_constant_output(--f, 3.0)
        self.check_constant_output(---f, -3.0)

        self.check_constant_output(-(f + f), -6.0)
        self.check_constant_output(-f + -f, -6.0)
        self.check_constant_output(-(-f + -f), 6.0)

        self.check_constant_output(f + -g, 1.0)
        self.check_constant_output(f - -g, 5.0)

        self.check_constant_output(-(f + g) + (f + g), 0.0)
        self.check_constant_output((f + g) + -(f + g), 0.0)
        self.check_constant_output(-(f + g) + -(f + g), -10.0)

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

    def test_pow(self):
        f, g = self.f, self.g

        self.check_constant_output(f ** g, 3.0 ** 2)
        self.check_constant_output(2 ** f, 2.0 ** 3)
        self.check_constant_output(f ** 2, 3.0 ** 2)

        self.check_constant_output((f + g) ** 2, (3.0 + 2.0) ** 2)
        self.check_constant_output(2 ** (f + g), 2 ** (3.0 + 2.0))

        self.check_constant_output(f ** (f ** g), 3.0 ** (3.0 ** 2.0))
        self.check_constant_output((f ** f) ** g, (3.0 ** 3.0) ** 2.0)

        self.check_constant_output((f ** g) ** (f ** g), 9.0 ** 9.0)
        self.check_constant_output((f ** g) ** (g ** f), 9.0 ** 8.0)
        self.check_constant_output((g ** f) ** (f ** g), 8.0 ** 9.0)
        self.check_constant_output((g ** f) ** (g ** f), 8.0 ** 8.0)

    def test_math_functions(self):
        f, g = self.f, self.g

        fake_raw_data = self.fake_raw_data
        alt_fake_raw_data = {
            self.f: full((5, 5), .5),
            self.g: full((5, 5), -.5),
        }

        for funcname in NUMEXPR_MATH_FUNCS:
            method = methodcaller(funcname)
            func = getattr(numpy, funcname)

            # These methods have domains in [0, 1], so we need alternate inputs
            # that are in the domain.
            if funcname in ('arcsin', 'arccos', 'arctanh'):
                self.fake_raw_data = alt_fake_raw_data
            else:
                self.fake_raw_data = fake_raw_data

            f_val = self.fake_raw_data[f][0, 0]
            g_val = self.fake_raw_data[g][0, 0]

            self.check_constant_output(method(f), func(f_val))
            self.check_constant_output(method(g), func(g_val))

            self.check_constant_output(method(f) + 1, func(f_val) + 1)
            self.check_constant_output(1 + method(f), 1 + func(f_val))

            self.check_constant_output(method(f + .25), func(f_val + .25))
            self.check_constant_output(method(.25 + f), func(.25 + f_val))

            self.check_constant_output(
                method(f) + method(g),
                func(f_val) + func(g_val),
            )
            self.check_constant_output(
                method(f + g),
                func(f_val + g_val),
            )
