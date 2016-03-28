from itertools import permutations
from operator import (
    add,
    ge,
    gt,
    le,
    lt,
    methodcaller,
    mul,
    ne,
    sub,
)
from string import ascii_uppercase
from unittest import TestCase

import numpy
from numpy import (
    arange,
    array,
    eye,
    float64,
    full,
    isnan,
    zeros,
)
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
)

from zipline.pipeline import Factor, Filter
from zipline.pipeline.expression import (
    NumericalExpression,
    NUMEXPR_MATH_FUNCS,
)
from zipline.testing import check_allclose
from zipline.utils.numpy_utils import datetime64ns_dtype, float64_dtype


class F(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class G(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class H(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class NonExprFilter(Filter):
    inputs = ()
    window_length = 0


class DateFactor(Factor):
    dtype = datetime64ns_dtype
    inputs = ()
    window_length = 0


class NumericalExpressionTestCase(TestCase):

    def setUp(self):
        self.dates = date_range('2014-01-01', periods=5, freq='D')
        self.assets = Int64Index(range(5))
        self.f = F()
        self.g = G()
        self.h = H()
        self.d = DateFactor()
        self.fake_raw_data = {
            self.f: full((5, 5), 3, float),
            self.g: full((5, 5), 2, float),
            self.h: full((5, 5), 1, float),
            self.d: full((5, 5), 0, dtype='datetime64[ns]'),
        }
        self.mask = DataFrame(True, index=self.dates, columns=self.assets)

    def check_output(self, expr, expected):
        result = expr._compute(
            [self.fake_raw_data[input_] for input_ in expr.inputs],
            self.mask.index,
            self.mask.columns,
            self.mask.values,
        )
        check_allclose(result, expected)

    def check_constant_output(self, expr, expected):
        self.assertFalse(isnan(expected))
        return self.check_output(expr, full((5, 5), expected, float))

    def test_validate_good(self):
        f = self.f
        g = self.g

        NumericalExpression("x_0", (f,), dtype=float64_dtype)
        NumericalExpression("x_0 ", (f,), dtype=float64_dtype)
        NumericalExpression("x_0 + x_0", (f,), dtype=float64_dtype)
        NumericalExpression("x_0 + 2", (f,), dtype=float64_dtype)
        NumericalExpression("2 * x_0", (f,), dtype=float64_dtype)
        NumericalExpression("x_0 + x_1", (f, g), dtype=float64_dtype)
        NumericalExpression("x_0 + x_1 + x_0", (f, g), dtype=float64_dtype)
        NumericalExpression("x_0 + 1 + x_1", (f, g), dtype=float64_dtype)

    def test_validate_bad(self):
        f, g, h = self.f, self.g, self.h

        # Too few inputs.
        with self.assertRaises(ValueError):
            NumericalExpression("x_0", (), dtype=float64_dtype)
        with self.assertRaises(ValueError):
            NumericalExpression("x_0 + x_1", (f,), dtype=float64_dtype)

        # Too many inputs.
        with self.assertRaises(ValueError):
            NumericalExpression("x_0", (f, g), dtype=float64_dtype)
        with self.assertRaises(ValueError):
            NumericalExpression("x_0 + x_1", (f, g, h), dtype=float64_dtype)

        # Invalid variable name.
        with self.assertRaises(ValueError):
            NumericalExpression("x_0x_1", (f,), dtype=float64_dtype)
        with self.assertRaises(ValueError):
            NumericalExpression("x_0x_1", (f, g), dtype=float64_dtype)

        # Variable index must start at 0.
        with self.assertRaises(ValueError):
            NumericalExpression("x_1", (f,), dtype=float64_dtype)

        # Scalar operands must be numeric.
        with self.assertRaises(TypeError):
            "2" + f
        with self.assertRaises(TypeError):
            f + "2"
        with self.assertRaises(TypeError):
            f > "2"

        # Boolean binary operators must be between filters.
        with self.assertRaises(TypeError):
            f + (f > 2)
        with self.assertRaises(TypeError):
            (f > f) > f

    def test_many_inputs(self):
        """
        Test adding NumericalExpressions with >10 inputs.
        """
        # Create an initial NumericalExpression by adding two factors together.
        f = self.f
        expr = f + f

        self.fake_raw_data = {f: full((5, 5), 0, float)}
        expected = 0

        # Alternate between adding and subtracting factors. Because subtraction
        # is not commutative, this ensures that we are combining factors in the
        # correct order.
        ops = (add, sub)

        for i, name in enumerate(ascii_uppercase):
            op = ops[i % 2]
            NewFactor = type(
                name,
                (Factor,),
                dict(dtype=float64_dtype, inputs=(), window_length=0),
            )
            new_factor = NewFactor()

            # Again we need a NumericalExpression, so add two factors together.
            new_expr = new_factor + new_factor
            self.fake_raw_data[new_factor] = full((5, 5), i + 1, float)
            expr = op(expr, new_expr)

            # Double the expected output since each factor is counted twice.
            expected = op(expected, (i + 1) * 2)

        self.check_output(expr, full((5, 5), expected, float))

    def test_combine_datetimes(self):
        with self.assertRaises(TypeError) as e:
            self.d + self.d
        message = e.exception.args[0]
        expected = (
            "Don't know how to compute datetime64[ns] + datetime64[ns].\n"
            "Arithmetic operators are only supported between Factors of dtype "
            "'float64'."
        )
        self.assertEqual(message, expected)

        # Confirm that * shows up in the error instead of +.
        with self.assertRaises(TypeError) as e:
            self.d * self.d
        message = e.exception.args[0]
        expected = (
            "Don't know how to compute datetime64[ns] * datetime64[ns].\n"
            "Arithmetic operators are only supported between Factors of dtype "
            "'float64'."
        )
        self.assertEqual(message, expected)

    def test_combine_datetime_with_float(self):
        # Test with both float-type factors and numeric values.
        for float_value in (self.f, float64(1.0), 1.0):
            for op, sym in ((add, '+'), (mul, '*')):
                with self.assertRaises(TypeError) as e:
                    op(self.f, self.d)
                message = e.exception.args[0]
                expected = (
                    "Don't know how to compute float64 {sym} datetime64[ns].\n"
                    "Arithmetic operators are only supported between Factors"
                    " of dtype 'float64'."
                ).format(sym=sym)
                self.assertEqual(message, expected)

                with self.assertRaises(TypeError) as e:
                    op(self.d, self.f)
                message = e.exception.args[0]
                expected = (
                    "Don't know how to compute datetime64[ns] {sym} float64.\n"
                    "Arithmetic operators are only supported between Factors"
                    " of dtype 'float64'."
                ).format(sym=sym)
                self.assertEqual(message, expected)

    def test_negate_datetime(self):
        with self.assertRaises(TypeError) as e:
            -self.d

        message = e.exception.args[0]
        expected = (
            "Can't apply unary operator '-' to instance of "
            "'DateFactor' with dtype 'datetime64[ns]'.\n"
            "'-' is only supported for Factors of dtype 'float64'."
        )
        self.assertEqual(message, expected)

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

    def test_mod(self):
        f, g = self.f, self.g

        self.check_constant_output(f % g, 3.0 % 2.0)
        self.check_constant_output(f % 2.0, 3.0 % 2.0)
        self.check_constant_output(g % f, 2.0 % 3.0)

        self.check_constant_output((f + g) % 2, (3.0 + 2.0) % 2)
        self.check_constant_output(2 % (f + g), 2 % (3.0 + 2.0))

        self.check_constant_output(f % (f % g), 3.0 % (3.0 % 2.0))
        self.check_constant_output((f % f) % g, (3.0 % 3.0) % 2.0)

        self.check_constant_output((f + g) % (f * g), 5.0 % 6.0)

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

    def test_comparisons(self):
        f, g, h = self.f, self.g, self.h
        self.fake_raw_data = {
            f: arange(25, dtype=float).reshape(5, 5),
            g: arange(25, dtype=float).reshape(5, 5) - eye(5),
            h: full((5, 5), 5, dtype=float),
        }
        f_data = self.fake_raw_data[f]
        g_data = self.fake_raw_data[g]

        cases = [
            # Sanity Check with hand-computed values.
            (f, g, eye(5), zeros((5, 5))),
            (f, 10, f_data, 10),
            (10, f, 10, f_data),
            (f, f, f_data, f_data),
            (f + 1, f, f_data + 1, f_data),
            (1 + f, f, 1 + f_data, f_data),
            (f, g, f_data, g_data),
            (f + 1, g, f_data + 1, g_data),
            (f, g + 1, f_data, g_data + 1),
            (f + 1, g + 1, f_data + 1, g_data + 1),
            ((f + g) / 2, f ** 2, (f_data + g_data) / 2, f_data ** 2),
        ]
        for op in (gt, ge, lt, le, ne):
            for expr_lhs, expr_rhs, expected_lhs, expected_rhs in cases:
                self.check_output(
                    op(expr_lhs, expr_rhs),
                    op(expected_lhs, expected_rhs),
                )

    def test_boolean_binops(self):
        f, g, h = self.f, self.g, self.h

        # Add a non-numexpr filter to ensure that we correctly handle
        # delegation to NumericalExpression.
        custom_filter = NonExprFilter()
        custom_filter_mask = array(
            [[0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 0, 0, 1, 0]],
            dtype=bool,
        )

        self.fake_raw_data = {
            f: arange(25, dtype=float).reshape(5, 5),
            g: arange(25, dtype=float).reshape(5, 5) - eye(5),
            h: full((5, 5), 5, dtype=float),
            custom_filter: custom_filter_mask,
        }

        # Should be True on the diagonal.
        eye_filter = (f > g)

        # Should be True in the first row only.
        first_row_filter = f < h

        eye_mask = eye(5, dtype=bool)

        first_row_mask = zeros((5, 5), dtype=bool)
        first_row_mask[0] = 1

        self.check_output(eye_filter, eye_mask)
        self.check_output(first_row_filter, first_row_mask)

        def gen_boolops(x, y, z):
            """
            Generate all possible interleavings of & and | between all possible
            orderings of x, y, and z.
            """
            for a, b, c in permutations([x, y, z]):
                yield (a & b) & c
                yield (a & b) | c
                yield (a | b) & c
                yield (a | b) | c
                yield a & (b & c)
                yield a & (b | c)
                yield a | (b & c)
                yield a | (b | c)

        exprs = gen_boolops(eye_filter, custom_filter, first_row_filter)
        arrays = gen_boolops(eye_mask, custom_filter_mask, first_row_mask)

        for expr, expected in zip(exprs, arrays):
            self.check_output(expr, expected)
