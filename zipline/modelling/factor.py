"""
factor.py
"""
from collections import Counter
from itertools import chain
import re
from operator import attrgetter

from six.moves import range

import numexpr
from numexpr.necompiler import getExprNames

from numpy import find_common_type
from zipline.modelling.computable import Term

_VARIABLE_NAME_RE = re.compile("^(x_)([0-9]+)$")
_NUMERIC_TYPES = (int, float, long)


_ops_to_right_bind_methods = {
    '+': '__radd__',
    '-': '__rsub__',
    '*': '__rmul__',
    '/': '__rdiv__',
    '**': '__rpow__',
}

NUMEXPR_MATH_FUNCS = {
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'log',
    'log10',
    'log1p',
    'exp',
    'expm1',
    'sqrt',
    'abs',
}


def _ensure_element(tup, elem):
    """
    Create a tuple containing all elements of tup, plus elem.

    Returns the new tuple and the index of elem in the new tuple.
    """
    try:
        return tup, tup.index(elem)
    except ValueError:
        return tuple(chain(tup, (elem,))), len(tup)


def _bad_op(op, left, right):
    return TypeError(
        "Can't compute {left} {op} {right}".format(
            op=op,
            left=type(left).__name__,
            right=type(right).__name__,
        )
    )


def _factor_binary_operators(op):
    """
    Factory function for making binary operator methods on Factor.

    Returns a pair of functions, binary_operator and rbinary_operator, suitable
    for implementing functions like __add__ and __radd__.
    """
    # When adding a factor to a NumericalExpression, we defer to the
    # right-binding implementation of the NumericalExpression operator.
    right_bind_method_getter = attrgetter(_ops_to_right_bind_methods[op])

    def binary_operator(self, other):
        if isinstance(other, NumericalExpression):
            # NumericalExpression overrides numerical ops to correctly handle
            # merging of inputs.  Look up and call the appropriate
            # right-binding operator with ourself as the input.
            return right_bind_method_getter(other)(self)
        elif isinstance(other, Factor):
            # FUTURE OPTIMIZATION: Detect commutative operations and normalize
            # term order for better caching.
            if other == self:
                return NumericalExpression(
                    "x_0 {op} x_0".format(op=op),
                    (self,),
                )
            return NumericalExpression(
                "x_0 {op} x_1".format(op=op),
                (self, other),
            )
        elif isinstance(other, _NUMERIC_TYPES):
            return NumericalExpression(
                "x_0 {op} {constant}".format(op=op, constant=other),
                binds=(self,),
            )
        raise _bad_op(op, self, other)

    def rbinary_operator(self, other):
        # Only have to handle the numeric case because in all other valid cases
        # the corresponding left-binding method will be called.
        if isinstance(other, _NUMERIC_TYPES):
            return NumericalExpression(
                "{constant} {op} x_0".format(op=op, constant=other),
                binds=(self,),
            )
        raise _bad_op(op, other, self)

    return binary_operator, rbinary_operator


def _factor_math_function(func):
    """
    Factory function for making mathematical function methods for Factor.
    """
    if func not in NUMEXPR_MATH_FUNCS:
        raise ValueError("Unsupported mathematical function '%s'" % func)

    def mathfunc(self):
        return NumericalExpression("{func}(x_0)".format(func=func), (self,))
    return mathfunc


def _factor_unary_operator(op):
    """
    Factory function for making unary operator methods for Factor.
    """
    # Only negate is currently supported for all our possible input types.
    valid_ops = {'-'}
    if op not in valid_ops:
        raise ValueError("Invalid unary operator %s." % op)

    def unary_operator(self):
        return NumericalExpression("{op}x_0".format(op=op), (self,))
    return unary_operator


def _numerical_expression_binary_operators(op):
    """
    Factory function for making binary operator methods on NumericalExpression.

    Returns a pair of functions, binary_operator and rbinary_operator, suitable
    for implementing functions like __add__ and __radd__.
    """
    def binary_operator(self, other):
        self_expr, other_expr, new_inputs = self._build_binary_op(op, other)
        return NumericalExpression(
            "({left}) {op} ({right})".format(
                left=self_expr,
                right=other_expr,
                op=op,
            ),
            new_inputs,
        )

    def rbinary_operator(self, other):
        self_expr, other_expr, new_inputs = self._build_binary_op(op, other)
        return NumericalExpression(
            "({left}) {op} ({right})".format(
                left=other_expr,
                right=self_expr,
                op=op,
            ),
            new_inputs,
        )

    return binary_operator, rbinary_operator


def _numerical_expression_unary_operator(op):
    """
    Factory function for producing unary operators on NumericalExpression.
    """
    def unary_operator(self):
        return NumericalExpression(
            "{op}({expr})".format(op=op, expr=self._expr),
            self.inputs,
        )
    return unary_operator


def _numerical_expression_math_function(func):
    """
    Factory function for producing mathematical function methods on
    NumericalExpression.
    """
    def mathfunc(self):
        return NumericalExpression(
            "{func}({expr})".format(func=func, expr=self._expr),
            self.inputs,
        )
    return mathfunc


class Factor(Term):
    """
    A transformation yielding a timeseries of scalar values associated with an
    Asset.
    """
    # Autogenerated binary operators.
    __add__, __radd__ = _factor_binary_operators('+')
    __sub__, __rsub__ = _factor_binary_operators('-')
    __mul__, __rmul__ = _factor_binary_operators('*')
    __div__, __rdiv__ = _factor_binary_operators('/')
    __pow__, __rpow__ = _factor_binary_operators('**')

    # Autogenerated unary operators.
    __neg__ = _factor_unary_operator("-")

    # Dynamically add mathematical functions as instance methods.
    clsdict = locals()
    for funcname in NUMEXPR_MATH_FUNCS:
        clsdict[funcname] = _factor_math_function(funcname)

    def greater_than(self, N):
        """
        Return a filter matching values greater than N.
        """
        pass


class NumericalExpression(Factor):
    """
    Factor binding to a numexpr expression.

    Parameters
    ----------
    expr : string
       A string suitable for passing to numexpr.  All variables in 'expr'
       should be of the form "x_i", where i is the index of the corresponding
       factor input in 'binds'.
    binds : tuple
       A tuple of factors to use as inputs.
    """
    __add__, __radd__ = _numerical_expression_binary_operators('+')
    __sub__, __rsub__ = _numerical_expression_binary_operators('-')
    __mul__, __rmul__ = _numerical_expression_binary_operators('*')
    __div__, __rdiv__ = _numerical_expression_binary_operators('/')
    __pow__, __rpow__ = _numerical_expression_binary_operators('**')
    __neg__ = _numerical_expression_unary_operator('-')

    # Dynamically add mathematical functions as instance methods.
    clsdict = locals()
    for funcname in NUMEXPR_MATH_FUNCS:
        clsdict[funcname] = _numerical_expression_math_function(funcname)

    def __new__(cls, expr, binds):
        return super(NumericalExpression, cls).__new__(
            cls,
            inputs=binds,
            window_length=0,
            domain=None,
            dtype=find_common_type(
                [factor.dtype for factor in binds],
                [],
            ),
            expr=expr,
        )

    @classmethod
    def static_identity(cls, expr, *args, **kwargs):
        return (
            super(NumericalExpression, cls).static_identity(*args, **kwargs),
            expr,
        )

    def _init(self, expr, *args, **kwargs):
        self._expr = expr
        return super(NumericalExpression, self)._init(*args, **kwargs)

    def _validate(self):
        """
        Ensure that our expression string has variables of the form x_0, x_1,
        ... x_(N - 1), where N is the length of our inputs.
        """
        variable_names, _unused = getExprNames(self._expr, {})
        expr_indices = []
        for name in variable_names:
            match = _VARIABLE_NAME_RE.match(name)
            if not match:
                raise ValueError("%r is not a valid variable name" % name)
            expr_indices.append(int(match.group(2)))

        expr_indices.sort()
        expected_indices = list(range(len(self.inputs)))
        if expr_indices != expected_indices:
            raise ValueError(
                "Expected %s for variable indices, but got %s" % (
                    expected_indices, expr_indices,
                )
            )

    def compute_from_arrays(self, arrays, outbuf, dates, assets):
        """
        Compute directly into outbuf via numexpr.
        """
        # This writes directly into our output buffer.
        numexpr.evaluate(
            self._expr,
            local_dict={
                "x_%d" % idx: array
                for idx, array in enumerate(arrays)
            },
            global_dict={},
            out=outbuf,
        )

    def _rebind_variables(self, new_inputs):
        """
        Return self._expr with all variables rebound to the indices implied by
        new_inputs.
        """
        expr = self._expr
        for idx, input_ in enumerate(self.inputs):
            old_varname = "x_%d" % idx
            # Temporarily rebind to x_temp_N so that we don't overwrite the
            # same value multiple times.
            temp_new_varname = "x_temp_%d" % new_inputs.index(input_)
            expr = expr.replace(old_varname, temp_new_varname)
        # Clear out the temp variables now that we've finished iteration.
        return expr.replace("_temp_", "_")

    def _merge_expressions(self, other):
        """
        Merge the inputs of two NumericalExpressions into a single input tuple,
        rewriting their respective string expressions to make input names
        resolve correctly.

        Returns a tuple of (new_self_expr, new_other_expr, new_inputs)
        """
        new_inputs = tuple(set(self.inputs).union(other.inputs))
        new_self_expr = self._rebind_variables(new_inputs)
        new_other_expr = other._rebind_variables(new_inputs)
        return new_self_expr, new_other_expr, new_inputs

    def _build_binary_op(self, op, other):
        """
        Compute new expression strings and a new inputs tuple for combining
        self and other with a binary operator.
        """
        if isinstance(other, NumericalExpression):
            self_expr, other_expr, new_inputs = self._merge_expressions(other)
        elif isinstance(other, Factor):
            self_expr = self._expr
            new_inputs, other_idx = _ensure_element(self.inputs, other)
            other_expr = "x_%d" % other_idx
        elif isinstance(other, _NUMERIC_TYPES):
            self_expr = self._expr
            other_expr = str(other)
            new_inputs = self.inputs
        else:
            raise _bad_op(op, other)
        return self_expr, other_expr, new_inputs


class TestFactor(Factor):
    """
    Base class for testing that asserts all inputs are correctly shaped.
    """

    def compute_from_windows(self, windows, outbuf, dates, assets):
        assert self.window_length > 0
        for idx, _ in enumerate(dates):
            result = self.from_windows(*(next(w) for w in windows))
            assert result.shape == (len(assets),)
            outbuf[idx] = result

        for window in windows:
            try:
                next(window)
            except StopIteration:
                pass
            else:
                raise AssertionError("window %s was not exhausted" % window)

    def compute_from_arrays(self, arrays, outbuf, dates, assets):
        assert self.window_length == 0
        for array in arrays:
            assert array.shape == len(dates), len(assets) == outbuf.shape
        outbuf[:] = self.from_arrays(*arrays)
