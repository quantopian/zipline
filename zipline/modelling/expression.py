"""
NumericalExpression term.
"""
from itertools import chain
import re

import numexpr
from numexpr.necompiler import getExprNames
from numpy import (
    empty,
    find_common_type,
)

from zipline.modelling.computable import Term

_VARIABLE_NAME_RE = re.compile("^(x_)([0-9]+)$")

# Map from op symbol to equivalent Python magic method name.
_ops_to_methods = {
    '+': '__add__',
    '-': '__sub__',
    '*': '__mul__',
    '/': '__div__',
    '**': '__pow__',
    '&': '__and__',
    '|': '__or__',
    '^': '__xor__',
    '<': '__lt__',
    '<=': '__le__',
    '==': '__eq__',
    '!=': '__ne__',
    '>=': '__ge__',
    '>': '__gt__',
}
# Map from op symbol to equivalent Python magic method name after flipping
# arguments.
_ops_to_commuted_methods = {
    '+': '__radd__',
    '-': '__rsub__',
    '*': '__rmul__',
    '/': '__rdiv__',
    '**': '__rpow__',
    '&': '__rand__',
    '|': '__ror__',
    '^': '__rxor__',
    '<': '__gt__',
    '<=': '__ge__',
    '==': '__eq__',
    '!=': '__ne__',
    '>=': '__le__',
    '>': '__lt__',
}
UNARY_OPS = {'-'}
MATH_BINOPS = {'+', '-', '*', '/', '**'}
FILTER_BINOPS = {'&', '|'}  # NumExpr doesn't support xor.
COMPARISONS = {'<', '<=', '!=', '>=', '>'}

NUMERIC_TYPES = (int, float, long)
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


def bad_op(op, left, right):
    """
    Called when a bad binary operation is encountered.

    Parameters
    ----------
    op : str
        The attempted operation
    left : zipline.computable.Term
        The left hand side of the operation.
    right : zipline.computable.Term
        The right hand side of the operation.

    Returns
    -------
    error : TypeError
        An instance of TypeError to raise.
    """
    return TypeError(
        "Can't compute {left} {op} {right}".format(
            op=op,
            left=type(left).__name__,
            right=type(right).__name__,
        )
    )


def method_name_for_op(op, commute=False):
    """
    Get the name of the Python magic method corresponding to `op`.

    Parameters
    ----------
    op : str {'+','-','*', '/','**','&','|','^','<','<=','==','!=','>=','>'}
        The requested operation.
    commute : bool
        Whether to return the name of an equivalent method after flipping args.

    Returns
    -------
    method_name : str
        The name of the Python magic method corresponding to `op`.
        If `commute` is True, returns the name of a method equivalent to `op`
        with inputs flipped.

    Examples
    --------
    >>> method_name_for_op('+')
    '__add__'
    >>> method_name_for_op('+', commute=True)
    '__radd__'
    >>> method_name_for_op('>')
    '__gt__'
    >>> method_name_for_op('>', commute=True)
    '__lt__'
    """
    if commute:
        return _ops_to_commuted_methods[op]
    return _ops_to_methods[op]


def is_comparison(op):
    return op in COMPARISONS


class NumericalExpression(Term):
    """
    Term binding to a numexpr expression.

    Parameters
    ----------
    expr : string
       A string suitable for passing to numexpr.  All variables in 'expr'
       should be of the form "x_i", where i is the index of the corresponding
       factor input in 'binds'.
    binds : tuple
       A tuple of factors to use as inputs.
    """

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

    def _init(self, expr, *args, **kwargs):
        self._expr = expr
        return super(NumericalExpression, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, expr, *args, **kwargs):
        return (
            super(NumericalExpression, cls).static_identity(*args, **kwargs),
            expr,
        )

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

    def compute_from_arrays(self, arrays, dtype, dates, assets):
        """
        Compute our stored expression string with numexpr.
        """
        out = empty((len(dates), len(assets)), dtype=dtype)
        # This writes directly into our output buffer.
        numexpr.evaluate(
            self._expr,
            local_dict={
                "x_%d" % idx: array
                for idx, array in enumerate(arrays)
            },
            global_dict={},
            out=out,
        )
        return out

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

    def build_binary_op(self, op, other):
        """
        Compute new expression strings and a new inputs tuple for combining
        self and other with a binary operator.
        """
        if isinstance(other, NumericalExpression):
            self_expr, other_expr, new_inputs = self._merge_expressions(other)
        elif isinstance(other, Term):
            self_expr = self._expr
            new_inputs, other_idx = _ensure_element(self.inputs, other)
            other_expr = "x_%d" % other_idx
        elif isinstance(other, NUMERIC_TYPES):
            self_expr = self._expr
            other_expr = str(other)
            new_inputs = self.inputs
        else:
            raise bad_op(op, other)
        return self_expr, other_expr, new_inputs

    @property
    def bindings(self):
        return {
            "x_%d" % i: input_
            for i, input_ in enumerate(self.inputs)
        }

    def __repr__(self):
        return "{typename}(expr='{expr}', bindings={bindings})".format(
            typename=type(self).__name__,
            expr=self._expr,
            bindings=self.bindings,
        )
