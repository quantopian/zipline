"""
factor.py
"""
from operator import attrgetter

from zipline.modelling.computable import Term
from zipline.modelling.expression import (
    bad_op,
    COMPARISONS,
    is_comparison,
    MATH_BINOPS,
    method_name_for_op,
    NUMERIC_TYPES,
    NumericalExpression,
    NUMEXPR_MATH_FUNCS,
    UNARY_OPS,
)
from zipline.modelling.filter import (
    NumExprFilter,
)


def binop_return_type(op):
    if is_comparison(op):
        return NumExprFilter
    else:
        return NumExprFactor


def binary_operator(op):
    """
    Factory function for making binary operator methods on a Factor subclass.

    Returns a function, "binary_operator" suitable for implementing functions
    like __add__.
    """
    # When combining a Factor with a NumericalExpression, we use this
    # attrgetter instance to defer to the commuted implementation of the
    # NumericalExpression operator.
    commuted_method_getter = attrgetter(method_name_for_op(op, commute=True))

    def binary_operator(self, other):
        # This can't be hoisted up a scope because the types returned by
        # binop_return_type aren't defined when the top-level function is
        # invoked in the class body of Factor.
        return_type = binop_return_type(op)
        if isinstance(self, NumExprFactor):
            self_expr, other_expr, new_inputs = self.build_binary_op(
                op, other,
            )
            return return_type(
                "({left}) {op} ({right})".format(
                    left=self_expr,
                    op=op,
                    right=other_expr,
                ),
                new_inputs,
            )
        elif isinstance(other, NumExprFactor):
            # NumericalExpression overrides ops to correctly handle merging of
            # inputs.  Look up and call the appropriate reflected operator with
            # ourself as the input.
            return commuted_method_getter(other)(self)
        elif isinstance(other, Factor):
            if self is other:
                return return_type(
                    "x_0 {op} x_0".format(op=op),
                    (self,),
                )
            return return_type(
                "x_0 {op} x_1".format(op=op),
                (self, other),
            )
        elif isinstance(other, NUMERIC_TYPES):
            return return_type(
                "x_0 {op} ({constant})".format(op=op, constant=other),
                binds=(self,),
            )
        raise bad_op(op, self, other)

    return binary_operator


def reflected_binary_operator(op):
    """
    Factory function for making binary operator methods on a Factor.

    Returns a function, "reflected_binary_operator" suitable for implementing
    functions like __radd__.
    """
    assert not is_comparison(op)

    def reflected_binary_operator(self, other):

        if isinstance(self, NumericalExpression):
            self_expr, other_expr, new_inputs = self.build_binary_op(
                op, other
            )
            return NumExprFactor(
                "({left}) {op} ({right})".format(
                    left=other_expr,
                    right=self_expr,
                    op=op,
                ),
                new_inputs,
            )

        # Only have to handle the numeric case because in all other valid cases
        # the corresponding left-binding method will be called.
        elif isinstance(other, NUMERIC_TYPES):
            return NumExprFactor(
                "{constant} {op} x_0".format(op=op, constant=other),
                binds=(self,),
            )
        raise bad_op(op, other, self)
    return reflected_binary_operator


def unary_operator(op):
    """
    Factory function for making unary operator methods for Factors.
    """
    # Only negate is currently supported for all our possible input types.
    valid_ops = {'-'}
    if op not in valid_ops:
        raise ValueError("Invalid unary operator %s." % op)

    def unary_operator(self):
        # This can't be hoisted up a scope because the types returned by
        # unary_op_return_type aren't defined when the top-level function is
        # invoked.
        if isinstance(self, NumericalExpression):
            return NumExprFactor(
                "{op}({expr})".format(op=op, expr=self._expr),
                self.inputs,
            )
        else:
            return NumExprFactor("{op}x_0".format(op=op), (self,))
    return unary_operator


def function_application(func):
    """
    Factory function for producing function application methods for Factor
    subclasses.
    """
    if func not in NUMEXPR_MATH_FUNCS:
        raise ValueError("Unsupported mathematical function '%s'" % func)

    def mathfunc(self):
        if isinstance(self, NumericalExpression):
            return NumExprFactor(
                "{func}({expr})".format(func=func, expr=self._expr),
                self.inputs,
            )
        else:
            return NumExprFactor("{func}(x_0)".format(func=func), (self,))
    return mathfunc


class Factor(Term):
    """
    A transformation yielding a timeseries of scalar values associated with an
    Asset.
    """
    # Dynamically add functions for creating NumExprFactor/NumExprFilter
    # instances.
    clsdict = locals()
    clsdict.update(
        {
            method_name_for_op(op): binary_operator(op)
            for op in MATH_BINOPS.union(COMPARISONS)
        }
    )
    clsdict.update(
        {
            method_name_for_op(op, commute=True): reflected_binary_operator(op)
            for op in MATH_BINOPS
        }
    )
    clsdict.update(
        {
            '__neg__': unary_operator(op)
            for op in UNARY_OPS
        }
    )
    clsdict.update(
        {
            funcname: function_application(funcname)
            for funcname in NUMEXPR_MATH_FUNCS
        }
    )


class NumExprFactor(Factor, NumericalExpression):
    """
    Factor computed from a numexpr expression.

    Parameters
    ----------
    expr : string
       A string suitable for passing to numexpr.  All variables in 'expr'
       should be of the form "x_i", where i is the index of the corresponding
       factor input in 'binds'.
    binds : tuple
       A tuple of factors to use as inputs.
    """
    pass


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
