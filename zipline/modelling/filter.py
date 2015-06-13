"""
filter.py
"""
from numpy import uint8
from operator import attrgetter

from zipline.modelling.computable import (
    Term,
)
from zipline.modelling.expression import (
    bad_op,
    FILTER_BINOPS,
    method_name_for_op,
    NumericalExpression,
)


def binary_operator(op):
    """
    Factory function for making binary operator methods on a Filter subclass.

    Returns a function "binary_operator" suitable for implementing functions
    like __and__ or __or__.
    """
    # When combining a Filter with a NumericalExpression, we use this
    # attrgetter instance to defer to the commuted interpretation of the
    # NumericalExpression operator.
    commuted_method_getter = attrgetter(method_name_for_op(op, commute=True))

    def binary_operator(self, other):
        if isinstance(self, NumericalExpression):
            self_expr, other_expr, new_inputs = self.build_binary_op(
                op, other,
            )
            return NumExprFilter(
                "({left}) {op} ({right})".format(
                    left=self_expr,
                    op=op,
                    right=other_expr,
                ),
                new_inputs,
            )
        elif isinstance(other, NumericalExpression):
            # NumericalExpression overrides numerical ops to correctly handle
            # merging of inputs.  Look up and call the appropriate
            # right-binding operator with ourself as the input.
            return commuted_method_getter(other)(self)
        elif isinstance(other, Filter):
            if self is other:
                return NumExprFilter(
                    "x_0 {op} x_0".format(op=op),
                    (self,),
                )
            return NumExprFilter(
                "x_0 {op} x_1".format(op=op),
                (self, other),
            )
        elif isinstance(other, int):  # Note that this is true for bool as well
            return NumExprFilter(
                "x_0 {op} ({constant})".format(op=op, constant=int(other)),
                binds=(self,),
            )
        raise bad_op(op, self, other)
    return binary_operator


class Filter(Term):
    """
    A boolean predicate on a universe of Assets.
    """
    window_length = 0
    domain = None
    dtype = uint8

    clsdict = locals()
    clsdict.update(
        {
            method_name_for_op(op): binary_operator(op)
            for op in FILTER_BINOPS
        }
    )


class NumExprFilter(Filter, NumericalExpression):
    """
    A Filter computed from a numexp expression.
    """
    pass
