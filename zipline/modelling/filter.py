"""
filter.py
"""
from numpy import (
    percentile,
    bool_,
)
from operator import attrgetter

from zipline.errors import (
    BadPercentileBounds,
)
from zipline.modelling.computable import (
    SingleInputMixin,
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
    domain = None
    dtype = bool_

    clsdict = locals()
    clsdict.update(
        {
            method_name_for_op(op): binary_operator(op)
            for op in FILTER_BINOPS
        }
    )


class NumExprFilter(Filter, NumericalExpression):
    """
    A Filter computed from a numexpr expression.
    """
    window_length = 0


class PercentileFilter(Filter, SingleInputMixin):
    """
    A Filter representing assets falling between percentile bounds of a Factor.

    Parameters
    ----------
    factor : zipline.modelling.factor.Factor
        The factor over which to compute percentile bounds.
    min_percentile : float [0.0, 1.0]
        The minimum percentile rank of an asset that will pass the filter.
    max_percentile : float [0.0, 1.0]
        The maxiumum percentile rank of an asset that will pass the filter.
    """

    def __new__(cls, rank_factor, min_percentile, max_percentile):
        return super(PercentileFilter, cls).__new__(
            cls,
            inputs=(rank_factor,),
            window_length=cls.window_length,
            domain=cls.domain,
            dtype=cls.dtype,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )

    def _init(self, min_percentile, max_percentile, *args, **kwargs):
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile
        return super(PercentileFilter, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, min_percentile, max_percentile, *args, **kwargs):
        return (
            super(PercentileFilter, cls).static_identity(*args, **kwargs),
            min_percentile,
            max_percentile,
        )

    def _validate(self):
        """
        Ensure that our percentile bounds are well-formed.
        """
        if not 0.0 <= self._min_percentile < self._max_percentile <= 100.0:
            raise BadPercentileBounds(
                min_percentile=self._min_percentile,
                max_percentile=self._max_percentile,
            )

    def compute_from_arrays(self, arrays, dtype, dates, assets):
        """
        For each row in the input, compute a mask of all values falling between
        the given percentiles.
        """
        # TODO: Review whether there's a better way of handling small numbers
        # of columns.
        data = arrays[0]
        lower_bounds, upper_bounds = percentile(
            data,
            [self._min_percentile, self._max_percentile],
            axis=1,
            keepdims=True,
        )
        return (lower_bounds <= data) & (data <= upper_bounds)
