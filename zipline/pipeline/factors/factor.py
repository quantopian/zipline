"""
factor.py
"""
from operator import attrgetter
from numbers import Number

from numpy import (
    apply_along_axis,
    float64,
    nan,
    inf,
)
from scipy.stats import rankdata

from zipline.errors import (
    UnknownRankMethod,
    UnsupportedDataType,
)
from zipline.lib.rank import rankdata_2d_ordinal
from zipline.pipeline.term import (
    CustomTermMixin,
    NotSpecified,
    RequiredWindowLengthMixin,
    SingleInputMixin,
    CompositeTerm,
)
from zipline.pipeline.expression import (
    BadBinaryOperator,
    COMPARISONS,
    is_comparison,
    MATH_BINOPS,
    method_name_for_op,
    NumericalExpression,
    NUMEXPR_MATH_FUNCS,
    UNARY_OPS,
)
from zipline.pipeline.filters import (
    NumExprFilter,
    PercentileFilter,
)
from zipline.utils.control_flow import nullctx


_RANK_METHODS = frozenset(['average', 'min', 'max', 'dense', 'ordinal'])


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
        elif isinstance(other, Number):
            return return_type(
                "x_0 {op} ({constant})".format(op=op, constant=other),
                binds=(self,),
            )
        raise BadBinaryOperator(op, self, other)

    binary_operator.__doc__ = "Binary Operator: '%s'" % op
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
        elif isinstance(other, Number):
            return NumExprFactor(
                "{constant} {op} x_0".format(op=op, constant=other),
                binds=(self,),
            )
        raise BadBinaryOperator(op, other, self)
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

    unary_operator.__doc__ = "Unary Operator: '%s'" % op
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


class Factor(CompositeTerm):
    """
    Pipeline API expression producing numerically-valued outputs.
    """
    dtype = float64

    # Dynamically add functions for creating NumExprFactor/NumExprFilter
    # instances.
    clsdict = locals()
    clsdict.update(
        {
            method_name_for_op(op): binary_operator(op)
            # Don't override __eq__ because it breaks comparisons on tuples of
            # Factors.
            for op in MATH_BINOPS.union(COMPARISONS - {'=='})
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

    __truediv__ = clsdict['__div__']
    __rtruediv__ = clsdict['__rdiv__']

    eq = binary_operator('==')

    def rank(self, method='ordinal', ascending=True, mask=NotSpecified):
        """
        Construct a new Factor representing the sorted rank of each column
        within each row.

        Parameters
        ----------
        method : str, {'ordinal', 'min', 'max', 'dense', 'average'}
            The method used to assign ranks to tied elements. See
            `scipy.stats.rankdata` for a full description of the semantics for
            each ranking method. Default is 'ordinal'.
        ascending : bool, optional
            Whether to return sorted rank in ascending or descending order.
            Default is True.
        mask : zipline.pipeline.Filter, optional
            A Filter representing assets to consider when computing ranks.
            If mask is supplied, ranks are computed ignoring any asset/date
            pairs for which `mask` produces a value of False.

        Returns
        -------
        ranks : zipline.pipeline.factors.Rank
            A new factor that will compute the ranking of the data produced by
            `self`.

        Notes
        -----
        The default value for `method` is different from the default for
        `scipy.stats.rankdata`.  See that function's documentation for a full
        description of the valid inputs to `method`.

        Missing or non-existent data on a given day will cause an asset to be
        given a rank of NaN for that day.

        See Also
        --------
        scipy.stats.rankdata
        zipline.lib.rank
        zipline.pipeline.factors.Rank
        """
        return Rank(self if ascending else -self, method=method, mask=mask)

    def top(self, N, mask=NotSpecified):
        """
        Construct a Filter matching the top N asset values of self each day.

        Parameters
        ----------
        N : int
            Number of assets passing the returned filter each day.
        mask : zipline.pipeline.Filter, optional
            A Filter representing assets to consider when computing ranks.
            If mask is supplied, top values are computed ignoring any
            asset/date pairs for which `mask` produces a value of False.

        Returns
        -------
        filter : zipline.pipeline.filters.Filter
        """
        return self.rank(ascending=False, mask=mask) <= N

    def bottom(self, N, mask=NotSpecified):
        """
        Construct a Filter matching the bottom N asset values of self each day.

        Parameters
        ----------
        N : int
            Number of assets passing the returned filter each day.
        mask : zipline.pipeline.Filter, optional
            A Filter representing assets to consider when computing ranks.
            If mask is supplied, bottom values are computed ignoring any
            asset/date pairs for which `mask` produces a value of False.

        Returns
        -------
        filter : zipline.pipeline.Filter
        """
        return self.rank(ascending=True, mask=mask) <= N

    def percentile_between(self,
                           min_percentile,
                           max_percentile,
                           mask=NotSpecified):
        """
        Construct a new Filter representing entries from the output of this
        Factor that fall within the percentile range defined by min_percentile
        and max_percentile.

        Parameters
        ----------
        min_percentile : float [0.0, 100.0]
            Return True for assets falling above this percentile in the data.
        max_percentile : float [0.0, 100.0]
            Return True for assets falling below this percentile in the data.
        mask : zipline.pipeline.Filter, optional
            A Filter representing assets to consider when percentile
            thresholds.  If mask is supplied, percentile cutoffs are computed
            each day using only assets for which `mask` returns True, and
            assets not passing `mask` will produce False in the output of this
            filter as well.

        Returns
        -------
        out : zipline.pipeline.filters.PercentileFilter
            A new filter that will compute the specified percentile-range mask.

        See Also
        --------
        zipline.pipeline.filters.PercentileFilter
        """
        return PercentileFilter(
            self,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            mask=mask,
        )

    def isnan(self):
        """
        A Filter producing True for all values where this Factor is NaN.
        """
        return self != self

    def notnan(self):
        """
        A Filter producing True for values where this Factor is not NaN.

        Returns
        -------
        nanfilter : zipline.pipeline.filters.Filter
        """
        return ~self.isnan()

    def isfinite(self):
        """
        A Filter producing True for values where this Factor is anything but
        NaN, inf, or -inf.
        """
        return (-inf < self) & (self < inf)


class NumExprFactor(NumericalExpression, Factor):
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

    Notes
    -----
    NumExprFactors are constructed by numerical operators like `+` and `-`.
    Users should rarely need to construct a NumExprFactor directly.
    """
    pass


class Rank(SingleInputMixin, Factor):
    """
    A Factor representing the row-wise rank data of another Factor.

    Parameters
    ----------
    factor : zipline.pipeline.factors.Factor
        The factor on which to compute ranks.
    method : str, {'average', 'min', 'max', 'dense', 'ordinal'}
        The method used to assign ranks to tied elements.  See
        `scipy.stats.rankdata` for a full description of the semantics for each
        ranking method.

    See Also
    --------
    scipy.stats.rankdata : Underlying ranking algorithm.
    zipline.factors.Factor.rank : Method-style interface to same functionality.

    Notes
    -----
    Most users should call Factor.rank rather than directly construct an
    instance of this class.
    """
    window_length = 0
    dtype = float64

    def __new__(cls, factor, method, mask):
        return super(Rank, cls).__new__(
            cls,
            inputs=(factor,),
            method=method,
            mask=mask,
        )

    def _init(self, method, *args, **kwargs):
        self._method = method
        return super(Rank, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, method, *args, **kwargs):
        return (
            super(Rank, cls).static_identity(*args, **kwargs),
            method,
        )

    def _validate(self):
        """
        Verify that the stored rank method is valid.
        """
        if self._method not in _RANK_METHODS:
            raise UnknownRankMethod(
                method=self._method,
                choices=set(_RANK_METHODS),
            )
        return super(Rank, self)._validate()

    def _compute(self, arrays, dates, assets, mask):
        """
        For each row in the input, compute a like-shaped array of per-row
        ranks.
        """
        inv_mask = ~mask
        data = arrays[0].copy()
        data[inv_mask] = nan
        # OPTIMIZATION: Fast path the default case with our own specialized
        # Cython implementation.
        if self._method == 'ordinal':
            result = rankdata_2d_ordinal(data)
        else:
            # FUTURE OPTIMIZATION:
            # Write a less general "apply to rows" method that doesn't do all
            # the extra work that apply_along_axis does.
            result = apply_along_axis(rankdata, 1, data, method=self._method)

        # rankdata will sort nan values into last place, but we want our
        # nans to propagate, so explicitly re-apply.
        result[inv_mask] = nan
        return result

    def __repr__(self):
        return "{type}({input_}, method='{method}', mask={mask})".format(
            type=type(self).__name__,
            input_=self.inputs[0],
            method=self._method,
            mask=self.mask,
        )


class CustomFactor(RequiredWindowLengthMixin, CustomTermMixin, Factor):
    '''
    Base class for user-defined Factors.

    Parameters
    ----------
    inputs : iterable, optional
        An iterable of `BoundColumn` instances (e.g. USEquityPricing.close),
        describing the data to load and pass to `self.compute`.  If this
        argument is passed to the CustomFactor constructor, we look for a
        class-level attribute named `inputs`.
    window_length : int, optional
        Number of rows of rows to pass for each input.  If this
        argument is not passed to the CustomFactor constructor, we look for a
        class-level attribute named `window_length`.

    Notes
    -----
    Users implementing their own Factors should subclass CustomFactor and
    implement a method named `compute` with the following signature:

    .. code-block:: python

        def compute(self, today, assets, out, *inputs):
           ...

    On each simulation date, ``compute`` will be called with the current date,
    an array of sids, an output array, and an input array for each expression
    passed as inputs to the CustomFactor constructor.

    The specific types of the values passed to `compute` are as follows::

        today : np.datetime64[ns]
            Row label for the last row of all arrays passed as `inputs`.
        assets : np.array[int64, ndim=1]
            Column labels for `out` and`inputs`.
        out : np.array[float64, ndim=1]
            Output array of the same shape as `assets`.  `compute` should write
            its desired return values into `out`.
        *inputs : tuple of np.array
            Raw data arrays corresponding to the values of `self.inputs`.

    ``compute`` functions should expect to be passed NaN values for dates on
    which no data was available for an asset.  This may include dates on which
    an asset did not yet exist.

    For example, if a CustomFactor requires 10 rows of close price data, and
    asset A started trading on Monday June 2nd, 2014, then on Tuesday, June
    3rd, 2014, the column of input data for asset A will have 9 leading NaNs
    for the preceding days on which data was not yet available.

    Examples
    --------

    A CustomFactor with pre-declared defaults:

    .. code-block:: python

        class TenDayRange(CustomFactor):
            """
            Computes the difference between the highest high in the last 10
            days and the lowest low.

            Pre-declares high and low as default inputs and `window_length` as
            10.
            """

            inputs = [USEquityPricing.high, USEquityPricing.low]
            window_length = 10

            def compute(self, today, assets, out, highs, lows):
                from numpy import nanmin, nanmax

                highest_highs = nanmax(highs, axis=0)
                lowest_lows = nanmin(lows, axis=0)
                out[:] = highest_highs - lowest_lows


        # Doesn't require passing inputs or window_length because they're
        # pre-declared as defaults for the TenDayRange class.
        ten_day_range = TenDayRange()

    A CustomFactor without defaults:

    .. code-block:: python

        class MedianValue(CustomFactor):
            """
            Computes the median value of an arbitrary single input over an
            arbitrary window..

            Does not declare any defaults, so values for `window_length` and
            `inputs` must be passed explicitly on every construction.
            """

            def compute(self, today, assets, out, data):
                from numpy import nanmedian
                out[:] = data.nanmedian(data, axis=0)

        # Values for `inputs` and `window_length` must be passed explicitly to
        # MedianValue.
        median_close10 = MedianValue([USEquityPricing.close], window_length=10)
        median_low15 = MedianValue([USEquityPricing.low], window_length=15)
    '''
    ctx = nullctx()

    def _validate(self):
        if self.dtype != float64:
            raise UnsupportedDataType(dtype=self.dtype)
        return super(CustomFactor, self)._validate()
