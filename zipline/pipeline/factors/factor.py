"""
factor.py
"""
from functools import wraps
from operator import attrgetter
from numbers import Number

from numpy import inf, where

from zipline.errors import UnknownRankMethod
from zipline.lib.normalize import naive_grouped_rowwise_apply
from zipline.lib.rank import masked_rankdata_2d
from zipline.pipeline.api_utils import restrict_to_dtype
from zipline.pipeline.classifiers import Classifier, Everything, Quantiles
from zipline.pipeline.mixins import (
    CustomTermMixin,
    LatestMixin,
    PositiveWindowLengthMixin,
    RestrictedDTypeMixin,
    SingleInputMixin,
)
from zipline.pipeline.term import (
    ComputableTerm,
    NotSpecified,
    NotSpecifiedType,
    Term,
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
    unary_op_name,
)
from zipline.pipeline.filters import (
    Filter,
    NumExprFilter,
    PercentileFilter,
    NullFilter,
)
from zipline.utils.functional import with_doc, with_name
from zipline.utils.input_validation import expect_types
from zipline.utils.math_utils import nanmean, nanstd
from zipline.utils.numpy_utils import (
    bool_dtype,
    categorical_dtype,
    coerce_to_dtype,
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
)


_RANK_METHODS = frozenset(['average', 'min', 'max', 'dense', 'ordinal'])


def coerce_numbers_to_my_dtype(f):
    """
    A decorator for methods whose signature is f(self, other) that coerces
    ``other`` to ``self.dtype``.

    This is used to make comparison operations between numbers and `Factor`
    instances work independently of whether the user supplies a float or
    integer literal.

    For example, if I write::

        my_filter = my_factor > 3

    my_factor probably has dtype float64, but 3 is an int, so we want to coerce
    to float64 before doing the comparison.
    """
    @wraps(f)
    def method(self, other):
        if isinstance(other, Number):
            other = coerce_to_dtype(self.dtype, other)
        return f(self, other)
    return method


def binop_return_type(op):
    if is_comparison(op):
        return NumExprFilter
    else:
        return NumExprFactor


def binop_return_dtype(op, left, right):
    """
    Compute the expected return dtype for the given binary operator.

    Parameters
    ----------
    op : str
        Operator symbol, (e.g. '+', '-', ...).
    left : numpy.dtype
        Dtype of left hand side.
    right : numpy.dtype
        Dtype of right hand side.

    Returns
    -------
    outdtype : numpy.dtype
        The dtype of the result of `left <op> right`.
    """
    if is_comparison(op):
        if left != right:
            raise TypeError(
                "Don't know how to compute {left} {op} {right}.\n"
                "Comparisons are only supported between Factors of equal "
                "dtypes.".format(left=left, op=op, right=right)
            )
        return bool_dtype

    elif left != float64_dtype or right != float64_dtype:
        raise TypeError(
            "Don't know how to compute {left} {op} {right}.\n"
            "Arithmetic operators are only supported between Factors of "
            "dtype 'float64'.".format(
                left=left.name,
                op=op,
                right=right.name,
            )
        )
    return float64_dtype


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

    @with_doc("Binary Operator: '%s'" % op)
    @with_name(method_name_for_op(op))
    @coerce_numbers_to_my_dtype
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
                dtype=binop_return_dtype(op, self.dtype, other.dtype),
            )
        elif isinstance(other, NumExprFactor):
            # NumericalExpression overrides ops to correctly handle merging of
            # inputs.  Look up and call the appropriate reflected operator with
            # ourself as the input.
            return commuted_method_getter(other)(self)
        elif isinstance(other, Term):
            if self is other:
                return return_type(
                    "x_0 {op} x_0".format(op=op),
                    (self,),
                    dtype=binop_return_dtype(op, self.dtype, other.dtype),
                )
            return return_type(
                "x_0 {op} x_1".format(op=op),
                (self, other),
                dtype=binop_return_dtype(op, self.dtype, other.dtype),
            )
        elif isinstance(other, Number):
            return return_type(
                "x_0 {op} ({constant})".format(op=op, constant=other),
                binds=(self,),
                # .dtype access is safe here because coerce_numbers_to_my_dtype
                # will convert any input numbers to numpy equivalents.
                dtype=binop_return_dtype(op, self.dtype, other.dtype)
            )
        raise BadBinaryOperator(op, self, other)

    return binary_operator


def reflected_binary_operator(op):
    """
    Factory function for making binary operator methods on a Factor.

    Returns a function, "reflected_binary_operator" suitable for implementing
    functions like __radd__.
    """
    assert not is_comparison(op)

    @with_name(method_name_for_op(op, commute=True))
    @coerce_numbers_to_my_dtype
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
                dtype=binop_return_dtype(op, other.dtype, self.dtype)
            )

        # Only have to handle the numeric case because in all other valid cases
        # the corresponding left-binding method will be called.
        elif isinstance(other, Number):
            return NumExprFactor(
                "{constant} {op} x_0".format(op=op, constant=other),
                binds=(self,),
                dtype=binop_return_dtype(op, other.dtype, self.dtype),
            )
        raise BadBinaryOperator(op, other, self)
    return reflected_binary_operator


def unary_operator(op):
    """
    Factory function for making unary operator methods for Factors.
    """
    # Only negate is currently supported.
    valid_ops = {'-'}
    if op not in valid_ops:
        raise ValueError("Invalid unary operator %s." % op)

    @with_doc("Unary Operator: '%s'" % op)
    @with_name(unary_op_name(op))
    def unary_operator(self):
        if self.dtype != float64_dtype:
            raise TypeError(
                "Can't apply unary operator {op!r} to instance of "
                "{typename!r} with dtype {dtypename!r}.\n"
                "{op!r} is only supported for Factors of dtype "
                "'float64'.".format(
                    op=op,
                    typename=type(self).__name__,
                    dtypename=self.dtype.name,
                )
            )

        # This can't be hoisted up a scope because the types returned by
        # unary_op_return_type aren't defined when the top-level function is
        # invoked.
        if isinstance(self, NumericalExpression):
            return NumExprFactor(
                "{op}({expr})".format(op=op, expr=self._expr),
                self.inputs,
                dtype=float64_dtype,
            )
        else:
            return NumExprFactor(
                "{op}x_0".format(op=op),
                (self,),
                dtype=float64_dtype,
            )
    return unary_operator


def function_application(func):
    """
    Factory function for producing function application methods for Factor
    subclasses.
    """
    if func not in NUMEXPR_MATH_FUNCS:
        raise ValueError("Unsupported mathematical function '%s'" % func)

    @with_name(func)
    def mathfunc(self):
        if isinstance(self, NumericalExpression):
            return NumExprFactor(
                "{func}({expr})".format(func=func, expr=self._expr),
                self.inputs,
                dtype=float64_dtype,
            )
        else:
            return NumExprFactor(
                "{func}(x_0)".format(func=func),
                (self,),
                dtype=float64_dtype,
            )
    return mathfunc


# Decorators for Factor methods.
if_not_float64_tell_caller_to_use_isnull = restrict_to_dtype(
    dtype=float64_dtype,
    message_template=(
        "{method_name}() was called on a factor of dtype {received_dtype}.\n"
        "{method_name}() is only defined for dtype {expected_dtype}."
        "To filter missing data, use isnull() or notnull()."
    )
)

float64_only = restrict_to_dtype(
    dtype=float64_dtype,
    message_template=(
        "{method_name}() is only defined on Factors of dtype {expected_dtype},"
        " but it was called on a Factor of dtype {received_dtype}."
    )
)


FACTOR_DTYPES = frozenset([datetime64ns_dtype, float64_dtype, int64_dtype])


class Factor(RestrictedDTypeMixin, ComputableTerm):
    """
    Pipeline API expression producing a numerical or date-valued output.

    Factors are the most commonly-used Pipeline term, representing the result
    of any computation producing a numerical result.

    Factors can be combined, both with other Factors and with scalar values,
    via any of the builtin mathematical operators (``+``, ``-``, ``*``, etc).
    This makes it easy to write complex expressions that combine multiple
    Factors.  For example, constructing a Factor that computes the average of
    two other Factors is simply::

        >>> f1 = SomeFactor(...)
        >>> f2 = SomeOtherFactor(...)
        >>> average = (f1 + f2) / 2.0

    Factors can also be converted into :class:`zipline.pipeline.Filter` objects
    via comparison operators: (``<``, ``<=``, ``!=``, ``eq``, ``>``, ``>=``).

    There are many natural operators defined on Factors besides the basic
    numerical operators. These include methods identifying missing or
    extreme-valued outputs (isnull, notnull, isnan, notnan), methods for
    normalizing outputs (rank, demean, zscore), and methods for constructing
    Filters based on rank-order properties of results (top, bottom,
    percentile_between).
    """
    ALLOWED_DTYPES = FACTOR_DTYPES  # Used by RestrictedDTypeMixin

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
            unary_op_name(op): unary_operator(op)
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

    @expect_types(
        mask=(Filter, NotSpecifiedType),
        groupby=(Classifier, NotSpecifiedType),
    )
    @float64_only
    def demean(self, mask=NotSpecified, groupby=NotSpecified):
        """
        Construct a Factor that computes ``self`` and subtracts the mean from
        row of the result.

        If ``mask`` is supplied, ignore values where ``mask`` returns False
        when computing row means, and output NaN anywhere the mask is False.

        If ``groupby`` is supplied, compute by partitioning each row based on
        the values produced by ``groupby``, de-meaning the partitioned arrays,
        and stitching the sub-results back together.

        Parameters
        ----------
        mask : zipline.pipeline.Filter, optional
            A Filter defining values to ignore when computing means.
        groupby : zipline.pipeline.Classifier, optional
            A classifier defining partitions over which to compute means.

        Example
        -------
        Let ``f`` be a Factor which would produce the following output::

                         AAPL   MSFT    MCD     BK
            2017-03-13    1.0    2.0    3.0    4.0
            2017-03-14    1.5    2.5    3.5    1.0
            2017-03-15    2.0    3.0    4.0    1.5
            2017-03-16    2.5    3.5    1.0    2.0

        Let ``c`` be a Classifier producing the following output::

                         AAPL   MSFT    MCD     BK
            2017-03-13      1      1      2      2
            2017-03-14      1      1      2      2
            2017-03-15      1      1      2      2
            2017-03-16      1      1      2      2

        Let ``m`` be a Filter producing the following output::

                         AAPL   MSFT    MCD     BK
            2017-03-13  False   True   True   True
            2017-03-14   True  False   True   True
            2017-03-15   True   True  False   True
            2017-03-16   True   True   True  False

        Then ``f.demean()`` will subtract the mean from each row produced by
        ``f``.

        ::

                         AAPL   MSFT    MCD     BK
            2017-03-13 -1.500 -0.500  0.500  1.500
            2017-03-14 -0.625  0.375  1.375 -1.125
            2017-03-15 -0.625  0.375  1.375 -1.125
            2017-03-16  0.250  1.250 -1.250 -0.250

        ``f.demean(mask=m)`` will subtract the mean from each row, but means
        will be calculated ignoring values on the diagonal, and NaNs will
        written to the diagonal in the output. Diagonal values are ignored
        because they are the locations where the mask ``m`` produced False.

        ::

                         AAPL   MSFT    MCD     BK
            2017-03-13    NaN -1.000  0.000  1.000
            2017-03-14 -0.500    NaN  1.500 -1.000
            2017-03-15 -0.166  0.833    NaN -0.666
            2017-03-16  0.166  1.166 -1.333    NaN

        ``f.demean(groupby=c)`` will subtract the group-mean of AAPL/MSFT and
        MCD/BK from their respective entries.  The AAPL/MSFT are grouped
        together because both assets always produce 1 in the output of the
        classifier ``c``.  Similarly, MCD/BK are grouped together because they
        always produce 2.

        ::

                         AAPL   MSFT    MCD     BK
            2017-03-13 -0.500  0.500 -0.500  0.500
            2017-03-14 -0.500  0.500  1.250 -1.250
            2017-03-15 -0.500  0.500  1.250 -1.250
            2017-03-16 -0.500  0.500 -0.500  0.500

        ``f.demean(mask=m, groupby=c)`` will also subtract the group-mean of
        AAPL/MSFT and MCD/BK, but means will be calculated ignoring values on
        the diagonal , and NaNs will be written to the diagonal in the output.

        ::

                         AAPL   MSFT    MCD     BK
            2017-03-13    NaN  0.000 -0.500  0.500
            2017-03-14  0.000    NaN  1.250 -1.250
            2017-03-15 -0.500  0.500    NaN  0.000
            2017-03-16 -0.500  0.500  0.000    NaN

        Notes
        -----
        Mean is sensitive to the magnitudes of outliers. When working with
        factor that can potentially produce large outliers, it is often useful
        to use the ``mask`` parameter to discard values at the extremes of the
        distribution::

            >>> base = MyFactor(...)
            >>> normalized = base.demean(mask=base.percentile_between(1, 99))

        ``demean()`` is only supported on Factors of dtype float64.

        See Also
        --------
        :meth:`pandas.DataFrame.groupby`
        """
        # This is a named function so that it has a __name__ for use in the
        # graph repr of GroupedRowTransform.
        def demean(row):
            return row - nanmean(row)

        return GroupedRowTransform(
            transform=demean,
            factor=self,
            mask=mask,
            groupby=groupby,
        )

    @expect_types(
        mask=(Filter, NotSpecifiedType),
        groupby=(Classifier, NotSpecifiedType),
    )
    @float64_only
    def zscore(self, mask=NotSpecified, groupby=NotSpecified):
        """
        Construct a Factor that Z-Scores each day's results.

        The Z-Score of a row is defined as::

            (row - row.mean()) / row.stddev()

        If ``mask`` is supplied, ignore values where ``mask`` returns False
        when computing row means and standard deviations, and output NaN
        anywhere the mask is False.

        If ``groupby`` is supplied, compute by partitioning each row based on
        the values produced by ``groupby``, z-scoring the partitioned arrays,
        and stitching the sub-results back together.

        Parameters
        ----------
        mask : zipline.pipeline.Filter, optional
            A Filter defining values to ignore when Z-Scoring.
        groupby : zipline.pipeline.Classifier, optional
            A classifier defining partitions over which to compute Z-Scores.

        Returns
        -------
        zscored : zipline.pipeline.Factor
            A Factor producing that z-scores the output of self.

        Notes
        -----
        Mean and standard deviation are sensitive to the magnitudes of
        outliers. When working with factor that can potentially produce large
        outliers, it is often useful to use the ``mask`` parameter to discard
        values at the extremes of the distribution::

            >>> base = MyFactor(...)
            >>> normalized = base.zscore(mask=base.percentile_between(1, 99))

        ``zscore()`` is only supported on Factors of dtype float64.

        Example
        -------
        See :meth:`~zipline.pipeline.factors.Factor.demean` for an in-depth
        example of the semantics for ``mask`` and ``groupby``.

        See Also
        --------
        :meth:`pandas.DataFrame.groupby`
        """
        # This is a named function so that it has a __name__ for use in the
        # graph repr of GroupedRowTransform.
        def zscore(row):
            return (row - nanmean(row)) / nanstd(row)

        return GroupedRowTransform(
            transform=zscore,
            factor=self,
            mask=mask,
            groupby=groupby,
            window_safe=True,
        )

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
        :func:`scipy.stats.rankdata`
        :class:`zipline.pipeline.factors.factor.Rank`
        """
        return Rank(self, method=method, ascending=ascending, mask=mask)

    @expect_types(bins=int, mask=(Filter, NotSpecifiedType))
    def quantiles(self, bins, mask=NotSpecified):
        """
        Construct a Classifier computing quantiles of the output of ``self``.

        Every non-NaN data point the output is labelled with an integer value
        from 0 to (bins - 1).  NaNs are labelled with -1.

        If ``mask`` is supplied, ignore data points in locations for which
        ``mask`` produces False, and emit a label of -1 at those locations.

        Parameters
        ----------
        bins : int
            Number of bins labels to compute.
        mask : zipline.pipeline.Filter, optional
            Mask of values to ignore when computing quantiles.

        Returns
        -------
        quantiles : zipline.pipeline.classifiers.Quantiles
            A Classifier producing integer labels ranging from 0 to (bins - 1).
        """
        if mask is NotSpecified:
            mask = self.mask
        return Quantiles(inputs=(self,), bins=bins, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType))
    def quartiles(self, mask=NotSpecified):
        """
        Construct a Classifier computing quartiles over the output of ``self``.

        Every non-NaN data point the output is labelled with a value of either
        0, 1, 2, or 3, corresponding to the first, second, third, or fourth
        quartile over each row.  NaN data points are labelled with -1.

        If ``mask`` is supplied, ignore data points in locations for which
        ``mask`` produces False, and emit a label of -1 at those locations.

        Parameters
        ----------
        mask : zipline.pipeline.Filter, optional
            Mask of values to ignore when computing quartiles.

        Returns
        -------
        quartiles : zipline.pipeline.classifiers.Quantiles
            A Classifier producing integer labels ranging from 0 to 3.
        """
        return self.quantiles(bins=4, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType))
    def quintiles(self, mask=NotSpecified):
        """
        Construct a Classifier computing quintile labels on ``self``.

        Every non-NaN data point the output is labelled with a value of either
        0, 1, 2, or 3, 4, corresonding to quintiles over each row.  NaN data
        points are labelled with -1.

        If ``mask`` is supplied, ignore data points in locations for which
        ``mask`` produces False, and emit a label of -1 at those locations.

        Parameters
        ----------
        mask : zipline.pipeline.Filter, optional
            Mask of values to ignore when computing quintiles.

        Returns
        -------
        quintiles : zipline.pipeline.classifiers.Quantiles
            A Classifier producing integer labels ranging from 0 to 4.
        """
        return self.quantiles(bins=5, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType))
    def deciles(self, mask=NotSpecified):
        """
        Construct a Classifier computing decile labels on ``self``.

        Every non-NaN data point the output is labelled with a value from 0 to
        9 corresonding to deciles over each row.  NaN data points are labelled
        with -1.

        If ``mask`` is supplied, ignore data points in locations for which
        ``mask`` produces False, and emit a label of -1 at those locations.

        Parameters
        ----------
        mask : zipline.pipeline.Filter, optional
            Mask of values to ignore when computing deciles.

        Returns
        -------
        deciles : zipline.pipeline.classifiers.Quantiles
            A Classifier producing integer labels ranging from 0 to 9.
        """
        return self.quantiles(bins=10, mask=mask)

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
            calculating thresholds.  If mask is supplied, percentile cutoffs
            are computed each day using only assets for which ``mask`` returns
            True.  Assets for which ``mask`` produces False will produce False
            in the output of this Factor as well.

        Returns
        -------
        out : zipline.pipeline.filters.PercentileFilter
            A new filter that will compute the specified percentile-range mask.

        See Also
        --------
        zipline.pipeline.filters.filter.PercentileFilter
        """
        return PercentileFilter(
            self,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            mask=mask,
        )

    def isnull(self):
        """
        A Filter producing True for values where this Factor has missing data.

        Equivalent to self.isnan() when ``self.dtype`` is float64.
        Otherwise equivalent to ``self.eq(self.missing_value)``.

        Returns
        -------
        filter : zipline.pipeline.filters.Filter
        """
        if self.dtype == float64_dtype:
            # Using isnan is more efficient when possible because we can fold
            # the isnan computation with other NumExpr expressions.
            return self.isnan()
        else:
            return NullFilter(self)

    def notnull(self):
        """
        A Filter producing True for values where this Factor has complete data.

        Equivalent to ``~self.isnan()` when ``self.dtype`` is float64.
        Otherwise equivalent to ``(self != self.missing_value)``.
        """
        return ~self.isnull()

    @if_not_float64_tell_caller_to_use_isnull
    def isnan(self):
        """
        A Filter producing True for all values where this Factor is NaN.

        Returns
        -------
        nanfilter : zipline.pipeline.filters.Filter
        """
        return self != self

    @if_not_float64_tell_caller_to_use_isnull
    def notnan(self):
        """
        A Filter producing True for values where this Factor is not NaN.

        Returns
        -------
        nanfilter : zipline.pipeline.filters.Filter
        """
        return ~self.isnan()

    @if_not_float64_tell_caller_to_use_isnull
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


class GroupedRowTransform(Factor):
    """
    A Factor that transforms an input factor by applying a row-wise
    shape-preserving transformation on classifier-defined groups of that
    Factor.

    This is most often useful for normalization operators like ``zscore`` or
    ``demean``.

    Parameters
    ----------
    transform : function[ndarray[ndim=1] -> ndarray[ndim=1]]
        Function to apply over each row group.
    factor : zipline.pipeline.Factor
        The factor providing baseline data to transform.
    mask : zipline.pipeline.Filter
        Mask of entries to ignore when calculating transforms.
    groupby : zipline.pipeline.Classifier
        Classifier partitioning ``factor`` into groups to use when calculating
        means.

    Notes
    -----
    Users should rarely construct instances of this factor directly.  Instead,
    they should construct instances via factor normalization methods like
    ``zscore`` and ``demean``.

    See Also
    --------
    zipline.pipeline.factors.Factor.zscore
    zipline.pipeline.factors.Factor.demean
    """
    window_length = 0

    def __new__(cls, transform, factor, mask, groupby, **kwargs):

        if mask is NotSpecified:
            mask = factor.mask
        else:
            mask = mask & factor.mask

        if groupby is NotSpecified:
            groupby = Everything(mask=mask)

        return super(GroupedRowTransform, cls).__new__(
            GroupedRowTransform,
            transform=transform,
            inputs=(factor, groupby),
            missing_value=factor.missing_value,
            mask=mask,
            dtype=factor.dtype,
            **kwargs
        )

    def _init(self, transform, *args, **kwargs):
        self._transform = transform
        return super(GroupedRowTransform, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, transform, *args, **kwargs):
        return (
            super(GroupedRowTransform, cls).static_identity(*args, **kwargs),
            transform,
        )

    def _compute(self, arrays, dates, assets, mask):
        data = arrays[0]
        groupby_expr = self.inputs[1]
        if groupby_expr.dtype == int64_dtype:
            group_labels = arrays[1]
            null_label = self.inputs[1].missing_value
        elif groupby_expr.dtype == categorical_dtype:
            # Coerce our LabelArray into an isomorphic array of ints.  This is
            # necessary because np.where doesn't know about LabelArrays or the
            # void dtype.
            group_labels = arrays[1].as_int_array()
            null_label = arrays[1].missing_value_code
        else:
            raise TypeError(
                "Unexpected groupby dtype: %s." % groupby_expr.dtype
            )

        # Make a copy with the null code written to masked locations.
        group_labels = where(mask, group_labels, null_label)

        return where(
            group_labels != null_label,
            naive_grouped_rowwise_apply(
                data=data,
                group_labels=group_labels,
                func=self._transform,
            ),
            self.missing_value,
        )

    @property
    def transform_name(self):
        return self._transform.__name__

    def short_repr(self):
        return type(self).__name__ + '(%r)' % self.transform_name


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
    :func:`scipy.stats.rankdata`
    :class:`Factor.rank`

    Notes
    -----
    Most users should call Factor.rank rather than directly construct an
    instance of this class.
    """
    window_length = 0
    dtype = float64_dtype
    window_safe = True

    def __new__(cls, factor, method, ascending, mask):
        return super(Rank, cls).__new__(
            cls,
            inputs=(factor,),
            method=method,
            ascending=ascending,
            mask=mask,
        )

    def _init(self, method, ascending, *args, **kwargs):
        self._method = method
        self._ascending = ascending
        return super(Rank, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, method, ascending, *args, **kwargs):
        return (
            super(Rank, cls).static_identity(*args, **kwargs),
            method,
            ascending,
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
        return masked_rankdata_2d(
            arrays[0],
            mask,
            self.inputs[0].missing_value,
            self._method,
            self._ascending,
        )

    def __repr__(self):
        return "{type}({input_}, method='{method}', mask={mask})".format(
            type=type(self).__name__,
            input_=self.inputs[0],
            method=self._method,
            mask=self.mask,
        )


class CustomFactor(PositiveWindowLengthMixin, CustomTermMixin, Factor):
    '''
    Base class for user-defined Factors.

    Parameters
    ----------
    inputs : iterable, optional
        An iterable of `BoundColumn` instances (e.g. USEquityPricing.close),
        describing the data to load and pass to `self.compute`.  If this
        argument is not passed to the CustomFactor constructor, we look for a
        class-level attribute named `inputs`.
    outputs : iterable[str], optional
        An iterable of strings which represent the names of each output this
        factor should compute and return. If this argument is not passed to the
        CustomFactor constructor, we look for a class-level attribute named
        `outputs`.
    window_length : int, optional
        Number of rows to pass for each input.  If this argument is not passed
        to the CustomFactor constructor, we look for a class-level attribute
        named `window_length`.
    mask : zipline.pipeline.Filter, optional
        A Filter describing the assets on which we should compute each day.
        Each call to ``CustomFactor.compute`` will only receive assets for
        which ``mask`` produced True on the day for which compute is being
        called.

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
        out : np.array[self.dtype, ndim=1]
            Output array of the same shape as `assets`.  `compute` should write
            its desired return values into `out`. If multiple outputs are
            specified, `compute` should write its desired return values into
            `out.<output_name>` for each output name in `self.outputs`.
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

    A CustomFactor with multiple outputs:

    .. code-block:: python

        class MultipleOutputs(CustomFactor):
            inputs = [USEquityPricing.close]
            outputs = ['alpha', 'beta']
            window_length = N

            def compute(self, today, assets, out, close):
                computed_alpha, computed_beta = some_function(close)
                out.alpha[:] = computed_alpha
                out.beta[:] = computed_beta

        # Each output is returned as its own Factor upon instantiation.
        alpha, beta = MultipleOutputs()

        # Equivalently, we can create a single factor instance and access each
        # output as an attribute of that instance.
        multiple_outputs = MultipleOutputs()
        alpha = multiple_outputs.alpha
        beta = multiple_outputs.beta

    Note: If a CustomFactor has multiple outputs, all outputs must have the
    same dtype. For instance, in the example above, if alpha is a float then
    beta must also be a float.
    '''
    dtype = float64_dtype

    def __getattr__(self, attribute_name):
        if self.outputs is NotSpecified:
            return getattr(super(CustomFactor, self), attribute_name)
        if attribute_name in self.outputs:
            return RecarrayField(factor=self, attribute=attribute_name)
        else:
            raise AttributeError(
                'Instance of {factor} has no output named {attr!r}.'
                ' Possible choices are: {choices}.'.format(
                    factor=type(self).__name__,
                    attr=attribute_name,
                    choices=self.outputs,
                )
            )

    def __iter__(self):
        if self.outputs is NotSpecified:
            raise ValueError(
                '{factor} does not have multiple outputs.'.format(
                    factor=type(self).__name__,
                )
            )
        return (RecarrayField(self, attr) for attr in self.outputs)


class RecarrayField(SingleInputMixin, Factor):

    def __new__(cls, factor, attribute):
        return super(RecarrayField, cls).__new__(
            cls,
            attribute=attribute,
            inputs=[factor],
            window_length=0,
            mask=factor.mask,
            dtype=factor.dtype,
            missing_value=factor.missing_value,
        )

    def _init(self, attribute, *args, **kwargs):
        self._attribute = attribute
        return super(RecarrayField, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, attribute, *args, **kwargs):
        return (
            super(RecarrayField, cls).static_identity(*args, **kwargs),
            attribute,
        )

    def _compute(self, windows, dates, assets, mask):
        return windows[0][self._attribute]


class Latest(LatestMixin, CustomFactor):
    """
    Factor producing the most recently-known value of `inputs[0]` on each day.

    The `.latest` attribute of DataSet columns returns an instance of this
    Factor.
    """
    window_length = 1

    def compute(self, today, assets, out, data):
        out[:] = data[-1]
