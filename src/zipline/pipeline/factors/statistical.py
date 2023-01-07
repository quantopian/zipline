from numexpr import evaluate
import numpy as np
from numpy import broadcast_arrays
from scipy.stats import (
    linregress,
    spearmanr,
)

from zipline.assets import Asset
from zipline.errors import IncompatibleTerms
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.filters import SingleAsset
from zipline.pipeline.mixins import StandardOutputs
from zipline.pipeline.sentinels import NotSpecified
from zipline.pipeline.term import AssetExists
from zipline.utils.input_validation import (
    expect_bounded,
    expect_dtypes,
    expect_types,
)
from zipline.utils.math_utils import nanmean
from zipline.utils.numpy_utils import (
    float64_dtype,
    int64_dtype,
)


from .basic import Returns


ALLOWED_DTYPES = (float64_dtype, int64_dtype)


class _RollingCorrelation(CustomFactor):
    @expect_dtypes(base_factor=ALLOWED_DTYPES, target=ALLOWED_DTYPES)
    @expect_bounded(correlation_length=(2, None))
    def __new__(cls, base_factor, target, correlation_length, mask=NotSpecified):
        if target.ndim == 2 and base_factor.mask is not target.mask:
            raise IncompatibleTerms(term_1=base_factor, term_2=target)

        return super(_RollingCorrelation, cls).__new__(
            cls,
            inputs=[base_factor, target],
            window_length=correlation_length,
            mask=mask,
        )


class RollingPearson(_RollingCorrelation):
    """
    A Factor that computes pearson correlation coefficients between the columns
    of a given Factor and either the columns of another Factor/BoundColumn or a
    slice/single column of data.

    Parameters
    ----------
    base_factor : zipline.pipeline.Factor
        The factor for which to compute correlations of each of its columns
        with `target`.
    target : zipline.pipeline.Term with a numeric dtype
        The term with which to compute correlations against each column of data
        produced by `base_factor`. This term may be a Factor, a BoundColumn or
        a Slice. If `target` is two-dimensional, correlations are computed
        asset-wise.
    correlation_length : int
        Length of the lookback window over which to compute each correlation
        coefficient.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets (columns) of `base_factor` should have
        their correlation with `target` computed each day.

    See Also
    --------
    :func:`scipy.stats.pearsonr`
    :meth:`Factor.pearsonr`
    :class:`zipline.pipeline.factors.RollingPearsonOfReturns`

    Notes
    -----
    Most users should call Factor.pearsonr rather than directly construct an
    instance of this class.
    """

    window_safe = True

    def compute(self, today, assets, out, base_data, target_data):
        vectorized_pearson_r(
            base_data,
            target_data,
            allowed_missing=0,
            out=out,
        )


class RollingSpearman(_RollingCorrelation):
    """
    A Factor that computes spearman rank correlation coefficients between the
    columns of a given Factor and either the columns of another
    Factor/BoundColumn or a slice/single column of data.

    Parameters
    ----------
    base_factor : zipline.pipeline.Factor
        The factor for which to compute correlations of each of its columns
        with `target`.
    target : zipline.pipeline.Term with a numeric dtype
        The term with which to compute correlations against each column of data
        produced by `base_factor`. This term may be a Factor, a BoundColumn or
        a Slice. If `target` is two-dimensional, correlations are computed
        asset-wise.
    correlation_length : int
        Length of the lookback window over which to compute each correlation
        coefficient.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets (columns) of `base_factor` should have
        their correlation with `target` computed each day.

    See Also
    --------
    :func:`scipy.stats.spearmanr`
    :meth:`Factor.spearmanr`
    :class:`zipline.pipeline.factors.RollingSpearmanOfReturns`

    Notes
    -----
    Most users should call Factor.spearmanr rather than directly construct an
    instance of this class.
    """

    window_safe = True

    def compute(self, today, assets, out, base_data, target_data):
        # If `target_data` is a Slice or single column of data, broadcast it
        # out to the same shape as `base_data`, then compute column-wise. This
        # is efficient because each column of the broadcasted array only refers
        # to a single memory location.
        target_data = broadcast_arrays(target_data, base_data)[0]
        for i in range(len(out)):
            out[i] = spearmanr(base_data[:, i], target_data[:, i])[0]


class RollingLinearRegression(CustomFactor):
    """
    A Factor that performs an ordinary least-squares regression predicting the
    columns of a given Factor from either the columns of another
    Factor/BoundColumn or a slice/single column of data.

    Parameters
    ----------
    dependent : zipline.pipeline.Factor
        The factor whose columns are the predicted/dependent variable of each
        regression with `independent`.
    independent : zipline.pipeline.slice.Slice or zipline.pipeline.Factor
        The factor/slice whose columns are the predictor/independent variable
        of each regression with `dependent`. If `independent` is a Factor,
        regressions are computed asset-wise.
    regression_length : int
        Length of the lookback window over which to compute each regression.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets (columns) of `dependent` should be
        regressed against `independent` each day.

    See Also
    --------
    :func:`scipy.stats.linregress`
    :meth:`Factor.linear_regression`
    :class:`zipline.pipeline.factors.RollingLinearRegressionOfReturns`

    Notes
    -----
    Most users should call Factor.linear_regression rather than directly
    construct an instance of this class.
    """

    outputs = ["alpha", "beta", "r_value", "p_value", "stderr"]

    @expect_dtypes(dependent=ALLOWED_DTYPES, independent=ALLOWED_DTYPES)
    @expect_bounded(regression_length=(2, None))
    def __new__(cls, dependent, independent, regression_length, mask=NotSpecified):
        if independent.ndim == 2 and dependent.mask is not independent.mask:
            raise IncompatibleTerms(term_1=dependent, term_2=independent)

        return super(RollingLinearRegression, cls).__new__(
            cls,
            inputs=[dependent, independent],
            window_length=regression_length,
            mask=mask,
        )

    def compute(self, today, assets, out, dependent, independent):
        alpha = out.alpha
        beta = out.beta
        r_value = out.r_value
        p_value = out.p_value
        stderr = out.stderr

        def regress(y, x):
            regr_results = linregress(y=y, x=x)
            # `linregress` returns its results in the following order:
            # slope, intercept, r-value, p-value, stderr
            alpha[i] = regr_results[1]
            beta[i] = regr_results[0]
            r_value[i] = regr_results[2]
            p_value[i] = regr_results[3]
            stderr[i] = regr_results[4]

        # If `independent` is a Slice or single column of data, broadcast it
        # out to the same shape as `dependent`, then compute column-wise. This
        # is efficient because each column of the broadcasted array only refers
        # to a single memory location.
        independent = broadcast_arrays(independent, dependent)[0]
        for i in range(len(out)):
            regress(y=dependent[:, i], x=independent[:, i])


class RollingPearsonOfReturns(RollingPearson):
    """
    Calculates the Pearson product-moment correlation coefficient of the
    returns of the given asset with the returns of all other assets.

    Pearson correlation is what most people mean when they say "correlation
    coefficient" or "R-value".

    Parameters
    ----------
    target : zipline.assets.Asset
        The asset to correlate with all other assets.
    returns_length : int >= 2
        Length of the lookback window over which to compute returns. Daily
        returns require a window length of 2.
    correlation_length : int >= 1
        Length of the lookback window over which to compute each correlation
        coefficient.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets should have their correlation with the
        target asset computed each day.

    Notes
    -----
    Computing this factor over many assets can be time consuming. It is
    recommended that a mask be used in order to limit the number of assets over
    which correlations are computed.

    Examples
    --------
    Let the following be example 10-day returns for three different assets::

                       SPY    MSFT     FB
        2017-03-13    -.03     .03    .04
        2017-03-14    -.02    -.03    .02
        2017-03-15    -.01     .02    .01
        2017-03-16       0    -.02    .01
        2017-03-17     .01     .04   -.01
        2017-03-20     .02    -.03   -.02
        2017-03-21     .03     .01   -.02
        2017-03-22     .04    -.02   -.02

    Suppose we are interested in SPY's rolling returns correlation with each
    stock from 2017-03-17 to 2017-03-22, using a 5-day look back window (that
    is, we calculate each correlation coefficient over 5 days of data). We can
    achieve this by doing::

        rolling_correlations = RollingPearsonOfReturns(
            target=sid(8554),
            returns_length=10,
            correlation_length=5,
        )

    The result of computing ``rolling_correlations`` from 2017-03-17 to
    2017-03-22 gives::

                       SPY   MSFT     FB
        2017-03-17       1    .15   -.96
        2017-03-20       1    .10   -.96
        2017-03-21       1   -.16   -.94
        2017-03-22       1   -.16   -.85

    Note that the column for SPY is all 1's, as the correlation of any data
    series with itself is always 1. To understand how each of the other values
    were calculated, take for example the .15 in MSFT's column. This is the
    correlation coefficient between SPY's returns looking back from 2017-03-17
    (-.03, -.02, -.01, 0, .01) and MSFT's returns (.03, -.03, .02, -.02, .04).

    See Also
    --------
    :class:`zipline.pipeline.factors.RollingSpearmanOfReturns`
    :class:`zipline.pipeline.factors.RollingLinearRegressionOfReturns`
    """

    def __new__(cls, target, returns_length, correlation_length, mask=NotSpecified):
        # Use the `SingleAsset` filter here because it protects against
        # inputting a non-existent target asset.
        returns = Returns(
            window_length=returns_length,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        return super(RollingPearsonOfReturns, cls).__new__(
            cls,
            base_factor=returns,
            target=returns[target],
            correlation_length=correlation_length,
            mask=mask,
        )


class RollingSpearmanOfReturns(RollingSpearman):
    """
    Calculates the Spearman rank correlation coefficient of the returns of the
    given asset with the returns of all other assets.

    Parameters
    ----------
    target : zipline.assets.Asset
        The asset to correlate with all other assets.
    returns_length : int >= 2
        Length of the lookback window over which to compute returns. Daily
        returns require a window length of 2.
    correlation_length : int >= 1
        Length of the lookback window over which to compute each correlation
        coefficient.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets should have their correlation with the
        target asset computed each day.

    Notes
    -----
    Computing this factor over many assets can be time consuming. It is
    recommended that a mask be used in order to limit the number of assets over
    which correlations are computed.

    See Also
    --------
    :class:`zipline.pipeline.factors.RollingPearsonOfReturns`
    :class:`zipline.pipeline.factors.RollingLinearRegressionOfReturns`
    """

    def __new__(cls, target, returns_length, correlation_length, mask=NotSpecified):
        # Use the `SingleAsset` filter here because it protects against
        # inputting a non-existent target asset.
        returns = Returns(
            window_length=returns_length,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        return super(RollingSpearmanOfReturns, cls).__new__(
            cls,
            base_factor=returns,
            target=returns[target],
            correlation_length=correlation_length,
            mask=mask,
        )


class RollingLinearRegressionOfReturns(RollingLinearRegression):
    """Perform an ordinary least-squares regression predicting the returns of all
    other assets on the given asset.

    Parameters
    ----------
    target : zipline.assets.Asset
        The asset to regress against all other assets.
    returns_length : int >= 2
        Length of the lookback window over which to compute returns. Daily
        returns require a window length of 2.
    regression_length : int >= 1
        Length of the lookback window over which to compute each regression.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets should be regressed against the target
        asset each day.

    Notes
    -----
    Computing this factor over many assets can be time consuming. It is
    recommended that a mask be used in order to limit the number of assets over
    which regressions are computed.

    This factor is designed to return five outputs:

    - alpha, a factor that computes the intercepts of each regression.
    - beta, a factor that computes the slopes of each regression.
    - r_value, a factor that computes the correlation coefficient of each
      regression.
    - p_value, a factor that computes, for each regression, the two-sided
      p-value for a hypothesis test whose null hypothesis is that the slope is
      zero.
    - stderr, a factor that computes the standard error of the estimate of each
      regression.

    For more help on factors with multiple outputs, see
    :class:`zipline.pipeline.CustomFactor`.

    Examples
    --------
    Let the following be example 10-day returns for three different assets::

                       SPY    MSFT     FB
        2017-03-13    -.03     .03    .04
        2017-03-14    -.02    -.03    .02
        2017-03-15    -.01     .02    .01
        2017-03-16       0    -.02    .01
        2017-03-17     .01     .04   -.01
        2017-03-20     .02    -.03   -.02
        2017-03-21     .03     .01   -.02
        2017-03-22     .04    -.02   -.02

    Suppose we are interested in predicting each stock's returns from SPY's
    over rolling 5-day look back windows. We can compute rolling regression
    coefficients (alpha and beta) from 2017-03-17 to 2017-03-22 by doing::

        regression_factor = RollingRegressionOfReturns(
            target=sid(8554),
            returns_length=10,
            regression_length=5,
        )
        alpha = regression_factor.alpha
        beta = regression_factor.beta

    The result of computing ``alpha`` from 2017-03-17 to 2017-03-22 gives::

                       SPY    MSFT     FB
        2017-03-17       0    .011   .003
        2017-03-20       0   -.004   .004
        2017-03-21       0    .007   .006
        2017-03-22       0    .002   .008

    And the result of computing ``beta`` from 2017-03-17 to 2017-03-22 gives::

                       SPY    MSFT     FB
        2017-03-17       1      .3   -1.1
        2017-03-20       1      .2     -1
        2017-03-21       1     -.3     -1
        2017-03-22       1     -.3    -.9

    Note that SPY's column for alpha is all 0's and for beta is all 1's, as the
    regression line of SPY with itself is simply the function y = x.

    To understand how each of the other values were calculated, take for
    example MSFT's ``alpha`` and ``beta`` values on 2017-03-17 (.011 and .3,
    respectively). These values are the result of running a linear regression
    predicting MSFT's returns from SPY's returns, using values starting at
    2017-03-17 and looking back 5 days. That is, the regression was run with
    x = [-.03, -.02, -.01, 0, .01] and y = [.03, -.03, .02, -.02, .04], and it
    produced a slope of .3 and an intercept of .011.

    See Also
    --------
    :class:`zipline.pipeline.factors.RollingPearsonOfReturns`
    :class:`zipline.pipeline.factors.RollingSpearmanOfReturns`
    """

    window_safe = True

    def __new__(cls, target, returns_length, regression_length, mask=NotSpecified):
        # Use the `SingleAsset` filter here because it protects against
        # inputting a non-existent target asset.
        returns = Returns(
            window_length=returns_length,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        return super(RollingLinearRegressionOfReturns, cls).__new__(
            cls,
            dependent=returns,
            independent=returns[target],
            regression_length=regression_length,
            mask=mask,
        )


class SimpleBeta(CustomFactor, StandardOutputs):
    """Factor producing the slope of a regression line between each asset's daily
    returns to the daily returns of a single "target" asset.

    Parameters
    ----------
    target : zipline.Asset
        Asset against which other assets should be regressed.
    regression_length : int
        Number of days of daily returns to use for the regression.
    allowed_missing_percentage : float, optional
        Percentage of returns observations (between 0 and 1) that are allowed
        to be missing when calculating betas. Assets with more than this
        percentage of returns observations missing will produce values of
        NaN. Default behavior is that 25% of inputs can be missing.
    """

    window_safe = True
    dtype = float64_dtype
    params = ("allowed_missing_count",)

    @expect_types(
        target=Asset,
        regression_length=int,
        allowed_missing_percentage=(int, float),
        __funcname="SimpleBeta",
    )
    @expect_bounded(
        regression_length=(3, None),
        allowed_missing_percentage=(0.0, 1.0),
        __funcname="SimpleBeta",
    )
    def __new__(cls, target, regression_length, allowed_missing_percentage=0.25):
        daily_returns = Returns(
            window_length=2,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        allowed_missing_count = int(allowed_missing_percentage * regression_length)
        return super(SimpleBeta, cls).__new__(
            cls,
            inputs=[daily_returns, daily_returns[target]],
            window_length=regression_length,
            allowed_missing_count=allowed_missing_count,
        )

    def compute(
        self, today, assets, out, all_returns, target_returns, allowed_missing_count
    ):
        vectorized_beta(
            dependents=all_returns,
            independent=target_returns,
            allowed_missing=allowed_missing_count,
            out=out,
        )

    def graph_repr(self):
        return "{}({!r}, {}, {})".format(
            type(self).__name__,
            str(self.target.symbol),  # coerce from unicode to str in py2.
            self.window_length,
            self.params["allowed_missing_count"],
        )

    @property
    def target(self):
        """Get the target of the beta calculation."""
        return self.inputs[1].asset

    def __repr__(self):
        return "{}({}, length={}, allowed_missing={})".format(
            type(self).__name__,
            self.target,
            self.window_length,
            self.params["allowed_missing_count"],
        )


def vectorized_beta(dependents, independent, allowed_missing, out=None):
    """Compute slopes of linear regressions between columns of ``dependents`` and
    ``independent``.

    Parameters
    ----------
    dependents : np.array[N, M]
        Array with columns of data to be regressed against ``independent``.
    independent : np.array[N, 1]
        Independent variable of the regression
    allowed_missing : int
        Number of allowed missing (NaN) observations per column. Columns with
        more than this many non-nan observations in either ``dependents`` or
        ``independents`` will output NaN as the regression coefficient.
    out : np.array[M] or None, optional
        Output array into which to write results.  If None, a new array is
        created and returned.

    Returns
    -------
    slopes : np.array[M]
        Linear regression coefficients for each column of ``dependents``.
    """
    # Cache these as locals since we're going to call them multiple times.
    nan = np.nan
    isnan = np.isnan
    N, M = dependents.shape

    if out is None:
        out = np.full(M, nan)

    # Copy N times as a column vector and fill with nans to have the same
    # missing value pattern as the dependent variable.
    #
    # PERF_TODO: We could probably avoid the space blowup by doing this in
    # Cython.

    # shape: (N, M)
    independent = np.where(
        isnan(dependents),
        nan,
        independent,
    )

    # Calculate beta as Cov(X, Y) / Cov(X, X).
    # https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line  # noqa
    #
    # NOTE: The usual formula for covariance is::
    #
    #    mean((X - mean(X)) * (Y - mean(Y)))
    #
    # However, we don't actually need to take the mean of both sides of the
    # product, because of the folllowing equivalence::
    #
    # Let X_res = (X - mean(X)).
    # We have:
    #
    #     mean(X_res * (Y - mean(Y))) = mean(X_res * (Y - mean(Y)))
    #                             (1) = mean((X_res * Y) - (X_res * mean(Y)))
    #                             (2) = mean(X_res * Y) - mean(X_res * mean(Y))
    #                             (3) = mean(X_res * Y) - mean(X_res) * mean(Y)
    #                             (4) = mean(X_res * Y) - 0 * mean(Y)
    #                             (5) = mean(X_res * Y)
    #
    #
    # The tricky step in the above derivation is step (4). We know that
    # mean(X_res) is zero because, for any X:
    #
    #     mean(X - mean(X)) = mean(X) - mean(X) = 0.
    #
    # The upshot of this is that we only have to center one of `independent`
    # and `dependent` when calculating covariances. Since we need the centered
    # `independent` to calculate its variance in the next step, we choose to
    # center `independent`.

    # shape: (N, M)
    ind_residual = independent - nanmean(independent, axis=0)

    # shape: (M,)
    covariances = nanmean(ind_residual * dependents, axis=0)

    # We end up with different variances in each column here because each
    # column may have a different subset of the data dropped due to missing
    # data in the corresponding dependent column.
    # shape: (M,)
    independent_variances = nanmean(ind_residual**2, axis=0)

    # shape: (M,)
    np.divide(covariances, independent_variances, out=out)

    # Write nans back to locations where we have more then allowed number of
    # missing entries.
    nanlocs = isnan(independent).sum(axis=0) > allowed_missing
    out[nanlocs] = nan

    return out


def vectorized_pearson_r(dependents, independents, allowed_missing, out=None):
    """Compute Pearson's r between columns of ``dependents`` and ``independents``.

    Parameters
    ----------
    dependents : np.array[N, M]
        Array with columns of data to be regressed against ``independent``.
    independents : np.array[N, M] or np.array[N, 1]
        Independent variable(s) of the regression. If a single column is
        passed, it is broadcast to the shape of ``dependents``.
    allowed_missing : int
        Number of allowed missing (NaN) observations per column. Columns with
        more than this many non-nan observations in either ``dependents`` or
        ``independents`` will output NaN as the correlation coefficient.
    out : np.array[M] or None, optional
        Output array into which to write results.  If None, a new array is
        created and returned.

    Returns
    -------
    correlations : np.array[M]
        Pearson correlation coefficients for each column of ``dependents``.

    See Also
    --------
    :class:`zipline.pipeline.factors.RollingPearson`
    :class:`zipline.pipeline.factors.RollingPearsonOfReturns`
    """
    nan = np.nan
    isnan = np.isnan
    N, M = dependents.shape

    if out is None:
        out = np.full(M, nan)

    if allowed_missing > 0:
        # If we're handling nans robustly, we need to mask both arrays to
        # locations where either was nan.
        either_nan = isnan(dependents) | isnan(independents)
        independents = np.where(either_nan, nan, independents)
        dependents = np.where(either_nan, nan, dependents)
        mean = nanmean
    else:
        # Otherwise, we can just use mean, which will give us a nan for any
        # column where there's ever a nan.
        mean = np.mean

    # Pearson R is Cov(X, Y) / StdDev(X) * StdDev(Y)
    # c.f. https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    ind_residual = independents - mean(independents, axis=0)
    dep_residual = dependents - mean(dependents, axis=0)

    ind_variance = mean(ind_residual**2, axis=0)
    dep_variance = mean(dep_residual**2, axis=0)

    covariances = mean(ind_residual * dep_residual, axis=0)

    evaluate(
        "where(mask, nan, cov / sqrt(ind_variance * dep_variance))",
        local_dict={
            "cov": covariances,
            "mask": isnan(independents).sum(axis=0) > allowed_missing,
            "nan": np.nan,
            "ind_variance": ind_variance,
            "dep_variance": dep_variance,
        },
        global_dict={},
        out=out,
    )
    return out
