
from scipy.stats import (
    linregress,
    pearsonr,
    spearmanr,
)

from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.filters import SingleAsset
from zipline.pipeline.mixins import SingleInputMixin
from zipline.pipeline.sentinels import NotSpecified
from zipline.pipeline.term import AssetExists
from zipline.utils.input_validation import expect_dtypes
from zipline.utils.numpy_utils import float64_dtype, int64_dtype

from .technical import Returns


ALLOWED_DTYPES = (float64_dtype, int64_dtype)


class _RollingCorrelation(CustomFactor, SingleInputMixin):

    @expect_dtypes(target_factor=ALLOWED_DTYPES, target_slice=ALLOWED_DTYPES)
    def __new__(cls,
                target_factor,
                target_slice,
                correlation_length,
                mask=NotSpecified):
        return super(_RollingCorrelation, cls).__new__(
            cls,
            inputs=[target_factor, target_slice],
            window_length=correlation_length,
            mask=mask,
        )


class RollingPearson(_RollingCorrelation):
    """
    A Factor that computes pearson correlation coefficients between a single
    column of data and the columns of another Factor.

    Parameters
    ----------
    target_factor : zipline.pipeline.factors.Factor
        The factor for which to compute correlations of each of its columns
        with `target_slice`.
    target_slice : zipline.pipeline.slice.Slice
        The column of data with which to compute correlations against each
        column of data produced by `target_factor`.
    correlation_length : int
        Length of the lookback window over which to compute each
        correlation coefficient.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets (columns) of `target_factor` should
        have their correlation with `target_slice` computed each day.

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
    def compute(self, today, assets, out, factor_data, slice_data):
        slice_data_column = slice_data[:, 0]
        for i in range(len(out)):
            # pearsonr returns the R-value and the P-value.
            out[i] = pearsonr(factor_data[:, i], slice_data_column)[0]


class RollingSpearman(_RollingCorrelation):
    """
    A Factor that computes spearman rank correlation coefficients between a
    single column of data and the columns of another Factor.

    Parameters
    ----------
    target_factor : zipline.pipeline.factors.Factor
        The factor for which to compute correlations of each of its columns
        with `target_slice`.
    target_slice : zipline.pipeline.slice.Slice
        The column of data with which to compute correlations against each
        column of data produced by `target_factor`.
    correlation_length : int
        Length of the lookback window over which to compute each
        correlation coefficient.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets (columns) of `target_factor` should
        have their correlation with `target_slice` computed each day.

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
    def compute(self, today, assets, out, factor_data, slice_data):
        slice_data_column = slice_data[:, 0]
        for i in range(len(out)):
            # spearmanr returns the R-value and the P-value.
            out[i] = spearmanr(factor_data[:, i], slice_data_column)[0]


class RollingLinearRegression(CustomFactor, SingleInputMixin):
    """
    A Factor that performs an ordinary least-squares regression predicting the
    columns of another Factor from a single column of data.

    Parameters
    ----------
    target_factor : zipline.pipeline.factors.Factor
        The factor whose columns are the predicted/dependent variable of each
        regression with `target_slice`.
    target_slice : zipline.pipeline.slice.Slice
        The column of data to use as the predictor/independent variable in
        each regression with the columns of `target_factor`.
    correlation_length : int
        Length of the lookback window over which to compute each regression.
    mask : zipline.pipeline.Filter, optional
        A Filter describing which assets (columns) of `target_factor` should be
        regressed against `target_slice` each day.

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
    outputs = ['alpha', 'beta', 'r_value', 'p_value', 'stderr']

    @expect_dtypes(target_factor=ALLOWED_DTYPES, target_slice=ALLOWED_DTYPES)
    def __new__(cls,
                target_factor,
                target_slice,
                regression_length,
                mask=NotSpecified):
        return super(RollingLinearRegression, cls).__new__(
            cls,
            inputs=[target_factor, target_slice],
            window_length=regression_length,
            mask=mask,
        )

    def compute(self, today, assets, out, factor_data, slice_data):
        slice_data_column = slice_data[:, 0]

        alpha = out.alpha
        beta = out.beta
        r_value = out.r_value
        p_value = out.p_value
        stderr = out.stderr
        for i in range(len(out)):
            regr_results = linregress(y=factor_data[:, i], x=slice_data_column)
            # `linregress` returns its results in the following order:
            # slope, intercept, r-value, p-value, stderr
            alpha[i] = regr_results[1]
            beta[i] = regr_results[0]
            r_value[i] = regr_results[2]
            p_value[i] = regr_results[3]
            stderr[i] = regr_results[4]


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

    Note
    ----
    Computing this factor over many assets can be time consuming. It is
    recommended that a mask be used in order to limit the number of assets over
    which correlations are computed.

    Example
    -------
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
            target=Equity(8554),
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
    def __new__(cls,
                target,
                returns_length,
                correlation_length,
                mask=NotSpecified):
        # Use the `SingleAsset` filter here because it protects against
        # inputting a non-existent target asset.
        returns = Returns(
            window_length=returns_length,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        return super(RollingPearsonOfReturns, cls).__new__(
            cls,
            target_factor=returns,
            target_slice=returns[target],
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

    Note
    ----
    Computing this factor over many assets can be time consuming. It is
    recommended that a mask be used in order to limit the number of assets over
    which correlations are computed.

    See Also
    --------
    :class:`zipline.pipeline.factors.RollingPearsonOfReturns`
    :class:`zipline.pipeline.factors.RollingLinearRegressionOfReturns`
    """
    def __new__(cls,
                target,
                returns_length,
                correlation_length,
                mask=NotSpecified):
        # Use the `SingleAsset` filter here because it protects against
        # inputting a non-existent target asset.
        returns = Returns(
            window_length=returns_length,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        return super(RollingSpearmanOfReturns, cls).__new__(
            cls,
            target_factor=returns,
            target_slice=returns[target],
            correlation_length=correlation_length,
            mask=mask,
        )


class RollingLinearRegressionOfReturns(RollingLinearRegression):
    """
    Perform an ordinary least-squares regression predicting the returns of all
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
    :class:`zipline.pipeline.factors.CustomFactor`.

    Example
    -------
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
            target=Equity(8554),
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
    def __new__(cls,
                target,
                returns_length,
                regression_length,
                mask=NotSpecified):
        # Use the `SingleAsset` filter here because it protects against
        # inputting a non-existent target asset.
        returns = Returns(
            window_length=returns_length,
            mask=(AssetExists() | SingleAsset(asset=target)),
        )
        return super(RollingLinearRegressionOfReturns, cls).__new__(
            cls,
            target_factor=returns,
            target_slice=returns[target],
            regression_length=regression_length,
            mask=mask,
        )
