"""
Technical Analysis Factors
--------------------------
"""
from numbers import Number
from numpy import (
    abs,
    arange,
    average,
    clip,
    corrcoef,
    diff,
    exp,
    fmax,
    full,
    inf,
    isnan,
    log,
    NINF,
    searchsorted,
    sqrt,
    sum as np_sum,
    where,
)
from numexpr import evaluate
from scipy.stats import linregress, spearmanr

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import SingleAsset
from zipline.pipeline.mixins import SingleInputMixin
from zipline.pipeline.term import NotSpecified
from zipline.utils.numpy_utils import ignore_nanwarnings
from zipline.utils.input_validation import expect_types
from zipline.utils.math_utils import (
    nanargmax,
    nanmax,
    nanmean,
    nansum,
)
from .factor import CustomFactor


class Returns(CustomFactor):
    """
    Calculates the percent change in close price over the given window_length.

    **Default Inputs**: [USEquityPricing.close]
    """
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close):
        out[:] = (close[-1] - close[0]) / close[0]


class RSI(CustomFactor, SingleInputMixin):
    """
    Relative Strength Index

    **Default Inputs**: [USEquityPricing.close]

    **Default Window Length**: 15
    """
    window_length = 15
    inputs = (USEquityPricing.close,)

    def compute(self, today, assets, out, closes):
        diffs = diff(closes, axis=0)
        ups = nanmean(clip(diffs, 0, inf), axis=0)
        downs = abs(nanmean(clip(diffs, -inf, 0), axis=0))
        return evaluate(
            "100 - (100 / (1 + (ups / downs)))",
            local_dict={'ups': ups, 'downs': downs},
            global_dict={},
            out=out,
        )


class SimpleMovingAverage(CustomFactor, SingleInputMixin):
    """
    Average Value of an arbitrary column

    **Default Inputs**: None

    **Default Window Length**: None
    """
    # numpy's nan functions throw warnings when passed an array containing only
    # nans, but they still returns the desired value (nan), so we ignore the
    # warning.
    ctx = ignore_nanwarnings()

    def compute(self, today, assets, out, data):
        out[:] = nanmean(data, axis=0)


class WeightedAverageValue(CustomFactor):
    """
    Helper for VWAP-like computations.

    **Default Inputs:** None

    **Default Window Length:** None
    """
    def compute(self, today, assets, out, base, weight):
        out[:] = nansum(base * weight, axis=0) / nansum(weight, axis=0)


class VWAP(WeightedAverageValue):
    """
    Volume Weighted Average Price

    **Default Inputs:** [USEquityPricing.close, USEquityPricing.volume]

    **Default Window Length:** None
    """
    inputs = (USEquityPricing.close, USEquityPricing.volume)


class MaxDrawdown(CustomFactor, SingleInputMixin):
    """
    Max Drawdown

    **Default Inputs:** None

    **Default Window Length:** None
    """
    ctx = ignore_nanwarnings()

    def compute(self, today, assets, out, data):
        drawdowns = fmax.accumulate(data, axis=0) - data
        drawdowns[isnan(drawdowns)] = NINF
        drawdown_ends = nanargmax(drawdowns, axis=0)

        # TODO: Accelerate this loop in Cython or Numba.
        for i, end in enumerate(drawdown_ends):
            peak = nanmax(data[:end + 1, i])
            out[i] = (peak - data[end, i]) / data[end, i]


class AverageDollarVolume(CustomFactor):
    """
    Average Daily Dollar Volume

    **Default Inputs:** [USEquityPricing.close, USEquityPricing.volume]

    **Default Window Length:** None
    """
    inputs = [USEquityPricing.close, USEquityPricing.volume]

    def compute(self, today, assets, out, close, volume):
        out[:] = nanmean(close * volume, axis=0)


class RollingPearsonOfReturns(SingleInputMixin, CustomFactor):
    """
    Calculates the Pearson product-moment correlation coefficient of the
    returns of the given asset with the returns of all other assets.

    **Default Inputs:** [Returns]

    Parameters
    ----------
    baseline_asset : zipline.assets.Asset
        The asset to correlate with all other assets.
    returns_length : int > 1
        Length of the lookback window over which to compute returns.
    correlation_length : int > 1
        Length of the lookback window over which to compute each correlation
        coefficient.

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
            returns_length=10, correlation_length=5,
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
    :class:`zipline.pipeline.factors.technical.RollingReturnsRegression`
    """
    params = ['baseline_asset']

    def __new__(cls,
                baseline_asset,
                returns_length,
                correlation_length,
                mask=NotSpecified,
                **kwargs):
        if mask is not NotSpecified:
            # Make sure we do not filter out the asset of interest.
            mask = mask | SingleAsset(asset=baseline_asset)
        return super(RollingPearsonOfReturns, cls).__new__(
            cls,
            baseline_asset=baseline_asset,
            inputs=[Returns(window_length=returns_length)],
            window_length=correlation_length,
            mask=mask,
            **kwargs
        )

    def compute(self, today, assets, out, data, baseline_asset):
        asset_col = searchsorted(list(assets), baseline_asset.sid)
        out[:] = corrcoef(data, rowvar=0)[asset_col]


class RollingSpearmanOfReturns(RollingPearsonOfReturns):
    """
    Calculates the Spearman rank correlation coefficient of the returns of the
    given asset with the returns of all other assets.

    **Default Inputs:** [Returns]

    Parameters
    ----------
    baseline_asset : zipline.assets.Asset
        The asset to correlate with all other assets.
    returns_length : int > 1
        Length of the lookback window over which to compute returns.
    correlation_length : int > 1
        Length of the lookback window over which to compute each correlation
        coefficient.

    See Also
    --------
    :class:`zipline.pipeline.factors.technical.RollingSpearmanOfReturns`
    """
    def compute(self, today, assets, out, data, baseline_asset):
        asset_col = searchsorted(list(assets), baseline_asset.sid)
        out[:] = spearmanr(data)[0][asset_col]


class SingleRegressionFactor(SingleInputMixin, CustomFactor):
    """
    Perform an ordinary least-squares regression predicting the returns of all
    other assets on the given asset.

    **Default Inputs:** [Returns]

    Parameters
    ----------
    asset : zipline.assets.Asset
        The asset to regress against all other assets.
    returns_length : int > 1
        Length of the lookback window over which to compute returns.
    regression_length : int > 1
        Length of the lookback window over which to compute each regression.
    dependent : bool, optional
        Determines if the given asset should be the independent or dependent
        variable in the regressions. When True, the given asset is the
        dependent variable. Default is False.

    Note
    ----
    This factor is designed to return two outputs:
        - alpha, a factor that computes the intercepts of the regressions.
        - beta, a factor that computes the slopes of the regressions.

    Example
    -------
    Let the following be example 10-day returns for three different stocks::

                      AAPL    MSFT     FB
        2017-03-13    -.03     .03    .04
        2017-03-14    -.02    -.03    .02
        2017-03-15    -.01     .02    .01
        2017-03-16       0    -.02    .01
        2017-03-17     .01     .04   -.01
        2017-03-20     .02    -.03   -.02
        2017-03-21     .03     .01   -.02
        2017-03-22     .04    -.02   -.02

    Suppose we are interested in predicting each stock's returns from AAPL's
    over rolling 5-day look back windows. We can compute rolling regression
    coefficients (alpha and beta) from 2017-03-17 to 2017-03-22 by doing::

        # Equivalent to:
        # alpha, beta = SingleRegressionFactor(
        #     asset=Equity(24),
        #     returns_length=10,
        #     regression_length=5,
        # )
        regression_factor = SingleRegressionFactor(
            asset=Equity(24),
            returns_length=10,
            regression_length=5,
        )
        alpha = regression_factor.alpha
        beta = regression_factor.beta

    The result of computing ``alpha`` from 2017-03-17 to 2017-03-22 gives::

                      AAPL    MSFT     FB
        2017-03-17       0    .011   .003
        2017-03-20       0   -.004   .004
        2017-03-21       0    .007   .006
        2017-03-22       0    .002   .008

    And the result of computing ``beta`` from 2017-03-17 to 2017-03-22 gives::

                      AAPL    MSFT     FB
        2017-03-17       1      .3   -1.1
        2017-03-20       1      .2     -1
        2017-03-21       1     -.3     -1
        2017-03-22       1     -.3    -.9

    Note that AAPL's column for alpha is all 0's and for beta is all 1's, as
    the regression line of a data set with itself is simply the function y = x.

    To understand how each of the other values were calculated, take for
    example MSFT's ``alpha`` and ``beta`` values on 2017-03-17 (.011 and .3,
    respectively). These values are the result of running a linear regression
    predicting MSFT's returns from AAPL's returns, using values starting at
    2017-03-17 and looking back 5 days. That is, the regression was run with
    x = [-.03, -.02, -.01, 0, .01] and y = [.03, -.03, .02, -.02, .04], and it
    produced a slope of .3 and an intercept of .011.

    See Also
    --------
    :class:`zipline.pipeline.factors.technical.CorrelationFactor`
    """
    params = ('asset', 'dependent')

    def __new__(cls,
                asset,
                returns_length,
                regression_length,
                dependent=False,
                mask=NotSpecified,
                dtype=NotSpecified,
                missing_value=NotSpecified,
                **kwargs):
        new_instance = super(SingleRegressionFactor, cls).__new__(
            cls,
            asset=asset,
            dependent=dependent,
            inputs=[Returns(window_length=returns_length)],
            outputs=['alpha', 'beta'],
            window_length=regression_length,
            mask=mask,
            dtype=dtype,
            missing_value=missing_value,
            **kwargs
        )
        return new_instance

    def compute(self, today, assets, out, returns, asset, dependent):
        asset_col = where(assets == asset.sid)[0][0]
        my_asset = returns[:, asset_col]
        for i in range(len(out)):
            other_asset = returns[:, i]
            if dependent:
                regr_results = linregress(y=my_asset, x=other_asset)
            else:
                regr_results = linregress(y=other_asset, x=my_asset)
            # `linregress` returns its results in the following order:
            # slope, intercept, r-value, p-value, stderr
            out.alpha[i] = regr_results[1]
            out.beta[i] = regr_results[0]


class _ExponentialWeightedFactor(SingleInputMixin, CustomFactor):
    """
    Base class for factors implementing exponential-weighted operations.

    **Default Inputs:** None

    **Default Window Length:** None

    Parameters
    ----------
    inputs : length-1 list or tuple of BoundColumn
        The expression over which to compute the average.
    window_length : int > 0
        Length of the lookback window over which to compute the average.
    decay_rate : float, 0 < decay_rate <= 1
        Weighting factor by which to discount past observations.

        When calculating historical averages, rows are multiplied by the
        sequence::

            decay_rate, decay_rate ** 2, decay_rate ** 3, ...

    Methods
    -------
    weights
    from_span
    from_halflife
    from_center_of_mass
    """
    params = ('decay_rate',)

    @staticmethod
    def weights(length, decay_rate):
        """
        Return weighting vector for an exponential moving statistic on `length`
        rows with a decay rate of `decay_rate`.
        """
        return full(length, decay_rate, float) ** arange(length + 1, 1, -1)

    @classmethod
    @expect_types(span=Number)
    def from_span(cls, inputs, window_length, span):
        """
        Convenience constructor for passing `decay_rate` in terms of `span`.

        Forwards `decay_rate` as `1 - (2.0 / (1 + span))`.  This provides the
        behavior equivalent to passing `span` to pandas.ewma.

        Example
        -------
        .. code-block:: python

            # Equivalent to:
            # my_ewma = EWMA(
            #    inputs=[USEquityPricing.close],
            #    window_length=30,
            #    decay_rate=(1 - (2.0 / (1 + 15.0))),
            # )
            my_ewma = EWMA.from_span(
                inputs=[USEquityPricing.close],
                window_length=30,
                span=15,
            )

        Note
        ----
        This classmethod is provided by both
        :class:`ExponentialWeightedMovingAverage` and
        :class:`ExponentialWeightedMovingStdDev`.
        """
        if span <= 1:
            raise ValueError(
                "`span` must be a positive number. %s was passed." % span
            )

        decay_rate = (1.0 - (2.0 / (1.0 + span)))
        assert 0.0 < decay_rate <= 1.0

        return cls(
            inputs=inputs,
            window_length=window_length,
            decay_rate=decay_rate,
        )

    @classmethod
    @expect_types(halflife=Number)
    def from_halflife(cls, inputs, window_length, halflife):
        """
        Convenience constructor for passing ``decay_rate`` in terms of half
        life.

        Forwards ``decay_rate`` as ``exp(log(.5) / halflife)``.  This provides
        the behavior equivalent to passing `halflife` to pandas.ewma.

        Example
        -------
        .. code-block:: python

            # Equivalent to:
            # my_ewma = EWMA(
            #    inputs=[USEquityPricing.close],
            #    window_length=30,
            #    decay_rate=np.exp(np.log(0.5) / 15),
            # )
            my_ewma = EWMA.from_halflife(
                inputs=[USEquityPricing.close],
                window_length=30,
                halflife=15,
            )

        Note
        ----
        This classmethod is provided by both
        :class:`ExponentialWeightedMovingAverage` and
        :class:`ExponentialWeightedMovingStdDev`.
        """
        if halflife <= 0:
            raise ValueError(
                "`span` must be a positive number. %s was passed." % halflife
            )
        decay_rate = exp(log(.5) / halflife)
        assert 0.0 < decay_rate <= 1.0

        return cls(
            inputs=inputs,
            window_length=window_length,
            decay_rate=decay_rate,
        )

    @classmethod
    def from_center_of_mass(cls, inputs, window_length, center_of_mass):
        """
        Convenience constructor for passing `decay_rate` in terms of center of
        mass.

        Forwards `decay_rate` as `1 - (1 / 1 + center_of_mass)`.  This provides
        behavior equivalent to passing `center_of_mass` to pandas.ewma.

        Example
        -------
        .. code-block:: python

            # Equivalent to:
            # my_ewma = EWMA(
            #    inputs=[USEquityPricing.close],
            #    window_length=30,
            #    decay_rate=(1 - (1 / 15.0)),
            # )
            my_ewma = EWMA.from_center_of_mass(
                inputs=[USEquityPricing.close],
                window_length=30,
                center_of_mass=15,
            )

        Note
        ----
        This classmethod is provided by both
        :class:`ExponentialWeightedMovingAverage` and
        :class:`ExponentialWeightedMovingStdDev`.
        """
        return cls(
            inputs=inputs,
            window_length=window_length,
            decay_rate=(1.0 - (1.0 / (1.0 + center_of_mass))),
        )


class ExponentialWeightedMovingAverage(_ExponentialWeightedFactor):
    """
    Exponentially Weighted Moving Average

    **Default Inputs:** None

    **Default Window Length:** None

    Parameters
    ----------
    inputs : length-1 list/tuple of BoundColumn
        The expression over which to compute the average.
    window_length : int > 0
        Length of the lookback window over which to compute the average.
    decay_rate : float, 0 < decay_rate <= 1
        Weighting factor by which to discount past observations.

        When calculating historical averages, rows are multiplied by the
        sequence::

            decay_rate, decay_rate ** 2, decay_rate ** 3, ...

    Notes
    -----
    - This class can also be imported under the name ``EWMA``.

    See Also
    --------
    :func:`pandas.ewma`
    """
    def compute(self, today, assets, out, data, decay_rate):
        out[:] = average(
            data,
            axis=0,
            weights=self.weights(len(data), decay_rate),
        )


class ExponentialWeightedMovingStdDev(_ExponentialWeightedFactor):
    """
    Exponentially Weighted Moving Standard Deviation

    **Default Inputs:** None

    **Default Window Length:** None

    Parameters
    ----------
    inputs : length-1 list/tuple of BoundColumn
        The expression over which to compute the average.
    window_length : int > 0
        Length of the lookback window over which to compute the average.
    decay_rate : float, 0 < decay_rate <= 1
        Weighting factor by which to discount past observations.

        When calculating historical averages, rows are multiplied by the
        sequence::

            decay_rate, decay_rate ** 2, decay_rate ** 3, ...

    Notes
    -----
    - This class can also be imported under the name ``EWMSTD``.

    See Also
    --------
    :func:`pandas.ewmstd`
    """

    def compute(self, today, assets, out, data, decay_rate):
        weights = self.weights(len(data), decay_rate)

        mean = average(data, axis=0, weights=weights)
        variance = average((data - mean) ** 2, axis=0, weights=weights)

        squared_weight_sum = (np_sum(weights) ** 2)
        bias_correction = (
            squared_weight_sum / (squared_weight_sum - np_sum(weights ** 2))
        )
        out[:] = sqrt(variance * bias_correction)


# Convenience aliases.
EWMA = ExponentialWeightedMovingAverage
EWMSTD = ExponentialWeightedMovingStdDev
