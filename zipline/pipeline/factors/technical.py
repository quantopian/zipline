"""
Technical Analysis Factors
--------------------------
"""
from __future__ import division

from numbers import Number
from numpy import (
    abs,
    arange,
    average,
    clip,
    diff,
    dstack,
    exp,
    fmax,
    full,
    inf,
    isnan,
    log,
    NINF,
    sqrt,
    sum as np_sum,
    nan
)
from numexpr import evaluate

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.mixins import SingleInputMixin
from zipline.utils.numpy_utils import ignore_nanwarnings
from zipline.utils.input_validation import expect_types
from zipline.utils.math_utils import (
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanstd,
    nansum,
    nanmin,
)
from .factor import CustomFactor

from talib import MACD


class Returns(CustomFactor):
    """
    Calculates the percent change in close price over the given window_length.

    **Default Inputs**: [USEquityPricing.close]
    """
    inputs = [USEquityPricing.close]
    window_safe = True

    def _validate(self):
        super(Returns, self)._validate()
        if self.window_length < 2:
            raise ValueError(
                "'Returns' expected a window length of at least 2, but was "
                "given {window_length}. For daily returns, use a window "
                "length of 2.".format(window_length=self.window_length)
            )

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
        out[:] = nansum(close * volume, axis=0) / len(close)


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
    def from_span(cls, inputs, window_length, span, **kwargs):
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
            **kwargs
        )

    @classmethod
    @expect_types(halflife=Number)
    def from_halflife(cls, inputs, window_length, halflife, **kwargs):
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
            **kwargs
        )

    @classmethod
    def from_center_of_mass(cls,
                            inputs,
                            window_length,
                            center_of_mass,
                            **kwargs):
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
            **kwargs
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


class LinearWeightedMovingAverage(CustomFactor, SingleInputMixin):
    """
    Weighted Average Value of an arbitrary column

    **Default Inputs**: None

    **Default Window Length**: None
    """
    # numpy's nan functions throw warnings when passed an array containing only
    # nans, but they still returns the desired value (nan), so we ignore the
    # warning.
    ctx = ignore_nanwarnings()

    def compute(self, today, assets, out, data):
        num_days = data.shape[0]

        # Initialize weights array
        weights = arange(1, num_days + 1, dtype=float).reshape(num_days, 1)

        # Compute normalizer
        normalizer = (num_days * (num_days + 1)) / 2

        # Weight the data
        weighted_data = data * weights

        # Compute weighted averages
        out[:] = nansum(weighted_data, axis=0) / normalizer


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


class BollingerBands(CustomFactor):
    """
    Bollinger Bands technical indicator.
    https://en.wikipedia.org/wiki/Bollinger_Bands

    **Default Inputs:** :data:`zipline.pipeline.data.USEquityPricing.close`

    Parameters
    ----------
    inputs : length-1 iterable[BoundColumn]
        The expression over which to compute bollinger bands.
    window_length : int > 0
        Length of the lookback window over which to compute the bollinger
        bands.
    k : float
        The number of standard deviations to add or subtract to create the
        upper and lower bands.
    """
    params = ('k',)
    inputs = (USEquityPricing.close,)
    outputs = 'lower', 'middle', 'upper'

    def compute(self, today, assets, out, close, k):
        difference = k * nanstd(close, axis=0)
        out.middle = middle = nanmean(close, axis=0)
        out.upper = middle + difference
        out.lower = middle - difference


class Aroon(CustomFactor):
    """
    Aroon technical indicator.
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/aroon-indicator  # noqa

    **Defaults Inputs:** USEquityPricing.low, USEquityPricing.high

    Parameters
    ----------
    window_length : int > 0
        Length of the lookback window over which to compute the Aroon
        indicator.
    """

    inputs = (USEquityPricing.low, USEquityPricing.high)
    outputs = ('down', 'up')

    def compute(self, today, assets, out, lows, highs):
        wl = self.window_length
        high_date_index = nanargmax(highs, axis=0)
        low_date_index = nanargmin(lows, axis=0)
        evaluate(
            '(100 * high_date_index) / (wl - 1)',
            local_dict={
                'high_date_index': high_date_index,
                'wl': wl,
            },
            out=out.up,
        )
        evaluate(
            '(100 * low_date_index) / (wl - 1)',
            local_dict={
                'low_date_index': low_date_index,
                'wl': wl,
            },
            out=out.down,
        )


class FastStochasticOscillator(CustomFactor):
    """
    Fast Stochastic Oscillator Indicator [%K, Momentum Indicator]
    https://wiki.timetotrade.eu/Stochastic

    This stochastic is considered volatile, and varies a lot when used in
    market analysis. It is recommended to use the slow stochastic oscillator
    or a moving average of the %K [%D].

    **Default Inputs:** :data: `zipline.pipeline.data.USEquityPricing.close`
                        :data: `zipline.pipeline.data.USEquityPricing.low`
                        :data: `zipline.pipeline.data.USEquityPricing.high`

    **Default Window Length:** 14

    Returns
    -------
    out: %K oscillator
    """
    inputs = (USEquityPricing.close, USEquityPricing.low, USEquityPricing.high)
    window_safe = True
    window_length = 14

    def compute(self, today, assets, out, closes, lows, highs):

        highest_highs = nanmax(highs, axis=0)
        lowest_lows = nanmin(lows, axis=0)
        today_closes = closes[-1]

        evaluate(
            '((tc - ll) / (hh - ll)) * 100',
            local_dict={
                'tc': today_closes,
                'll': lowest_lows,
                'hh': highest_highs,
            },
            global_dict={},
            out=out,
        )


class IchimokuKinkoHyo(CustomFactor):
    """Compute the various metrics for the Ichimoku Kinko Hyo (Ichimoku Cloud).
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud  # noqa

    **Default Inputs:** :data:`zipline.pipeline.data.USEquityPricing.high`
                        :data:`zipline.pipeline.data.USEquityPricing.low`
                        :data:`zipline.pipeline.data.USEquityPricing.close`
    **Default Window Length:** 52

    Parameters
    ----------
    window_length : int > 0
        The length the the window for the senkou span b.
    tenkan_sen_length : int >= 0, <= window_length
        The length of the window for the tenkan-sen.
    kijun_sen_length : int >= 0, <= window_length
        The length of the window for the kijou-sen.
    chikou_span_length : int >= 0, <= window_length
        The lag for the chikou span.
    """

    params = {
        'tenkan_sen_length': 9,
        'kijun_sen_length': 26,
        'chikou_span_length': 26,
    }
    inputs = USEquityPricing.high, USEquityPricing.close
    outputs = (
        'tenkan_sen',
        'kijun_sen',
        'senkou_span_a',
        'senkou_span_b',
        'chikou_span',
    )
    window_length = 52

    def _validate(self):
        super(IchimokuKinkoHyo, self)._validate()
        for k, v in self.params.items():
            if v > self.window_length:
                raise ValueError(
                    '%s must be <= the window_length: %s > %s' % (
                        k, v, self.window_length,
                    ),
                )

    def compute(self,
                today,
                assets,
                out,
                high,
                low,
                close,
                tenkan_sen_length,
                kijun_sen_length,
                chikou_span_length):

        out.tenkan_sen = tenkan_sen = (
            high[-tenkan_sen_length:].max(axis=0) +
            low[-tenkan_sen_length:].min(axis=0)
        ) / 2
        out.kijun_sen = kijun_sen = (
            high[-kijun_sen_length:].max(axis=0) +
            low[-kijun_sen_length:].min(axis=0)
        ) / 2
        out.senkou_span_a = (tenkan_sen + kijun_sen) / 2
        out.senkou_span_b = (high.max(axis=0) + low.min(axis=0)) / 2
        out.chikou_span = close[chikou_span_length]


class RateOfChangePercentage(CustomFactor):
    """
    Rate of change Percentage
    ROC measures the percentage change in price from one period to the next.
    The ROC calculation compares the current price with the price `n`
    periods ago.
    Formula for calculation: ((price - prevPrice) / prevPrice) * 100
    price - the current price
    prevPrice - the price n days ago, equals window length
    """
    def compute(self, today, assets, out, close):
        today_close = close[-1]
        prev_close = close[0]
        evaluate('((tc - pc) / pc) * 100',
                 local_dict={
                     'tc': today_close,
                     'pc': prev_close
                 },
                 global_dict={},
                 out=out,
                 )


class TrueRange(CustomFactor):
    """
    True Range

    A technical indicator originally developed by J. Welles Wilder, Jr.
    Indicates the true degree of daily price change in an underlying.

    **Default Inputs:** :data:`zipline.pipeline.data.USEquityPricing.high`
                        :data:`zipline.pipeline.data.USEquityPricing.low`
                        :data:`zipline.pipeline.data.USEquityPricing.close`
    **Default Window Length:** 2
    """
    inputs = (
        USEquityPricing.high,
        USEquityPricing.low,
        USEquityPricing.close,
    )
    window_length = 2

    def compute(self, today, assets, out, highs, lows, closes):
        high_to_low = highs[1:] - lows[1:]
        high_to_prev_close = abs(highs[1:] - closes[:-1])
        low_to_prev_close = abs(lows[1:] - closes[:-1])
        out[:] = nanmax(
            dstack((
                high_to_low,
                high_to_prev_close,
                low_to_prev_close,
            )),
            2
        )



class MovingAverageConvergenceDivergence(CustomFactor):
    """
    Moving Average Convergence/Divergence (MACD)
    https://en.wikipedia.org/wiki/MACD

    A technical indicator originally developed by Gerald Appel in the late
    1970's. MACD shows the relationship between two moving averages and
    reveals changes in the strength, direction, momentum, and duration of a
    trend in a stock's price.

    **Default Inputs:** :data:`zipline.pipeline.data.USEquityPricing.close`
    **Default Window Length:** None

    Parameters
    ----------
    fast_period : int >= 0, <= window_length
        The window length for the "fast" EMA.
    slow_period : int >= 0, <= window_length
        The window length for the "slow" EMA.
    signal_period' : int >= 0, <= slow_period
        The window length for the signal line.

    Returns
    -------
    MACD: The difference between "fast" EMA and "slow" EMA.
    signal: The signal_period length period EMA of the MACD line.
    hist: Difference between MACD and signal. (Divergence series)
    """
    inputs = [USEquityPricing.close]

    params = {'fast_period': 12,
              'slow_period': 26,
              'signal_period': 9}

    outputs = ('MACD', 'signal', 'hist')

    def calculate_macd(self, col, fast, slow, signal):
        try:
            macd, sig, hist = MACD(col,
                                   fastperiod=fast,
                                   slowperiod=slow,
                                   signalperiod=signal)
            return macd[-1], sig[-1], hist[-1]
        except:
            return nan, nan, nan

    def compute(self, today, assets, out, close, fast_period, slow_period,
                signal_period):
        n = len(close)
        macd, sig, hist = zip(*map(self.calculate_macd,
                                   close.T,
                                   [fast_period]*n,
                                   [slow_period]*n,
                                   [signal_period]*n))
        out.MACD[:] = macd
        out.signal[:] = sig
        out.hist[:] = hist


class AnnualVolatility(CustomFactor):
    """
    Volatility
    https://en.wikipedia.org/wiki/Volatility_(finance)

    The degree of variation of a series over time as measured by the standard
    deviation of the data over the course of a year.

    **Default Inputs:** :data:`zipline.pipeline.data.USEquityPricing.close`
    """
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, closes):
        out[:] = nanstd(closes, ddof=1, axis=0) * (252 ** 0.5)
