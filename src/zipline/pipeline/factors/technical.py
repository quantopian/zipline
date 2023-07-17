"""
Technical Analysis Factors
--------------------------
"""
from numpy import (
    abs,
    average,
    clip,
    diff,
    dstack,
    inf,
)
from numexpr import evaluate

from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.mixins import SingleInputMixin
from zipline.utils.input_validation import expect_bounded
from zipline.utils.math_utils import (
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanstd,
    nanmin,
)
from zipline.utils.numpy_utils import rolling_window

from .basic import exponential_weights
from .basic import (  # noqa reexport
    # These are re-exported here for backwards compatibility with the old
    # definition site.
    LinearWeightedMovingAverage,
    MaxDrawdown,
    SimpleMovingAverage,
    VWAP,
    WeightedAverageValue,
)


class RSI(SingleInputMixin, CustomFactor):
    """
    Relative Strength Index

    **Default Inputs**: :data:`zipline.pipeline.data.EquityPricing.close`

    **Default Window Length**: 15
    """

    window_length = 15
    inputs = (EquityPricing.close,)
    window_safe = True

    def compute(self, today, assets, out, closes):
        diffs = diff(closes, axis=0)
        ups = nanmean(clip(diffs, 0, inf), axis=0)
        downs = abs(nanmean(clip(diffs, -inf, 0), axis=0))
        return evaluate(
            "100 - (100 / (1 + (ups / downs)))",
            local_dict={"ups": ups, "downs": downs},
            global_dict={},
            out=out,
        )


class BollingerBands(CustomFactor):
    """
    Bollinger Bands technical indicator.
    https://en.wikipedia.org/wiki/Bollinger_Bands

    **Default Inputs:** :data:`zipline.pipeline.data.EquityPricing.close`

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

    params = ("k",)
    inputs = (EquityPricing.close,)
    outputs = "lower", "middle", "upper"

    def compute(self, today, assets, out, close, k):
        difference = k * nanstd(close, axis=0)
        out.middle = middle = nanmean(close, axis=0)
        out.upper = middle + difference
        out.lower = middle - difference


class Aroon(CustomFactor):
    """
    Aroon technical indicator.
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/aroon-indicator

    **Defaults Inputs:** :data:`zipline.pipeline.data.EquityPricing.low`, \
                         :data:`zipline.pipeline.data.EquityPricing.high`

    Parameters
    ----------
    window_length : int > 0
        Length of the lookback window over which to compute the Aroon
        indicator.
    """  # noqa

    inputs = (EquityPricing.low, EquityPricing.high)
    outputs = ("down", "up")

    def compute(self, today, assets, out, lows, highs):
        wl = self.window_length
        high_date_index = nanargmax(highs, axis=0)
        low_date_index = nanargmin(lows, axis=0)
        evaluate(
            "(100 * high_date_index) / (wl - 1)",
            local_dict={
                "high_date_index": high_date_index,
                "wl": wl,
            },
            out=out.up,
        )
        evaluate(
            "(100 * low_date_index) / (wl - 1)",
            local_dict={
                "low_date_index": low_date_index,
                "wl": wl,
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

    **Default Inputs:** :data:`zipline.pipeline.data.EquityPricing.close`, \
                        :data:`zipline.pipeline.data.EquityPricing.low`, \
                        :data:`zipline.pipeline.data.EquityPricing.high`

    **Default Window Length:** 14

    Returns
    -------
    out: %K oscillator
    """

    inputs = (EquityPricing.close, EquityPricing.low, EquityPricing.high)
    window_safe = True
    window_length = 14

    def compute(self, today, assets, out, closes, lows, highs):

        highest_highs = nanmax(highs, axis=0)
        lowest_lows = nanmin(lows, axis=0)
        today_closes = closes[-1]

        evaluate(
            "((tc - ll) / (hh - ll)) * 100",
            local_dict={
                "tc": today_closes,
                "ll": lowest_lows,
                "hh": highest_highs,
            },
            global_dict={},
            out=out,
        )


class IchimokuKinkoHyo(CustomFactor):
    """Compute the various metrics for the Ichimoku Kinko Hyo (Ichimoku Cloud).
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    **Default Inputs:** :data:`zipline.pipeline.data.EquityPricing.high`, \
                        :data:`zipline.pipeline.data.EquityPricing.low`, \
                        :data:`zipline.pipeline.data.EquityPricing.close`

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
    """  # noqa

    params = {
        "tenkan_sen_length": 9,
        "kijun_sen_length": 26,
        "chikou_span_length": 26,
    }
    inputs = (EquityPricing.high, EquityPricing.low, EquityPricing.close)
    outputs = (
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_span",
    )
    window_length = 52

    def _validate(self):
        super(IchimokuKinkoHyo, self)._validate()
        for k, v in self.params.items():
            if v > self.window_length:
                raise ValueError(
                    "%s must be <= the window_length: %s > %s"
                    % (
                        k,
                        v,
                        self.window_length,
                    ),
                )

    def compute(
        self,
        today,
        assets,
        out,
        high,
        low,
        close,
        tenkan_sen_length,
        kijun_sen_length,
        chikou_span_length,
    ):

        out.tenkan_sen = tenkan_sen = (
            high[-tenkan_sen_length:].max(axis=0) + low[-tenkan_sen_length:].min(axis=0)
        ) / 2
        out.kijun_sen = kijun_sen = (
            high[-kijun_sen_length:].max(axis=0) + low[-kijun_sen_length:].min(axis=0)
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
        evaluate(
            "((tc - pc) / pc) * 100",
            local_dict={"tc": today_close, "pc": prev_close},
            global_dict={},
            out=out,
        )


class TrueRange(CustomFactor):
    """
    True Range

    A technical indicator originally developed by J. Welles Wilder, Jr.
    Indicates the true degree of daily price change in an underlying.

    **Default Inputs:** :data:`zipline.pipeline.data.EquityPricing.high`, \
                        :data:`zipline.pipeline.data.EquityPricing.low`, \
                        :data:`zipline.pipeline.data.EquityPricing.close`

    **Default Window Length:** 2
    """

    inputs = (
        EquityPricing.high,
        EquityPricing.low,
        EquityPricing.close,
    )
    window_length = 2

    def compute(self, today, assets, out, highs, lows, closes):
        high_to_low = highs[1:] - lows[1:]
        high_to_prev_close = abs(highs[1:] - closes[:-1])
        low_to_prev_close = abs(lows[1:] - closes[:-1])
        out[:] = nanmax(
            dstack(
                (
                    high_to_low,
                    high_to_prev_close,
                    low_to_prev_close,
                )
            ),
            2,
        )


class MovingAverageConvergenceDivergenceSignal(CustomFactor):
    """
    Moving Average Convergence/Divergence (MACD) Signal line
    https://en.wikipedia.org/wiki/MACD

    A technical indicator originally developed by Gerald Appel in the late
    1970's. MACD shows the relationship between two moving averages and
    reveals changes in the strength, direction, momentum, and duration of a
    trend in a stock's price.

    **Default Inputs:** :data:`zipline.pipeline.data.EquityPricing.close`

    Parameters
    ----------
    fast_period : int > 0, optional
        The window length for the "fast" EWMA. Default is 12.
    slow_period : int > 0, > fast_period, optional
        The window length for the "slow" EWMA. Default is 26.
    signal_period : int > 0, < fast_period, optional
        The window length for the signal line. Default is 9.

    Notes
    -----
    Unlike most pipeline expressions, this factor does not accept a
    ``window_length`` parameter. ``window_length`` is inferred from
    ``slow_period`` and ``signal_period``.
    """

    inputs = (EquityPricing.close,)
    # We don't use the default form of `params` here because we want to
    # dynamically calculate `window_length` from the period lengths in our
    # __new__.
    params = ("fast_period", "slow_period", "signal_period")

    @expect_bounded(
        __funcname="MACDSignal",
        fast_period=(1, None),  # These must all be >= 1.
        slow_period=(1, None),
        signal_period=(1, None),
    )
    def __new__(cls, fast_period=12, slow_period=26, signal_period=9, *args, **kwargs):

        if slow_period <= fast_period:
            raise ValueError(
                "'slow_period' must be greater than 'fast_period', but got\n"
                "slow_period={slow}, fast_period={fast}".format(
                    slow=slow_period,
                    fast=fast_period,
                )
            )

        return super(MovingAverageConvergenceDivergenceSignal, cls).__new__(
            cls,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            window_length=slow_period + signal_period - 1,
            *args,
            **kwargs,
        )

    def _ewma(self, data, length):
        decay_rate = 1.0 - (2.0 / (1.0 + length))
        return average(data, axis=1, weights=exponential_weights(length, decay_rate))

    def compute(
        self, today, assets, out, close, fast_period, slow_period, signal_period
    ):
        slow_EWMA = self._ewma(rolling_window(close, slow_period), slow_period)
        fast_EWMA = self._ewma(
            rolling_window(close, fast_period)[-signal_period:], fast_period
        )
        macd = fast_EWMA - slow_EWMA
        out[:] = self._ewma(macd.T, signal_period)


# Convenience aliases.
MACDSignal = MovingAverageConvergenceDivergenceSignal
