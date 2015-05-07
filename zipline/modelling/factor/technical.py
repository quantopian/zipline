"""
Technical Analysis Factors
--------------------------
"""
from bottleneck import (
    nanargmax,
    nanmax,
    nanmean,
    nansum,
)
from numpy import (
    clip,
    diff,
    fmax,
    inf,
    isnan,
    NINF,
)
from numexpr import evaluate

from zipline.data.equities import USEquityPricing
from zipline.modelling.term import SingleInputMixin
from zipline.utils.control_flow import ignore_nanwarnings
from .factor import CustomFactor


class RSI(CustomFactor, SingleInputMixin):
    """
    Factor computing rolling relative-strength index on a DataSet.

    Default Input: USEquityPricing.close
    Default Window Length: 14
    """
    window_length = 14
    inputs = (USEquityPricing.close,)

    def compute(self, today, assets, out, closes):
        diffs = diff(closes)
        ups = nanmean(clip(diffs, 0, inf), axis=0)
        downs = nanmean(clip(diffs, -inf, 0), axis=0)
        return evaluate(
            "100 - (100 / (1 + (ups / downs)))",
            locals_dict={'ups': ups, 'downs': downs},
            globals_dict={},
            out=out,
        )


class SimpleMovingAverage(CustomFactor, SingleInputMixin):
    """
    Factor computing moving averages on a DataSet.
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
    """
    def compute(self, today, assets, out, base, weight):
        out[:] = nansum(base * weight, axis=0) / nansum(weight, axis=0)


class VWAP(WeightedAverageValue):
    """
    Volume-weighted average price
    """
    inputs = (USEquityPricing.close, USEquityPricing.volume)


class MaxDrawdown(CustomFactor, SingleInputMixin):
    """
    Max Drawdown over a window
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
