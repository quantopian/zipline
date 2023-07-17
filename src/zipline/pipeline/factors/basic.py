"""Simple common factors.
"""
from numbers import Number
from numpy import (
    arange,
    average,
    clip,
    copyto,
    exp,
    fmax,
    full,
    isnan,
    log,
    NINF,
    sqrt,
    sum as np_sum,
    unique,
    errstate as np_errstate,
)

from zipline.pipeline.data import EquityPricing
from zipline.utils.input_validation import expect_types
from zipline.utils.math_utils import (
    nanargmax,
    nanmax,
    nanmean,
    nanstd,
    nansum,
)
from zipline.utils.numpy_utils import (
    float64_dtype,
    ignore_nanwarnings,
)

from .factor import CustomFactor
from ..mixins import SingleInputMixin


class Returns(CustomFactor):
    """
    Calculates the percent change in close price over the given window_length.

    **Default Inputs**: [EquityPricing.close]
    """

    inputs = [EquityPricing.close]
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


class PercentChange(SingleInputMixin, CustomFactor):
    """
    Calculates the percent change over the given window_length.

    **Default Inputs:** None

    **Default Window Length:** None

    Notes
    -----
    Percent change is calculated as ``(new - old) / abs(old)``.
    """

    window_safe = True

    def _validate(self):
        super(PercentChange, self)._validate()
        if self.window_length < 2:
            raise ValueError(
                "'PercentChange' expected a window length"
                "of at least 2, but was given {window_length}. "
                "For daily percent change, use a window "
                "length of 2.".format(window_length=self.window_length)
            )

    def compute(self, today, assets, out, values):
        with np_errstate(divide="ignore", invalid="ignore"):
            out[:] = (values[-1] - values[0]) / abs(values[0])


class DailyReturns(Returns):
    """
    Calculates daily percent change in close price.

    **Default Inputs**: [EquityPricing.close]
    """

    inputs = [EquityPricing.close]
    window_safe = True
    window_length = 2


class SimpleMovingAverage(SingleInputMixin, CustomFactor):
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

    **Default Inputs:** [EquityPricing.close, EquityPricing.volume]

    **Default Window Length:** None
    """

    inputs = (EquityPricing.close, EquityPricing.volume)


class MaxDrawdown(SingleInputMixin, CustomFactor):
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
            peak = nanmax(data[: end + 1, i])
            out[i] = (peak - data[end, i]) / data[end, i]


class AverageDollarVolume(CustomFactor):
    """
    Average Daily Dollar Volume

    **Default Inputs:** [EquityPricing.close, EquityPricing.volume]

    **Default Window Length:** None
    """

    inputs = [EquityPricing.close, EquityPricing.volume]

    def compute(self, today, assets, out, close, volume):
        out[:] = nansum(close * volume, axis=0) / len(close)


def exponential_weights(length, decay_rate):
    """
    Build a weight vector for an exponentially-weighted statistic.

    The resulting ndarray is of the form::

        [decay_rate ** length, ..., decay_rate ** 2, decay_rate]

    Parameters
    ----------
    length : int
        The length of the desired weight vector.
    decay_rate : float
        The rate at which entries in the weight vector increase or decrease.

    Returns
    -------
    weights : ndarray[float64]
    """
    return full(length, decay_rate, float64_dtype) ** arange(length + 1, 1, -1)


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

    params = ("decay_rate",)

    @classmethod
    @expect_types(span=Number)
    def from_span(cls, inputs, window_length, span, **kwargs):
        """
        Convenience constructor for passing `decay_rate` in terms of `span`.

        Forwards `decay_rate` as `1 - (2.0 / (1 + span))`.  This provides the
        behavior equivalent to passing `span` to pandas.ewma.

        Examples
        --------
        .. code-block:: python

            # Equivalent to:
            # my_ewma = EWMA(
            #    inputs=[EquityPricing.close],
            #    window_length=30,
            #    decay_rate=(1 - (2.0 / (1 + 15.0))),
            # )
            my_ewma = EWMA.from_span(
                inputs=[EquityPricing.close],
                window_length=30,
                span=15,
            )

        Notes
        -----
        This classmethod is provided by both
        :class:`ExponentialWeightedMovingAverage` and
        :class:`ExponentialWeightedMovingStdDev`.
        """
        if span <= 1:
            raise ValueError("`span` must be a positive number. %s was passed." % span)

        decay_rate = 1.0 - (2.0 / (1.0 + span))
        assert 0.0 < decay_rate <= 1.0

        return cls(
            inputs=inputs, window_length=window_length, decay_rate=decay_rate, **kwargs
        )

    @classmethod
    @expect_types(halflife=Number)
    def from_halflife(cls, inputs, window_length, halflife, **kwargs):
        """
        Convenience constructor for passing ``decay_rate`` in terms of half
        life.

        Forwards ``decay_rate`` as ``exp(log(.5) / halflife)``.  This provides
        the behavior equivalent to passing `halflife` to pandas.ewma.

        Examples
        --------
        .. code-block:: python

            # Equivalent to:
            # my_ewma = EWMA(
            #    inputs=[EquityPricing.close],
            #    window_length=30,
            #    decay_rate=np.exp(np.log(0.5) / 15),
            # )
            my_ewma = EWMA.from_halflife(
                inputs=[EquityPricing.close],
                window_length=30,
                halflife=15,
            )

        Notes
        -----
        This classmethod is provided by both
        :class:`ExponentialWeightedMovingAverage` and
        :class:`ExponentialWeightedMovingStdDev`.
        """
        if halflife <= 0:
            raise ValueError(
                "`span` must be a positive number. %s was passed." % halflife
            )
        decay_rate = exp(log(0.5) / halflife)
        assert 0.0 < decay_rate <= 1.0

        return cls(
            inputs=inputs, window_length=window_length, decay_rate=decay_rate, **kwargs
        )

    @classmethod
    def from_center_of_mass(cls, inputs, window_length, center_of_mass, **kwargs):
        """
        Convenience constructor for passing `decay_rate` in terms of center of
        mass.

        Forwards `decay_rate` as `1 - (1 / 1 + center_of_mass)`.  This provides
        behavior equivalent to passing `center_of_mass` to pandas.ewma.

        Examples
        --------
        .. code-block:: python

            # Equivalent to:
            # my_ewma = EWMA(
            #    inputs=[EquityPricing.close],
            #    window_length=30,
            #    decay_rate=(1 - (1 / 15.0)),
            # )
            my_ewma = EWMA.from_center_of_mass(
                inputs=[EquityPricing.close],
                window_length=30,
                center_of_mass=15,
            )

        Notes
        -----
        This classmethod is provided by both
        :class:`ExponentialWeightedMovingAverage` and
        :class:`ExponentialWeightedMovingStdDev`.
        """
        return cls(
            inputs=inputs,
            window_length=window_length,
            decay_rate=(1.0 - (1.0 / (1.0 + center_of_mass))),
            **kwargs,
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
    :meth:`pandas.DataFrame.ewm`
    """

    def compute(self, today, assets, out, data, decay_rate):
        out[:] = average(
            data,
            axis=0,
            weights=exponential_weights(len(data), decay_rate),
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
    :func:`pandas.DataFrame.ewm`
    """

    def compute(self, today, assets, out, data, decay_rate):
        weights = exponential_weights(len(data), decay_rate)

        mean = average(data, axis=0, weights=weights)
        variance = average((data - mean) ** 2, axis=0, weights=weights)

        squared_weight_sum = np_sum(weights) ** 2
        bias_correction = squared_weight_sum / (
            squared_weight_sum - np_sum(weights**2)
        )
        out[:] = sqrt(variance * bias_correction)


class LinearWeightedMovingAverage(SingleInputMixin, CustomFactor):
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
        ndays = data.shape[0]

        # Initialize weights array
        weights = arange(1, ndays + 1, dtype=float64_dtype).reshape(ndays, 1)

        # Compute normalizer
        normalizer = (ndays * (ndays + 1)) / 2

        # Weight the data
        weighted_data = data * weights

        # Compute weighted averages
        out[:] = nansum(weighted_data, axis=0) / normalizer


class AnnualizedVolatility(CustomFactor):
    """
    Volatility. The degree of variation of a series over time as measured by
    the standard deviation of daily returns.
    https://en.wikipedia.org/wiki/Volatility_(finance)

    **Default Inputs:** [Returns(window_length=2)]

    Parameters
    ----------
    annualization_factor : float, optional
        The number of time units per year. Defaults is 252, the number of NYSE
        trading days in a normal year.
    """

    inputs = [Returns(window_length=2)]
    params = {"annualization_factor": 252.0}
    window_length = 252

    def compute(self, today, assets, out, returns, annualization_factor):
        out[:] = nanstd(returns, axis=0) * (annualization_factor**0.5)


class PeerCount(SingleInputMixin, CustomFactor):
    """
    Peer Count of distinct categories in a given classifier.  This factor
    is returned by the classifier instance method peer_count()

    **Default Inputs:** None

    **Default Window Length:** 1
    """

    window_length = 1

    def _validate(self):
        super(PeerCount, self)._validate()
        if self.window_length != 1:
            raise ValueError(
                "'PeerCount' expected a window length of 1, but was given"
                "{window_length}.".format(window_length=self.window_length)
            )

    def compute(self, today, assets, out, classifier_values):
        # Convert classifier array to group label int array
        group_labels, null_label = self.inputs[0]._to_integral(classifier_values[0])
        _, inverse, counts = unique(  # Get counts, idx of unique groups
            group_labels,
            return_counts=True,
            return_inverse=True,
        )
        copyto(out, counts[inverse], where=(group_labels != null_label))


# Convenience aliases
EWMA = ExponentialWeightedMovingAverage
EWMSTD = ExponentialWeightedMovingStdDev


class Clip(CustomFactor):
    """
    Clip (limit) the values in a factor.

    Given an interval, values outside the interval are clipped to the interval
    edges. For example, if an interval of ``[0, 1]`` is specified, values
    smaller than 0 become 0, and values larger than 1 become 1.

    **Default Window Length:** 1

    Parameters
    ----------
    min_bound : float
        The minimum value to use.
    max_bound : float
        The maximum value to use.

    Notes
    -----
    To only clip values on one side, ``-np.inf` and ``np.inf`` may be passed.
    For example, to only clip the maximum value but not clip a minimum value:

    .. code-block:: python

       Clip(inputs=[factor], min_bound=-np.inf, max_bound=user_provided_max)

    See Also
    --------
    numpy.clip
    """

    window_length = 1
    params = ("min_bound", "max_bound")

    def compute(self, today, assets, out, values, min_bound, max_bound):
        clip(values[-1], min_bound, max_bound, out=out)
