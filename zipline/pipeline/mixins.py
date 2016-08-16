"""
Mixins classes for use with Filters and Factors.
"""
from operator import attrgetter

from numpy import (
    array,
    full,
    recarray,
    vstack,
)
from pandas import NaT as pd_NaT

from zipline.errors import (
    WindowLengthNotPositive,
    UnsupportedDataType,
    NoFurtherDataError,
)
from zipline.utils.control_flow import nullctx
from zipline.utils.input_validation import expect_element, expect_types
from zipline.utils.numpy_utils import changed_locations
from zipline.utils.pandas_utils import nearest_unequal_elements

from .sentinels import NotSpecified
from .term import Term


class PositiveWindowLengthMixin(object):
    """
    Validation mixin enforcing that a Term gets a positive WindowLength
    """
    def _validate(self):
        super(PositiveWindowLengthMixin, self)._validate()
        if not self.windowed:
            raise WindowLengthNotPositive(window_length=self.window_length)


class SingleInputMixin(object):
    """
    Validation mixin enforcing that a Term gets a length-1 inputs list.
    """
    def _validate(self):
        super(SingleInputMixin, self)._validate()
        num_inputs = len(self.inputs)
        if num_inputs != 1:
            raise ValueError(
                "{typename} expects only one input, "
                "but received {num_inputs} instead.".format(
                    typename=type(self).__name__,
                    num_inputs=num_inputs
                )
            )


class StandardOutputs(object):
    """
    Validation mixin enforcing that a Term cannot produce non-standard outputs.
    """
    def _validate(self):
        super(StandardOutputs, self)._validate()
        if self.outputs is not NotSpecified:
            raise ValueError(
                "{typename} does not support custom outputs,"
                " but received custom outputs={outputs}.".format(
                    typename=type(self).__name__,
                    outputs=self.outputs,
                )
            )


class RestrictedDTypeMixin(object):
    """
    Validation mixin enforcing that a term has a specific dtype.
    """
    ALLOWED_DTYPES = NotSpecified

    def _validate(self):
        super(RestrictedDTypeMixin, self)._validate()
        assert self.ALLOWED_DTYPES is not NotSpecified, (
            "ALLOWED_DTYPES not supplied on subclass "
            "of RestrictedDTypeMixin: %s." % type(self).__name__
        )

        if self.dtype not in self.ALLOWED_DTYPES:
            raise UnsupportedDataType(
                typename=type(self).__name__,
                dtype=self.dtype,
            )


class CustomTermMixin(object):
    """
    Mixin for user-defined rolling-window Terms.

    Implements `_compute` in terms of a user-defined `compute` function, which
    is mapped over the input windows.

    Used by CustomFactor, CustomFilter, CustomClassifier, etc.
    """
    ctx = nullctx()

    def __new__(cls,
                inputs=NotSpecified,
                outputs=NotSpecified,
                window_length=NotSpecified,
                mask=NotSpecified,
                dtype=NotSpecified,
                missing_value=NotSpecified,
                ndim=NotSpecified,
                **kwargs):

        unexpected_keys = set(kwargs) - set(cls.params)
        if unexpected_keys:
            raise TypeError(
                "{termname} received unexpected keyword "
                "arguments {unexpected}".format(
                    termname=cls.__name__,
                    unexpected={k: kwargs[k] for k in unexpected_keys},
                )
            )

        return super(CustomTermMixin, cls).__new__(
            cls,
            inputs=inputs,
            outputs=outputs,
            window_length=window_length,
            mask=mask,
            dtype=dtype,
            missing_value=missing_value,
            ndim=ndim,
            **kwargs
        )

    def compute(self, today, assets, out, *arrays):
        """
        Override this method with a function that writes a value into `out`.
        """
        raise NotImplementedError()

    def _allocate_output(self, windows, shape):
        """
        Allocate an output array whose rows should be passed to `self.compute`.

        The resulting array must have a shape of ``shape``.

        If we have standard outputs (i.e. self.outputs is NotSpecified), the
        default is an empty ndarray whose dtype is ``self.dtype``.

        If we have an outputs tuple, the default is an empty recarray with
        ``self.outputs`` as field names. Each field will have dtype
        ``self.dtype``.

        This can be overridden to control the kind of array constructed
        (e.g. to produce a LabelArray instead of an ndarray).
        """
        missing_value = self.missing_value
        outputs = self.outputs
        if outputs is not NotSpecified:
            out = recarray(
                shape,
                formats=[self.dtype.str] * len(outputs),
                names=outputs,
            )
            out[:] = missing_value
        else:
            out = full(shape, missing_value, dtype=self.dtype)
        return out

    def _format_inputs(self, windows, column_mask):
        inputs = []
        for input_ in windows:
            window = next(input_)
            if window.shape[1] == 1:
                # Do not mask single-column inputs.
                inputs.append(window)
            else:
                inputs.append(window[:, column_mask])
        return inputs

    def _compute(self, windows, dates, assets, mask):
        """
        Call the user's `compute` function on each window with a pre-built
        output array.
        """
        format_inputs = self._format_inputs
        compute = self.compute
        params = self.params
        ndim = self.ndim

        shape = (len(mask), 1) if ndim == 1 else mask.shape
        out = self._allocate_output(windows, shape)

        with self.ctx:
            for idx, date in enumerate(dates):
                # Never apply a mask to 1D outputs.
                out_mask = array([True]) if ndim == 1 else mask[idx]

                # Mask our inputs as usual.
                inputs_mask = mask[idx]

                masked_assets = assets[inputs_mask]
                out_row = out[idx][out_mask]
                inputs = format_inputs(windows, inputs_mask)

                compute(date, masked_assets, out_row, *inputs, **params)
                out[idx][out_mask] = out_row
        return out

    def short_repr(self):
        return type(self).__name__ + '(%d)' % self.window_length


class LatestMixin(SingleInputMixin):
    """
    Mixin for behavior shared by Custom{Factor,Filter,Classifier}.
    """
    window_length = 1

    def compute(self, today, assets, out, data):
        out[:] = data[-1]

    def _validate(self):
        super(LatestMixin, self)._validate()
        if self.inputs[0].dtype != self.dtype:
            raise TypeError(
                "{name} expected an input of dtype {expected}, "
                "but got {actual} instead.".format(
                    name=type(self).__name__,
                    expected=self.dtype,
                    actual=self.inputs[0].dtype,
                )
            )


_dt_to_period = {
    'Y': attrgetter('year'),
    'Q': attrgetter('quarter'),
    'M': attrgetter('month'),
    'W': attrgetter('week'),
}


def select_sampling_indices(dates, frequency):
    """
    Choose entries from ``dates`` to use for downsampling at ``frequency``.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Dates from which to select sample choices.
    frequency : {'Y', 'Q', 'M', 'W'}
        Frequency at which samples are to be taken.

    Returns
    -------
    indices : np.array[int64]
        An array condtaining indices of dates on which samples should be taken.

        The resulting index will always include 0 as a sample index, and it
        will include the first date of each subsequent year/quarter/month/week,
        as determined by ``frequency``.

    Notes
    -----
    This function assumes that ``dates`` does not have large gaps.

    In particular, it assumes that the maximum distance between any two entries
    in ``dates`` is never greater than a year, which we rely on because we use
    ``np.diff(dates.{quarter,month,week})`` to find dates where the sampling
    period has changed.
    """
    return changed_locations(
        _dt_to_period[frequency](dates),
        include_first=True
    )


class DownsampledMixin(StandardOutputs):
    """
    Mixin for behavior shared by Downsampled{Factor,Filter,Classifier}

    A downsampled term is a wrapper around the "real" term that performs actual
    computation. The downsampler is responsible for calling the real term's
    `compute` method at selected intervals and forward-filling the computed
    values.

    Downsampling is not currently supported for terms with multiple outputs.
    """
    # There's no reason to take a window of a downsampled term.  The whole
    # point is that you're re-using the same result multiple times.
    window_safe = False

    @expect_types(term=Term)
    @expect_element(frequency=frozenset(_dt_to_period))
    def __new__(cls, term, frequency):
        return super(DownsampledMixin, cls).__new__(
            cls,
            inputs=term.inputs,
            outputs=term.outputs,
            window_length=term.window_length,
            mask=term.mask,
            frequency=frequency,
            wrapped_term=term,
            dtype=term.dtype,
            missing_value=term.missing_value,
            ndim=term.ndim,
        )

    def _init(self, frequency, wrapped_term, *args, **kwargs):
        self._frequency = frequency
        self._wrapped_term = wrapped_term
        return super(DownsampledMixin, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, frequency, wrapped_term, *args, **kwargs):
        return (
            super(DownsampledMixin, cls)._static_identity(*args, **kwargs),
            frequency,
            wrapped_term,
        )

    def compute_extra_rows(self,
                           all_dates,
                           start_date,
                           end_date,
                           min_extra_rows):
        """
        Ensure that min_extra_rows pushes us back to a computation date.

        Parameters
        ----------
        all_dates : pd.DatetimeIndex
            The trading sessions against which ``self`` will be computed.
        start_date : pd.Timestamp
            The first date for which final output is requested.
        end_date : pd.Timestamp
            The last date for which final output is requested.
        min_extra_rows : int
            The minimum number of extra rows required of ``self``, as
            determined by other terms that depend on ``self``.

        Returns
        -------
        extra_rows : int
            The number of extra rows to compute.  This will be the minimum
            number of rows required to make our computed start_date fall on a
            recomputation date.
        """
        try:
            current_start_pos = all_dates.get_loc(start_date) - min_extra_rows
            if current_start_pos < 0:
                raise NoFurtherDataError(
                    initial_message="Insufficient data to compute Pipeline:",
                    first_date=all_dates[0],
                    lookback_start=start_date,
                    lookback_length=min_extra_rows,
                )
        except KeyError:
            before, after = nearest_unequal_elements(all_dates, start_date)
            raise ValueError(
                "Pipeline start_date {start_date} is not in calendar.\n"
                "Latest date before start_date is {before}.\n"
                "Earliest date after start_date is {after}.".format(
                    start_date=start_date,
                    before=before,
                    after=after,
                )
            )

        # Our possible target dates are all the dates on or before the current
        # starting position.
        # TODO: Consider bounding this below by self.window_length
        candidates = all_dates[:current_start_pos + 1]

        # Choose the latest date in the candidates that is the start of a new
        # period at our frequency.
        choices = select_sampling_indices(candidates, self._frequency)

        # If we have choices, the last choice is the first date if the
        # period containing current_start_date.  Choose it.
        new_start_date = candidates[choices[-1]]

        # Add the difference between the new and old start dates to get the
        # number of rows for the new start_date.
        new_start_pos = all_dates.get_loc(new_start_date)
        assert new_start_pos <= current_start_pos, \
            "Computed negative extra rows!"

        return min_extra_rows + (current_start_pos - new_start_pos)

    def _compute(self, windows, dates, assets, mask):
        """
        Compute by delegating to self._wrapped_term._compute on sample dates.

        On non-sample dates, forward-fill from previously-computed samples.
        """
        to_sample = dates[select_sampling_indices(dates, self._frequency)]
        assert to_sample[0] == dates[0], \
            "Misaligned sampling dates in %s." % type(self).__name__

        real_compute = self._wrapped_term._compute

        results = []
        samples = iter(to_sample)
        next_sample = next(samples)
        for i, compute_date in enumerate(dates):
            if next_sample == compute_date:
                results.append(
                    real_compute(
                        windows,
                        dates[i:i + 1],
                        assets,
                        mask[i:i + 1],
                    )
                )
                try:
                    next_sample = next(samples)
                except StopIteration:
                    # No more samples to take. Set next_sample to Nat, which
                    # compares False with any other datetime.
                    next_sample = pd_NaT
            else:
                # Copy results from previous sample period.
                results.append(results[-1])

                # Force adjusted arrays forward one tick.
                for w in windows:
                    next(w)

        # We should have exhausted our sample dates.
        try:
            next_sample = next(samples)
        except StopIteration:
            pass
        else:
            raise AssertionError("Unconsumed sample date: %s" % next_sample)

        # Concatenate stored results.
        return vstack(results)
