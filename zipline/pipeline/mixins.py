"""
Mixins classes for use with Filters and Factors.
"""
from numpy import full_like

from zipline.utils.control_flow import nullctx
from zipline.errors import WindowLengthNotPositive, UnsupportedDataType

from .term import NotSpecified


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
                typename=type(self.__name__),
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
                window_length=NotSpecified,
                dtype=NotSpecified,
                missing_value=NotSpecified,
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
            window_length=window_length,
            dtype=dtype,
            missing_value=missing_value,
            **kwargs
        )

    def compute(self, today, assets, out, *arrays):
        """
        Override this method with a function that writes a value into `out`.
        """
        raise NotImplementedError()

    def _compute(self, windows, dates, assets, mask):
        """
        Call the user's `compute` function on each window with a pre-built
        output array.
        """
        # TODO: Make mask available to user's `compute`.
        compute = self.compute
        missing_value = self.missing_value
        params = self.params
        out = full_like(mask, missing_value, dtype=self.dtype)
        with self.ctx:
            # TODO: Consider pre-filtering columns that are all-nan at each
            # time-step?
            for idx, date in enumerate(dates):
                compute(
                    date,
                    assets,
                    out[idx],
                    *(next(w) for w in windows),
                    **params
                )
        out[~mask] = missing_value
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
