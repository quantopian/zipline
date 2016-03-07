"""
classifier.py
"""
from numpy import zeros, where

from zipline.errors import UnsupportedDataType
from zipline.pipeline.term import ComputableTerm
from zipline.utils.numpy_utils import int64_dtype

from ..mixins import CustomTermMixin, PositiveWindowLengthMixin


class Classifier(ComputableTerm):

    def _validate(self):
        # Run superclass validation first so that we handle `dtype not passed`
        # before this.
        retval = super(Classifier, self)._validate()
        # TODO: Support strings here.
        if self.dtype != int64_dtype:
            raise UnsupportedDataType(
                typename=type(self).__name__,
                dtype=self.dtype
            )
        return retval


class Everything(Classifier):
    """
    A trivial classifier that classifies everything the same.
    """
    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1

    def _compute(self, arrays, dates, assets, mask):
        return where(
            mask,
            zeros(shape=mask.shape, dtype=int64_dtype),
            self.missing_value,
        )


class CustomClassifier(PositiveWindowLengthMixin, CustomTermMixin, Classifier):
    """
    Base class for user-defined Classifiers.

    See Also
    --------
    zipline.pipeline.CustomFactor
    zipline.pipeline.CustomFilter
    """
    pass
