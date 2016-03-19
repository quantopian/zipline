"""
classifier.py
"""
from numpy import zeros, where

from zipline.pipeline.term import ComputableTerm
from zipline.utils.numpy_utils import int64_dtype

from ..mixins import (
    CustomTermMixin,
    LatestMixin,
    PositiveWindowLengthMixin,
    RestrictedDTypeMixin
)


class Classifier(RestrictedDTypeMixin, ComputableTerm):
    ALLOWED_DTYPES = (int64_dtype,)  # Used by RestrictedDTypeMixin


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


class Latest(LatestMixin, CustomClassifier):
    """
    A classifier producing the latest value of an input.

    See Also
    --------
    zipline.pipeline.data.dataset.BoundColumn.latest
    zipline.pipeline.factors.factor.Latest
    zipline.pipeline.filters.filter.Latest
    """
