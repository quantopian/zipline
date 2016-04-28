"""
classifier.py
"""
from functools import wraps
from numbers import Number
import operator

from numpy import where, isnan, nan, zeros

from zipline.lib.labelarray import LabelArray
from zipline.lib.quantiles import quantiles
from zipline.pipeline.api_utils import restrict_to_dtype
from zipline.pipeline.term import ComputableTerm, NotSpecified
from zipline.utils.compat import unicode
from zipline.utils.input_validation import expect_types
from zipline.utils.numpy_utils import (
    categorical_dtype,
    int64_dtype,
)

from ..filters import Filter, NullFilter, NumExprFilter
from ..mixins import (
    CustomTermMixin,
    LatestMixin,
    PositiveWindowLengthMixin,
    RestrictedDTypeMixin,
    SingleInputMixin,
)


strings_only = restrict_to_dtype(
    dtype=categorical_dtype,
    message_template=(
        "{method_name}() is only defined on Classifiers producing strings"
        " but it was called on a Factor of dtype {received_dtype}."
    )
)


class Classifier(RestrictedDTypeMixin, ComputableTerm):
    """
    A Pipeline expression computing a categorical output.

    Classifiers are most commonly useful for describing grouping keys for
    complex transformations on Factor outputs. For example, Factor.demean() and
    Factor.zscore() can be passed a Classifier in their ``groupby`` argument,
    indicating that means/standard deviations should be computed on assets for
    which the classifier produced the same label.
    """
    # Used by RestrictedDTypeMixin
    ALLOWED_DTYPES = (int64_dtype, categorical_dtype)
    categories = NotSpecified

    def isnull(self):
        """
        A Filter producing True for values where this term has missing data.
        """
        return NullFilter(self)

    def notnull(self):
        """
        A Filter producing True for values where this term has complete data.
        """
        return ~self.isnull()

    # We explicitly don't support classifier to classifier comparisons, since
    # the stored values likely don't mean the same thing. This may be relaxed
    # in the future, but for now we're starting conservatively.
    def eq(self, other):
        """
        Construct a Filter returning True for asset/date pairs where the output
        of ``self`` matches ``other.
        """
        # We treat this as an error because missing_values have NaN semantics,
        # which means this would return an array of all False, which is almost
        # certainly not what the user wants.
        if other == self.missing_value:
            raise ValueError(
                "Comparison against self.missing_value ({value!r}) in"
                " {typename}.eq().\n"
                "Missing values have NaN semantics, so the "
                "requested comparison would always produce False.\n"
                "Use the isnull() method to check for missing values.".format(
                    value=other,
                    typename=(type(self).__name__),
                )
            )

        if isinstance(other, Number) != (self.dtype == int64_dtype):
            raise InvalidClassifierComparison(self, other)

        if isinstance(other, Number):
            return NumExprFilter.create(
                "x_0 == {other}".format(other=int(other)),
                binds=(self,),
            )
        else:
            return ScalarStringPredicate(
                classifier=self,
                op=operator.eq,
                compval=other,
            )

    def __ne__(self, other):
        """
        Construct a Filter returning True for asset/date pairs where the output
        of ``self`` matches ``other.
        """
        if isinstance(other, Number) != (self.dtype == int64_dtype):
            raise InvalidClassifierComparison(self, other)

        if isinstance(other, Number):
            return NumExprFilter.create(
                "((x_0 != {other}) & (x_0 != {missing}))".format(
                    other=int(other),
                    missing=self.missing_value,
                ),
                binds=(self,),
            )
        else:
            return ScalarStringPredicate(
                classifier=self,
                op=operator.ne,
                compval=other,
            )

    def _string_predicate(f):
        """
        Decorator for converting a function from (LabelArray, str) -> bool
        into a Classifier method that returns a ScalarStringPredicate filter.

        This mainly exists to avoid replicating shared boilerplate
        (e.g. argument type validation).
        """
        @wraps(f)
        @expect_types(compval=(bytes, unicode))
        @strings_only
        def method(self, compval):
            return ScalarStringPredicate(
                classifier=self,
                op=f,
                compval=compval,
            )
        return method

    @_string_predicate
    @expect_types(label_array=LabelArray)
    def startswith(label_array, other):
        return label_array.startswith(other)

    @_string_predicate
    @expect_types(label_array=LabelArray)
    def endswith(label_array, other):
        return label_array.endswith(other)

    @_string_predicate
    @expect_types(label_array=LabelArray)
    def contains(label_array, other):
        return label_array.contains(other)

    del _string_predicate

    def postprocess(self, data):
        if self.dtype == int64_dtype:
            return data
        if not isinstance(data, LabelArray):
            raise AssertionError("Expected a LabelArray, got %s." % type(data))
        return data.as_categorical()


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


class Quantiles(SingleInputMixin, Classifier):
    """
    A classifier computing quantiles over an input.
    """
    params = ('bins',)
    dtype = int64_dtype
    window_length = 0
    missing_value = -1

    def _compute(self, arrays, dates, assets, mask):
        data = arrays[0]
        bins = self.params['bins']
        to_bin = where(mask, data, nan)
        result = quantiles(to_bin, bins)
        # Write self.missing_value into nan locations, whether they were
        # generated by our input mask or not.
        result[isnan(result)] = self.missing_value
        return result.astype(int64_dtype)

    def short_repr(self):
        return type(self).__name__ + '(%d)' % self.params['bins']


class ScalarStringPredicate(SingleInputMixin, Filter):
    """
    A filter that applies a function from (LabelArray, str) -> ndarray[bool].

    Examples include ``==, !=, startswith, and endswith``.

    This exists because we represent string arrays with
    ``zipline.lib.LabelArray``s, which numexpr doesn't know about, so we can't
    use the generic NumExprFilter implementation here.
    """
    window_length = 0

    @expect_types(classifier=Classifier, compval=(bytes, unicode))
    def __new__(cls, classifier, op, compval):
        return super(ScalarStringPredicate, cls).__new__(
            ScalarStringPredicate,
            compval=compval,
            op=op,
            inputs=(classifier,),
            mask=classifier.mask,
        )

    def _init(self, op, compval, *args, **kwargs):
        self._op = op
        self._compval = compval
        return super(ScalarStringPredicate, self)._init(*args, **kwargs)

    @classmethod
    def static_identity(cls, op, compval, *args, **kwargs):
        return (
            super(ScalarStringPredicate, cls).static_identity(*args, **kwargs),
            op,
            compval,
        )

    def _compute(self, arrays, dates, assets, mask):
        data = arrays[0]
        return (
            self._op(data, self._compval)
            & (data != self.inputs[0].missing_value)
            & mask
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
    def _allocate_output(self, windows, shape):
        """
        Override the default array allocation to produce a LabelArray when we
        have a string-like dtype.
        """
        if self.dtype == int64_dtype:
            return super(Latest, self)._allocate_output(windows, shape)

        # This is a little bit of a hack.  We might not know what the
        # categories for a LabelArray are until it's actually been loaded, so
        # we need to look at the underlying data.
        return windows[0].data.empty_like(shape)


class InvalidClassifierComparison(TypeError):
    def __init__(self, classifier, compval):
        super(InvalidClassifierComparison, self).__init__(
            "Can't compare classifier of dtype"
            " {dtype} to value {value} of type {type}.".format(
                dtype=classifier.dtype,
                value=compval,
                type=type(compval).__name__,
            )
        )
