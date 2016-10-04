import numpy as np

from zipline.testing.predicates import assert_equal
from zipline.pipeline import Classifier, Factor, Filter
from zipline.utils.numpy_utils import float64_dtype, int64_dtype

from .base import BasePipelineTestCase


class WithAlias(object):

    def test_alias(self):
        f = self.Term()
        alias = f.alias('ayy lmao')

        f_values = np.random.RandomState(5).randn(5, 5)

        self.check_terms(
            terms={
                'f_alias': alias,
            },
            expected={
                'f_alias': f_values,
            },
            initial_workspace={f: f_values},
            mask=self.build_mask(np.ones((5, 5))),
        )

    def test_repr(self):
        assert_equal(
            repr(self.Term().alias('ayy lmao')),
            "Aliased%s(..., name='ayy lmao')" % self.Term.__base__.__name__,
        )


class TestFactorAlias(WithAlias, BasePipelineTestCase):
    class Term(Factor):
        dtype = float64_dtype
        inputs = ()
        window_length = 0


class TestFilterAlias(WithAlias, BasePipelineTestCase):
    class Term(Filter):
        inputs = ()
        window_length = 0


class TestClassifierAlias(WithAlias, BasePipelineTestCase):
    class Term(Classifier):
        dtype = int64_dtype
        inputs = ()
        window_length = 0
        missing_value = -1
