import numpy as np

from zipline.pipeline import Classifier
from zipline.testing import parameter_space
from zipline.utils.numpy_utils import int64_dtype

from .base import BasePipelineTestCase


class ClassifierTestCase(BasePipelineTestCase):

    @parameter_space(mv=[-1, 0, 1, 999])
    def test_isnull(self, mv):

        class C(Classifier):
            dtype = int64_dtype
            missing_value = mv
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of missing and non-missing values.
        data = np.array([[-1,  1,  0, 2],
                         [3,   0,  1, 0],
                         [-5,  0, -1, 0],
                         [-3,  1,  2, 2]], dtype=int64_dtype)

        self.check_terms(
            terms={
                'isnull': c.isnull(),
                'notnull': c.notnull()
            },
            expected={
                'isnull': data == mv,
                'notnull': data != mv,
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(compval=[0, 1, 999])
    def test_eq(self, compval):

        class C(Classifier):
            dtype = int64_dtype
            missing_value = -1
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = np.array([[-1,  1,  0, 2],
                         [3,   0,  1, 0],
                         [-5,  0, -1, 0],
                         [-3,  1,  2, 2]], dtype=int64_dtype)

        self.check_terms(
            terms={
                'eq': c.eq(compval),
            },
            expected={
                'eq': (data == compval),
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

    @parameter_space(missing=[-1, 0, 1])
    def test_disallow_comparison_to_missing_value(self, missing):
        class C(Classifier):
            dtype = int64_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        with self.assertRaises(ValueError) as e:
            C().eq(missing)
        errmsg = str(e.exception)
        self.assertEqual(
            errmsg,
            "Comparison against self.missing_value ({v}) in C.eq().\n"
            "Missing values have NaN semantics, so the requested comparison"
            " would always produce False.\n"
            "Use the isnull() method to check for missing values.".format(
                v=missing,
            ),
        )

    @parameter_space(compval=[0, 1, 999], missing=[-1, 0, 999])
    def test_not_equal(self, compval, missing):

        class C(Classifier):
            dtype = int64_dtype
            missing_value = missing
            inputs = ()
            window_length = 0

        c = C()

        # There's no significance to the values here other than that they
        # contain a mix of the comparison value and other values.
        data = np.array([[-1,  1,  0, 2],
                         [3,   0,  1, 0],
                         [-5,  0, -1, 0],
                         [-3,  1,  2, 2]], dtype=int64_dtype)

        self.check_terms(
            terms={
                'ne': c != compval,
            },
            expected={
                'ne': (data != compval) & (data != C.missing_value),
            },
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )
