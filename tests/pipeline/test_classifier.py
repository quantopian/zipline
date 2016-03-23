import numpy as np

from zipline.pipeline import Classifier, TermGraph
from zipline.testing import check_arrays, parameter_space
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

        # There's no significance to the values here other than that they
        # contain a mix of missing and non-missing values.
        data = np.array([[-1,  1,  0, 2],
                         [3,   0,  1, 0],
                         [-5,  0, -1, 0],
                         [-3,  1,  2, 2]], dtype=int)

        c = C()
        graph = TermGraph(
            {
                'isnull': c.isnull(),
                'notnull': c.notnull()
            }
        )

        results = self.run_graph(
            graph,
            initial_workspace={c: data},
            mask=self.build_mask(self.ones_mask(shape=data.shape)),
        )

        check_arrays(results['isnull'], (data == mv))
        check_arrays(results['notnull'], (data != mv))
