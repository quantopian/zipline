"""
Tests for Factor terms.
"""
from nose_parameterized import parameterized

from numpy import arange, array, empty, eye, nan, ones, datetime64
from numpy.random import randn, seed

from zipline.errors import UnknownRankMethod
from zipline.pipeline import Factor, Filter, TermGraph
from zipline.pipeline.factors import RSI
from zipline.utils.test_utils import check_allclose, check_arrays

from .base import BasePipelineTestCase


class F(Factor):
    inputs = ()
    window_length = 0


class Mask(Filter):
    inputs = ()
    window_length = 0


class FactorTestCase(BasePipelineTestCase):

    def setUp(self):
        super(FactorTestCase, self).setUp()
        self.f = F()

    def test_bad_input(self):
        with self.assertRaises(UnknownRankMethod):
            self.f.rank("not a real rank method")

    def test_rank_ascending(self):

        # Generated with:
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]], dtype=float)
        expected_ranks = {
            'ordinal': array([[1., 3., 4., 5., 2.],
                              [2., 4., 5., 1., 3.],
                              [3., 5., 1., 2., 4.],
                              [4., 1., 2., 3., 5.],
                              [1., 3., 4., 5., 2.]]),
            'average': array([[1.5, 3., 4., 5., 1.5],
                              [2.5, 4., 5., 1., 2.5],
                              [3.5, 5., 1., 2., 3.5],
                              [4.5, 1., 2., 3., 4.5],
                              [1.5, 3., 4., 5., 1.5]]),
            'min': array([[1., 3., 4., 5., 1.],
                          [2., 4., 5., 1., 2.],
                          [3., 5., 1., 2., 3.],
                          [4., 1., 2., 3., 4.],
                          [1., 3., 4., 5., 1.]]),
            'max': array([[2., 3., 4., 5., 2.],
                          [3., 4., 5., 1., 3.],
                          [4., 5., 1., 2., 4.],
                          [5., 1., 2., 3., 5.],
                          [2., 3., 4., 5., 2.]]),
            'dense': array([[1., 2., 3., 4., 1.],
                            [2., 3., 4., 1., 2.],
                            [3., 4., 1., 2., 3.],
                            [4., 1., 2., 3., 4.],
                            [1., 2., 3., 4., 1.]]),
        }

        def check(terms):
            graph = TermGraph(terms)
            results = self.run_graph(
                graph,
                initial_workspace={self.f: data},
                mask=self.build_mask(ones((5, 5))),
            )
            for method in terms:
                check_arrays(results[method], expected_ranks[method])

        check({meth: self.f.rank(method=meth) for meth in expected_ranks})
        check({
            meth: self.f.rank(method=meth, ascending=True)
            for meth in expected_ranks
        })
        # Not passing a method should default to ordinal.
        check({'ordinal': self.f.rank()})
        check({'ordinal': self.f.rank(ascending=True)})

    def test_rank_descending(self):

        # Generated with:
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]], dtype=float)
        expected_ranks = {
            'ordinal': array([[4., 3., 2., 1., 5.],
                              [3., 2., 1., 5., 4.],
                              [2., 1., 5., 4., 3.],
                              [1., 5., 4., 3., 2.],
                              [4., 3., 2., 1., 5.]]),
            'average': array([[4.5, 3., 2., 1., 4.5],
                              [3.5, 2., 1., 5., 3.5],
                              [2.5, 1., 5., 4., 2.5],
                              [1.5, 5., 4., 3., 1.5],
                              [4.5, 3., 2., 1., 4.5]]),
            'min': array([[4., 3., 2., 1., 4.],
                          [3., 2., 1., 5., 3.],
                          [2., 1., 5., 4., 2.],
                          [1., 5., 4., 3., 1.],
                          [4., 3., 2., 1., 4.]]),
            'max': array([[5., 3., 2., 1., 5.],
                          [4., 2., 1., 5., 4.],
                          [3., 1., 5., 4., 3.],
                          [2., 5., 4., 3., 2.],
                          [5., 3., 2., 1., 5.]]),
            'dense': array([[4., 3., 2., 1., 4.],
                            [3., 2., 1., 4., 3.],
                            [2., 1., 4., 3., 2.],
                            [1., 4., 3., 2., 1.],
                            [4., 3., 2., 1., 4.]]),
        }

        def check(terms):
            graph = TermGraph(terms)
            results = self.run_graph(
                graph,
                initial_workspace={self.f: data},
                mask=self.build_mask(ones((5, 5))),
            )
            for method in terms:
                check_arrays(results[method], expected_ranks[method])

        check({
            meth: self.f.rank(method=meth, ascending=False)
            for meth in expected_ranks
        })
        # Not passing a method should default to ordinal.
        check({'ordinal': self.f.rank(ascending=False)})

    def test_rank_after_mask(self):
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]], dtype=float)
        mask_data = ~eye(5, dtype=bool)
        initial_workspace = {self.f: data, Mask(): mask_data}

        graph = TermGraph(
            {
                "ascending_nomask": self.f.rank(ascending=True),
                "ascending_mask": self.f.rank(ascending=True, mask=Mask()),
                "descending_nomask": self.f.rank(ascending=False),
                "descending_mask": self.f.rank(ascending=False, mask=Mask()),
            }
        )

        expected = {
            "ascending_nomask": array([[1., 3., 4., 5., 2.],
                                       [2., 4., 5., 1., 3.],
                                       [3., 5., 1., 2., 4.],
                                       [4., 1., 2., 3., 5.],
                                       [1., 3., 4., 5., 2.]]),
            "descending_nomask": array([[4., 3., 2., 1., 5.],
                                        [3., 2., 1., 5., 4.],
                                        [2., 1., 5., 4., 3.],
                                        [1., 5., 4., 3., 2.],
                                        [4., 3., 2., 1., 5.]]),
            # Diagonal should be all nans, and anything whose rank was less
            # than the diagonal in the unmasked calc should go down by 1.
            "ascending_mask": array([[nan, 2., 3., 4., 1.],
                                     [2., nan, 4., 1., 3.],
                                     [2., 4., nan, 1., 3.],
                                     [3., 1., 2., nan, 4.],
                                     [1., 2., 3., 4., nan]]),
            "descending_mask": array([[nan, 3., 2., 1., 4.],
                                      [2., nan, 1., 4., 3.],
                                      [2., 1., nan, 4., 3.],
                                      [1., 4., 3., nan, 2.],
                                      [4., 3., 2., 1., nan]]),
        }

        results = self.run_graph(
            graph,
            initial_workspace,
            mask=self.build_mask(ones((5, 5))),
        )
        for method in results:
            check_arrays(expected[method], results[method])

    @parameterized.expand([
        # Test cases computed by doing:
        # from numpy.random import seed, randn
        # from talib import RSI
        # seed(seed_value)
        # data = abs(randn(15, 3))
        # expected = [RSI(data[:, i])[-1] for i in range(3)]
        (100, array([41.032913785966, 51.553585468393, 51.022005016446])),
        (101, array([43.506969935466, 46.145367530182, 50.57407044197])),
        (102, array([46.610102205934, 47.646892444315, 52.13182788538])),
    ])
    def test_rsi(self, seed_value, expected):

        rsi = RSI()

        today = datetime64(1, 'ns')
        assets = arange(3)
        out = empty((3,), dtype=float)

        seed(seed_value)  # Seed so we get deterministic results.
        test_data = abs(randn(15, 3))

        out = empty((3,), dtype=float)
        rsi.compute(today, assets, out, test_data)

        check_allclose(expected, out)
