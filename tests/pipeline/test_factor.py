"""
Tests for Factor terms.
"""
from itertools import product
from nose_parameterized import parameterized

from numpy import (
    arange,
    array,
    datetime64,
    empty,
    eye,
    nan,
    ones,
)
from numpy.random import randn, seed

from zipline.errors import UnknownRankMethod
from zipline.lib.rank import masked_rankdata_2d
from zipline.pipeline import Factor, Filter, TermGraph
from zipline.pipeline.factors import (
    Returns,
    RSI,
)
from zipline.utils.test_utils import check_allclose, check_arrays
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    float64_dtype,
    NaTns,
)

from .base import BasePipelineTestCase


class F(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class Mask(Filter):
    inputs = ()
    window_length = 0


for_each_factor_dtype = parameterized.expand([
    ('datetime64[ns]', datetime64ns_dtype),
    ('float', float64_dtype),
])


class FactorTestCase(BasePipelineTestCase):

    def setUp(self):
        super(FactorTestCase, self).setUp()
        self.f = F()

    def test_bad_input(self):
        with self.assertRaises(UnknownRankMethod):
            self.f.rank("not a real rank method")

    @for_each_factor_dtype
    def test_rank_ascending(self, name, factor_dtype):

        f = F(dtype=factor_dtype)

        # Generated with:
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]], dtype=factor_dtype)

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
                initial_workspace={f: data},
                mask=self.build_mask(ones((5, 5))),
            )
            for method in terms:
                check_arrays(results[method], expected_ranks[method])

        check({meth: f.rank(method=meth) for meth in expected_ranks})
        check({
            meth: f.rank(method=meth, ascending=True)
            for meth in expected_ranks
        })
        # Not passing a method should default to ordinal.
        check({'ordinal': f.rank()})
        check({'ordinal': f.rank(ascending=True)})

    @for_each_factor_dtype
    def test_rank_descending(self, name, factor_dtype):

        f = F(dtype=factor_dtype)

        # Generated with:
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]], dtype=factor_dtype)
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
                initial_workspace={f: data},
                mask=self.build_mask(ones((5, 5))),
            )
            for method in terms:
                check_arrays(results[method], expected_ranks[method])

        check({
            meth: f.rank(method=meth, ascending=False)
            for meth in expected_ranks
        })
        # Not passing a method should default to ordinal.
        check({'ordinal': f.rank(ascending=False)})

    @for_each_factor_dtype
    def test_rank_after_mask(self, name, factor_dtype):

        f = F(dtype=factor_dtype)
        # data = arange(25).reshape(5, 5).transpose() % 4
        data = array([[0, 1, 2, 3, 0],
                      [1, 2, 3, 0, 1],
                      [2, 3, 0, 1, 2],
                      [3, 0, 1, 2, 3],
                      [0, 1, 2, 3, 0]], dtype=factor_dtype)
        mask_data = ~eye(5, dtype=bool)
        initial_workspace = {f: data, Mask(): mask_data}

        graph = TermGraph(
            {
                "ascending_nomask": f.rank(ascending=True),
                "ascending_mask": f.rank(ascending=True, mask=Mask()),
                "descending_nomask": f.rank(ascending=False),
                "descending_mask": f.rank(ascending=False, mask=Mask()),
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

    @parameterized.expand([
        (100, 15),
        (101, 4),
        (102, 100),
        ])
    def test_returns(self, seed_value, window_length):

        returns = Returns(window_length=window_length)

        today = datetime64(1, 'ns')
        assets = arange(3)
        out = empty((3,), dtype=float)

        seed(seed_value)  # Seed so we get deterministic results.
        test_data = abs(randn(window_length, 3))

        # Calculate the expected returns
        expected = (test_data[-1] - test_data[0]) / test_data[0]

        out = empty((3,), dtype=float)
        returns.compute(today, assets, out, test_data)

        check_allclose(expected, out)

    def gen_ranking_cases():
        seeds = range(int(1e4), int(1e5), int(1e4))
        methods = ('ordinal', 'average')
        use_mask_values = (True, False)
        set_missing_values = (True, False)
        ascending_values = (True, False)
        return product(
            seeds,
            methods,
            use_mask_values,
            set_missing_values,
            ascending_values,
        )

    @parameterized.expand(gen_ranking_cases())
    def test_masked_rankdata_2d(self,
                                seed_value,
                                method,
                                use_mask,
                                set_missing,
                                ascending):
        eyemask = ~eye(5, dtype=bool)
        nomask = ones((5, 5), dtype=bool)

        seed(seed_value)
        asfloat = (randn(5, 5) * seed_value)
        asdatetime = (asfloat).copy().view('datetime64[ns]')

        mask = eyemask if use_mask else nomask
        if set_missing:
            asfloat[:, 2] = nan
            asdatetime[:, 2] = NaTns

        float_result = masked_rankdata_2d(
            data=asfloat,
            mask=mask,
            missing_value=nan,
            method=method,
            ascending=True,
        )
        datetime_result = masked_rankdata_2d(
            data=asdatetime,
            mask=mask,
            missing_value=NaTns,
            method=method,
            ascending=True,
        )

        check_arrays(float_result, datetime_result)
