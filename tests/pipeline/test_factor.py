"""
Tests for Factor terms.
"""
from itertools import product
from nose_parameterized import parameterized

from numpy import (
    apply_along_axis,
    arange,
    array,
    datetime64,
    empty,
    eye,
    nan,
    nanmean,
    nanstd,
    ones,
    where,
)
from numpy.random import randn, seed

from zipline.errors import UnknownRankMethod
from zipline.lib.rank import masked_rankdata_2d
from zipline.lib.normalize import naive_grouped_rowwise_apply as grouped_apply
from zipline.pipeline import Classifier, Factor, Filter, TermGraph
from zipline.pipeline.factors import (
    Returns,
    RSI,
)
from zipline.testing import (
    check_allclose,
    check_arrays,
    parameter_space,
)
from zipline.utils.numpy_utils import (
    datetime64ns_dtype,
    float64_dtype,
    int64_dtype,
    NaTns,
)

from .base import BasePipelineTestCase


class F(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class C(Classifier):
    dtype = int64_dtype
    missing_value = -1
    inputs = ()
    window_length = 0


class OtherC(Classifier):
    dtype = int64_dtype
    missing_value = -1
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

    @parameter_space(method_name=['isnan', 'notnan', 'isfinite'])
    def test_float64_only_ops(self, method_name):
        class NotFloat(Factor):
            dtype = datetime64ns_dtype
            inputs = ()
            window_length = 0

        nf = NotFloat()
        meth = getattr(nf, method_name)
        with self.assertRaises(TypeError):
            meth()

    @parameter_space(custom_missing_value=[-1, 0])
    def test_isnull_int_dtype(self, custom_missing_value):

        class CustomMissingValue(Factor):
            dtype = int64_dtype
            window_length = 0
            missing_value = custom_missing_value
            inputs = ()

        factor = CustomMissingValue()

        data = arange(25).reshape(5, 5)
        data[eye(5, dtype=bool)] = custom_missing_value

        graph = TermGraph(
            {
                'isnull': factor.isnull(),
                'notnull': factor.notnull(),
            }
        )

        results = self.run_graph(
            graph,
            initial_workspace={factor: data},
            mask=self.build_mask(ones((5, 5))),
        )
        check_arrays(results['isnull'], eye(5, dtype=bool))
        check_arrays(results['notnull'], ~eye(5, dtype=bool))

    def test_isnull_datetime_dtype(self):
        class DatetimeFactor(Factor):
            dtype = datetime64ns_dtype
            window_length = 0
            inputs = ()

        factor = DatetimeFactor()

        data = arange(25).reshape(5, 5).astype('datetime64[ns]')
        data[eye(5, dtype=bool)] = NaTns

        graph = TermGraph(
            {
                'isnull': factor.isnull(),
                'notnull': factor.notnull(),
            }
        )

        results = self.run_graph(
            graph,
            initial_workspace={factor: data},
            mask=self.build_mask(ones((5, 5))),
        )
        check_arrays(results['isnull'], eye(5, dtype=bool))
        check_arrays(results['notnull'], ~eye(5, dtype=bool))

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

    @parameter_space(
        seed_value=range(1, 2),
        normalizer_name_and_func=[
            ('demean', lambda row: row - nanmean(row)),
            ('zscore', lambda row: (row - nanmean(row)) / nanstd(row)),
        ],
        add_nulls_to_factor=(False, True,)
    )
    def test_normalizations(self,
                            seed_value,
                            normalizer_name_and_func,
                            add_nulls_to_factor):

        name, func = normalizer_name_and_func

        shape = (7, 7)

        # All Trues.
        nomask = self.ones_mask(shape=shape)
        # Falses on main diagonal.
        eyemask = self.eye_mask(shape=shape)
        # Falses on other diagonal.
        eyemask_T = eyemask.T
        # Falses on both diagonals.
        xmask = eyemask & eyemask_T

        # Block of random data.
        factor_data = self.randn_data(seed=seed_value, shape=shape)
        if add_nulls_to_factor:
            factor_data = where(eyemask, factor_data, nan)

        # Cycles of 0, 1, 2, 0, 1, 2, ...
        classifier_data = (
            (self.arange_data(shape=shape, dtype=int) + seed_value) % 3
        )
        # With -1s on main diagonal.
        classifier_data_eyenulls = where(eyemask, classifier_data, -1)
        # With -1s on opposite diagonal.
        classifier_data_eyenulls_T = where(eyemask_T, classifier_data, -1)
        # With -1s on both diagonals.
        classifier_data_xnulls = where(xmask, classifier_data, -1)

        f = self.f
        c = C()
        c_with_nulls = OtherC()
        m = Mask()
        method = getattr(f, name)
        terms = {
            'vanilla': method(),
            'masked': method(mask=m),
            'grouped': method(groupby=c),
            'grouped_with_nulls': method(groupby=c_with_nulls),
            'both': method(mask=m, groupby=c),
            'both_with_nulls': method(mask=m, groupby=c_with_nulls),
        }

        expected = {
            'vanilla': apply_along_axis(func, 1, factor_data,),
            'masked': where(
                eyemask,
                grouped_apply(factor_data, eyemask, func),
                nan,
            ),
            'grouped': grouped_apply(
                factor_data,
                classifier_data,
                func,
            ),
            # If the classifier has nulls, we should get NaNs in the
            # corresponding locations in the output.
            'grouped_with_nulls': where(
                eyemask_T,
                grouped_apply(factor_data, classifier_data_eyenulls_T, func),
                nan,
            ),
            # Passing a mask with a classifier should behave as though the
            # classifier had nulls where the mask was False.
            'both': where(
                eyemask,
                grouped_apply(
                    factor_data,
                    classifier_data_eyenulls,
                    func,
                ),
                nan,
            ),
            'both_with_nulls': where(
                xmask,
                grouped_apply(
                    factor_data,
                    classifier_data_xnulls,
                    func,
                ),
                nan,
            )
        }

        graph = TermGraph(terms)
        results = self.run_graph(
            graph,
            initial_workspace={
                f: factor_data,
                c: classifier_data,
                c_with_nulls: classifier_data_eyenulls_T,
                Mask(): eyemask,
            },
            mask=self.build_mask(nomask),
        )

        for key in expected:
            check_arrays(expected[key], results[key])

    @parameter_space(method_name=['demean', 'zscore'])
    def test_cant_normalize_non_float(self, method_name):
        class DateFactor(Factor):
            dtype = datetime64ns_dtype
            inputs = ()
            window_length = 0

        d = DateFactor()
        with self.assertRaises(TypeError) as e:
            getattr(d, method_name)()

        errmsg = str(e.exception)
        expected = (
            "{normalizer}() is only defined on Factors of dtype float64,"
            " but it was called on a Factor of dtype datetime64[ns]."
        ).format(normalizer=method_name)

        self.assertEqual(errmsg, expected)
