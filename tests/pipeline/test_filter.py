"""
Tests for filter terms.
"""
from itertools import product
from operator import and_

from numpy import (
    arange,
    argsort,
    array,
    eye,
    float64,
    full_like,
    inf,
    isfinite,
    nan,
    nanpercentile,
    ones,
    ones_like,
    putmask,
)
from numpy.random import randn, seed as random_seed

from zipline.errors import BadPercentileBounds
from zipline.pipeline import Filter, Factor, TermGraph
from zipline.testing import check_arrays
from zipline.utils.numpy_utils import float64_dtype
from .base import BasePipelineTestCase, with_default_shape


def rowwise_rank(array, mask=None):
    """
    Take a 2D array and return the 0-indexed sorted position of each element in
    the array for each row.

    Example
    -------
    In [5]: data
    Out[5]:
    array([[-0.141, -1.103, -1.0171,  0.7812,  0.07  ],
           [ 0.926,  0.235, -0.7698,  1.4552,  0.2061],
           [ 1.579,  0.929, -0.557 ,  0.7896, -1.6279],
           [-1.362, -2.411, -1.4604,  1.4468, -0.1885],
           [ 1.272,  1.199, -3.2312, -0.5511, -1.9794]])

    In [7]: argsort(argsort(data))
    Out[7]:
    array([[2, 0, 1, 4, 3],
           [3, 2, 0, 4, 1],
           [4, 3, 1, 2, 0],
           [2, 0, 1, 4, 3],
           [4, 3, 0, 2, 1]])
    """
    # note that unlike scipy.stats.rankdata, the output here is 0-indexed, not
    # 1-indexed.
    return argsort(argsort(array))


class SomeFactor(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class SomeOtherFactor(Factor):
    dtype = float64_dtype
    inputs = ()
    window_length = 0


class Mask(Filter):
    inputs = ()
    window_length = 0


class FilterTestCase(BasePipelineTestCase):

    def init_instance_fixtures(self):
        super(FilterTestCase, self).init_instance_fixtures()
        self.f = SomeFactor()
        self.g = SomeOtherFactor()

    @with_default_shape
    def randn_data(self, seed, shape):
        """
        Build a block of testing data from numpy.random.randn.
        """
        random_seed(seed)
        return randn(*shape)

    def test_bad_percentiles(self):
        f = self.f

        bad_percentiles = [
            (-.1, 10),
            (10, 100.1),
            (20, 10),
            (50, 50),
        ]
        for min_, max_ in bad_percentiles:
            with self.assertRaises(BadPercentileBounds):
                f.percentile_between(min_, max_)

    def test_top_and_bottom(self):
        data = self.randn_data(seed=5)  # Fix a seed for determinism.

        mask_data = ones_like(data, dtype=bool)
        mask_data[:, 0] = False

        nan_data = data.copy()
        nan_data[:, 0] = nan

        mask = Mask()
        workspace = {self.f: data, mask: mask_data}

        methods = ['top', 'bottom']
        counts = 2, 3, 10
        term_combos = list(product(methods, counts, [True, False]))

        def termname(method, count, masked):
            return '_'.join([method, str(count), 'mask' if masked else ''])

        # Add a term for each permutation of top/bottom, count, and
        # mask/no_mask.
        terms = {}
        for method, count, masked in term_combos:
            kwargs = {'N': count}
            if masked:
                kwargs['mask'] = mask
            term = getattr(self.f, method)(**kwargs)
            terms[termname(method, count, masked)] = term

        results = self.run_graph(TermGraph(terms), initial_workspace=workspace)

        def expected_result(method, count, masked):
            # Ranking with a mask is equivalent to ranking with nans applied on
            # the masked values.
            to_rank = nan_data if masked else data

            if method == 'top':
                return rowwise_rank(-to_rank) < count
            elif method == 'bottom':
                return rowwise_rank(to_rank) < count

        for method, count, masked in term_combos:
            result = results[termname(method, count, masked)]

            # Check that `min(c, num_assets)` assets passed each day.
            passed_per_day = result.sum(axis=1)
            check_arrays(
                passed_per_day,
                full_like(passed_per_day, min(count, data.shape[1])),
            )

            expected = expected_result(method, count, masked)
            check_arrays(result, expected)

    def test_bottom(self):
        counts = 2, 3, 10
        data = self.randn_data(seed=5)  # Arbitrary seed choice.
        results = self.run_graph(
            TermGraph(
                {'bottom_' + str(c): self.f.bottom(c) for c in counts}
            ),
            initial_workspace={self.f: data},
        )
        for c in counts:
            result = results['bottom_' + str(c)]

            # Check that `min(c, num_assets)` assets passed each day.
            passed_per_day = result.sum(axis=1)
            check_arrays(
                passed_per_day,
                full_like(passed_per_day, min(c, data.shape[1])),
            )

            # Check that the bottom `c` assets passed.
            expected = rowwise_rank(data) < c
            check_arrays(result, expected)

    def test_percentile_between(self):

        quintiles = range(5)
        filter_names = ['pct_' + str(q) for q in quintiles]
        iter_quintiles = zip(filter_names, quintiles)

        graph = TermGraph(
            {
                name: self.f.percentile_between(q * 20.0, (q + 1) * 20.0)
                for name, q in zip(filter_names, quintiles)
            }
        )

        # Test with 5 columns and no NaNs.
        eye5 = eye(5, dtype=float64)
        results = self.run_graph(
            graph,
            initial_workspace={self.f: eye5},
            mask=self.build_mask(ones((5, 5))),
        )
        for name, quintile in iter_quintiles:
            result = results[name]
            if quintile < 4:
                # There are four 0s and one 1 in each row, so the first 4
                # quintiles should be all the locations with zeros in the input
                # array.
                check_arrays(result, ~eye5.astype(bool))
            else:
                # The top quintile should match the sole 1 in each row.
                check_arrays(result, eye5.astype(bool))

        # Test with 6 columns, no NaNs, and one masked entry per day.
        eye6 = eye(6, dtype=float64)
        mask = array([[1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1],
                      [1, 0, 1, 1, 1, 1],
                      [1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 0, 1]], dtype=bool)

        results = self.run_graph(
            graph,
            initial_workspace={self.f: eye6},
            mask=self.build_mask(mask)
        )
        for name, quintile in iter_quintiles:
            result = results[name]
            if quintile < 4:
                # Should keep all values that were 0 in the base data and were
                # 1 in the mask.
                check_arrays(result, mask & (~eye6.astype(bool))),
            else:
                # Should keep all the 1s in the base data.
                check_arrays(result, eye6.astype(bool))

        # Test with 6 columns, no mask, and one NaN per day.  Should have the
        # same outcome as if we had masked the NaNs.
        # In particular, the NaNs should never pass any filters.
        eye6_withnans = eye6.copy()
        putmask(eye6_withnans, ~mask, nan)
        results = self.run_graph(
            graph,
            initial_workspace={self.f: eye6},
            mask=self.build_mask(mask)
        )
        for name, quintile in iter_quintiles:
            result = results[name]
            if quintile < 4:
                # Should keep all values that were 0 in the base data and were
                # 1 in the mask.
                check_arrays(result, mask & (~eye6.astype(bool))),
            else:
                # Should keep all the 1s in the base data.
                check_arrays(result, eye6.astype(bool))

    def test_percentile_nasty_partitions(self):
        # Test percentile with nasty partitions: divide up 5 assets into
        # quartiles.
        # There isn't a nice mathematical definition of correct behavior here,
        # so for now we guarantee the behavior of numpy.nanpercentile.  This is
        # mostly for regression testing in case we write our own specialized
        # percentile calculation at some point in the future.

        data = arange(25, dtype=float).reshape(5, 5) % 4
        quartiles = range(4)
        filter_names = ['pct_' + str(q) for q in quartiles]

        graph = TermGraph(
            {
                name: self.f.percentile_between(q * 25.0, (q + 1) * 25.0)
                for name, q in zip(filter_names, quartiles)
            }
        )
        results = self.run_graph(
            graph,
            initial_workspace={self.f: data},
            mask=self.build_mask(ones((5, 5))),
        )

        for name, quartile in zip(filter_names, quartiles):
            result = results[name]
            lower = quartile * 25.0
            upper = (quartile + 1) * 25.0
            expected = and_(
                nanpercentile(data, lower, axis=1, keepdims=True) <= data,
                data <= nanpercentile(data, upper, axis=1, keepdims=True),
            )
            check_arrays(result, expected)

    def test_percentile_after_mask(self):
        f_input = eye(5)
        g_input = arange(25, dtype=float).reshape(5, 5)
        initial_mask = self.build_mask(ones((5, 5)))

        custom_mask = self.f < 1
        without_mask = self.g.percentile_between(80, 100)
        with_mask = self.g.percentile_between(80, 100, mask=custom_mask)

        graph = TermGraph(
            {
                'custom_mask': custom_mask,
                'without': without_mask,
                'with': with_mask,
            }
        )

        results = self.run_graph(
            graph,
            initial_workspace={self.f: f_input, self.g: g_input},
            mask=initial_mask,
        )

        # First should pass everything but the diagonal.
        check_arrays(results['custom_mask'], ~eye(5, dtype=bool))

        # Second should pass the largest value each day.  Each row is strictly
        # increasing, so we always select the last value.
        expected_without = array(
            [[0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1]],
            dtype=bool,
        )
        check_arrays(results['without'], expected_without)

        # When sequencing, we should remove the diagonal as an option before
        # computing percentiles.  On the last day, we should get the
        # second-largest value, rather than the largest.
        expected_with = array(
            [[0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0]],  # Different from previous!
            dtype=bool,
        )
        check_arrays(results['with'], expected_with)

    def test_isnan(self):
        data = self.randn_data(seed=10)
        diag = eye(*data.shape, dtype=bool)
        data[diag] = nan

        results = self.run_graph(
            TermGraph({
                'isnan': self.f.isnan(),
                'isnull': self.f.isnull(),
            }),
            initial_workspace={self.f: data},
        )
        check_arrays(results['isnan'], diag)
        check_arrays(results['isnull'], diag)

    def test_notnan(self):
        data = self.randn_data(seed=10)
        diag = eye(*data.shape, dtype=bool)
        data[diag] = nan

        results = self.run_graph(
            TermGraph({
                'notnan': self.f.notnan(),
                'notnull': self.f.notnull(),
            }),
            initial_workspace={self.f: data},
        )
        check_arrays(results['notnan'], ~diag)
        check_arrays(results['notnull'], ~diag)

    def test_isfinite(self):
        data = self.randn_data(seed=10)
        data[:, 0] = nan
        data[:, 2] = inf
        data[:, 4] = -inf

        results = self.run_graph(
            TermGraph({'isfinite': self.f.isfinite()}),
            initial_workspace={self.f: data},
        )
        check_arrays(results['isfinite'], isfinite(data))
