"""
Base class for Pipeline API unit tests.
"""
import numpy as np
import pandas as pd

from zipline.lib.labelarray import LabelArray
from zipline.pipeline import ExecutionPlan
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.hooks import NoHooks
from zipline.pipeline.term import AssetExists, InputDates
from zipline.testing.core import check_arrays
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithTradingSessions,
    ZiplineTestCase,
)
from zipline.utils.compat import wraps
from zipline.utils.functional import dzip_exact
from zipline.utils.pandas_utils import explode


def with_defaults(**default_funcs):
    """
    Decorator for providing dynamic default values for a method.

    Usages:

    @with_defaults(foo=lambda self: self.x + self.y)
    def func(self, foo):
        ...

    If a value is passed for `foo`, it will be used. Otherwise the function
    supplied to `with_defaults` will be called with `self` as an argument.
    """

    def decorator(f):
        @wraps(f)
        def method(self, *args, **kwargs):
            for name, func in default_funcs.items():
                if name not in kwargs:
                    kwargs[name] = func(self)
            return f(self, *args, **kwargs)

        return method

    return decorator


with_default_shape = with_defaults(shape=lambda self: self.default_shape)


class BaseUSEquityPipelineTestCase(
    WithTradingSessions, WithAssetFinder, ZiplineTestCase
):
    START_DATE = pd.Timestamp("2014")
    END_DATE = pd.Timestamp("2014-12-31")
    ASSET_FINDER_EQUITY_SIDS = list(range(20))

    @classmethod
    def init_class_fixtures(cls):
        super(BaseUSEquityPipelineTestCase, cls).init_class_fixtures()

        cls.default_asset_exists_mask = cls.asset_finder.lifetimes(
            cls.nyse_sessions[-30:],
            include_start_date=False,
            country_codes={cls.ASSET_FINDER_COUNTRY_CODE},
        )

    @property
    def default_shape(self):
        """Default shape for methods that build test data."""
        return self.default_asset_exists_mask.shape

    def run_graph(self, graph, initial_workspace, mask=None):
        """
        Compute the given TermGraph, seeding the workspace of our engine with
        `initial_workspace`.

        Parameters
        ----------
        graph : zipline.pipeline.graph.ExecutionPlan
            Graph to run.
        initial_workspace : dict
            Initial workspace to forward to SimplePipelineEngine.compute_chunk.
        mask : DataFrame, optional
            This is a value to pass to `initial_workspace` as the mask from
            `AssetExists()`.  Defaults to a frame of shape `self.default_shape`
            containing all True values.

        Returns
        -------
        results : dict
            Mapping from termname -> computed result.
        """

        def get_loader(c):
            raise AssertionError("run_graph() should not require any loaders!")

        engine = SimplePipelineEngine(
            get_loader,
            self.asset_finder,
            default_domain=US_EQUITIES,
        )
        if mask is None:
            mask = self.default_asset_exists_mask

        dates, sids, mask_values = explode(mask)

        initial_workspace.setdefault(AssetExists(), mask_values)
        initial_workspace.setdefault(InputDates(), dates)

        refcounts = graph.initial_refcounts(initial_workspace)
        execution_order = graph.execution_order(initial_workspace, refcounts)

        return engine.compute_chunk(
            graph=graph,
            dates=dates,
            sids=sids,
            workspace=initial_workspace,
            execution_order=execution_order,
            refcounts=refcounts,
            hooks=NoHooks(),
        )

    def run_terms(self, terms, initial_workspace, mask):
        start_date, end_date = mask.index[[0, -1]]
        graph = ExecutionPlan(
            domain=US_EQUITIES,
            terms=terms,
            start_date=start_date,
            end_date=end_date,
        )

        return self.run_graph(graph, initial_workspace, mask)

    def check_terms(self, terms, expected, initial_workspace, mask, check=check_arrays):
        """
        Compile the given terms into a TermGraph, compute it with
        initial_workspace, and compare the results with ``expected``.
        """
        results = self.run_terms(terms, initial_workspace, mask)

        for key, (res, exp) in dzip_exact(results, expected).items():
            check(res, exp)

        return results

    def build_mask(self, array):
        """
        Helper for constructing an AssetExists mask from a boolean-coercible
        array.
        """
        ndates, nassets = array.shape
        return pd.DataFrame(
            array,
            # Use the **last** N dates rather than the first N so that we have
            # space for lookbacks.
            index=self.nyse_sessions[-ndates:],
            columns=self.ASSET_FINDER_EQUITY_SIDS[:nassets],
            dtype=bool,
        )

    @with_default_shape
    def arange_data(self, shape, dtype=np.float64):
        """
        Build a block of testing data from numpy.arange.
        """
        return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    @with_default_shape
    def randn_data(self, seed, shape):
        """
        Build a block of random numerical data.
        """
        return np.random.RandomState(seed).randn(*shape)

    @with_default_shape
    def rand_ints(self, seed, shape, low=0, high=10):
        """
        Build a block of random numerical data.
        """
        rand = np.random.RandomState(seed)
        return rand.randint(low, high, shape, dtype="i8")

    @with_default_shape
    def rand_datetimes(self, seed, shape):
        ints = self.rand_ints(seed=seed, shape=shape, low=0, high=10000)
        return ints.astype("datetime64[D]").astype("datetime64[ns]")

    @with_default_shape
    def rand_categoricals(self, categories, seed, shape, missing_value=None):
        """Build a block of random categorical data.

        Categories should not include ``missing_value``.
        """
        categories = list(categories) + [missing_value]
        data = np.random.RandomState(seed).choice(categories, shape)
        return LabelArray(
            data,
            missing_value=missing_value,
            categories=categories,
        )

    @with_default_shape
    def rand_mask(self, seed, shape):
        """Build a block of random boolean data."""
        return np.random.RandomState(seed).randint(0, 2, shape).astype(bool)

    @with_default_shape
    def eye_mask(self, shape):
        """
        Build a mask using np.eye.
        """
        return ~np.eye(*shape, dtype=bool)

    @with_default_shape
    def ones_mask(self, shape):
        return np.ones(shape, dtype=bool)
