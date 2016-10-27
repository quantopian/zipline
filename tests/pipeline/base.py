"""
Base class for Pipeline API unit tests.
"""
from functools import wraps

import numpy as np
from numpy import arange, prod
from pandas import DataFrame, Timestamp
from six import iteritems

from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline import ExecutionPlan
from zipline.pipeline.term import AssetExists, InputDates
from zipline.testing import (
    check_arrays,
    ExplodingObject,
)
from zipline.testing.fixtures import (
    WithAssetFinder,
    WithTradingSessions,
    ZiplineTestCase,
)

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
            for name, func in iteritems(default_funcs):
                if name not in kwargs:
                    kwargs[name] = func(self)
            return f(self, *args, **kwargs)
        return method
    return decorator


with_default_shape = with_defaults(shape=lambda self: self.default_shape)


class BasePipelineTestCase(WithTradingSessions,
                           WithAssetFinder,
                           ZiplineTestCase):
    START_DATE = Timestamp('2014', tz='UTC')
    END_DATE = Timestamp('2014-12-31', tz='UTC')
    ASSET_FINDER_EQUITY_SIDS = list(range(20))

    @classmethod
    def init_class_fixtures(cls):
        super(BasePipelineTestCase, cls).init_class_fixtures()

        cls.default_asset_exists_mask = cls.asset_finder.lifetimes(
            cls.nyse_sessions[-30:],
            include_start_date=False,
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
        graph : zipline.pipeline.graph.TermGraph
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
        engine = SimplePipelineEngine(
            lambda column: ExplodingObject(),
            self.nyse_sessions,
            self.asset_finder,
        )
        if mask is None:
            mask = self.default_asset_exists_mask

        dates, assets, mask_values = explode(mask)

        initial_workspace.setdefault(AssetExists(), mask_values)
        initial_workspace.setdefault(InputDates(), dates)

        return engine.compute_chunk(
            graph,
            dates,
            assets,
            initial_workspace,
        )

    def check_terms(self,
                    terms,
                    expected,
                    initial_workspace,
                    mask,
                    check=check_arrays):
        """
        Compile the given terms into a TermGraph, compute it with
        initial_workspace, and compare the results with ``expected``.
        """
        start_date, end_date = mask.index[[0, -1]]
        graph = ExecutionPlan(
            terms,
            all_dates=self.nyse_sessions,
            start_date=start_date,
            end_date=end_date,
        )

        results = self.run_graph(graph, initial_workspace, mask)
        for key, (res, exp) in dzip_exact(results, expected).items():
            check(res, exp)

        return results

    def build_mask(self, array):
        """
        Helper for constructing an AssetExists mask from a boolean-coercible
        array.
        """
        ndates, nassets = array.shape
        return DataFrame(
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
        return arange(prod(shape), dtype=dtype).reshape(shape)

    @with_default_shape
    def randn_data(self, seed, shape):
        """
        Build a block of testing data from a seeded RandomState.
        """
        return np.random.RandomState(seed).randn(*shape)

    @with_default_shape
    def eye_mask(self, shape):
        """
        Build a mask using np.eye.
        """
        return ~np.eye(*shape, dtype=bool)

    @with_default_shape
    def ones_mask(self, shape):
        return np.ones(shape, dtype=bool)
