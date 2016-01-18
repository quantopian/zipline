"""
Base class for Pipeline API  unittests.
"""
from functools import wraps
from unittest import TestCase

from numpy import arange, prod
from pandas import date_range, Int64Index, DataFrame
from six import iteritems

from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.term import AssetExists
from zipline.utils.pandas_utils import explode
from zipline.utils.test_utils import (
    ExplodingObject,
    make_simple_equity_info,
    tmp_asset_finder,
)
from zipline.utils.tradingcalendar import trading_day


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


class BasePipelineTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.__calendar = date_range('2014', '2015', freq=trading_day)
        cls.__assets = assets = Int64Index(arange(1, 20))
        cls.__tmp_finder_ctx = tmp_asset_finder(
            equities=make_simple_equity_info(
                assets,
                cls.__calendar[0],
                cls.__calendar[-1],
            )
        )
        cls.__finder = cls.__tmp_finder_ctx.__enter__()
        cls.__mask = cls.__finder.lifetimes(
            cls.__calendar[-30:],
            include_start_date=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.__tmp_finder_ctx.__exit__()

    @property
    def default_shape(self):
        """Default shape for methods that build test data."""
        return self.__mask.shape

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
            self.__calendar,
            self.__finder,
        )
        if mask is None:
            mask = self.__mask

        dates, assets, mask_values = explode(mask)
        initial_workspace.setdefault(AssetExists(), mask_values)
        return engine.compute_chunk(
            graph,
            dates,
            assets,
            initial_workspace,
        )

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
            index=self.__calendar[-ndates:],
            columns=self.__assets[:nassets],
            dtype=bool,
        )

    @with_default_shape
    def arange_data(self, shape, dtype=float):
        """
        Build a block of testing data from numpy.arange.
        """
        return arange(prod(shape), dtype=dtype).reshape(shape)
