"""
Base class for FFC unit tests.
"""
from functools import wraps
from unittest import TestCase

from numpy import arange, prod
from numpy.random import randn, seed as random_seed
from pandas import date_range, Int64Index, DataFrame
from six import iteritems

from zipline.assets import AssetFinder
from zipline.modelling.engine import SimpleFFCEngine
from zipline.modelling.graph import TermGraph
from zipline.utils.test_utils import make_simple_asset_info, ExplodingObject
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


class BaseFFCTestCase(TestCase):

    def setUp(self):
        self.__calendar = date_range('2014', '2015', freq=trading_day)
        self.__assets = assets = Int64Index(arange(1, 20))
        self.__finder = AssetFinder(
            make_simple_asset_info(
                assets,
                self.__calendar[0],
                self.__calendar[-1],
            ),
            db_path=':memory:',
            create_table=True,
        )
        self.__mask = self.__finder.lifetimes(self.__calendar[-10:])

    @property
    def default_shape(self):
        """Default shape for methods that build test data."""
        return self.__mask.shape

    def run_terms(self, terms, initial_workspace, mask=None):
        """
        Compute the given terms, seeding the workspace of our FFCEngine with
        `initial_workspace`.

        Parameters
        ----------
        terms : dict
            Mapping from termname -> term object.

        Returns
        -------
        results : dict
            Mapping from termname -> computed result.
        """
        engine = SimpleFFCEngine(
            ExplodingObject(),
            self.__calendar,
            self.__finder,
        )
        mask = mask if mask is not None else self.__mask
        return engine.compute_chunk(TermGraph(terms), mask, initial_workspace)

    def build_mask(self, array):
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

    @with_default_shape
    def randn_data(self, seed, shape):
        """
        Build a block of testing data from numpy.random.randn.
        """
        random_seed(seed)
        return randn(*shape)
