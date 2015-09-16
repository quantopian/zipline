"""
Base class for FFC unit tests.
"""
from functools import wraps
from unittest import TestCase

from numpy import arange, prod
from pandas import date_range, Int64Index, DataFrame
from six import iteritems

from zipline.finance.trading import TradingEnvironment
from zipline.modelling.engine import SimpleFFCEngine
from zipline.modelling.graph import TermGraph
from zipline.modelling.term import AssetExists
from zipline.utils.pandas_utils import explode
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

        # Set up env for test
        env = TradingEnvironment()
        env.write_data(
            equities_df=make_simple_asset_info(
                assets,
                self.__calendar[0],
                self.__calendar[-1],
            ),
        )
        self.__finder = env.asset_finder

        # Use a 30-day period at the end of the year by default.
        self.__mask = self.__finder.lifetimes(
            self.__calendar[-30:],
            include_start_date=False,
        )

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
        initial_workspace : dict
            Initial workspace to forward to SimpleFFCEngine.compute_chunk.
        mask : DataFrame, optional
            This is a value to pass to `initial_workspace` as the mask from
            `AssetExists()`.  Defaults to a frame of shape `self.default_shape`
            containing all True values.

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
        if mask is None:
            mask = self.__mask

        dates, assets, mask_values = explode(mask)
        initial_workspace.setdefault(AssetExists(), mask_values)
        return engine.compute_chunk(
            TermGraph(terms),
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
