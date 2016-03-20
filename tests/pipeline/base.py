"""
Base class for Pipeline API  unittests.
"""
import abc
from functools import wraps
from unittest import TestCase

from nose_parameterized import parameterized
import numpy as np
from numpy import arange, prod
import pandas as pd
from pandas import date_range, Int64Index, DataFrame
from pandas.util.testing import assert_series_equal
from six import iteritems

from zipline.pipeline import Pipeline
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.term import AssetExists
from zipline.testing import (
    ExplodingObject,
    gen_calendars,
    make_simple_equity_info,
    num_days_in_range,
    tmp_asset_finder,
)

from zipline.utils.numpy_utils import (
    NaTD,
    make_datetime64D
)
from zipline.utils.pandas_utils import explode
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


class EventLoaderCommonMixin(object):
    @abc.abstractproperty
    def get_sids(cls):
        raise NotImplementedError('get_sids')

    @classmethod
    def get_equity_info(cls):
        return make_simple_equity_info(
            cls.get_sids(),
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )

    def zip_with_floats(self, dates, flts):
        return pd.Series(flts, index=dates).astype('float')

    def num_days_between(self, dates, start_date, end_date):
        return num_days_in_range(dates, start_date, end_date)

    def zip_with_dates(self, index_dates, dts):
        return pd.Series(pd.to_datetime(dts), index=index_dates)

    def loader_args(self, dates):
        """Construct the base  object to pass to the loader.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The dates we can serve.

        Returns
        -------
        args : tuple[any]
            The arguments to forward to the loader positionally.
        """
        return dates, self.dataset

    def setup_engine(self, dates):
        """
        Make a Pipeline Enigne object based on the given dates.
        """
        loader = self.loader_type(*self.loader_args(dates))
        return SimplePipelineEngine(lambda _: loader, dates, self.finder)

    @staticmethod
    def _compute_busday_offsets(announcement_dates):
        """
        Compute expected business day offsets from a DataFrame of announcement
        dates.
        """
        # Column-vector of dates on which factor `compute` will be called.
        raw_call_dates = announcement_dates.index.values.astype(
            'datetime64[D]'
        )[:, None]

        # 2D array of dates containining expected nexg announcement.
        raw_announce_dates = (
            announcement_dates.values.astype('datetime64[D]')
        )

        # Set NaTs to 0 temporarily because busday_count doesn't support NaT.
        # We fill these entries with NaNs later.
        whereNaT = raw_announce_dates == NaTD
        raw_announce_dates[whereNaT] = make_datetime64D(0)

        # The abs call here makes it so that we can use this function to
        # compute offsets for both next and previous earnings (previous
        # earnings offsets come back negative).
        expected = abs(np.busday_count(
            raw_call_dates,
            raw_announce_dates
        ).astype(float))

        expected[whereNaT] = np.nan
        return pd.DataFrame(
            data=expected,
            columns=announcement_dates.columns,
            index=announcement_dates.index,
        )

    @parameterized.expand(gen_calendars(
        '2014-01-01',
        '2014-01-31',
        critical_dates=pd.to_datetime([
            '2014-01-05',
            '2014-01-10',
            '2014-01-15',
            '2014-01-20',
        ], utc=True),
    ))
    def test_compute(self, dates):
        engine = self.setup_engine(dates)
        self.setup(dates)

        pipe = Pipeline(
            columns=self.pipeline_columns
        )

        result = engine.run_pipeline(
            pipe,
            start_date=dates[0],
            end_date=dates[-1],
        )

        for sid in self.get_sids():
            for col_name in self.cols.keys():
                assert_series_equal(result[col_name].xs(sid, level=1),
                                    self.cols[col_name][sid],
                                    check_names=False)
