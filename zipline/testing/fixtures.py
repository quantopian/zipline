from unittest import TestCase

from contextlib2 import ExitStack
from logbook import NullHandler
from nose_parameterized import parameterized
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from six import with_metaclass

from .core import tmp_asset_finder, make_simple_equity_info, gen_calendars
from ..finance.trading import TradingEnvironment
from ..utils import tradingcalendar, factory
from ..utils.final import FinalMeta, final

from zipline.pipeline import Pipeline, SimplePipelineEngine
from zipline.utils.numpy_utils import make_datetime64D
from zipline.utils.numpy_utils import NaTD
from zipline.pipeline.common import TS_FIELD_NAME
from zipline.pipeline.loaders.utils import (
    get_values_for_date_ranges,
    zip_with_dates
)


class ZiplineTestCase(with_metaclass(FinalMeta, TestCase)):
    """
    Shared extensions to core unittest.TestCase.

    Overrides the default unittest setUp/tearDown functions with versions that
    use ExitStack to correctly clean up resources, even in the face of
    exceptions that occur during setUp/setUpClass.

    Subclasses **should not override setUp or setUpClass**!

    Instead, they should implement `init_instance_fixtures` for per-test-method
    resources, and `init_class_fixtures` for per-class resources.

    Resources that need to be cleaned up should be registered using
    either `enter_{class,instance}_context` or `add_{class,instance}_callback}.
    """

    @final
    @classmethod
    def setUpClass(cls):
        cls._class_teardown_stack = ExitStack()
        try:
            cls._base_init_fixtures_was_called = False
            cls.init_class_fixtures()
            assert cls._base_init_fixtures_was_called, (
                "ZiplineTestCase.init_class_fixtures() was not called.\n"
                "This probably means that you overrode init_class_fixtures"
                " without calling super()."
            )
        except:
            cls.tearDownClass()
            raise

    @classmethod
    def init_class_fixtures(cls):
        """
        Override and implement this classmethod to register resources that
        should be created and/or torn down on a per-class basis.

        Subclass implementations of this should always invoke this with super()
        to ensure that fixture mixins work properly.
        """
        cls._base_init_fixtures_was_called = True

    @final
    @classmethod
    def tearDownClass(cls):
        cls._class_teardown_stack.close()

    @final
    @classmethod
    def enter_class_context(cls, context_manager):
        """
        Enter a context manager to be exited during the tearDownClass
        """
        return cls._class_teardown_stack.enter_context(context_manager)

    @final
    @classmethod
    def add_class_callback(cls, callback):
        """
        Register a callback to be executed during tearDownClass.

        Parameters
        ----------
        callback : callable
            The callback to invoke at the end of the test suite.
        """
        return cls._class_teardown_stack.callback(callback)

    @final
    def setUp(self):
        self._instance_teardown_stack = ExitStack()
        try:
            self._init_instance_fixtures_was_called = False
            self.init_instance_fixtures()
            assert self._init_instance_fixtures_was_called, (
                "ZiplineTestCase.init_instance_fixtures() was not"
                " called.\n"
                "This probably means that you overrode"
                " init_instance_fixtures without calling super()."
            )
        except:
            self.tearDown()
            raise

    def init_instance_fixtures(self):
        self._init_instance_fixtures_was_called = True

    @final
    def tearDown(self):
        self._instance_teardown_stack.close()

    @final
    def enter_instance_context(self, context_manager):
        """
        Enter a context manager that should be exited during tearDown.
        """
        return self._instance_teardown_stack.enter_context(context_manager)

    @final
    def add_instance_callback(self, callback):
        """
        Register a callback to be executed during tearDown.

        Parameters
        ----------
        callback : callable
            The callback to invoke at the end of each test.
        """
        return self._instance_teardown_stack.callback(callback)


class WithLogger(object):
    """
    ZiplineTestCase mixin providing cls.log_handler as an instance-level
    fixture.

    After init_instance_fixtures has been called `self.log_handler` will be a
    new ``logbook.NullHandler``.

    This behavior may be overridden by defining a ``make_log_handler`` class
    method which returns a new logbook.LogHandler instance.
    """
    make_log_handler = NullHandler

    @classmethod
    def init_class_fixtures(cls):
        super(WithLogger, cls).init_class_fixtures()

        cls.log_handler = cls.enter_class_context(
            cls.make_log_handler().applicationbound(),
        )


class WithAssetFinder(object):
    """
    ZiplineTestCase mixin providing cls.asset_finder as a class-level fixture.

    After init_class_fixtures has been called, `cls.asset_finder` is populated
    with an AssetFinder. The default finder is the result of calling
    `tmp_asset_finder` with arguments generated as follows::

       equities=cls.make_equities_info(),
       futures=cls.make_futures_info(),
       exchanges=cls.make_exchanges_info(),
       root_symbols=cls.make_root_symbols_info(),

    Each of these methods may be overridden with a function returning a
    alternative dataframe of data to write.

    The top-level creation behavior can be altered by overriding
    `make_asset_finder` as a class method.

    See Also
    --------
    zipline.testing.make_simple_equity_info
    zipline.testing.make_jagged_equity_info
    zipline.testing.make_rotating_equity_info
    zipline.testing.make_future_info
    zipline.testing.make_commodity_future_info
    """
    @classmethod
    def _make_info(cls):
        return None

    make_equities_info = _make_info
    make_futures_info = _make_info
    make_exchanges_info = _make_info
    make_root_symbols_info = _make_info

    del _make_info

    @classmethod
    def make_asset_finder(cls):
        return cls.enter_class_context(tmp_asset_finder(
            equities=cls.equities_info,
            futures=cls.futures_info,
            exchanges=cls.exchanges_info,
            root_symbols=cls.root_symbols_info,
        ))

    @classmethod
    def init_class_fixtures(cls):
        super(WithAssetFinder, cls).init_class_fixtures()

        # TODO: Move this to consumers that actually depend on it.
        #       These are misleading if make_asset_finder is overridden.
        cls.equities_info = cls.make_equities_info()
        cls.futures_info = cls.make_futures_info()
        cls.exchanges_info = cls.make_exchanges_info()
        cls.root_symbols_info = cls.make_root_symbols_info()
        cls.asset_finder = cls.make_asset_finder()


class WithTradingEnvironment(WithAssetFinder):
    """
    ZiplineTestCase mixin providing cls.env as a class-level fixture.

    After ``init_class_fixtures`` has been called, `cls.env` is populated
    with a trading environment whose `asset_finder` is the result of
    `cls.make_asset_finder`.

    The ``load`` function may be provided by overriding the
    ``make_load_function`` class method.

    This behavior can be altered by overriding `make_trading_environment` as a
    class method.
    """
    @classmethod
    def make_load_function(cls):
        return None

    @classmethod
    def make_trading_environment(cls):
        return TradingEnvironment(
            load=cls.make_load_function(),
            asset_db_path=cls.asset_finder.engine,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithTradingEnvironment, cls).init_class_fixtures()
        cls.env = cls.make_trading_environment()


class WithSimParams(WithTradingEnvironment):
    """
    ZiplineTestCase mixin providing cls.sim_params as a class level fixture.

    The arguments used to construct the trading environment may be overridded
    by putting ``SIM_PARAMS_{argname}`` in the class dict except for the
    trading environment which is overridden with the mechanisms provided by
    ``WithTradingEnvironment``.
    """
    SIM_PARAMS_YEAR = None
    SIM_PARAMS_START = pd.Timestamp('2006-01-01')
    SIM_PARAMS_END = pd.Timestamp('2006-12-31')
    SIM_PARAMS_CAPITAL_BASE = float("1.0e5")
    SIM_PARAMS_NUM_DAYS = None
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    SIM_PARAMS_EMISSION_RATE = 'daily'

    @classmethod
    def init_class_fixtures(cls):
        super(WithSimParams, cls).init_class_fixtures()
        cls.sim_params = factory.create_simulation_parameters(
            year=cls.SIM_PARAMS_YEAR,
            start=cls.SIM_PARAMS_START,
            end=cls.SIM_PARAMS_END,
            capital_base=cls.SIM_PARAMS_CAPITAL_BASE,
            data_frequency=cls.SIM_PARAMS_DATA_FREQUENCY,
            emission_rate=cls.SIM_PARAMS_EMISSION_RATE,
            env=cls.env,
        )


class WithNYSETradingDays(object):
    """
    ZiplineTestCase mixin providing cls.trading_days as a class-level fixture.

    After init_class_fixtures has been called, `cls.trading_days` is populated
    with a DatetimeIndex containing NYSE calendar trading days ranging from:

    (DATA_MAX_DAY - (cls.TRADING_DAY_COUNT) -> DATA_MAX_DAY)

    The default value of TRADING_DAY_COUNT is 126 (half a trading-year).
    Inheritors can override TRADING_DAY_COUNT to request more or less data.
    """
    DATA_MAX_DAY = pd.Timestamp('2016-01-04')
    TRADING_DAY_COUNT = 126

    @classmethod
    def init_class_fixtures(cls):
        super(WithNYSETradingDays, cls).init_class_fixtures()

        all_days = tradingcalendar.trading_days
        end_loc = all_days.get_loc(cls.DATA_MAX_DAY)
        start_loc = end_loc - cls.TRADING_DAY_COUNT

        cls.trading_days = all_days[start_loc:end_loc + 1]


class WithPipelineEventDataLoader(WithAssetFinder):
    """
    ZiplineTestCase mixin providing common test methods/behaviors for event
    data loaders.

    `get_sids` must return the sids being tested.
    `get_dataset` must return {sid -> pd.DataFrame}
    `loader_type` must return the loader class to use for loading the dataset
    `make_asset_finder` returns a default asset finder which can be overridden.
    """
    @classmethod
    def get_sids(cls):
        return range(0, 5)

    @classmethod
    def get_dataset(cls):
        return {sid: pd.DataFrame() for sid in cls.get_sids()}

    @classmethod
    def loader_type(self):
        return None

    @classmethod
    def make_equities_info(cls):
        return make_simple_equity_info(
            cls.get_sids(),
            start_date=pd.Timestamp('2013-01-01', tz='UTC'),
            end_date=pd.Timestamp('2015-01-01', tz='UTC'),
        )

    def pipeline_event_loader_args(self, dates):
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
        return dates, self.get_dataset()

    def pipeline_event_setup_engine(self, dates):
        """
        Make a Pipeline Enigne object based on the given dates.
        """
        loader = self.loader_type(*self.pipeline_event_loader_args(dates))
        return SimplePipelineEngine(lambda _: loader, dates, self.asset_finder)

    def get_sids_to_frames(self,
                           zip_date_index_with_vals,
                           vals,
                           date_intervals,
                           dates):
        """
        Construct a DataFrame that maps sid to the expected values for the
        given dates.

        Parameters
        ----------
        zip_date_index_with_vals: callable
            A function that returns a series of `vals` repeated based on the
            number of days in the date interval for each val, indexed by the
            dates in `dates`.
        vals: iterable
            An iterable with values that correspond to each interval in
            `date_intervals`.
        date_intervals: list
            A list of date intervals for each sid that correspond to values in
            `vals`.
        dates: DatetimeIndex
            The dates which will serve as the index for each Series for each
            sid in the DataFrame.
        """
        frame = pd.DataFrame({sid: get_values_for_date_ranges(
            zip_date_index_with_vals,
            vals[sid],
            date_intervals[sid],
            dates
        ) for sid in self.get_sids()[:-1]})
        frame[self.get_sids()[-1]] = zip_date_index_with_vals(
            dates, ['NaN'] * len(dates)
        )
        return frame

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
        engine = self.pipeline_event_setup_engine(dates)
        cols = self.setup(dates)

        pipe = Pipeline(
            columns=self.pipeline_columns
        )

        result = engine.run_pipeline(
            pipe,
            start_date=dates[0],
            end_date=dates[-1],
        )

        for sid in self.get_sids():
            for col_name in cols.keys():
                assert_series_equal(result[col_name].xs(sid, level=1),
                                    cols[col_name][sid],
                                    check_names=False)


class WithNextAndPreviousEventDataLoader(WithPipelineEventDataLoader):
    """
    ZiplineTestCase mixin extending common functionality for event data
    loader tests that have both next and previous events.

    `base_cases` should be used as the template to test cases that combine
    knowledge date (timestamp) and some 'other_date' in various ways.
    `next_date_intervals` gives the date intervals for the next event based
    on the dates given in `base_cases`.
    `next_dates` gives the next date from `other_date` which is known about at
    each interval.
    `prev_date_intervals` gives the date intervals for each sid for the
    previous event based on the dates given in `base_cases`.
    `prev_dates` gives the previous date from `other_date` which is known
    about at each interval.
    `get_expected_previous_event_dates` is a convenience function that fills
    a DataFrame with the previously known dates for each sid for the given
    dates.
    `get_expected_next_event_dates` is a convenience function that fills
    a DataFrame with the next known dates for each sid for the given
    dates.
    """
    base_cases = [
        # K1--K2--A1--A2.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
            'other_date': pd.to_datetime(['2014-01-15', '2014-01-20']),
        }),
        # K1--K2--A2--A1.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-10']),
            'other_date': pd.to_datetime(['2014-01-20', '2014-01-15']),
        }),
        # K1--A1--K2--A2.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05', '2014-01-15']),
            'other_date': pd.to_datetime(['2014-01-10', '2014-01-20']),
        }),
        # K1 == K2.
        pd.DataFrame({
            TS_FIELD_NAME: pd.to_datetime(['2014-01-05'] * 2),
            'other_date': pd.to_datetime(['2014-01-10', '2014-01-15']),
        }),
        pd.DataFrame(
            columns=['other_date',
                     TS_FIELD_NAME],
            dtype='datetime64[ns]'
        ),
    ]

    next_date_intervals = [
        [[None, '2014-01-04'],
         ['2014-01-05', '2014-01-15'],
         ['2014-01-16', '2014-01-20'],
         ['2014-01-21', None]],
        [[None, '2014-01-04'],
         ['2014-01-05', '2014-01-09'],
         ['2014-01-10', '2014-01-15'],
         ['2014-01-16', '2014-01-20'],
         ['2014-01-21', None]],
        [[None, '2014-01-04'],
         ['2014-01-05', '2014-01-10'],
         ['2014-01-11', '2014-01-14'],
         ['2014-01-15', '2014-01-20'],
         ['2014-01-21', None]],
        [[None, '2014-01-04'],
         ['2014-01-05', '2014-01-10'],
         ['2014-01-11', '2014-01-15'],
         ['2014-01-16', None]]
    ]

    next_dates = [
        ['NaT', '2014-01-15', '2014-01-20', 'NaT'],
        ['NaT', '2014-01-20', '2014-01-15', '2014-01-20', 'NaT'],
        ['NaT', '2014-01-10', 'NaT', '2014-01-20', 'NaT'],
        ['NaT', '2014-01-10', '2014-01-15', 'NaT'],
        ['NaT']
    ]

    prev_date_intervals = [
        [[None, '2014-01-14'],
         ['2014-01-15', '2014-01-19'],
         ['2014-01-20', None]],
        [[None, '2014-01-14'],
         ['2014-01-15', '2014-01-19'],
         ['2014-01-20', None]],
        [[None, '2014-01-09'],
         ['2014-01-10', '2014-01-19'],
         ['2014-01-20', None]],
        [[None, '2014-01-09'],
         ['2014-01-10', '2014-01-14'],
         ['2014-01-15', None]]
    ]

    prev_dates = [
        ['NaT', '2014-01-15', '2014-01-20'],
        ['NaT', '2014-01-15', '2014-01-20'],
        ['NaT', '2014-01-10', '2014-01-20'],
        ['NaT', '2014-01-10', '2014-01-15'],
        ['NaT']
    ]

    def get_expected_previous_event_dates(self, dates):
        return self.get_sids_to_frames(
            zip_with_dates,
            self.prev_dates,
            self.prev_date_intervals,
            dates
        )

    def get_expected_next_event_dates(self, dates):
        return self.get_sids_to_frames(
            zip_with_dates,
            self.next_dates,
            self.next_date_intervals,
            dates
        )
