import sqlite3
from unittest import TestCase

from contextlib2 import ExitStack
from logbook import NullHandler
from nose_parameterized import parameterized
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
import responses
from six import with_metaclass, iteritems

from ..assets.synthetic import make_simple_equity_info
from .core import (
    create_daily_bar_data,
    create_minute_bar_data,
    gen_calendars,
    tmp_asset_finder,
    tmp_dir,
)
from ..data.data_portal import DataPortal
from ..data.us_equity_pricing import (
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
)
from ..finance.trading import TradingEnvironment
from ..data.us_equity_pricing import (
    BcolzDailyBarReader,
    BcolzDailyBarWriter,
)
from ..data.minute_bars import (
    BcolzMinuteBarReader,
    BcolzMinuteBarWriter,
    US_EQUITIES_MINUTES_PER_DAY
)
from ..utils import tradingcalendar, factory
from ..utils.classproperty import classproperty
from ..utils.final import FinalMeta, final
from zipline.pipeline import Pipeline, SimplePipelineEngine
from zipline.utils.numpy_utils import make_datetime64D
from zipline.utils.numpy_utils import NaTD


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
        # Hold a set of all the "static" attributes on the class. These are
        # things that are not populated after the class was created like
        # methods or other class level attributes.
        cls._static_class_attributes = set(vars(cls))
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
        for name in set(vars(cls)) - cls._static_class_attributes:
            # Remove all of the attributes that were added after the class was
            # constructed. This cleans up any large test data that is class
            # scoped while still allowing subclasses to access class level
            # attributes.
            delattr(cls, name)

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
    def add_instance_callback(self, callback, *args, **kwargs):
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
            equities=cls.make_equities_info(),
            futures=cls.make_futures_info(),
            exchanges=cls.make_exchanges_info(),
            root_symbols=cls.make_root_symbols_info(),
        ))

    @classmethod
    def init_class_fixtures(cls):
        super(WithAssetFinder, cls).init_class_fixtures()
        cls.asset_finder = cls.make_asset_finder()

    @classproperty
    def equities_info(cls):
        return cls.asset_finder.equities_info

    @classproperty
    def futures_info(cls):
        return cls.asset_finder.futures_info


class WithTradingEnvironment(WithAssetFinder):
    """
    ZiplineTestCase mixin providing cls.env as a class-level fixture.

    After ``init_class_fixtures`` has been called, `cls.env` is populated
    with a trading environment whose `asset_finder` is the result of
    `cls.make_asset_finder`.

    The ``load`` function may be provided by overriding the
    ``make_load_function`` class method.

    """
    TRADING_ENV_MIN_DATE = None
    TRADING_ENV_MAX_DATE = None
    TRADING_ENV_TRADING_CALENDAR = tradingcalendar

    @classmethod
    def make_load_function(cls):
        return None

    @classmethod
    def make_trading_environment(cls):
        return TradingEnvironment(
            load=cls.make_load_function(),
            asset_db_path=cls.asset_finder.engine,
            min_date=cls.TRADING_ENV_MIN_DATE,
            max_date=cls.TRADING_ENV_MAX_DATE,
            env_trading_calendar=cls.TRADING_ENV_TRADING_CALENDAR,
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
    SIM_PARAMS_START = pd.Timestamp('2006-01-03', tz='utc')
    SIM_PARAMS_END = pd.Timestamp('2006-12-29', tz='utc')
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
            num_days=cls.SIM_PARAMS_NUM_DAYS,
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
    DATA_MAX_DAY = pd.Timestamp('2016-01-04', tz='utc')
    TRADING_DAY_COUNT = 126

    @classmethod
    def init_class_fixtures(cls):
        super(WithNYSETradingDays, cls).init_class_fixtures()

        all_days = tradingcalendar.trading_days
        end_loc = all_days.get_loc(cls.DATA_MAX_DAY)
        start_loc = end_loc - cls.TRADING_DAY_COUNT

        cls.trading_days = all_days[start_loc:end_loc + 1]


class WithTmpDir(object):
    """
    ZiplineTestCase mixing providing cls.tmpdir as a class-level fixture.

    After init_class_fixtures has been called, `cls.tmpdir` is populated with
    a `testfixtures.TempDirectory` object whose path is `cls.TMP_DIR_PATH`.

    The default value of TMP_DIR_PATH is None which will create a unique
    directory in /tmp.
    """
    TMP_DIR_PATH = None

    @classmethod
    def init_class_fixtures(cls):
        super(WithTmpDir, cls).init_class_fixtures()
        cls.tmpdir = cls.enter_class_context(tmp_dir(cls.TMP_DIR_PATH))


class WithInstanceTmpDir(object):
    """
    ZiplineTestCase mixing providing self.tmpdir as an instance-level fixture.

    After init_instance_fixtures has been called, `self.tmpdir` is populated
    with a `testfixtures.TempDirectory` object whose path is
    `cls.TMP_DIR_PATH`.

    The default value of TMP_DIR_PATH is None which will create a unique
    directory in /tmp.
    """
    TMP_DIR_PATH = None

    def init_instance_fixtures(self):
        super(WithInstanceTmpDir, self).init_instance_fixtures()
        self.tmpdir = self.enter_instance_context(tmp_dir(self.TMP_DIR_PATH))


class WithBcolzDailyBarReader(WithTmpDir, WithSimParams):
    """
    ZiplineTestCase mixin providing cls.bcolz_daily_bar_path,
    cls.bcolz_daily_bar_ctable, and cls.bcolz_daily_bar_reader class level
    fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_daily_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_DAILY_BAR_PATH)`.
    - `cls.bcolz_daily_bar_ctable` is populated with data returned from
      `cls.make_daily_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_daily_bar_data`.
    - `cls.bcolz_daily_bar_reader` is a daily bar reader pointing to the
      directory that was just written to.

    See Also
    --------
    WithBcolzMinuteBarReader
    WithDataPortal
    """
    BCOLZ_DAILY_BAR_PATH = 'daily_equity_pricing.bcolz'
    BCOLZ_DAILY_BAR_LOOKBACK_DAYS = 0
    BCOLZ_DAILY_BAR_FROM_CSVS = False
    BCOLZ_DAILY_BAR_USE_FULL_CALENDAR = False

    @classmethod
    def make_daily_bar_data(cls):
        return create_daily_bar_data(
            cls.bcolz_daily_bar_days,
            cls.asset_finder.sids,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithBcolzDailyBarReader, cls).init_class_fixtures()
        cls.bcolz_daily_bar_path = p = cls.tmpdir.makedir(
            cls.BCOLZ_DAILY_BAR_PATH,
        )
        cls.bcolz_daily_bar_days = days = (
            cls.env.trading_days
            if cls.BCOLZ_DAILY_BAR_USE_FULL_CALENDAR else
            cls.env.days_in_range(
                cls.env.trading_days[
                    cls.env.get_index(cls.sim_params.period_start) -
                    cls.BCOLZ_DAILY_BAR_LOOKBACK_DAYS
                ],
                cls.sim_params.period_end,
            )
        )
        cls.bcolz_daily_bar_ctable = t = getattr(
            BcolzDailyBarWriter(p, days),
            'write_csvs' if cls.BCOLZ_DAILY_BAR_FROM_CSVS else 'write',
        )(cls.make_daily_bar_data())

        cls.bcolz_daily_bar_reader = BcolzDailyBarReader(t)


class WithBcolzMinuteBarReader(WithTmpDir, WithSimParams):
    """
    ZiplineTestCase mixin providing cls.bcolz_minute_bar_path,
    cls.bcolz_minute_bar_ctable, and cls.bcolz_minute_bar_reader class level
    fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_minute_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_MINUTE_BAR_PATH)`.
    - `cls.bcolz_minute_bar_ctable` is populated with data returned from
      `cls.make_minute_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_minute_bar_data`.
    - `cls.bcolz_minute_bar_reader` is a minute bar reader pointing to the
      directory that was just written to.

    See Also
    --------
    WithBcolzDailyBarReader
    WithDataPortal
    """
    BCOLZ_MINUTE_BAR_PATH = 'minute_equity_pricing.bcolz'
    BCOLZ_MINUTE_BAR_LOOKBACK_DAYS = 0
    BCOLZ_MINUTE_BAR_USE_FULL_CALENDAR = False

    @classmethod
    def make_minute_bar_data(cls):
        return create_minute_bar_data(
            cls.env.minutes_for_days_in_range(
                cls.bcolz_minute_bar_days[0],
                cls.bcolz_minute_bar_days[-1],
            ),
            cls.asset_finder.sids,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithBcolzMinuteBarReader, cls).init_class_fixtures()
        cls.bcolz_minute_bar_path = p = cls.tmpdir.makedir(
            cls.BCOLZ_MINUTE_BAR_PATH,
        )
        cls.bcolz_minute_bar_days = days = (
            cls.env.trading_days
            if cls.BCOLZ_MINUTE_BAR_USE_FULL_CALENDAR else
            cls.env.days_in_range(
                cls.env.trading_days[
                    cls.env.get_index(cls.sim_params.period_start) -
                    cls.BCOLZ_MINUTE_BAR_LOOKBACK_DAYS
                ],
                cls.sim_params.period_end,
            )
        )
        writer = BcolzMinuteBarWriter(
            days[0],
            p,
            cls.env.open_and_closes.market_open.loc[days],
            cls.env.open_and_closes.market_close.loc[days],
            US_EQUITIES_MINUTES_PER_DAY
        )
        cls.bcolz_minute_bar_data = df_dict = cls.make_minute_bar_data()
        for sid, df in iteritems(df_dict):
            writer.write(sid, df)

        cls.bcolz_minute_bar_reader = BcolzMinuteBarReader(p)


class WithAdjustmentReader(WithBcolzDailyBarReader):
    """
    ZiplineTestCase mixin providing cls.adjustment_reader as a class level
    fixture.

    After init_class_fixtures has been called, `cls.adjustment_reader` will be
    populated with a new SQLiteAdjustmentReader object. The data that will be
    written can be passed by overriding `make_{field}_data` where field may
    be `splits`, `mergers` `dividends`, or `stock_dividends`.
    The daily bar reader used for this adjustment reader may be customized
    by overriding `make_adjustment_writer_daily_bar_reader`. This is useful
    to providing a `MockDailyBarReader`.

    For more advanced configuration, `make_adjustment_writer` may be
    overwritten directly.

    See Also
    --------
    zipline.testing.MockDailyBarReader
    """
    @classmethod
    def _make_data(cls):
        return None

    make_splits_data = _make_data
    make_mergers_data = _make_data
    make_dividends_data = _make_data
    make_stock_dividends_data = _make_data

    del _make_data

    @classmethod
    def make_adjustment_writer(cls):
        return SQLiteAdjustmentWriter(
            cls.adjustment_db_conn,
            cls.make_adjustment_writer_daily_bar_reader(),
            cls.bcolz_daily_bar_days,
        )

    @classmethod
    def make_adjustment_writer_daily_bar_reader(cls):
        return cls.bcolz_daily_bar_reader

    @classmethod
    def init_class_fixtures(cls):
        super(WithAdjustmentReader, cls).init_class_fixtures()
        cls.adjustment_db_conn = conn = sqlite3.connect(':memory:')
        cls.make_adjustment_writer().write(
            splits=cls.make_splits_data(),
            mergers=cls.make_mergers_data(),
            dividends=cls.make_dividends_data(),
            stock_dividends=cls.make_stock_dividends_data(),
        )
        cls.adjustment_reader = SQLiteAdjustmentReader(conn)


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


class WithDataPortal(WithAdjustmentReader, WithBcolzMinuteBarReader):
    """
    ZiplineTestCase mixin providing self.data_portal as an instance level
    fixture.

    After init_instance_fixtures has been called, `self.data_portal` will be
    populated with a new data portal created by passing in the class's
    trading env, `cls.bcolz_minute_bar_reader`, `cls.bcolz_daily_bar_reader`,
    and `cls.adjustment_reader`.

    Any of the three readers may be set to false by overriding:
    DATA_PORTAL_USE_DAILY_DATA = False
    DATA_PORTAL_USE_MINUTE_DATA = False
    DATA_PORTAL_USE_ADJUSTMENTS = False
    """
    DATA_PORTAL_USE_DAILY_DATA = True
    DATA_PORTAL_USE_MINUTE_DATA = True
    DATA_PORTAL_USE_ADJUSTMENTS = True

    @classmethod
    def make_data_portal(cls):
        return DataPortal(
            cls.env,
            equity_daily_reader=(
                cls.bcolz_daily_bar_reader
                if cls.DATA_PORTAL_USE_DAILY_DATA else
                None
            ),
            equity_minute_reader=(
                cls.bcolz_minute_bar_reader
                if cls.DATA_PORTAL_USE_MINUTE_DATA else
                None
            ),
            adjustment_reader=(
                cls.adjustment_reader
                if cls.DATA_PORTAL_USE_ADJUSTMENTS else
                None
            ),
        )

    def init_instance_fixtures(self):
        super(WithDataPortal, self).init_instance_fixtures()
        self.data_portal = self.make_data_portal()


class WithResponses(object):
    """
    ZiplineTestCase mixin that provides self.responses as an instance
    fixture.

    After init_instance_fixtures has been called, `self.responses` will be
    a new `responses.RequestsMock` object. Users may add new endpoints to this
    with the `self.responses.add` method.
    """
    def init_instance_fixtures(self):
        super(WithResponses, self).init_instance_fixtures()
        self.responses = self.enter_instance_context(
            responses.RequestsMock(),
        )
