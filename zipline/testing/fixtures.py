from abc import ABCMeta, abstractproperty
import sqlite3
from unittest import TestCase

from contextlib2 import ExitStack
from logbook import NullHandler, Logger
from nose_parameterized import parameterized
from pandas.util.testing import assert_series_equal
from six import with_metaclass
from toolz import flip
import numpy as np
import pandas as pd
import responses

from .core import (
    create_daily_bar_data,
    create_minute_bar_data,
    tmp_dir,
)
from ..data.data_portal import DataPortal
from ..data.us_equity_pricing import (
    SQLiteAdjustmentReader,
    SQLiteAdjustmentWriter,
)
from ..data.us_equity_pricing import (
    BcolzDailyBarReader,
    BcolzDailyBarWriter,
)
from ..data.minute_bars import (
    BcolzMinuteBarReader,
    BcolzMinuteBarWriter,
    US_EQUITIES_MINUTES_PER_DAY
)

from ..finance.trading import TradingEnvironment
from ..utils import tradingcalendar, factory
from ..utils.classproperty import classproperty
from ..utils.final import FinalMeta, final
from ..utils.metautils import with_metaclasses
from .core import tmp_asset_finder, make_simple_equity_info, gen_calendars
from zipline.pipeline import Pipeline, SimplePipelineEngine
from zipline.pipeline.loaders.testing import make_seeded_random_loader
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
    _in_setup = False

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
        if cls._in_setup:
            raise ValueError(
                'Called init_class_fixtures from init_instance_fixtures.'
                'Did you write super(..., self).init_class_fixtures() instead'
                ' of super(..., self).init_instance_fixtures()?',
            )
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
        if cls._in_setup:
            raise ValueError(
                'Attempted to enter a class context in init_instance_fixtures.'
                '\nDid you mean to call enter_instance_context?',
            )
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
        if cls._in_setup:
            raise ValueError(
                'Attempted to add a class callback in init_instance_fixtures.'
                '\nDid you mean to call add_instance_callback?',
            )
        return cls._class_teardown_stack.callback(callback)

    @final
    def setUp(self):
        type(self)._in_setup = True
        self._pre_setup_attrs = set(vars(self))
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
        finally:
            type(self)._in_setup = False

    def init_instance_fixtures(self):
        self._init_instance_fixtures_was_called = True

    @final
    def tearDown(self):
        self._instance_teardown_stack.close()
        for attr in set(vars(self)) - self._pre_setup_attrs:
            delattr(self, attr)

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


def alias(attr_name):
    """Make a fixture attribute an alias of another fixture's attribute by
    default.

    Parameters
    ----------
    attr_name : str
        The name of the attribute to alias.

    Returns
    -------
    p : classproperty
        A class property that does the property aliasing.

    Examples
    --------
    >>> class C(object):
    ...     attr = 1
    ...
    >>> class D(object):
    ...     attr_alias = alias('attr')
    ...
    >>> D.attr
    1
    >>> D.attr_alias
    1
    >>> class E(D):
    ...     attr_alias = 2
    ...
    >>> E.attr
    1
    >>> E.attr_alias
    2
    """
    return classproperty(flip(getattr, attr_name))


class WithDefaultDateBounds(object):
    """
    ZiplineTestCase mixin which makes it possible to synchronize date bounds
    across fixtures.

    Attributes
    ----------
    START_DATE : datetime
    END_DATE : datetime
        The date bounds to be used for fixtures that want to have consistent
        dates.
    """
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-12-29', tz='utc')


class WithLogger(object):
    """
    ZiplineTestCase mixin providing cls.log_handler as an instance-level
    fixture.

    After init_instance_fixtures has been called `self.log_handler` will be a
    new ``logbook.NullHandler``.

    Methods
    -------
    make_log_handler() -> logbook.LogHandler
        A class method which constructs the new log handler object. By default
        this will construct a ``NullHandler``.
    """
    make_log_handler = NullHandler

    @classmethod
    def init_class_fixtures(cls):
        super(WithLogger, cls).init_class_fixtures()
        cls.log = Logger()
        cls.log_handler = cls.enter_class_context(
            cls.make_log_handler().applicationbound(),
        )


class WithAssetFinder(WithDefaultDateBounds):
    """
    ZiplineTestCase mixin providing cls.asset_finder as a class-level fixture.

    After init_class_fixtures has been called, `cls.asset_finder` is populated
    with an AssetFinder.

    Attributes
    ----------
    ASSET_FINDER_EQUITY_SIDS : iterable[int]
        The default sids to construct equity data for.
    ASSET_FINDER_EQUITY_SYMBOLS : iterable[str]
        The default symbols to use for the equities.
    ASSET_FINDER_EQUITY_START_DATE : datetime
        The default start date to create equity data for. This defaults to
        ``START_DATE``.
    ASSET_FINDER_EQUITY_END_DATE : datetime
        The default end date to create equity data for. This defaults to
        ``END_DATE``.

    Methods
    -------
    make_equity_info() -> pd.DataFrame
        A class method which constructs the dataframe of equity info to write
        to the class's asset db. By default this is empty.
    make_futures_info() -> pd.DataFrame
        A class method which constructs the dataframe of futures contract info
        to write to the class's asset db. By default this is empty.
    make_exchanges_info() -> pd.DataFrame
        A class method which constructs the dataframe of exchange information
        to write to the class's assets db. By default this is empty.
    make_root_symbols_info() -> pd.DataFrame
        A class method which constructs the dataframe of root symbols
        information to write to the class's assets db. By default this is
        empty.
    make_asset_finder() -> pd.DataFrame
        A class method which constructs the actual asset finder object to use
        for the class. If this method is overridden then the ``make_*_info``
        methods may not be respected.

    See Also
    --------
    zipline.testing.make_simple_equity_info
    zipline.testing.make_jagged_equity_info
    zipline.testing.make_rotating_equity_info
    zipline.testing.make_future_info
    zipline.testing.make_commodity_future_info
    """
    ASSET_FINDER_EQUITY_SIDS = ord('A'), ord('B'), ord('C')
    ASSET_FINDER_EQUITY_SYMBOLS = None
    ASSET_FINDER_EQUITY_START_DATE = alias('START_DATE')
    ASSET_FINDER_EQUITY_END_DATE = alias('END_DATE')

    @classmethod
    def _make_info(cls):
        return None

    make_futures_info = _make_info
    make_exchanges_info = _make_info
    make_root_symbols_info = _make_info

    del _make_info

    @classmethod
    def make_equity_info(cls):
        return make_simple_equity_info(
            cls.ASSET_FINDER_EQUITY_SIDS,
            cls.ASSET_FINDER_EQUITY_START_DATE,
            cls.ASSET_FINDER_EQUITY_END_DATE,
            cls.ASSET_FINDER_EQUITY_SYMBOLS,
        )

    @classmethod
    def make_asset_finder(cls):
        return cls.enter_class_context(tmp_asset_finder(
            equities=cls.make_equity_info(),
            futures=cls.make_futures_info(),
            exchanges=cls.make_exchanges_info(),
            root_symbols=cls.make_root_symbols_info(),
        ))

    @classmethod
    def init_class_fixtures(cls):
        super(WithAssetFinder, cls).init_class_fixtures()
        cls.asset_finder = cls.make_asset_finder()


class WithTradingEnvironment(WithAssetFinder):
    """
    ZiplineTestCase mixin providing cls.env as a class-level fixture.

    After ``init_class_fixtures`` has been called, `cls.env` is populated
    with a trading environment whose `asset_finder` is the result of
    `cls.make_asset_finder`.

    Attributes
    ----------
    TRADING_ENV_MIN_DATE : datetime
        The min_date to forward to the constructed TradingEnvironment.
    TRADING_ENV_MAX_DATE : datetime
        The max date to forward to the constructed TradingEnvironment.
    TRADING_ENV_TRADING_CALENDAR : pd.DatetimeIndex
        The trading calendar to use for the class's TradingEnvironment.

    Methods
    -------
    make_load_function() -> callable
        A class method that returns the ``load`` argument to pass to the
        constructor of ``TradingEnvironment`` for this class.
        The signature for the callable returned is:
        ``(datetime, pd.DatetimeIndex, str) -> (pd.Series, pd.DataFrame)``
    make_trading_environment() -> TradingEnvironment
        A class method that constructs the trading environment for the class.
        If this is overridden then ``make_load_function`` or the class
        attributes may not be respected.

    See Also
    --------
    :class:`zipline.finance.trading.TradingEnvironment`
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

    Attributes
    ----------
    SIM_PARAMS_YEAR : int
    SIM_PARAMS_CAPITAL_BASE : float
    SIM_PARAMS_NUM_DAYS : int
    SIM_PARAMS_DATA_FREQUENCY : {'daily', 'minute'}
    SIM_PARAMS_EMISSION_RATE : {'daily', 'minute'}
        Forwarded to ``factory.create_simulation_parameters``.

    SIM_PARAMS_START : datetime
    SIM_PARAMS_END : datetime
        Forwarded to ``factory.create_simulation_parameters``. If not
        explicitly overridden these will be ``START_DATE`` and ``END_DATE``

    See Also
    --------
    zipline.utils.factory.create_simulation_parameters
    """
    SIM_PARAMS_YEAR = None
    SIM_PARAMS_CAPITAL_BASE = 1.0e5
    SIM_PARAMS_NUM_DAYS = None
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    SIM_PARAMS_EMISSION_RATE = 'daily'

    SIM_PARAMS_START = alias('START_DATE')
    SIM_PARAMS_END = alias('END_DATE')

    @classmethod
    def make_simparams(cls):
        return factory.create_simulation_parameters(
            year=cls.SIM_PARAMS_YEAR,
            start=cls.SIM_PARAMS_START,
            end=cls.SIM_PARAMS_END,
            num_days=cls.SIM_PARAMS_NUM_DAYS,
            capital_base=cls.SIM_PARAMS_CAPITAL_BASE,
            data_frequency=cls.SIM_PARAMS_DATA_FREQUENCY,
            emission_rate=cls.SIM_PARAMS_EMISSION_RATE,
            env=cls.env,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithSimParams, cls).init_class_fixtures()
        cls.sim_params = cls.make_simparams()


class WithNYSETradingDays(object):
    """
    ZiplineTestCase mixin providing cls.trading_days as a class-level fixture.

    After init_class_fixtures has been called, `cls.trading_days` is populated
    with a DatetimeIndex containing NYSE calendar trading days ranging from:

    (DATA_MAX_DAY - (cls.TRADING_DAY_COUNT) -> DATA_MAX_DAY)

    Attributes
    ----------
    DATA_MAX_DAY : datetime
        The most recent trading day in the calendar.
    TRADING_DAY_COUNT : int
        The number of days to put in the calendar. The default value of
        ``TRADING_DAY_COUNT`` is 126 (half a trading-year). Inheritors can
        override TRADING_DAY_COUNT to request more or less data.
    """
    DATA_MIN_DAY = alias('START_DATE')
    DATA_MAX_DAY = alias('END_DATE')

    @classmethod
    def init_class_fixtures(cls):
        super(WithNYSETradingDays, cls).init_class_fixtures()

        all_days = tradingcalendar.trading_days
        start_loc = all_days.get_loc(cls.DATA_MIN_DAY, 'bfill')
        end_loc = all_days.get_loc(cls.DATA_MAX_DAY, 'ffill')

        cls.trading_days = all_days[start_loc:end_loc + 1]


class WithTmpDir(object):
    """
    ZiplineTestCase mixing providing cls.tmpdir as a class-level fixture.

    After init_class_fixtures has been called, `cls.tmpdir` is populated with
    a `testfixtures.TempDirectory` object whose path is `cls.TMP_DIR_PATH`.

    Attributes
    ----------
    TMP_DIR_PATH : str
        The path to the new directory to create. By default this is None
        which will create a unique directory in /tmp.
    """
    TMP_DIR_PATH = None

    @classmethod
    def init_class_fixtures(cls):
        super(WithTmpDir, cls).init_class_fixtures()
        cls.tmpdir = cls.enter_class_context(
            tmp_dir(path=cls.TMP_DIR_PATH),
        )


class WithInstanceTmpDir(object):
    """
    ZiplineTestCase mixing providing self.tmpdir as an instance-level fixture.

    After init_instance_fixtures has been called, `self.tmpdir` is populated
    with a `testfixtures.TempDirectory` object whose path is
    `cls.TMP_DIR_PATH`.

    Attributes
    ----------
    INSTANCE_TMP_DIR_PATH : str
        The path to the new directory to create. By default this is None
        which will create a unique directory in /tmp.
    """
    INSTANCE_TMP_DIR_PATH = None

    def init_instance_fixtures(self):
        super(WithInstanceTmpDir, self).init_instance_fixtures()
        self.instance_tmpdir = self.enter_instance_context(
            tmp_dir(path=self.INSTANCE_TMP_DIR_PATH),
        )


class WithBcolzDailyBarReader(WithTradingEnvironment, WithTmpDir):
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

    Attributes
    ----------
    BCOLZ_DAILY_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    BCOLZ_DAILY_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day. This is used
        when a test needs to use history, in which case this should be set to
        the largest history window that will be
        requested.
    BCOLZ_DAILY_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``bcolz_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``BCOLZ_DAILY_BAR_LOOKBACK_DAYS``.
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD : int
        If this flag is set, use the value as the `read_all_threshold`
        parameter to BcolzDailyBarReader, otherwise use the default value.

    Methods
    -------
    make_daily_bar_data() -> iterable[(int, pd.DataFrame)]
        A class method that returns an iterator of (sid, dataframe) pairs
        which will be written to the bcolz files that the class's
        ``BcolzDailyBarReader`` will read from. By default this creates
        some simple sythetic data with
        :func:`~zipline.testing.create_daily_bar_data`

    See Also
    --------
    WithBcolzMinuteBarReader
    WithDataPortal
    zipline.testing.create_daily_bar_data
    """
    BCOLZ_DAILY_BAR_PATH = 'daily_equity_pricing.bcolz'
    BCOLZ_DAILY_BAR_LOOKBACK_DAYS = 0
    BCOLZ_DAILY_BAR_USE_FULL_CALENDAR = False
    BCOLZ_DAILY_BAR_START_DATE = alias('START_DATE')
    BCOLZ_DAILY_BAR_END_DATE = alias('END_DATE')
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD = None
    # allows WithBcolzDailyBarReaderFromCSVs to call the `write_csvs` method
    # without needing to reimplement `init_class_fixtures`
    _write_method_name = 'write'

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
        if cls.BCOLZ_DAILY_BAR_USE_FULL_CALENDAR:
            days = cls.env.trading_days
        else:
            days = cls.env.days_in_range(
                cls.env.trading_days[
                    cls.env.get_index(cls.BCOLZ_DAILY_BAR_START_DATE) -
                    cls.BCOLZ_DAILY_BAR_LOOKBACK_DAYS
                ],
                cls.BCOLZ_DAILY_BAR_END_DATE,
            )
        cls.bcolz_daily_bar_days = days
        cls.bcolz_daily_bar_ctable = t = getattr(
            BcolzDailyBarWriter(p, days),
            cls._write_method_name,
        )(cls.make_daily_bar_data())

        if cls.BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD is not None:
            cls.bcolz_daily_bar_reader = BcolzDailyBarReader(
                t, cls.BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD)
        else:
            cls.bcolz_daily_bar_reader = BcolzDailyBarReader(t)


class WithBcolzDailyBarReaderFromCSVs(WithBcolzDailyBarReader):
    """
    ZiplineTestCase mixin that provides cls.bcolz_daily_bar_reader from a
    mapping of sids to CSV file paths.
    """
    _write_method_name = 'write_csvs'


class WithBcolzMinuteBarReader(WithTradingEnvironment, WithTmpDir):
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

    Attributes
    ----------
    BCOLZ_MINUTE_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    BCOLZ_MINUTE_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day.
        This is used when a test needs to use history, in which case this
        should be set to the largest history window that will be requested.
    BCOLZ_MINUTE_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``bcolz_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``BCOLZ_MINUTE_BAR_LOOKBACK_DAYS``.

    Methods
    -------
    make_minute_bar_data() -> iterable[(int, pd.DataFrame)]
        A class method that returns a dict mapping sid to dataframe
        which will be written to the bcolz files that the class's
        ``BcolzMinuteBarReader`` will read from. By default this creates
        some simple sythetic data with
        :func:`~zipline.testing.create_minute_bar_data`

    See Also
    --------
    WithBcolzDailyBarReader
    WithDataPortal
    zipline.testing.create_minute_bar_data
    """
    BCOLZ_MINUTE_BAR_PATH = 'minute_equity_pricing.bcolz'
    BCOLZ_MINUTE_BAR_LOOKBACK_DAYS = 0
    BCOLZ_MINUTE_BAR_USE_FULL_CALENDAR = False
    BCOLZ_MINUTE_BAR_START_DATE = alias('START_DATE')
    BCOLZ_MINUTE_BAR_END_DATE = alias('END_DATE')

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
        if cls.BCOLZ_MINUTE_BAR_USE_FULL_CALENDAR:
            days = cls.env.trading_days
        else:
            days = cls.env.days_in_range(
                cls.env.trading_days[
                    cls.env.get_index(cls.BCOLZ_MINUTE_BAR_START_DATE) -
                    cls.BCOLZ_MINUTE_BAR_LOOKBACK_DAYS
                ],
                cls.BCOLZ_MINUTE_BAR_END_DATE,
            )
        cls.bcolz_minute_bar_days = days
        writer = BcolzMinuteBarWriter(
            days[0],
            p,
            cls.env.open_and_closes.market_open.loc[days],
            cls.env.open_and_closes.market_close.loc[days],
            US_EQUITIES_MINUTES_PER_DAY
        )
        writer.write(cls.make_minute_bar_data())

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

    Methods
    -------
    make_splits_data() -> pd.DataFrame
        A class method that returns a dataframe of splits data to write to the
        class's adjustment db. By default this is empty.
    make_mergers_data() -> pd.DataFrame
        A class method that returns a dataframe of mergers data to write to the
        class's adjustment db. By default this is empty.
    make_dividends_data() -> pd.DataFrame
        A class method that returns a dataframe of dividends data to write to
        the class's adjustment db. By default this is empty.
    make_stock_dividends_data() -> pd.DataFrame
        A class method that returns a dataframe of stock dividends data to
        write to the class's adjustment db. By default this is empty.
    make_adjustment_writer_daily_bar_reader() -> pd.DataFrame
        A class method that returns the daily bar reader to use for the class's
        adjustment writer. By default this is the class's actual
        ``bcolz_daily_bar_reader`` as inherited from
        ``WithBcolzDailyBarReader``. This should probably not be overridden;
        however, some tests used a ``MockDailyBarReader`` for this.
    make_adjustment_writer(conn: sqlite3.Connection) -> AdjustmentWriter
        A class method that constructs the adjustment which will be used
        to write the data into the connection to be used by the class's
        adjustment reader.

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
    def make_adjustment_writer(cls, conn):
        return SQLiteAdjustmentWriter(
            conn,
            cls.make_adjustment_writer_daily_bar_reader(),
            cls.bcolz_daily_bar_days,
        )

    @classmethod
    def make_adjustment_writer_daily_bar_reader(cls):
        return cls.bcolz_daily_bar_reader

    @classmethod
    def init_class_fixtures(cls):
        super(WithAdjustmentReader, cls).init_class_fixtures()
        conn = sqlite3.connect(':memory:')
        cls.make_adjustment_writer(conn).write(
            splits=cls.make_splits_data(),
            mergers=cls.make_mergers_data(),
            dividends=cls.make_dividends_data(),
            stock_dividends=cls.make_stock_dividends_data(),
        )
        cls.adjustment_reader = SQLiteAdjustmentReader(conn)


class WithPipelineEventDataLoader(
        with_metaclasses((type(ZiplineTestCase), ABCMeta), WithAssetFinder)):
    """
    ZiplineTestCase mixin providing common test methods/behaviors for event
    data loaders.

    Attributes
    ----------
    loader_type : PipelineLoader
        The type of loader to use. This must be overridden by subclasses.

    Methods
    -------
    get_sids() -> iterable[int]
        Class method which returns the sids that need to be available to the
        tests.
    get_dataset() -> dict[int -> pd.DataFrmae]
        Class method which returns a mapping from sid to data for that sid.
        By default this is empty for every sid.
    pipeline_event_loader_args(dates: pd.DatetimeIndex) -> tuple[any]
        The arguments to pass to the ``loader_type`` to construct the pipeline
        loader for this test.
    """
    @classmethod
    def get_sids(cls):
        return range(0, 5)

    @classmethod
    def get_dataset(cls):
        return {sid: pd.DataFrame() for sid in cls.get_sids()}

    @abstractproperty
    def loader_type(self):
        raise NotImplementedError('loader_type')

    @classmethod
    def make_equity_info(cls):
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
                           dates,
                           dtype_name,
                           missing_dtype):
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
        dtype_name: str
            The name of the dtype of the values in `vals`.
        missing_dtype: str
            The name of the value that should be used as the missing value
            for the dtype of `vals` - e.g., 'NaN' for floats.
        """
        frame = pd.DataFrame({sid: get_values_for_date_ranges(
            zip_date_index_with_vals,
            vals[sid],
            pd.DatetimeIndex(list(zip(*date_intervals[sid]))[0]),
            pd.DatetimeIndex(list(zip(*date_intervals[sid]))[1]),
            dates
        ).astype(dtype_name) for sid in self.get_sids()[:-1]})
        frame[self.get_sids()[-1]] = zip_date_index_with_vals(
            dates, [missing_dtype] * len(dates)
        ).astype(dtype_name)
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
                assert_series_equal(result[col_name].unstack(1)[sid],
                                    cols[col_name][sid],
                                    check_names=False)


class WithSeededRandomPipelineEngine(WithNYSETradingDays, WithAssetFinder):
    """
    ZiplineTestCase mixin providing class-level fixtures for running pipelines
    against deterministically-generated random data.

    Attributes
    ----------
    SEEDED_RANDOM_PIPELINE_SEED : int
        Fixture input. Random seed used to initialize the random state loader.
    seeded_random_loader : SeededRandomLoader
        Fixture output. Loader capable of providing columns for
        zipline.pipeline.data.testing.TestingDataSet.
    seeded_random_engine : SimplePipelineEngine
        Fixture output.  A pipeline engine that will use seeded_random_loader
        as its only data provider.

    Methods
    -------
    run_pipeline(start_date, end_date)
        Run a pipeline with self.seeded_random_engine.

    See Also
    --------
    zipline.pipeline.loaders.synthetic.SeededRandomLoader
    zipline.pipeline.loaders.testing.make_seeded_random_loader
    zipline.pipeline.engine.SimplePipelineEngine
    """
    SEEDED_RANDOM_PIPELINE_SEED = 42

    @classmethod
    def init_class_fixtures(cls):
        super(WithSeededRandomPipelineEngine, cls).init_class_fixtures()
        cls._sids = cls.asset_finder.sids
        cls.seeded_random_loader = loader = make_seeded_random_loader(
            cls.SEEDED_RANDOM_PIPELINE_SEED,
            cls.trading_days,
            cls._sids,
        )
        cls.seeded_random_engine = SimplePipelineEngine(
            get_loader=lambda column: loader,
            calendar=cls.trading_days,
            asset_finder=cls.asset_finder,
        )

    def raw_expected_values(self, column, start_date, end_date):
        """
        Get an array containing the raw values we expect to be produced for the
        given dates between start_date and end_date, inclusive.
        """
        all_values = self.seeded_random_loader.values(
            column.dtype,
            self.trading_days,
            self._sids,
        )
        row_slice = self.trading_days.slice_indexer(start_date, end_date)
        return all_values[row_slice]

    def run_pipeline(self, pipeline, start_date, end_date):
        """
        Run a pipeline with self.seeded_random_engine.
        """
        if start_date not in self.trading_days:
            raise AssertionError("Start date not in calendar: %s" % start_date)
        if end_date not in self.trading_days:
            raise AssertionError("Start date not in calendar: %s" % start_date)
        return self.seeded_random_engine.run_pipeline(
            pipeline,
            start_date,
            end_date,
        )


class WithDataPortal(WithBcolzMinuteBarReader, WithAdjustmentReader):
    """
    ZiplineTestCase mixin providing self.data_portal as an instance level
    fixture.

    After init_instance_fixtures has been called, `self.data_portal` will be
    populated with a new data portal created by passing in the class's
    trading env, `cls.bcolz_minute_bar_reader`, `cls.bcolz_daily_bar_reader`,
    and `cls.adjustment_reader`.

    Attributes
    ----------
    DATA_PORTAL_USE_DAILY_DATA : bool
        Should the daily bar reader be used? Defaults to True.
    DATA_PORTAL_USE_MINUTE_DATA : bool
        Should the minute bar reader be used? Defaults to True.
    DATA_PORTAL_USE_ADJUSTMENTS : bool
        Should the adjustment reader be used? Defaults to True.

    Methods
    -------
    make_data_portal() -> DataPortal
        Method which returns the data portal to be used for each test case.
        If this is overridden, the ``DATA_PORTAL_USE_*`` attributes may not
        be respected.
    """
    DATA_PORTAL_USE_DAILY_DATA = True
    DATA_PORTAL_USE_MINUTE_DATA = True
    DATA_PORTAL_USE_ADJUSTMENTS = True

    def make_data_portal(self):
        return DataPortal(
            self.env,
            equity_daily_reader=(
                self.bcolz_daily_bar_reader
                if self.DATA_PORTAL_USE_DAILY_DATA else
                None
            ),
            equity_minute_reader=(
                self.bcolz_minute_bar_reader
                if self.DATA_PORTAL_USE_MINUTE_DATA else
                None
            ),
            adjustment_reader=(
                self.adjustment_reader
                if self.DATA_PORTAL_USE_ADJUSTMENTS else
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
        [['2014-01-01', '2014-01-04'],
         ['2014-01-05', '2014-01-15'],
         ['2014-01-16', '2014-01-20'],
         ['2014-01-21', '2014-01-31']],
        [['2014-01-01', '2014-01-04'],
         ['2014-01-05', '2014-01-09'],
         ['2014-01-10', '2014-01-15'],
         ['2014-01-16', '2014-01-20'],
         ['2014-01-21', '2014-01-31']],
        [['2014-01-01', '2014-01-04'],
         ['2014-01-05', '2014-01-10'],
         ['2014-01-11', '2014-01-14'],
         ['2014-01-15', '2014-01-20'],
         ['2014-01-21', '2014-01-31']],
        [['2014-01-01', '2014-01-04'],
         ['2014-01-05', '2014-01-10'],
         ['2014-01-11', '2014-01-15'],
         ['2014-01-16', '2014-01-31']]
    ]

    next_dates = [
        ['NaT', '2014-01-15', '2014-01-20', 'NaT'],
        ['NaT', '2014-01-20', '2014-01-15', '2014-01-20', 'NaT'],
        ['NaT', '2014-01-10', 'NaT', '2014-01-20', 'NaT'],
        ['NaT', '2014-01-10', '2014-01-15', 'NaT'],
        ['NaT']
    ]

    prev_date_intervals = [
        [['2014-01-01', '2014-01-14'],
         ['2014-01-15', '2014-01-19'],
         ['2014-01-20', '2014-01-31']],
        [['2014-01-01', '2014-01-14'],
         ['2014-01-15', '2014-01-19'],
         ['2014-01-20', '2014-01-31']],
        [['2014-01-01', '2014-01-09'],
         ['2014-01-10', '2014-01-19'],
         ['2014-01-20', '2014-01-31']],
        [['2014-01-01', '2014-01-09'],
         ['2014-01-10', '2014-01-14'],
         ['2014-01-15', '2014-01-31']]
    ]

    prev_dates = [
        ['NaT', '2014-01-15', '2014-01-20'],
        ['NaT', '2014-01-15', '2014-01-20'],
        ['NaT', '2014-01-10', '2014-01-20'],
        ['NaT', '2014-01-10', '2014-01-15'],
        ['NaT']
    ]

    def get_expected_previous_event_dates(self, dates, dtype_name,
                                          missing_dtype):
        return self.get_sids_to_frames(
            zip_with_dates,
            self.prev_dates,
            self.prev_date_intervals,
            dates,
            dtype_name,
            missing_dtype
        )

    def get_expected_next_event_dates(self, dates, dtype_name, missing_dtype):
        return self.get_sids_to_frames(
            zip_with_dates,
            self.next_dates,
            self.next_date_intervals,
            dates,
            dtype_name,
            missing_dtype
        )
