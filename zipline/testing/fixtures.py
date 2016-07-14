import sqlite3
from unittest import TestCase

from contextlib2 import ExitStack
from logbook import NullHandler, Logger
from six import with_metaclass
from toolz import flip
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
from ..utils import factory
from ..utils.classproperty import classproperty
from ..utils.final import FinalMeta, final
from .core import tmp_asset_finder, make_simple_equity_info
from zipline.pipeline import SimplePipelineEngine
from zipline.pipeline.loaders.testing import make_seeded_random_loader
from zipline.utils.calendars import (
    get_calendar,
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
        # We need to get this before it's deleted by the loop.
        stack = cls._class_teardown_stack
        for name in set(vars(cls)) - cls._static_class_attributes:
            # Remove all of the attributes that were added after the class was
            # constructed. This cleans up any large test data that is class
            # scoped while still allowing subclasses to access class level
            # attributes.
            delattr(cls, name)
        stack.close()

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
        # We need to get this before it's deleted by the loop.
        stack = self._instance_teardown_stack
        for attr in set(vars(self)) - self._pre_setup_attrs:
            delattr(self, attr)
        stack.close()

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
    >>> class D(C):
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
    make_asset_finder_db_url() -> string
        A class method which returns the URL at which to create the SQLAlchemy
        engine. By default provides a URL for an in-memory database.
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
    def make_asset_finder_db_url(cls):
        return 'sqlite:///:memory:'

    @classmethod
    def make_asset_finder(cls):
        return cls.enter_class_context(tmp_asset_finder(
            url=cls.make_asset_finder_db_url(),
            equities=cls.make_equity_info(),
            futures=cls.make_futures_info(),
            exchanges=cls.make_exchanges_info(),
            root_symbols=cls.make_root_symbols_info(),
        ))

    @classmethod
    def init_class_fixtures(cls):
        super(WithAssetFinder, cls).init_class_fixtures()
        cls.asset_finder = cls.make_asset_finder()


class WithTradingCalendar(object):
    """
    ZiplineTestCase mixing providing cls.trading_calendar as a class-level
    fixture.

    After ``init_class_fixtures`` has been called, `cls.trading_calendar` is
    populated with a trading calendar.

    Attributes
    ----------
    TRADING_CALENDAR_STR : str
        The identifier of the calendar to use.
    """
    TRADING_CALENDAR_STR = 'NYSE'

    @classmethod
    def init_class_fixtures(cls):
        super(WithTradingCalendar, cls).init_class_fixtures()
        cls.trading_calendar = get_calendar(cls.TRADING_CALENDAR_STR)


class WithTradingEnvironment(WithAssetFinder, WithTradingCalendar):
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

    @classmethod
    def make_load_function(cls):
        return None

    @classmethod
    def make_trading_environment(cls):
        return TradingEnvironment(
            load=cls.make_load_function(),
            asset_db_path=cls.asset_finder.engine,
            trading_calendar=cls.trading_calendar,
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
            trading_calendar=cls.trading_calendar,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithSimParams, cls).init_class_fixtures()
        cls.sim_params = cls.make_simparams()


class WithNYSETradingDays(WithTradingCalendar):
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

        all_days = cls.trading_calendar.all_sessions
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


class WithEquityDailyBarData(WithTradingEnvironment):
    """
    ZiplineTestCase mixin providing cls.make_equity_daily_bar_data.

    Attributes
    ----------
    EQUITY_DAILY_BAR_START_DATE : Timestamp
        The date at to which to start creating data. This defaults to
        ``START_DATE``.
    EQUITY_DAILY_BAR_END_DATE = Timestamp
        The end date up to which to create data. This defaults to ``END_DATE``.
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE : bool
        If this flag is set, `make_equity_daily_bar_data` will read data from
        the minute bars defined by `WithMinuteBarData`.
        The current default is `False`, but could be `True` in the future.

    Methods
    -------
    make_equity_daily_bar_data() -> iterable[(int, pd.DataFrame)]
        A class method that returns an iterator of (sid, dataframe) pairs
        which will be written to the bcolz files that the class's
        ``BcolzDailyBarReader`` will read from. By default this creates
        some simple sythetic data with
        :func:`~zipline.testing.create_daily_bar_data`

    See Also
    --------
    WithEquityMinuteBarData
    zipline.testing.create_daily_bar_data
    """
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = 0

    EQUITY_DAILY_BAR_USE_FULL_CALENDAR = False
    EQUITY_DAILY_BAR_START_DATE = alias('START_DATE')
    EQUITY_DAILY_BAR_END_DATE = alias('END_DATE')
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = None

    @classmethod
    def _make_equity_daily_bar_from_minute(cls):
        assets = cls.asset_finder.retrieve_all(cls.asset_finder.sids)
        ohclv_how = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            # TODO: Change test data so that large minute volumes are not used,
            # so that 'sum' can be used without going over the uint limit.
            # When that data is changed, this function can and should be moved
            # to the `data` module so that loaders and tests can use the same
            # source from minute logic.
            'volume': 'last'
        }
        mm = cls.trading_calendar.all_minutes
        m_opens = cls.trading_calendar.schedule.market_open
        m_closes = cls.trading_calendar.schedule.market_close

        minute_data = dict(cls.make_equity_minute_bar_data())

        for asset in assets:
            first_minute = m_opens.loc[asset.start_date]
            last_minute = m_closes.loc[asset.end_date]
            asset_df = minute_data[asset]
            slicer = asset_df.index.slice_indexer(first_minute, last_minute)
            asset_df = asset_df[slicer]
            minutes = mm[mm.slice_indexer(start=first_minute,
                                          end=last_minute)]
            asset_df = asset_df.reindex(minutes)
            yield asset.sid, asset_df.resample('1d', how=ohclv_how).dropna()

    @classmethod
    def make_equity_daily_bar_data(cls):
        # Requires a WithEquityMinuteBarData to come before in the MRO.
        # Resample that data so that daily and minute bar data are aligned.
        if cls.EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE:
            return cls._make_equity_daily_bar_from_minute()
        else:
            return create_daily_bar_data(
                cls.equity_daily_bar_days,
                cls.asset_finder.sids,
            )

    @classmethod
    def init_class_fixtures(cls):
        super(WithEquityDailyBarData, cls).init_class_fixtures()
        if cls.EQUITY_DAILY_BAR_USE_FULL_CALENDAR:
            days = cls.trading_calendar.all_sessions
        else:
            if cls.trading_calendar.is_session(
                    cls.EQUITY_DAILY_BAR_START_DATE
            ):
                first_session = cls.EQUITY_DAILY_BAR_START_DATE
            else:
                first_session = cls.trading_calendar.minute_to_session_label(
                    pd.Timestamp(cls.EQUITY_DAILY_BAR_START_DATE)
                )

            if cls.EQUITY_DAILY_BAR_LOOKBACK_DAYS > 0:
                first_session = cls.trading_calendar.sessions_window(
                    first_session,
                    -1 * cls.EQUITY_DAILY_BAR_LOOKBACK_DAYS
                )[0]

            days = cls.trading_calendar.sessions_in_range(
                first_session,
                cls.EQUITY_DAILY_BAR_END_DATE,
            )

        cls.equity_daily_bar_days = days


class WithBcolzEquityDailyBarReader(WithEquityDailyBarData, WithTmpDir):
    """
    ZiplineTestCase mixin providing cls.bcolz_daily_bar_path,
    cls.bcolz_daily_bar_ctable, and cls.bcolz_equity_daily_bar_reader
    class level fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_daily_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_DAILY_BAR_PATH)`.
    - `cls.bcolz_daily_bar_ctable` is populated with data returned from
      `cls.make_equity_daily_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_equity_daily_bar_data`.
    - `cls.bcolz_equity_daily_bar_reader` is a daily bar reader
       pointing to the directory that was just written to.

    Attributes
    ----------
    BCOLZ_DAILY_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    EQUITY_DAILY_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day. This is used
        when a test needs to use history, in which case this should be set to
        the largest history window that will be
        requested.
    EQUITY_DAILY_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``equity_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``EQUITY_DAILY_BAR_LOOKBACK_DAYS``.
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD : int
        If this flag is set, use the value as the `read_all_threshold`
        parameter to BcolzDailyBarReader, otherwise use the default
        value.
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE : bool
        If this flag is set, `make_equity_daily_bar_data` will read data from
        the minute bar reader defined by a `WithBcolzEquityMinuteBarReader`.

    Methods
    -------
    make_bcolz_daily_bar_rootdir_path() -> string
        A class method that returns the path for the rootdir of the daily
        bars ctable. By default this is a subdirectory BCOLZ_DAILY_BAR_PATH in
        the shared temp directory.

    See Also
    --------
    WithBcolzEquityMinuteBarReader
    WithDataPortal
    zipline.testing.create_daily_bar_data
    """
    BCOLZ_DAILY_BAR_PATH = 'daily_equity_pricing.bcolz'
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD = None
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = False
    # allows WithBcolzEquityDailyBarReaderFromCSVs to call the
    # `write_csvs`method without needing to reimplement `init_class_fixtures`
    _write_method_name = 'write'

    @classmethod
    def make_bcolz_daily_bar_rootdir_path(cls):
        return cls.tmpdir.makedir(cls.BCOLZ_DAILY_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        super(WithBcolzEquityDailyBarReader, cls).init_class_fixtures()
        cls.bcolz_daily_bar_path = p = cls.make_bcolz_daily_bar_rootdir_path()
        days = cls.equity_daily_bar_days

        cls.bcolz_daily_bar_ctable = t = getattr(
            BcolzDailyBarWriter(p, days, cls.trading_calendar),
            cls._write_method_name,
        )(cls.make_equity_daily_bar_data())

        if cls.BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD is not None:
            cls.bcolz_equity_daily_bar_reader = BcolzDailyBarReader(
                t, cls.BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD)
        else:
            cls.bcolz_equity_daily_bar_reader = BcolzDailyBarReader(t)


class WithBcolzEquityDailyBarReaderFromCSVs(WithBcolzEquityDailyBarReader):
    """
    ZiplineTestCase mixin that provides
    cls.bcolz_equity_daily_bar_reader from a mapping of sids to CSV
    file paths.
    """
    _write_method_name = 'write_csvs'


class WithEquityMinuteBarData(WithTradingEnvironment):
    """
    ZiplineTestCase mixin providing cls.equity_minute_bar_days.

    After init_class_fixtures has been called:
    - `cls.equyt_minute_bar_days` has the range over which data has been
       generated.

    Attributes
    ----------
    EQUITY_MINUTE_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day.
        This is used when a test needs to use history, in which case this
        should be set to the largest history window that will be requested.
    EQUITY_MINUTE_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``equity_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``EQUITY_MINUTE_BAR_LOOKBACK_DAYS``.
    EQUITY_MINUTE_BAR_START_DATE : Timestamp
        The date at to which to start creating data. This defaults to
        ``START_DATE``.
    EQUITY_MINUTE_BAR_END_DATE = Timestamp
        The end date up to which to create data. This defaults to ``END_DATE``.

    Methods
    -------
    make_equity_minute_bar_data() -> iterable[(int, pd.DataFrame)]
        A class method that returns a dict mapping sid to dataframe
        which will be written to into the the format of the inherited
        class which writes the minute bar data for use by a reader.
        By default this creates some simple sythetic data with
        :func:`~zipline.testing.create_minute_bar_data`

    See Also
    --------
    WithEquityDailyBarData
    zipline.testing.create_minute_bar_data
    """

    EQUITY_MINUTE_BAR_LOOKBACK_DAYS = 0
    EQUITY_MINUTE_BAR_USE_FULL_CALENDAR = False
    EQUITY_MINUTE_BAR_START_DATE = alias('START_DATE')
    EQUITY_MINUTE_BAR_END_DATE = alias('END_DATE')

    @classmethod
    def make_equity_minute_bar_data(cls):
        return create_minute_bar_data(
            cls.trading_calendar.minutes_for_sessions_in_range(
                cls.equity_minute_bar_days[0],
                cls.equity_minute_bar_days[-1],
            ),
            cls.asset_finder.sids,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(WithEquityMinuteBarData, cls).init_class_fixtures()
        if cls.EQUITY_MINUTE_BAR_USE_FULL_CALENDAR:
            days = cls.trading_calendar.all_execution_days
        else:
            first_session = cls.trading_calendar.minute_to_session_label(
                pd.Timestamp(cls.EQUITY_MINUTE_BAR_START_DATE)
            )

            if cls.EQUITY_MINUTE_BAR_LOOKBACK_DAYS > 0:
                first_session = cls.trading_calendar.sessions_window(
                    first_session,
                    -1 * cls.EQUITY_MINUTE_BAR_LOOKBACK_DAYS
                )[0]

            days = cls.trading_calendar.sessions_in_range(
                first_session,
                cls.EQUITY_MINUTE_BAR_END_DATE
            )

        cls.equity_minute_bar_days = days


class WithBcolzEquityMinuteBarReader(WithEquityMinuteBarData, WithTmpDir):
    """
    ZiplineTestCase mixin providing cls.bcolz_minute_bar_path,
    cls.bcolz_minute_bar_ctable, and cls.bcolz_equity_minute_bar_reader
    class level fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_minute_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_MINUTE_BAR_PATH)`.
    - `cls.bcolz_minute_bar_ctable` is populated with data returned from
      `cls.make_equity_minute_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_equity_minute_bar_data`.
    - `cls.bcolz_equity_minute_bar_reader` is a minute bar reader
       pointing to the directory that was just written to.

    Attributes
    ----------
    BCOLZ_MINUTE_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    EQUITY_MINUTE_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day.
        This is used when a test needs to use history, in which case this
        should be set to the largest history window that will be requested.
    BCOLZ_MINUTE_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``equity_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``EQUITY_MINUTE_BAR_LOOKBACK_DAYS``.

    Methods
    -------
    make_bcolz_minute_bar_rootdir_path() -> string
        A class method that returns the path for the directory that contains
        the minute bar ctables. By default this is a subdirectory
        BCOLZ_MINUTE_BAR_PATH in the shared temp directory.

    See Also
    --------
    WithBcolzEquityDailyBarReader
    WithDataPortal
    zipline.testing.create_minute_bar_data
    """
    BCOLZ_MINUTE_BAR_PATH = 'minute_equity_pricing.bcolz'

    @classmethod
    def make_bcolz_minute_bar_rootdir_path(cls):
        return cls.tmpdir.makedir(cls.BCOLZ_MINUTE_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        super(WithBcolzEquityMinuteBarReader, cls).init_class_fixtures()
        cls.bcolz_minute_bar_path = p = \
            cls.make_bcolz_minute_bar_rootdir_path()
        days = cls.equity_minute_bar_days

        writer = BcolzMinuteBarWriter(
            days[0],
            p,
            cls.trading_calendar.schedule.market_open.loc[days],
            cls.trading_calendar.schedule.market_close.loc[days],
            US_EQUITIES_MINUTES_PER_DAY
        )
        writer.write(cls.make_equity_minute_bar_data())

        cls.bcolz_equity_minute_bar_reader = \
            BcolzMinuteBarReader(p)


class WithAdjustmentReader(WithBcolzEquityDailyBarReader):
    """
    ZiplineTestCase mixin providing cls.adjustment_reader as a class level
    fixture.

    After init_class_fixtures has been called, `cls.adjustment_reader` will be
    populated with a new SQLiteAdjustmentReader object. The data that will be
    written can be passed by overriding `make_{field}_data` where field may
    be `splits`, `mergers` `dividends`, or `stock_dividends`.
    The daily bar reader used for this adjustment reader may be customized
    by overriding `make_adjustment_writer_equity_daily_bar_reader`.
    This is useful to providing a `MockDailyBarReader`.

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
    make_adjustment_db_conn_str() -> string
        A class method that returns the sqlite3 connection string for the
        database in to which the adjustments will be written. By default this
        is an in-memory database.
    make_adjustment_writer_equity_daily_bar_reader() -> pd.DataFrame
        A class method that returns the daily bar reader to use for the class's
        adjustment writer. By default this is the class's actual
        ``bcolz_equity_daily_bar_reader`` as inherited from
        ``WithBcolzEquityDailyBarReader``. This should probably not be
          overridden; however, some tests used a ``MockDailyBarReader``
         for this.
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
            cls.make_adjustment_writer_equity_daily_bar_reader(),
            cls.equity_daily_bar_days,
        )

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        return cls.bcolz_equity_daily_bar_reader

    @classmethod
    def make_adjustment_db_conn_str(cls):
        return ':memory:'

    @classmethod
    def init_class_fixtures(cls):
        super(WithAdjustmentReader, cls).init_class_fixtures()
        conn = sqlite3.connect(cls.make_adjustment_db_conn_str())
        cls.make_adjustment_writer(conn).write(
            splits=cls.make_splits_data(),
            mergers=cls.make_mergers_data(),
            dividends=cls.make_dividends_data(),
            stock_dividends=cls.make_stock_dividends_data(),
        )
        cls.adjustment_reader = SQLiteAdjustmentReader(conn)


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


class WithDataPortal(WithAdjustmentReader,
                     # Ordered so that bcolz minute reader is used first.
                     WithBcolzEquityMinuteBarReader):
    """
    ZiplineTestCase mixin providing self.data_portal as an instance level
    fixture.

    After init_instance_fixtures has been called, `self.data_portal` will be
    populated with a new data portal created by passing in the class's
    trading env, `cls.bcolz_equity_minute_bar_reader`,
    `cls.bcolz_equity_daily_bar_reader`, and `cls.adjustment_reader`.

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

    DATA_PORTAL_FIRST_TRADING_DAY = None

    def make_data_portal(self):
        if self.DATA_PORTAL_FIRST_TRADING_DAY is None:
            if self.DATA_PORTAL_USE_MINUTE_DATA:
                self.DATA_PORTAL_FIRST_TRADING_DAY = (
                    self.bcolz_equity_minute_bar_reader.
                    first_trading_day)
            elif self.DATA_PORTAL_USE_DAILY_DATA:
                self.DATA_PORTAL_FIRST_TRADING_DAY = (
                    self.bcolz_equity_daily_bar_reader.
                    first_trading_day)

        return DataPortal(
            self.env.asset_finder,
            self.trading_calendar,
            first_trading_day=self.DATA_PORTAL_FIRST_TRADING_DAY,
            equity_daily_reader=(
                self.bcolz_equity_daily_bar_reader
                if self.DATA_PORTAL_USE_DAILY_DATA else
                None
            ),
            equity_minute_reader=(
                self.bcolz_equity_minute_bar_reader
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
