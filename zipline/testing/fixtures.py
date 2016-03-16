from unittest import TestCase

from contextlib2 import ExitStack
from logbook import NullHandler
import pandas as pd
from six import with_metaclass

from .core import tmp_asset_finder
from ..finance.trading import TradingEnvironment
from ..utils import tradingcalendar, factory
from ..utils.final import FinalMeta, final


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
        super(WithLogger, cls).init_instance_fixtures()

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
