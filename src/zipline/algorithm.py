#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterable
from collections import namedtuple
from copy import copy
import warnings
from datetime import tzinfo, time, timezone
import logging
import pytz
import pandas as pd
import numpy as np

from itertools import chain, repeat

from zipline.utils.calendar_utils import get_calendar, days_at_time

from zipline._protocol import handle_non_market_minutes
from zipline.errors import (
    AttachPipelineAfterInitialize,
    CannotOrderDelistedAsset,
    DuplicatePipelineName,
    IncompatibleCommissionModel,
    IncompatibleSlippageModel,
    NoSuchPipeline,
    OrderDuringInitialize,
    OrderInBeforeTradingStart,
    PipelineOutputDuringInitialize,
    RegisterAccountControlPostInit,
    RegisterTradingControlPostInit,
    ScheduleFunctionInvalidCalendar,
    SetBenchmarkOutsideInitialize,
    SetCancelPolicyPostInit,
    SetCommissionPostInit,
    SetSlippagePostInit,
    UnsupportedCancelPolicy,
    UnsupportedDatetimeFormat,
    UnsupportedOrderParameters,
    ZeroCapitalError,
)
from zipline.finance.blotter import SimulationBlotter
from zipline.finance.controls import (
    LongOnly,
    MaxOrderCount,
    MaxOrderSize,
    MaxPositionSize,
    MaxLeverage,
    MinLeverage,
    RestrictedListOrder,
)
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.finance.asset_restrictions import Restrictions
from zipline.finance.cancel_policy import NeverCancel, CancelPolicy
from zipline.finance.asset_restrictions import (
    NoRestrictions,
    StaticRestrictions,
    SecurityListRestrictions,
)
from zipline.assets import Asset, Equity, Future
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.finance.metrics import MetricsTracker, load as load_metrics_set
from zipline.pipeline import Pipeline
import zipline.pipeline.domain as domain
from zipline.pipeline.engine import (
    ExplodingPipelineEngine,
    SimplePipelineEngine,
)
from zipline.utils.api_support import (
    api_method,
    require_initialized,
    require_not_initialized,
    ZiplineAPI,
    disallowed_in_before_trading_start,
)
from zipline.utils.compat import ExitStack
from zipline.utils.date_utils import make_utc_aware
from zipline.utils.input_validation import (
    coerce_string,
    ensure_upper_case,
    error_keywords,
    expect_dtypes,
    expect_types,
    optional,
    optionally,
)
from zipline.utils.numpy_utils import int64_dtype
from zipline.utils.cache import ExpiringCache

import zipline.utils.events
from zipline.utils.events import (
    EventManager,
    make_eventrule,
    date_rules,
    time_rules,
    calendars,
    AfterOpen,
    BeforeClose,
)
from zipline.utils.math_utils import (
    tolerant_equals,
    round_if_near_integer,
)
from zipline.utils.preprocess import preprocess
from zipline.utils.security_list import SecurityList

import zipline.protocol
from zipline.sources.requests_csv import PandasRequestsCSV

from zipline.gens.sim_engine import MinuteSimulationClock
from zipline.sources.benchmark_source import BenchmarkSource
from zipline.zipline_warnings import ZiplineDeprecationWarning

log = logging.getLogger("ZiplineLog")

# For creating and storing pipeline instances
AttachedPipeline = namedtuple("AttachedPipeline", "pipe chunks eager")


class NoBenchmark(ValueError):
    def __init__(self):
        super(NoBenchmark, self).__init__(
            "Must specify either benchmark_sid or benchmark_returns.",
        )


class TradingAlgorithm:
    """A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``initialize`` unless listed below.
    initialize : callable[context -> None], optional
        Function that is called at the start of the simulation to
        setup the initial context.
    handle_data : callable[(context, data) -> None], optional
        Function called on every bar. This is where most logic should be
        implemented.
    before_trading_start : callable[(context, data) -> None], optional
        Function that is called before any bars have been processed each
        day.
    analyze : callable[(context, DataFrame) -> None], optional
        Function that is called at the end of the backtest. This is passed
        the context and the performance results for the backtest.
    script : str, optional
        Algoscript that contains the definitions for the four algorithm
        lifecycle functions and any supporting code.
    namespace : dict, optional
        The namespace to execute the algoscript in. By default this is an
        empty namespace that will include only python built ins.
    algo_filename : str, optional
        The filename for the algoscript. This will be used in exception
        tracebacks. default: '<string>'.
    data_frequency : {'daily', 'minute'}, optional
        The duration of the bars.
    equities_metadata : dict or DataFrame or file-like object, optional
        If dict is provided, it must have the following structure:
        * keys are the identifiers
        * values are dicts containing the metadata, with the metadata
          field name as the key
        If pandas.DataFrame is provided, it must have the
        following structure:
        * column names must be the metadata fields
        * index must be the different asset identifiers
        * array contents should be the metadata value
        If an object with a ``read`` method is provided, ``read`` must
        return rows containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    futures_metadata : dict or DataFrame or file-like object, optional
        The same layout as ``equities_metadata`` except that it is used
        for futures information.
    identifiers : list, optional
        Any asset identifiers that are not provided in the
        equities_metadata, but will be traded by this TradingAlgorithm.
    get_pipeline_loader : callable[BoundColumn -> PipelineLoader], optional
        The function that maps pipeline columns to their loaders.
    create_event_context : callable[BarData -> context manager], optional
        A function used to create a context mananger that wraps the
        execution of all events that are scheduled for a bar.
        This function will be passed the data for the bar and should
        return the actual context manager that will be entered.
    history_container_class : type, optional
        The type of history container to use. default: HistoryContainer
    platform : str, optional
        The platform the simulation is running on. This can be queried for
        in the simulation with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
        default: 'zipline'
    adjustment_reader : AdjustmentReader
        The interface to the adjustments.
    """

    def __init__(
        self,
        sim_params,
        data_portal=None,
        asset_finder=None,
        # Algorithm API
        namespace=None,
        script=None,
        algo_filename=None,
        initialize=None,
        handle_data=None,
        before_trading_start=None,
        analyze=None,
        #
        trading_calendar=None,
        metrics_set=None,
        blotter=None,
        blotter_class=None,
        cancel_policy=None,
        benchmark_sid=None,
        benchmark_returns=None,
        platform="zipline",
        capital_changes=None,
        get_pipeline_loader=None,
        create_event_context=None,
        **initialize_kwargs,
    ):
        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        # List of account controls to be checked on each bar.
        self.account_controls = []

        self._recorded_vars = {}
        self.namespace = namespace or {}

        self._platform = platform
        self.logger = None

        # XXX: This is kind of a mess.
        # We support passing a data_portal in `run`, but we need an asset
        # finder earlier than that to look up assets for things like
        # set_benchmark.
        self.data_portal = data_portal

        if self.data_portal is None:
            if asset_finder is None:
                raise ValueError(
                    "Must pass either data_portal or asset_finder "
                    "to TradingAlgorithm()"
                )
            self.asset_finder = asset_finder
        else:
            # Raise an error if we were passed two different asset finders.
            # There's no world where that's a good idea.
            if (
                asset_finder is not None
                and asset_finder is not data_portal.asset_finder
            ):
                raise ValueError("Inconsistent asset_finders in TradingAlgorithm()")
            self.asset_finder = data_portal.asset_finder

        self.benchmark_returns = benchmark_returns

        # XXX: This is also a mess. We should remove all of this and only allow
        #      one way to pass a calendar.
        #
        # We have a required sim_params argument as well as an optional
        # trading_calendar argument, but sim_params has a trading_calendar
        # attribute. If the user passed trading_calendar explicitly, make sure
        # it matches their sim_params. Otherwise, just use what's in their
        # sim_params.
        self.sim_params = sim_params
        if trading_calendar is None:
            self.trading_calendar = sim_params.trading_calendar
        elif trading_calendar.name == sim_params.trading_calendar.name:
            self.trading_calendar = sim_params.trading_calendar
        else:
            raise ValueError(
                "Conflicting calendars: trading_calendar={}, but "
                "sim_params.trading_calendar={}".format(
                    trading_calendar.name,
                    self.sim_params.trading_calendar.name,
                )
            )

        self.metrics_tracker = None
        self._last_sync_time = pd.NaT
        self._metrics_set = metrics_set
        if self._metrics_set is None:
            self._metrics_set = load_metrics_set("default")

        # Initialize Pipeline API data.
        self.init_engine(get_pipeline_loader)
        self._pipelines = {}

        # Create an already-expired cache so that we compute the first time
        # data is requested.
        self._pipeline_cache = ExpiringCache()

        if blotter is not None:
            self.blotter = blotter
        else:
            cancel_policy = cancel_policy or NeverCancel()
            blotter_class = blotter_class or SimulationBlotter
            self.blotter = blotter_class(cancel_policy=cancel_policy)

        # The symbol lookup date specifies the date to use when resolving
        # symbols to sids, and can be set using set_symbol_lookup_date()
        self._symbol_lookup_date = None

        # If string is passed in, execute and get reference to
        # functions.
        self.algoscript = script

        self._initialize = None
        self._before_trading_start = None
        self._analyze = None

        self._in_before_trading_start = False

        self.event_manager = EventManager(create_event_context)

        self._handle_data = None

        def noop(*args, **kwargs):
            pass

        if self.algoscript is not None:
            unexpected_api_methods = set()
            if initialize is not None:
                unexpected_api_methods.add("initialize")
            if handle_data is not None:
                unexpected_api_methods.add("handle_data")
            if before_trading_start is not None:
                unexpected_api_methods.add("before_trading_start")
            if analyze is not None:
                unexpected_api_methods.add("analyze")

            if unexpected_api_methods:
                raise ValueError(
                    "TradingAlgorithm received a script and the following API"
                    " methods as functions:\n{funcs}".format(
                        funcs=unexpected_api_methods,
                    )
                )

            if algo_filename is None:
                algo_filename = "<string>"
            code = compile(self.algoscript, algo_filename, "exec")
            exec(code, self.namespace)

            self._initialize = self.namespace.get("initialize", noop)
            self._handle_data = self.namespace.get("handle_data", noop)
            self._before_trading_start = self.namespace.get(
                "before_trading_start",
            )
            # Optional analyze function, gets called after run
            self._analyze = self.namespace.get("analyze")

        else:
            self._initialize = initialize or (lambda self: None)
            self._handle_data = handle_data
            self._before_trading_start = before_trading_start
            self._analyze = analyze

        self.event_manager.add_event(
            zipline.utils.events.Event(
                zipline.utils.events.Always(),
                # We pass handle_data.__func__ to get the unbound method.
                # We will explicitly pass the algorithm to bind it again.
                self.handle_data.__func__,
            ),
            prepend=True,
        )

        if self.sim_params.capital_base <= 0:
            raise ZeroCapitalError()

        # Prepare the algo for initialization
        self.initialized = False

        self.initialize_kwargs = initialize_kwargs or {}

        self.benchmark_sid = benchmark_sid

        # A dictionary of capital changes, keyed by timestamp, indicating the
        # target/delta of the capital changes, along with values
        self.capital_changes = capital_changes or {}

        # A dictionary of the actual capital change deltas, keyed by timestamp
        self.capital_change_deltas = {}

        self.restrictions = NoRestrictions()

    def init_engine(self, get_loader):
        """Construct and store a PipelineEngine from loader.

        If get_loader is None, constructs an ExplodingPipelineEngine
        """
        if get_loader is not None:
            self.engine = SimplePipelineEngine(
                get_loader,
                self.asset_finder,
                self.default_pipeline_domain(self.trading_calendar),
            )
        else:
            self.engine = ExplodingPipelineEngine()

    def initialize(self, *args, **kwargs):
        """Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)

    def before_trading_start(self, data):
        self.compute_eager_pipelines()

        if self._before_trading_start is None:
            return

        self._in_before_trading_start = True

        with handle_non_market_minutes(
            data
        ) if self.data_frequency == "minute" else ExitStack():
            self._before_trading_start(self, data)

        self._in_before_trading_start = False

    def handle_data(self, data):
        if self._handle_data:
            self._handle_data(self, data)

    def analyze(self, perf):
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

    def __repr__(self):
        """N.B. this does not yet represent a string that can be used
        to instantiate an exact copy of an algorithm.

        However, it is getting close, and provides some value as something
        that can be inspected interactively.
        """
        return """
{class_name}(
    capital_base={capital_base}
    sim_params={sim_params},
    initialized={initialized},
    slippage_models={slippage_models},
    commission_models={commission_models},
    blotter={blotter},
    recorded_vars={recorded_vars})
""".strip().format(
            class_name=self.__class__.__name__,
            capital_base=self.sim_params.capital_base,
            sim_params=repr(self.sim_params),
            initialized=self.initialized,
            slippage_models=repr(self.blotter.slippage_models),
            commission_models=repr(self.blotter.commission_models),
            blotter=repr(self.blotter),
            recorded_vars=repr(self.recorded_vars),
        )

    def _create_clock(self):
        """If the clock property is not set, then create one based on frequency."""
        market_closes = self.trading_calendar.schedule.loc[
            self.sim_params.sessions, "close"
        ]
        market_opens = self.trading_calendar.first_minutes.loc[self.sim_params.sessions]
        minutely_emission = False

        if self.sim_params.data_frequency == "minute":
            minutely_emission = self.sim_params.emission_rate == "minute"

            # The calendar's execution times are the minutes over which we
            # actually want to run the clock. Typically the execution times
            # simply adhere to the market open and close times. In the case of
            # the futures calendar, for example, we only want to simulate over
            # a subset of the full 24 hour calendar, so the execution times
            # dictate a market open time of 6:31am US/Eastern and a close of
            # 5:00pm US/Eastern.
            if self.trading_calendar.name == "us_futures":
                execution_opens = self.trading_calendar.execution_time_from_open(
                    market_opens
                )
                execution_closes = self.trading_calendar.execution_time_from_close(
                    market_closes
                )
            else:
                execution_opens = market_opens
                execution_closes = market_closes
        else:
            # in daily mode, we want to have one bar per session, timestamped
            # as the last minute of the session.
            if self.trading_calendar.name == "us_futures":
                execution_closes = self.trading_calendar.execution_time_from_close(
                    market_closes
                )
                execution_opens = execution_closes
            else:
                execution_closes = market_closes
                execution_opens = market_closes

        # FIXME generalize these values
        before_trading_start_minutes = days_at_time(
            self.sim_params.sessions,
            time(8, 45),
            "US/Eastern",
            day_offset=0,
        )

        return MinuteSimulationClock(
            self.sim_params.sessions,
            execution_opens,
            execution_closes,
            before_trading_start_minutes,
            minute_emission=minutely_emission,
        )

    def _create_benchmark_source(self):
        if self.benchmark_sid is not None:
            benchmark_asset = self.asset_finder.retrieve_asset(self.benchmark_sid)
            benchmark_returns = None
        else:
            benchmark_asset = None
            benchmark_returns = self.benchmark_returns
        return BenchmarkSource(
            benchmark_asset=benchmark_asset,
            benchmark_returns=benchmark_returns,
            trading_calendar=self.trading_calendar,
            sessions=self.sim_params.sessions,
            data_portal=self.data_portal,
            emission_rate=self.sim_params.emission_rate,
        )

    def _create_metrics_tracker(self):
        return MetricsTracker(
            trading_calendar=self.trading_calendar,
            first_session=self.sim_params.start_session,
            last_session=self.sim_params.end_session,
            capital_base=self.sim_params.capital_base,
            emission_rate=self.sim_params.emission_rate,
            data_frequency=self.sim_params.data_frequency,
            asset_finder=self.asset_finder,
            metrics=self._metrics_set,
        )

    def _create_generator(self, sim_params):
        if sim_params is not None:
            self.sim_params = sim_params

        self.metrics_tracker = metrics_tracker = self._create_metrics_tracker()

        # Set the dt initially to the period start by forcing it to change.
        self.on_dt_changed(self.sim_params.start_session)

        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True

        benchmark_source = self._create_benchmark_source()

        self.trading_client = AlgorithmSimulator(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            benchmark_source,
            self.restrictions,
        )

        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()

    def compute_eager_pipelines(self):
        """Compute any pipelines attached with eager=True."""
        for name, pipe in self._pipelines.items():
            if pipe.eager:
                self.pipeline_output(name)

    def get_generator(self):
        """Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

    def run(self, data_portal=None):
        """Run the algorithm."""
        # HACK: I don't think we really want to support passing a data portal
        # this late in the long term, but this is needed for now for backwards
        # compat downstream.
        if data_portal is not None:
            self.data_portal = data_portal
            self.asset_finder = data_portal.asset_finder
        elif self.data_portal is None:
            raise RuntimeError(
                "No data portal in TradingAlgorithm.run().\n"
                "Either pass a DataPortal to TradingAlgorithm() or to run()."
            )
        else:
            assert (
                self.asset_finder is not None
            ), "Have data portal without asset_finder."

        # Create zipline and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        try:
            perfs = []
            for perf in self.get_generator():
                perfs.append(perf)

            # convert perf dict to pandas dataframe
            daily_stats = self._create_daily_stats(perfs)

            self.analyze(daily_stats)
        finally:
            self.data_portal = None
            self.metrics_tracker = None

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        # TODO: the loop here could overwrite expected properties
        # of daily_perf. Could potentially raise or log a
        # warning.
        for perf in perfs:
            if "daily_perf" in perf:
                perf["daily_perf"].update(perf["daily_perf"].pop("recorded_vars"))
                perf["daily_perf"].update(perf["cumulative_risk_metrics"])
                daily_perfs.append(perf["daily_perf"])
            else:
                self.risk_report = perf

        daily_dts = pd.DatetimeIndex([p["period_close"] for p in daily_perfs])
        daily_dts = make_utc_aware(daily_dts)
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        return daily_stats

    def calculate_capital_changes(
        self, dt, emission_rate, is_interday, portfolio_value_adjustment=0.0
    ):
        """If there is a capital change for a given dt, this means the the change
        occurs before `handle_data` on the given dt. In the case of the
        change being a target value, the change will be computed on the
        portfolio value according to prices at the given dt

        `portfolio_value_adjustment`, if specified, will be removed from the
        portfolio_value of the cumulative performance when calculating deltas
        from target capital changes.
        """

        # CHECK is try/catch faster than search?

        try:
            capital_change = self.capital_changes[dt]
        except KeyError:
            return

        self._sync_last_sale_prices()
        if capital_change["type"] == "target":
            target = capital_change["value"]
            capital_change_amount = target - (
                self.portfolio.portfolio_value - portfolio_value_adjustment
            )

            log.info(
                "Processing capital change to target %s at %s. Capital "
                "change delta is %s" % (target, dt, capital_change_amount)
            )
        elif capital_change["type"] == "delta":
            target = None
            capital_change_amount = capital_change["value"]
            log.info(
                "Processing capital change of delta %s at %s"
                % (capital_change_amount, dt)
            )
        else:
            log.error(
                "Capital change %s does not indicate a valid type "
                "('target' or 'delta')" % capital_change
            )
            return

        self.capital_change_deltas.update({dt: capital_change_amount})
        self.metrics_tracker.capital_change(capital_change_amount)

        yield {
            "capital_change": {
                "date": dt,
                "type": "cash",
                "target": target,
                "delta": capital_change_amount,
            }
        }

    @api_method
    def get_environment(self, field="platform"):
        """Query the execution environment.

        Parameters
        ----------
        field : {'platform', 'arena', 'data_frequency', 'start', 'end',
        'capital_base', 'platform', '*'}

        The field to query. The options have the following meanings:

        - arena : str
          The arena from the simulation parameters. This will normally
          be ``'backtest'`` but some systems may use this distinguish
          live trading from backtesting.
        - data_frequency : {'daily', 'minute'}
          data_frequency tells the algorithm if it is running with
          daily data or minute data.
        - start : datetime
          The start date for the simulation.
        - end : datetime
          The end date for the simulation.
        - capital_base : float
          The starting capital for the simulation.
        -platform : str
          The platform that the code is running on. By default, this
          will be the string 'zipline'. This can allow algorithms to
          know if they are running on the Quantopian platform instead.
        - * : dict[str -> any]
          Returns all the fields in a dictionary.

        Returns
        -------
        val : any
            The value for the field queried. See above for more information.

        Raises
        ------
        ValueError
            Raised when ``field`` is not a valid option.
        """
        env = {
            "arena": self.sim_params.arena,
            "data_frequency": self.sim_params.data_frequency,
            "start": self.sim_params.first_open,
            "end": self.sim_params.last_close,
            "capital_base": self.sim_params.capital_base,
            "platform": self._platform,
        }
        if field == "*":
            return env
        else:
            try:
                return env[field]
            except KeyError as exc:
                raise ValueError(
                    "%r is not a valid field for get_environment" % field,
                ) from exc

    @api_method
    def fetch_csv(
        self,
        url,
        pre_func=None,
        post_func=None,
        date_column="date",
        date_format=None,
        timezone=str(timezone.utc),
        symbol=None,
        mask=True,
        symbol_column=None,
        special_params_checker=None,
        country_code=None,
        **kwargs,
    ):
        """Fetch a csv from a remote url and register the data so that it is
        queryable from the ``data`` object.

        Parameters
        ----------
        url : str
            The url of the csv file to load.
        pre_func : callable[pd.DataFrame -> pd.DataFrame], optional
            A callback to allow preprocessing the raw data returned from
            fetch_csv before dates are paresed or symbols are mapped.
        post_func : callable[pd.DataFrame -> pd.DataFrame], optional
            A callback to allow postprocessing of the data after dates and
            symbols have been mapped.
        date_column : str, optional
            The name of the column in the preprocessed dataframe containing
            datetime information to map the data.
        date_format : str, optional
            The format of the dates in the ``date_column``. If not provided
            ``fetch_csv`` will attempt to infer the format. For information
            about the format of this string, see :func:`pandas.read_csv`.
        timezone : tzinfo or str, optional
            The timezone for the datetime in the ``date_column``.
        symbol : str, optional
            If the data is about a new asset or index then this string will
            be the name used to identify the values in ``data``. For example,
            one may use ``fetch_csv`` to load data for VIX, then this field
            could be the string ``'VIX'``.
        mask : bool, optional
            Drop any rows which cannot be symbol mapped.
        symbol_column : str
            If the data is attaching some new attribute to each asset then this
            argument is the name of the column in the preprocessed dataframe
            containing the symbols. This will be used along with the date
            information to map the sids in the asset finder.
        country_code : str, optional
            Country code to use to disambiguate symbol lookups.
        **kwargs
            Forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        csv_data_source : zipline.sources.requests_csv.PandasRequestsCSV
            A requests source that will pull data from the url specified.
        """
        if country_code is None:
            country_code = self.default_fetch_csv_country_code(
                self.trading_calendar,
            )

        # Show all the logs every time fetcher is used.
        csv_data_source = PandasRequestsCSV(
            url,
            pre_func,
            post_func,
            self.asset_finder,
            self.trading_calendar.day,
            self.sim_params.start_session,
            self.sim_params.end_session,
            date_column,
            date_format,
            timezone,
            symbol,
            mask,
            symbol_column,
            data_frequency=self.data_frequency,
            country_code=country_code,
            special_params_checker=special_params_checker,
            **kwargs,
        )

        # ingest this into dataportal
        self.data_portal.handle_extra_source(csv_data_source.df, self.sim_params)

        return csv_data_source

    def add_event(self, rule, callback):
        """Adds an event to the algorithm's EventManager.

        Parameters
        ----------
        rule : EventRule
            The rule for when the callback should be triggered.
        callback : callable[(context, data) -> None]
            The function to execute when the rule is triggered.
        """
        self.event_manager.add_event(
            zipline.utils.events.Event(rule, callback),
        )

    @api_method
    def schedule_function(
        self,
        func,
        date_rule=None,
        time_rule=None,
        half_days=True,
        calendar=None,
    ):
        """Schedule a function to be called repeatedly in the future.

        Parameters
        ----------
        func : callable
            The function to execute when the rule is triggered. ``func`` should
            have the same signature as ``handle_data``.
        date_rule : zipline.utils.events.EventRule, optional
            Rule for the dates on which to execute ``func``. If not
            passed, the function will run every trading day.
        time_rule : zipline.utils.events.EventRule, optional
            Rule for the time at which to execute ``func``. If not passed, the
            function will execute at the end of the first market minute of the
            day.
        half_days : bool, optional
            Should this rule fire on half days? Default is True.
        calendar : Sentinel, optional
            Calendar used to compute rules that depend on the trading calendar.

        See Also
        --------
        :class:`zipline.api.date_rules`
        :class:`zipline.api.time_rules`
        """

        # When the user calls schedule_function(func, <time_rule>), assume that
        # the user meant to specify a time rule but no date rule, instead of
        # a date rule and no time rule as the signature suggests
        if isinstance(date_rule, (AfterOpen, BeforeClose)) and not time_rule:
            warnings.warn(
                "Got a time rule for the second positional argument "
                "date_rule. You should use keyword argument "
                "time_rule= when calling schedule_function without "
                "specifying a date_rule",
                stacklevel=3,
            )

        date_rule = date_rule or date_rules.every_day()
        time_rule = (
            (time_rule or time_rules.every_minute())
            if self.sim_params.data_frequency == "minute"
            else
            # If we are in daily mode the time_rule is ignored.
            time_rules.every_minute()
        )

        # Check the type of the algorithm's schedule before pulling calendar
        # Note that the ExchangeTradingSchedule is currently the only
        # TradingSchedule class, so this is unlikely to be hit
        if calendar is None:
            cal = self.trading_calendar
        elif calendar is calendars.US_EQUITIES:
            cal = get_calendar("XNYS")
        elif calendar is calendars.US_FUTURES:
            cal = get_calendar("us_futures")
        else:
            raise ScheduleFunctionInvalidCalendar(
                given_calendar=calendar,
                allowed_calendars=("[calendars.US_EQUITIES, calendars.US_FUTURES]"),
            )

        self.add_event(
            make_eventrule(date_rule, time_rule, cal, half_days),
            func,
        )

    @api_method
    def record(self, *args, **kwargs):
        """Track and record values each day.

        Parameters
        ----------
        **kwargs
            The names and values to record.

        Notes
        -----
        These values will appear in the performance packets and the performance
        dataframe passed to ``analyze`` and returned from
        :func:`~zipline.run_algorithm`.
        """
        # Make 2 objects both referencing the same iterator
        args = [iter(args)] * 2

        # Zip generates list entries by calling `next` on each iterator it
        # receives.  In this case the two iterators are the same object, so the
        # call to next on args[0] will also advance args[1], resulting in zip
        # returning (a,b) (c,d) (e,f) rather than (a,a) (b,b) (c,c) etc.
        positionals = zip(*args)
        for name, value in chain(positionals, kwargs.items()):
            self._recorded_vars[name] = value

    @api_method
    def set_benchmark(self, benchmark):
        """Set the benchmark asset.

        Parameters
        ----------
        benchmark : zipline.assets.Asset
            The asset to set as the new benchmark.

        Notes
        -----
        Any dividends payed out for that new benchmark asset will be
        automatically reinvested.
        """
        if self.initialized:
            raise SetBenchmarkOutsideInitialize()

        self.benchmark_sid = benchmark

    @api_method
    @preprocess(root_symbol_str=ensure_upper_case)
    def continuous_future(
        self, root_symbol_str, offset=0, roll="volume", adjustment="mul"
    ):
        """Create a specifier for a continuous contract.

        Parameters
        ----------
        root_symbol_str : str
            The root symbol for the future chain.

        offset : int, optional
            The distance from the primary contract. Default is 0.

        roll_style : str, optional
            How rolls are determined. Default is 'volume'.

        adjustment : str, optional
            Method for adjusting lookback prices between rolls. Options are
            'mul', 'add', and None. Default is 'mul'.

        Returns
        -------
        continuous_future : zipline.assets.ContinuousFuture
            The continuous future specifier.
        """
        return self.asset_finder.create_continuous_future(
            root_symbol_str,
            offset,
            roll,
            adjustment,
        )

    @api_method
    @preprocess(
        symbol_str=ensure_upper_case,
        country_code=optionally(ensure_upper_case),
    )
    def symbol(self, symbol_str, country_code=None):
        """Lookup an Equity by its ticker symbol.

        Parameters
        ----------
        symbol_str : str
            The ticker symbol for the equity to lookup.
        country_code : str or None, optional
            A country to limit symbol searches to.

        Returns
        -------
        equity : zipline.assets.Equity
            The equity that held the ticker symbol on the current
            symbol lookup date.

        Raises
        ------
        SymbolNotFound
            Raised when the symbols was not held on the current lookup date.

        See Also
        --------
        :func:`zipline.api.set_symbol_lookup_date`
        """
        # If the user has not set the symbol lookup date,
        # use the end_session as the date for symbol->sid resolution.
        _lookup_date = (
            self._symbol_lookup_date
            if self._symbol_lookup_date is not None
            else self.sim_params.end_session
        )

        return self.asset_finder.lookup_symbol(
            symbol_str,
            as_of_date=_lookup_date,
            country_code=country_code,
        )

    @api_method
    def symbols(self, *args, **kwargs):
        """Lookup multuple Equities as a list.

        Parameters
        ----------
        *args : iterable[str]
            The ticker symbols to lookup.
        country_code : str or None, optional
            A country to limit symbol searches to.

        Returns
        -------
        equities : list[zipline.assets.Equity]
            The equities that held the given ticker symbols on the current
            symbol lookup date.

        Raises
        ------
        SymbolNotFound
            Raised when one of the symbols was not held on the current
            lookup date.

        See Also
        --------
        :func:`zipline.api.set_symbol_lookup_date`
        """
        return [self.symbol(identifier, **kwargs) for identifier in args]

    @api_method
    def sid(self, sid):
        """Lookup an Asset by its unique asset identifier.

        Parameters
        ----------
        sid : int
            The unique integer that identifies an asset.

        Returns
        -------
        asset : zipline.assets.Asset
            The asset with the given ``sid``.

        Raises
        ------
        SidsNotFound
            When a requested ``sid`` does not map to any asset.
        """
        return self.asset_finder.retrieve_asset(sid)

    @api_method
    @preprocess(symbol=ensure_upper_case)
    def future_symbol(self, symbol):
        """Lookup a futures contract with a given symbol.

        Parameters
        ----------
        symbol : str
            The symbol of the desired contract.

        Returns
        -------
        future : zipline.assets.Future
            The future that trades with the name ``symbol``.

        Raises
        ------
        SymbolNotFound
            Raised when no contract named 'symbol' is found.
        """
        return self.asset_finder.lookup_future_symbol(symbol)

    def _calculate_order_value_amount(self, asset, value):
        """Calculates how many shares/contracts to order based on the type of
        asset being ordered.
        """
        # Make sure the asset exists, and that there is a last price for it.
        # FIXME: we should use BarData's can_trade logic here, but I haven't
        # yet found a good way to do that.
        normalized_date = self.trading_calendar.minute_to_session(self.datetime)

        if normalized_date < asset.start_date:
            raise CannotOrderDelistedAsset(
                msg="Cannot order {0}, as it started trading on"
                " {1}.".format(asset.symbol, asset.start_date)
            )
        elif normalized_date > asset.end_date:
            raise CannotOrderDelistedAsset(
                msg="Cannot order {0}, as it stopped trading on"
                " {1}.".format(asset.symbol, asset.end_date)
            )
        else:
            last_price = self.trading_client.current_data.current(asset, "price")

            if np.isnan(last_price):
                raise CannotOrderDelistedAsset(
                    msg="Cannot order {0} on {1} as there is no last "
                    "price for the security.".format(asset.symbol, self.datetime)
                )

        if tolerant_equals(last_price, 0):
            zero_message = "Price of 0 for {psid}; can't infer value".format(psid=asset)
            if self.logger:
                self.logger.debug(zero_message)
            # Don't place any order
            return 0

        value_multiplier = asset.price_multiplier

        return value / (last_price * value_multiplier)

    def _can_order_asset(self, asset):
        if not isinstance(asset, Asset):
            raise UnsupportedOrderParameters(
                msg="Passing non-Asset argument to 'order()' is not supported."
                " Use 'sid()' or 'symbol()' methods to look up an Asset."
            )

        if asset.auto_close_date:
            # TODO FIXME TZ MESS
            day = self.trading_calendar.minute_to_session(self.get_datetime())

            if day > min(asset.end_date, asset.auto_close_date):
                # If we are after the asset's end date or auto close date, warn
                # the user that they can't place an order for this asset, and
                # return None.
                log.warning(
                    "Cannot place order for {0}, as it has de-listed. "
                    "Any existing positions for this asset will be "
                    "liquidated on "
                    "{1}.".format(asset.symbol, asset.auto_close_date)
                )

                return False

        return True

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order(self, asset, amount, limit_price=None, stop_price=None, style=None):
        """Place an order for a fixed number of shares.

        Parameters
        ----------
        asset : Asset
            The asset to be ordered.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle, optional
            The execution style for the order.

        Returns
        -------
        order_id : str or None
            The unique identifier for this order, or None if no order was
            placed.

        Notes
        -----
        The ``limit_price`` and ``stop_price`` arguments provide shorthands for
        passing common execution styles. Passing ``limit_price=N`` is
        equivalent to ``style=LimitOrder(N)``. Similarly, passing
        ``stop_price=M`` is equivalent to ``style=StopOrder(M)``, and passing
        ``limit_price=N`` and ``stop_price=M`` is equivalent to
        ``style=StopLimitOrder(N, M)``. It is an error to pass both a ``style``
        and ``limit_price`` or ``stop_price``.

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order_value`
        :func:`zipline.api.order_percent`
        """
        if not self._can_order_asset(asset):
            return None

        amount, style = self._calculate_order(
            asset, amount, limit_price, stop_price, style
        )
        return self.blotter.order(asset, amount, style)

    def _calculate_order(
        self, asset, amount, limit_price=None, stop_price=None, style=None
    ):
        amount = self.round_order(amount)

        # Raises a ZiplineError if invalid parameters are detected.
        self.validate_order_params(asset, amount, limit_price, stop_price, style)

        # Convert deprecated limit_price and stop_price parameters to use
        # ExecutionStyle objects.
        style = self.__convert_order_params_for_blotter(
            asset, limit_price, stop_price, style
        )
        return amount, style

    @staticmethod
    def round_order(amount):
        """Convert number of shares to an integer.

        By default, truncates to the integer share count that's either within
        .0001 of amount or closer to zero.

        E.g. 3.9999 -> 4.0; 5.5 -> 5.0; -5.5 -> -5.0
        """
        return int(round_if_near_integer(amount))

    def validate_order_params(self, asset, amount, limit_price, stop_price, style):
        """
        Helper method for validating parameters to the order API function.

        Raises an UnsupportedOrderParameters if invalid arguments are found.
        """

        if not self.initialized:
            raise OrderDuringInitialize(
                msg="order() can only be called from within handle_data()"
            )

        if style:
            if limit_price:
                raise UnsupportedOrderParameters(
                    msg="Passing both limit_price and style is not supported."
                )

            if stop_price:
                raise UnsupportedOrderParameters(
                    msg="Passing both stop_price and style is not supported."
                )

        for control in self.trading_controls:
            control.validate(
                asset,
                amount,
                self.portfolio,
                self.get_datetime(),
                self.trading_client.current_data,
            )

    @staticmethod
    def __convert_order_params_for_blotter(asset, limit_price, stop_price, style):
        """Helper method for converting deprecated limit_price and stop_price
        arguments into ExecutionStyle instances.

        This function assumes that either style == None or (limit_price,
        stop_price) == (None, None).
        """
        if style:
            assert (limit_price, stop_price) == (None, None)
            return style
        if limit_price and stop_price:
            return StopLimitOrder(limit_price, stop_price, asset=asset)
        if limit_price:
            return LimitOrder(limit_price, asset=asset)
        if stop_price:
            return StopOrder(stop_price, asset=asset)
        else:
            return MarketOrder()

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_value(self, asset, value, limit_price=None, stop_price=None, style=None):
        """Place an order for a fixed amount of money.

        Equivalent to ``order(asset, value / data.current(asset, 'price'))``.

        Parameters
        ----------
        asset : Asset
            The asset to be ordered.
        value : float
            Amount of value of ``asset`` to be transacted. The number of shares
            bought or sold will be equal to ``value / current_price``.
        limit_price : float, optional
            Limit price for the order.
        stop_price : float, optional
            Stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_percent`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_value_amount(asset, value)
        return self.order(
            asset,
            amount,
            limit_price=limit_price,
            stop_price=stop_price,
            style=style,
        )

    @property
    def recorded_vars(self):
        return copy(self._recorded_vars)

    def _sync_last_sale_prices(self, dt=None):
        """Sync the last sale prices on the metrics tracker to a given
        datetime.

        Parameters
        ----------
        dt : datetime
            The time to sync the prices to.

        Notes
        -----
        This call is cached by the datetime. Repeated calls in the same bar
        are cheap.
        """
        if dt is None:
            dt = self.datetime

        if dt != self._last_sync_time:
            self.metrics_tracker.sync_last_sale_prices(
                dt,
                self.data_portal,
            )
            self._last_sync_time = dt

    @property
    def portfolio(self):
        self._sync_last_sale_prices()
        return self.metrics_tracker.portfolio

    @property
    def account(self):
        self._sync_last_sale_prices()
        return self.metrics_tracker.account

    def set_logger(self, logger):
        self.logger = logger

    def on_dt_changed(self, dt):
        """Callback triggered by the simulation loop whenever the current dt
        changes.

        Any logic that should happen exactly once at the start of each datetime
        group should happen here.
        """
        self.datetime = dt
        self.blotter.set_date(dt)

    @api_method
    @preprocess(tz=coerce_string(pytz.timezone))
    @expect_types(tz=optional(tzinfo))
    def get_datetime(self, tz=None):
        """Returns the current simulation datetime.

        Parameters
        ----------
        tz : tzinfo or str, optional
            The timezone to return the datetime in. This defaults to utc.

        Returns
        -------
        dt : datetime
            The current simulation datetime converted to ``tz``.
        """
        dt = self.datetime
        assert dt.tzinfo == timezone.utc, "Algorithm should have a utc datetime"
        if tz is not None:
            dt = dt.astimezone(tz)
        return dt

    @api_method
    def set_slippage(self, us_equities=None, us_futures=None):
        """Set the slippage models for the simulation.

        Parameters
        ----------
        us_equities : EquitySlippageModel
            The slippage model to use for trading US equities.
        us_futures : FutureSlippageModel
            The slippage model to use for trading US futures.

        Notes
        -----
        This function can only be called during
        :func:`~zipline.api.initialize`.

        See Also
        --------
        :class:`zipline.finance.slippage.SlippageModel`
        """
        if self.initialized:
            raise SetSlippagePostInit()

        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleSlippageModel(
                    asset_type="equities",
                    given_model=us_equities,
                    supported_asset_types=us_equities.allowed_asset_types,
                )
            self.blotter.slippage_models[Equity] = us_equities

        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleSlippageModel(
                    asset_type="futures",
                    given_model=us_futures,
                    supported_asset_types=us_futures.allowed_asset_types,
                )
            self.blotter.slippage_models[Future] = us_futures

    @api_method
    def set_commission(self, us_equities=None, us_futures=None):
        """Sets the commission models for the simulation.

        Parameters
        ----------
        us_equities : EquityCommissionModel
            The commission model to use for trading US equities.
        us_futures : FutureCommissionModel
            The commission model to use for trading US futures.

        Notes
        -----
        This function can only be called during
        :func:`~zipline.api.initialize`.

        See Also
        --------
        :class:`zipline.finance.commission.PerShare`
        :class:`zipline.finance.commission.PerTrade`
        :class:`zipline.finance.commission.PerDollar`
        """
        if self.initialized:
            raise SetCommissionPostInit()

        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleCommissionModel(
                    asset_type="equities",
                    given_model=us_equities,
                    supported_asset_types=us_equities.allowed_asset_types,
                )
            self.blotter.commission_models[Equity] = us_equities

        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleCommissionModel(
                    asset_type="futures",
                    given_model=us_futures,
                    supported_asset_types=us_futures.allowed_asset_types,
                )
            self.blotter.commission_models[Future] = us_futures

    @api_method
    def set_cancel_policy(self, cancel_policy):
        """Sets the order cancellation policy for the simulation.

        Parameters
        ----------
        cancel_policy : CancelPolicy
            The cancellation policy to use.

        See Also
        --------
        :class:`zipline.api.EODCancel`
        :class:`zipline.api.NeverCancel`
        """
        if not isinstance(cancel_policy, CancelPolicy):
            raise UnsupportedCancelPolicy()

        if self.initialized:
            raise SetCancelPolicyPostInit()

        self.blotter.cancel_policy = cancel_policy

    @api_method
    def set_symbol_lookup_date(self, dt):
        """Set the date for which symbols will be resolved to their assets
        (symbols may map to different firms or underlying assets at
        different times)

        Parameters
        ----------
        dt : datetime
            The new symbol lookup date.
        """
        try:
            self._symbol_lookup_date = pd.Timestamp(dt).tz_localize("UTC")
        except TypeError:
            self._symbol_lookup_date = pd.Timestamp(dt).tz_convert("UTC")
        except ValueError as exc:
            raise UnsupportedDatetimeFormat(
                input=dt, method="set_symbol_lookup_date"
            ) from exc

    @property
    def data_frequency(self):
        return self.sim_params.data_frequency

    @data_frequency.setter
    def data_frequency(self, value):
        assert value in ("daily", "minute")
        self.sim_params.data_frequency = value

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_percent(
        self, asset, percent, limit_price=None, stop_price=None, style=None
    ):
        """Place an order in the specified asset corresponding to the given
        percent of the current portfolio value.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        percent : float
            The percentage of the portfolio value to allocate to ``asset``.
            This is specified as a decimal, for example: 0.50 means 50%.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_value`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_percent_amount(asset, percent)
        return self.order(
            asset,
            amount,
            limit_price=limit_price,
            stop_price=stop_price,
            style=style,
        )

    def _calculate_order_percent_amount(self, asset, percent):
        value = self.portfolio.portfolio_value * percent
        return self._calculate_order_value_amount(asset, value)

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target(
        self, asset, target, limit_price=None, stop_price=None, style=None
    ):
        """Place an order to adjust a position to a target number of shares. If
        the position doesn't already exist, this is equivalent to placing a new
        order. If the position does exist, this is equivalent to placing an
        order for the difference between the target number of shares and the
        current number of shares.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        target : int
            The desired number of shares of ``asset``.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.


        Notes
        -----
        ``order_target`` does not take into account any open orders. For
        example:

        .. code-block:: python

           order_target(sid(0), 10)
           order_target(sid(0), 10)

        This code will result in 20 shares of ``sid(0)`` because the first
        call to ``order_target`` will not have been filled when the second
        ``order_target`` call is made.

        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_target_percent`
        :func:`zipline.api.order_target_value`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_target_amount(asset, target)
        return self.order(
            asset,
            amount,
            limit_price=limit_price,
            stop_price=stop_price,
            style=style,
        )

    def _calculate_order_target_amount(self, asset, target):
        if asset in self.portfolio.positions:
            current_position = self.portfolio.positions[asset].amount
            target -= current_position

        return target

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target_value(
        self, asset, target, limit_price=None, stop_price=None, style=None
    ):
        """Place an order to adjust a position to a target value. If
        the position doesn't already exist, this is equivalent to placing a new
        order. If the position does exist, this is equivalent to placing an
        order for the difference between the target value and the
        current value.
        If the Asset being ordered is a Future, the 'target value' calculated
        is actually the target exposure, as Futures have no 'value'.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        target : float
            The desired total value of ``asset``.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        ``order_target_value`` does not take into account any open orders. For
        example:

        .. code-block:: python

           order_target_value(sid(0), 10)
           order_target_value(sid(0), 10)

        This code will result in 20 dollars of ``sid(0)`` because the first
        call to ``order_target_value`` will not have been filled when the
        second ``order_target_value`` call is made.

        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_target`
        :func:`zipline.api.order_target_percent`
        """
        if not self._can_order_asset(asset):
            return None

        target_amount = self._calculate_order_value_amount(asset, target)
        amount = self._calculate_order_target_amount(asset, target_amount)
        return self.order(
            asset,
            amount,
            limit_price=limit_price,
            stop_price=stop_price,
            style=style,
        )

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target_percent(
        self, asset, target, limit_price=None, stop_price=None, style=None
    ):
        """Place an order to adjust a position to a target percent of the
        current portfolio value. If the position doesn't already exist, this is
        equivalent to placing a new order. If the position does exist, this is
        equivalent to placing an order for the difference between the target
        percent and the current percent.

        Parameters
        ----------
        asset : Asset
            The asset that this order is for.
        target : float
            The desired percentage of the portfolio value to allocate to
            ``asset``. This is specified as a decimal, for example:
            0.50 means 50%.
        limit_price : float, optional
            The limit price for the order.
        stop_price : float, optional
            The stop price for the order.
        style : ExecutionStyle
            The execution style for the order.

        Returns
        -------
        order_id : str
            The unique identifier for this order.

        Notes
        -----
        ``order_target_value`` does not take into account any open orders. For
        example:

        .. code-block:: python

           order_target_percent(sid(0), 10)
           order_target_percent(sid(0), 10)

        This code will result in 20% of the portfolio being allocated to sid(0)
        because the first call to ``order_target_percent`` will not have been
        filled when the second ``order_target_percent`` call is made.

        See :func:`zipline.api.order` for more information about
        ``limit_price``, ``stop_price``, and ``style``

        See Also
        --------
        :class:`zipline.finance.execution.ExecutionStyle`
        :func:`zipline.api.order`
        :func:`zipline.api.order_target`
        :func:`zipline.api.order_target_value`
        """
        if not self._can_order_asset(asset):
            return None

        amount = self._calculate_order_target_percent_amount(asset, target)
        return self.order(
            asset,
            amount,
            limit_price=limit_price,
            stop_price=stop_price,
            style=style,
        )

    def _calculate_order_target_percent_amount(self, asset, target):
        target_amount = self._calculate_order_percent_amount(asset, target)
        return self._calculate_order_target_amount(asset, target_amount)

    @api_method
    @expect_types(share_counts=pd.Series)
    @expect_dtypes(share_counts=int64_dtype)
    def batch_market_order(self, share_counts):
        """Place a batch market order for multiple assets.

        Parameters
        ----------
        share_counts : pd.Series[Asset -> int]
            Map from asset to number of shares to order for that asset.

        Returns
        -------
        order_ids : pd.Index[str]
            Index of ids for newly-created orders.
        """
        style = MarketOrder()
        order_args = [
            (asset, amount, style) for (asset, amount) in share_counts.items() if amount
        ]
        return self.blotter.batch_order(order_args)

    @error_keywords(
        sid="Keyword argument `sid` is no longer supported for "
        "get_open_orders. Use `asset` instead."
    )
    @api_method
    def get_open_orders(self, asset=None):
        """Retrieve all of the current open orders.

        Parameters
        ----------
        asset : Asset
            If passed and not None, return only the open orders for the given
            asset instead of all open orders.

        Returns
        -------
        open_orders : dict[list[Order]] or list[Order]
            If no asset is passed this will return a dict mapping Assets
            to a list containing all the open orders for the asset.
            If an asset is passed then this will return a list of the open
            orders for this asset.
        """
        if asset is None:
            return {
                key: [order.to_api_obj() for order in orders]
                for key, orders in self.blotter.open_orders.items()
                if orders
            }
        if asset in self.blotter.open_orders:
            orders = self.blotter.open_orders[asset]
            return [order.to_api_obj() for order in orders]
        return []

    @api_method
    def get_order(self, order_id):
        """Lookup an order based on the order id returned from one of the
        order functions.

        Parameters
        ----------
        order_id : str
            The unique identifier for the order.

        Returns
        -------
        order : Order
            The order object.
        """
        if order_id in self.blotter.orders:
            return self.blotter.orders[order_id].to_api_obj()

    @api_method
    def cancel_order(self, order_param):
        """Cancel an open order.

        Parameters
        ----------
        order_param : str or Order
            The order_id or order object to cancel.
        """
        order_id = order_param
        if isinstance(order_param, zipline.protocol.Order):
            order_id = order_param.id

        self.blotter.cancel(order_id)

    ####################
    # Account Controls #
    ####################

    def register_account_control(self, control):
        """
        Register a new AccountControl to be checked on each bar.
        """
        if self.initialized:
            raise RegisterAccountControlPostInit()
        self.account_controls.append(control)

    def validate_account_controls(self):
        for control in self.account_controls:
            control.validate(
                self.portfolio,
                self.account,
                self.get_datetime(),
                self.trading_client.current_data,
            )

    @api_method
    def set_max_leverage(self, max_leverage):
        """Set a limit on the maximum leverage of the algorithm.

        Parameters
        ----------
        max_leverage : float
            The maximum leverage for the algorithm. If not provided there will
            be no maximum.
        """
        control = MaxLeverage(max_leverage)
        self.register_account_control(control)

    @api_method
    def set_min_leverage(self, min_leverage, grace_period):
        """Set a limit on the minimum leverage of the algorithm.

        Parameters
        ----------
        min_leverage : float
            The minimum leverage for the algorithm.
        grace_period : pd.Timedelta
            The offset from the start date used to enforce a minimum leverage.
        """
        deadline = self.sim_params.start_session + grace_period
        control = MinLeverage(min_leverage, deadline)
        self.register_account_control(control)

    ####################
    # Trading Controls #
    ####################

    def register_trading_control(self, control):
        """
        Register a new TradingControl to be checked prior to order calls.
        """
        if self.initialized:
            raise RegisterTradingControlPostInit()
        self.trading_controls.append(control)

    @api_method
    def set_max_position_size(
        self, asset=None, max_shares=None, max_notional=None, on_error="fail"
    ):
        """Set a limit on the number of shares and/or dollar value held for the
        given sid. Limits are treated as absolute values and are enforced at
        the time that the algo attempts to place an order for sid. This means
        that it's possible to end up with more than the max number of shares
        due to splits/dividends, and more than the max notional due to price
        improvement.

        If an algorithm attempts to place an order that would result in
        increasing the absolute value of shares/dollar value exceeding one of
        these limits, raise a TradingControlException.

        Parameters
        ----------
        asset : Asset, optional
            If provided, this sets the guard only on positions in the given
            asset.
        max_shares : int, optional
            The maximum number of shares to hold for an asset.
        max_notional : float, optional
            The maximum value to hold for an asset.
        """
        control = MaxPositionSize(
            asset=asset,
            max_shares=max_shares,
            max_notional=max_notional,
            on_error=on_error,
        )
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(
        self, asset=None, max_shares=None, max_notional=None, on_error="fail"
    ):
        """Set a limit on the number of shares and/or dollar value of any single
        order placed for sid.  Limits are treated as absolute values and are
        enforced at the time that the algo attempts to place an order for sid.

        If an algorithm attempts to place an order that would result in
        exceeding one of these limits, raise a TradingControlException.

        Parameters
        ----------
        asset : Asset, optional
            If provided, this sets the guard only on positions in the given
            asset.
        max_shares : int, optional
            The maximum number of shares that can be ordered at one time.
        max_notional : float, optional
            The maximum value that can be ordered at one time.
        """
        control = MaxOrderSize(
            asset=asset,
            max_shares=max_shares,
            max_notional=max_notional,
            on_error=on_error,
        )
        self.register_trading_control(control)

    @api_method
    def set_max_order_count(self, max_count, on_error="fail"):
        """Set a limit on the number of orders that can be placed in a single
        day.

        Parameters
        ----------
        max_count : int
            The maximum number of orders that can be placed on any single day.
        """
        control = MaxOrderCount(on_error, max_count)
        self.register_trading_control(control)

    @api_method
    def set_do_not_order_list(self, restricted_list, on_error="fail"):
        """Set a restriction on which assets can be ordered.

        Parameters
        ----------
        restricted_list : container[Asset], SecurityList
            The assets that cannot be ordered.
        """
        if isinstance(restricted_list, SecurityList):
            warnings.warn(
                "`set_do_not_order_list(security_lists.leveraged_etf_list)` "
                "is deprecated. Use `set_asset_restrictions("
                "security_lists.restrict_leveraged_etfs)` instead.",
                category=ZiplineDeprecationWarning,
                stacklevel=2,
            )
            restrictions = SecurityListRestrictions(restricted_list)
        else:
            warnings.warn(
                "`set_do_not_order_list(container_of_assets)` is deprecated. "
                "Create a zipline.finance.asset_restrictions."
                "StaticRestrictions object with a container of assets and use "
                "`set_asset_restrictions(StaticRestrictions("
                "container_of_assets))` instead.",
                category=ZiplineDeprecationWarning,
                stacklevel=2,
            )
            restrictions = StaticRestrictions(restricted_list)

        self.set_asset_restrictions(restrictions, on_error)

    @api_method
    @expect_types(
        restrictions=Restrictions,
        on_error=str,
    )
    def set_asset_restrictions(self, restrictions, on_error="fail"):
        """Set a restriction on which assets can be ordered.

        Parameters
        ----------
        restricted_list : Restrictions
            An object providing information about restricted assets.

        See Also
        --------
        zipline.finance.asset_restrictions.Restrictions
        """
        control = RestrictedListOrder(on_error, restrictions)
        self.register_trading_control(control)
        self.restrictions |= restrictions

    @api_method
    def set_long_only(self, on_error="fail"):
        """Set a rule specifying that this algorithm cannot take short
        positions.
        """
        self.register_trading_control(LongOnly(on_error))

    ##############
    # Pipeline API
    ##############
    @api_method
    @require_not_initialized(AttachPipelineAfterInitialize())
    @expect_types(
        pipeline=Pipeline,
        name=str,
        chunks=(int, Iterable, type(None)),
    )
    def attach_pipeline(self, pipeline, name, chunks=None, eager=True):
        """Register a pipeline to be computed at the start of each day.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to have computed.
        name : str
            The name of the pipeline.
        chunks : int or iterator, optional
            The number of days to compute pipeline results for. Increasing
            this number will make it longer to get the first results but
            may improve the total runtime of the simulation. If an iterator
            is passed, we will run in chunks based on values of the iterator.
            Default is True.
        eager : bool, optional
            Whether or not to compute this pipeline prior to
            before_trading_start.

        Returns
        -------
        pipeline : Pipeline
            Returns the pipeline that was attached unchanged.

        See Also
        --------
        :func:`zipline.api.pipeline_output`
        """
        if chunks is None:
            # Make the first chunk smaller to get more immediate results:
            # (one week, then every half year)
            chunks = chain([5], repeat(126))
        elif isinstance(chunks, int):
            chunks = repeat(chunks)

        if name in self._pipelines:
            raise DuplicatePipelineName(name=name)

        self._pipelines[name] = AttachedPipeline(pipeline, iter(chunks), eager)

        # Return the pipeline to allow expressions like
        # p = attach_pipeline(Pipeline(), 'name')
        return pipeline

    @api_method
    @require_initialized(PipelineOutputDuringInitialize())
    def pipeline_output(self, name):
        """Get results of the pipeline attached by with name ``name``.

        Parameters
        ----------
        name : str
            Name of the pipeline from which to fetch results.

        Returns
        -------
        results : pd.DataFrame
            DataFrame containing the results of the requested pipeline for
            the current simulation date.

        Raises
        ------
        NoSuchPipeline
            Raised when no pipeline with the name `name` has been registered.

        See Also
        --------
        :func:`zipline.api.attach_pipeline`
        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
        """
        try:
            pipe, chunks, _ = self._pipelines[name]
        except KeyError as exc:
            raise NoSuchPipeline(
                name=name,
                valid=list(self._pipelines.keys()),
            ) from exc
        return self._pipeline_output(pipe, chunks, name)

    def _pipeline_output(self, pipeline, chunks, name):
        """Internal implementation of `pipeline_output`."""
        # TODO FIXME TZ MESS
        today = self.get_datetime().normalize().tz_localize(None)
        try:
            data = self._pipeline_cache.get(name, today)
        except KeyError:
            # Calculate the next block.
            data, valid_until = self.run_pipeline(
                pipeline,
                today,
                next(chunks),
            )
            self._pipeline_cache.set(name, data, valid_until)

        # Now that we have a cached result, try to return the data for today.
        try:
            return data.loc[today]
        except KeyError:
            # This happens if no assets passed the pipeline screen on a given
            # day.
            return pd.DataFrame(index=[], columns=data.columns)

    def run_pipeline(self, pipeline, start_session, chunksize):
        """Compute `pipeline`, providing values for at least `start_date`.

        Produces a DataFrame containing data for days between `start_date` and
        `end_date`, where `end_date` is defined by:

            `end_date = min(start_date + chunksize trading days,
                            simulation_end)`

        Returns
        -------
        (data, valid_until) : tuple (pd.DataFrame, pd.Timestamp)

        See Also
        --------
        PipelineEngine.run_pipeline
        """
        sessions = self.trading_calendar.sessions

        # Load data starting from the previous trading day...
        start_date_loc = sessions.get_loc(start_session)

        # ...continuing until either the day before the simulation end, or
        # until chunksize days of data have been loaded.
        sim_end_session = self.sim_params.end_session

        end_loc = min(start_date_loc + chunksize, sessions.get_loc(sim_end_session))

        end_session = sessions[end_loc]

        return (
            self.engine.run_pipeline(pipeline, start_session, end_session),
            end_session,
        )

    @staticmethod
    def default_pipeline_domain(calendar):
        """Get a default pipeline domain for algorithms running on ``calendar``.

        This will be used to infer a domain for pipelines that only use generic
        datasets when running in the context of a TradingAlgorithm.
        """
        return _DEFAULT_DOMAINS.get(calendar.name, domain.GENERIC)

    @staticmethod
    def default_fetch_csv_country_code(calendar):
        """Get a default country_code to use for fetch_csv symbol lookups.

        This will be used to disambiguate symbol lookups for fetch_csv calls if
        our asset db contains entries with the same ticker spread across
        multiple
        """
        return _DEFAULT_FETCH_CSV_COUNTRY_CODES.get(calendar.name)

    ##################
    # End Pipeline API
    ##################

    @classmethod
    def all_api_methods(cls):
        """Return a list of all the TradingAlgorithm API methods."""
        return [fn for fn in vars(cls).values() if getattr(fn, "is_api_method", False)]


# Map from calendar name to default domain for that calendar.
_DEFAULT_DOMAINS = {d.calendar_name: d for d in domain.BUILT_IN_DOMAINS}
# Map from calendar name to default country code for that calendar.
_DEFAULT_FETCH_CSV_COUNTRY_CODES = {
    d.calendar_name: d.country_code for d in domain.BUILT_IN_DOMAINS
}
# Include us_futures, which doesn't have a pipeline domain.
_DEFAULT_FETCH_CSV_COUNTRY_CODES["us_futures"] = "US"
