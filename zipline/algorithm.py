#
# Copyright 2014 Quantopian, Inc.
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
from copy import copy
import warnings

import pytz
import pandas as pd
import numpy as np

from datetime import datetime

from itertools import groupby, chain
from six.moves import filter
from six import (
    exec_,
    iteritems,
    itervalues,
    string_types,
)
from operator import attrgetter


from zipline.errors import (
    AddTermPostInit,
    OrderDuringInitialize,
    OverrideCommissionPostInit,
    OverrideSlippagePostInit,
    RegisterAccountControlPostInit,
    RegisterTradingControlPostInit,
    UnsupportedCommissionModel,
    UnsupportedOrderParameters,
    UnsupportedSlippageModel,
)
from zipline.finance.trading import TradingEnvironment
from zipline.finance.blotter import Blotter
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance.controls import (
    LongOnly,
    MaxOrderCount,
    MaxOrderSize,
    MaxPositionSize,
    MaxLeverage,
    RestrictedListOrder
)
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.finance.performance import PerformanceTracker
from zipline.finance.slippage import (
    VolumeShareSlippage,
    SlippageModel,
    transact_partial
)
from zipline.assets import Asset, Future
from zipline.assets.futures import FutureChain
from zipline.gens.composites import date_sorted_sources
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.modelling.engine import (
    NoOpFFCEngine,
    SimpleFFCEngine,
)
from zipline.sources import DataFrameSource, DataPanelSource
from zipline.utils.api_support import (
    api_method,
    require_not_initialized,
    ZiplineAPI,
)
import zipline.utils.events
from zipline.utils.events import (
    EventManager,
    make_eventrule,
    DateRuleFactory,
    TimeRuleFactory,
)
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.math_utils import tolerant_equals

import zipline.protocol
from zipline.protocol import Event

from zipline.history import HistorySpec
from zipline.history.history_container import HistoryContainer

DEFAULT_CAPITAL_BASE = float("1.0e5")


class TradingAlgorithm(object):
    """
    Base class for trading algorithms. Inherit and overload
    initialize() and handle_data(data).

    A new algorithm could look like this:
    ```
    from zipline.api import order, symbol

    def initialize(context):
        context.sid = symbol('AAPL')
        context.amount = 100

    def handle_data(context, data):
        sid = context.sid
        amount = context.amount
        order(sid, amount)
    ```
    To then to run this algorithm pass these functions to
    TradingAlgorithm:

    my_algo = TradingAlgorithm(initialize, handle_data)
    stats = my_algo.run(data)

    """

    def __init__(self, *args, **kwargs):
        """Initialize sids and other state variables.

        :Arguments:
        :Optional:
            initialize : function
                Function that is called with a single
                argument at the begninning of the simulation.
            handle_data : function
                Function that is called with 2 arguments
                (context and data) on every bar.
            script : str
                Algoscript that contains initialize and
                handle_data function definition.
            data_frequency : {'daily', 'minute'}
               The duration of the bars.
            capital_base : float <default: 1.0e5>
               How much capital to start with.
            instant_fill : bool <default: False>
               Whether to fill orders immediately or on next bar.
            asset_finder : An AssetFinder object
                A new AssetFinder object to be used in this TradingEnvironment
            asset_metadata: can be either:
                            - dict
                            - pandas.DataFrame
                            - object with 'read' property
                If dict is provided, it must have the following structure:
                * keys are the identifiers
                * values are dicts containing the metadata, with the metadata
                  field name as the key
                If pandas.DataFrame is provided, it must have the
                following structure:
                * column names must be the metadata fields
                * index must be the different asset identifiers
                * array contents should be the metadata value
                If an object with a 'read' property is provided, 'read' must
                return rows containing at least one of 'sid' or 'symbol' along
                with the other metadata fields.
            identifiers : List
                Any asset identifiers that are not provided in the
                asset_metadata, but will be traded by this TradingAlgorithm
        """
        self.sources = []

        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        # List of account controls to be checked on each bar.
        self.account_controls = []

        self._recorded_vars = {}
        self.namespace = kwargs.get('namespace', {})

        self._platform = kwargs.pop('platform', 'zipline')

        self.logger = None

        self.benchmark_return_source = None

        # default components for transact
        self.slippage = VolumeShareSlippage()
        self.commission = PerShare()

        self.instant_fill = kwargs.pop('instant_fill', False)

        # set the capital base
        self.capital_base = kwargs.pop('capital_base', DEFAULT_CAPITAL_BASE)
        self.sim_params = kwargs.pop('sim_params', None)
        if self.sim_params is None:
            self.sim_params = create_simulation_parameters(
                capital_base=self.capital_base,
                start=kwargs.pop('start', None),
                end=kwargs.pop('end', None)
            )
        self.perf_tracker = PerformanceTracker(self.sim_params)

        # Update the TradingEnvironment with the provided asset metadata
        self.trading_environment = kwargs.pop('env',
                                              TradingEnvironment.instance())
        self.trading_environment.write_data(
            equities_data=kwargs.pop('asset_metadata', {}),
            equities_identifiers=kwargs.pop('identifiers', []),
        )

        # Pull in the environment's new AssetFinder for quick reference
        self.asset_finder = self.trading_environment.asset_finder
        self.init_engine(kwargs.pop('ffc_loader', None))

        # Maps from name to Term
        self._filters = {}
        self._factors = {}
        self._classifiers = {}

        self.blotter = kwargs.pop('blotter', None)
        if not self.blotter:
            self.blotter = Blotter()

        # Set the dt initally to the period start by forcing it to change
        self.on_dt_changed(self.sim_params.period_start)

        self.portfolio_needs_update = True
        self.account_needs_update = True
        self.performance_needs_update = True
        self._portfolio = None
        self._account = None

        self.history_container_class = kwargs.pop(
            'history_container_class', HistoryContainer,
        )
        self.history_container = None
        self.history_specs = {}

        # If string is passed in, execute and get reference to
        # functions.
        self.algoscript = kwargs.pop('script', None)

        self._initialize = None
        self._before_trading_start = None
        self._analyze = None

        self.event_manager = EventManager()

        if self.algoscript is not None:
            filename = kwargs.pop('algo_filename', None)
            if filename is None:
                filename = '<string>'
            code = compile(self.algoscript, filename, 'exec')
            exec_(code, self.namespace)
            self._initialize = self.namespace.get('initialize')
            if 'handle_data' not in self.namespace:
                raise ValueError('You must define a handle_data function.')
            else:
                self._handle_data = self.namespace['handle_data']

            self._before_trading_start = \
                self.namespace.get('before_trading_start')
            # Optional analyze function, gets called after run
            self._analyze = self.namespace.get('analyze')

        elif kwargs.get('initialize') and kwargs.get('handle_data'):
            if self.algoscript is not None:
                raise ValueError('You can not set script and \
                initialize/handle_data.')
            self._initialize = kwargs.pop('initialize')
            self._handle_data = kwargs.pop('handle_data')
            self._before_trading_start = kwargs.pop('before_trading_start',
                                                    None)

        self.event_manager.add_event(
            zipline.utils.events.Event(
                zipline.utils.events.Always(),
                # We pass handle_data.__func__ to get the unbound method.
                # We will explicitly pass the algorithm to bind it again.
                self.handle_data.__func__,
            ),
            prepend=True,
        )

        # If method not defined, NOOP
        if self._initialize is None:
            self._initialize = lambda x: None

        # Alternative way of setting data_frequency for backwards
        # compatibility.
        if 'data_frequency' in kwargs:
            self.data_frequency = kwargs.pop('data_frequency')

        self._most_recent_data = None

        # Prepare the algo for initialization
        self.initialized = False
        self.initialize_args = args
        self.initialize_kwargs = kwargs

    def init_engine(self, loader):
        """
        Construct and save an FFCEngine from loader.

        If loader is None, constructs a NoOpFFCEngine.
        """
        if loader is not None:
            self.engine = SimpleFFCEngine(
                loader,
                self.trading_environment.trading_days,
                self.asset_finder,
            )
        else:
            self.engine = NoOpFFCEngine()

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self)

    def before_trading_start(self):
        if self._before_trading_start is None:
            return

        self._before_trading_start(self)

    def handle_data(self, data):
        self._most_recent_data = data
        if self.history_container:
            self.history_container.update(data, self.datetime)

        self._handle_data(self, data)

        # Unlike trading controls which remain constant unless placing an
        # order, account controls can change each bar. Thus, must check
        # every bar no matter if the algorithm places an order or not.
        self.validate_account_controls()

    def analyze(self, perf):
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

    def __repr__(self):
        """
        N.B. this does not yet represent a string that can be used
        to instantiate an exact copy of an algorithm.

        However, it is getting close, and provides some value as something
        that can be inspected interactively.
        """
        return """
{class_name}(
    capital_base={capital_base}
    sim_params={sim_params},
    initialized={initialized},
    slippage={slippage},
    commission={commission},
    blotter={blotter},
    recorded_vars={recorded_vars})
""".strip().format(class_name=self.__class__.__name__,
                   capital_base=self.capital_base,
                   sim_params=repr(self.sim_params),
                   initialized=self.initialized,
                   slippage=repr(self.slippage),
                   commission=repr(self.commission),
                   blotter=repr(self.blotter),
                   recorded_vars=repr(self.recorded_vars))

    def _create_data_generator(self, source_filter, sim_params=None):
        """
        Create a merged data generator using the sources attached to this
        algorithm.

        ::source_filter:: is a method that receives events in date
        sorted order, and returns True for those events that should be
        processed by the zipline, and False for those that should be
        skipped.
        """
        if sim_params is None:
            sim_params = self.sim_params

        if self.benchmark_return_source is None:
            if sim_params.data_frequency == 'minute' or \
               sim_params.emission_rate == 'minute':
                def update_time(date):
                    return self.trading_environment.get_open_and_close(date)[1]
            else:
                def update_time(date):
                    return date
            benchmark_return_source = [
                Event({'dt': update_time(dt),
                       'returns': ret,
                       'type': zipline.protocol.DATASOURCE_TYPE.BENCHMARK,
                       'source_id': 'benchmarks'})
                for dt, ret in
                self.trading_environment.benchmark_returns.iteritems()
                if dt.date() >= sim_params.period_start.date() and
                dt.date() <= sim_params.period_end.date()
            ]
        else:
            benchmark_return_source = self.benchmark_return_source

        date_sorted = date_sorted_sources(*self.sources)

        if source_filter:
            date_sorted = filter(source_filter, date_sorted)

        with_benchmarks = date_sorted_sources(benchmark_return_source,
                                              date_sorted)

        # Group together events with the same dt field. This depends on the
        # events already being sorted.
        return groupby(with_benchmarks, attrgetter('dt'))

    def _create_generator(self, sim_params, source_filter=None):
        """
        Create a basic generator setup using the sources to this algorithm.

        ::source_filter:: is a method that receives events in date
        sorted order, and returns True for those events that should be
        processed by the zipline, and False for those that should be
        skipped.
        """

        if not self.initialized:
            self.initialize(*self.initialize_args, **self.initialize_kwargs)
            self.initialized = True

        if self.perf_tracker is None:
            # HACK: When running with the `run` method, we set perf_tracker to
            # None so that it will be overwritten here.
            self.perf_tracker = PerformanceTracker(sim_params)

        self.portfolio_needs_update = True
        self.account_needs_update = True
        self.performance_needs_update = True

        self.data_gen = self._create_data_generator(source_filter, sim_params)

        self.trading_client = AlgorithmSimulator(self, sim_params)

        transact_method = transact_partial(self.slippage, self.commission)
        self.set_transact(transact_method)

        return self.trading_client.transform(self.data_gen)

    def get_generator(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

    # TODO: make a new subclass, e.g. BatchAlgorithm, and move
    # the run method to the subclass, and refactor to put the
    # generator creation logic into get_generator.
    def run(self, source, overwrite_sim_params=True,
            benchmark_return_source=None):
        """Run the algorithm.

        :Arguments:
            source : can be either:
                     - pandas.DataFrame
                     - zipline source
                     - list of sources

               If pandas.DataFrame is provided, it must have the
               following structure:
               * column names must be the different asset identifiers
               * index must be DatetimeIndex
               * array contents should be price info.

        :Returns:
            daily_stats : pandas.DataFrame
              Daily performance metrics such as returns, alpha etc.

        """

        # Ensure that source is a DataSource object
        if isinstance(source, list):
            if overwrite_sim_params:
                warnings.warn("""List of sources passed, will not attempt to extract start and end
 dates. Make sure to set the correct fields in sim_params passed to
 __init__().""", UserWarning)
                overwrite_sim_params = False
        elif isinstance(source, pd.DataFrame):
            # if DataFrame provided, map columns to sids and wrap
            # in DataFrameSource
            copy_frame = source.copy()
            self.trading_environment.write_data(
                equities_identifiers=source.columns)
            copy_frame.columns = \
                self.asset_finder.map_identifier_index_to_sids(
                    source.columns, source.index[0]
                )
            source = DataFrameSource(copy_frame)

        elif isinstance(source, pd.Panel):
            # If Panel provided, map items to sids and wrap
            # in DataPanelSource
            copy_panel = source.copy()
            self.trading_environment.write_data(
                equities_identifiers=source.items)
            copy_panel.items = self.asset_finder.map_identifier_index_to_sids(
                source.items, source.major_axis[0]
            )
            source = DataPanelSource(copy_panel)

        if isinstance(source, list):
            self.set_sources(source)
        else:
            self.set_sources([source])

        # Override sim_params if params are provided by the source.
        if overwrite_sim_params:
            if hasattr(source, 'start'):
                self.sim_params.period_start = source.start
            if hasattr(source, 'end'):
                self.sim_params.period_end = source.end
            # Changing period_start and period_close might require updating
            # of first_open and last_close.
            self.sim_params._update_internal()

        # The sids field of the source is the reference for the universe at
        # the start of the run
        self._current_universe = set()
        for source in self.sources:
            for sid in source.sids:
                self._current_universe.add(sid)
        # Check that all sids from the source are accounted for in
        # the AssetFinder. This retrieve call will raise an exception if the
        # sid is not found.
        for sid in self._current_universe:
            self.asset_finder.retrieve_asset(sid)

        # force a reset of the performance tracker, in case
        # this is a repeat run of the algorithm.
        self.perf_tracker = None

        # create zipline
        self.gen = self._create_generator(self.sim_params)

        # Create history containers
        if self.history_specs:
            self.history_container = self.history_container_class(
                self.history_specs,
                self.current_universe(),
                self.sim_params.first_open,
                self.sim_params.data_frequency,
            )

        # loop through simulated_trading, each iteration returns a
        # perf dictionary
        perfs = []
        for perf in self.gen:
            perfs.append(perf)

        # convert perf dict to pandas dataframe
        daily_stats = self._create_daily_stats(perfs)

        self.analyze(daily_stats)

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        # TODO: the loop here could overwrite expected properties
        # of daily_perf. Could potentially raise or log a
        # warning.
        for perf in perfs:
            if 'daily_perf' in perf:

                perf['daily_perf'].update(
                    perf['daily_perf'].pop('recorded_vars')
                )
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])
            else:
                self.risk_report = perf

        daily_dts = [np.datetime64(perf['period_close'], utc=True)
                     for perf in daily_perfs]
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)

        return daily_stats

    @api_method
    def add_transform(self, transform, days=None):
        """
        Ensures that the history container will have enough size to service
        a simple transform.

        :Arguments:
            transform : string
                The transform to add. must be an element of:
                {'mavg', 'stddev', 'vwap', 'returns'}.
            days : int <default=None>
                The maximum amount of days you will want for this transform.
                This is not needed for 'returns'.
        """
        if transform not in {'mavg', 'stddev', 'vwap', 'returns'}:
            raise ValueError('Invalid transform')

        if transform == 'returns':
            if days is not None:
                raise ValueError('returns does use days')

            self.add_history(2, '1d', 'price')
            return
        elif days is None:
            raise ValueError('no number of days specified')

        if self.sim_params.data_frequency == 'daily':
            mult = 1
            freq = '1d'
        else:
            mult = 390
            freq = '1m'

        bars = mult * days
        self.add_history(bars, freq, 'price')

        if transform == 'vwap':
            self.add_history(bars, freq, 'volume')

    @api_method
    def get_environment(self, field='platform'):
        env = {
            'arena': self.sim_params.arena,
            'data_frequency': self.sim_params.data_frequency,
            'start': self.sim_params.first_open,
            'end': self.sim_params.last_close,
            'capital_base': self.sim_params.capital_base,
            'platform': self._platform
        }
        if field == '*':
            return env
        else:
            return env[field]

    def add_event(self, rule=None, callback=None):
        """
        Adds an event to the algorithm's EventManager.
        """
        self.event_manager.add_event(
            zipline.utils.events.Event(rule, callback),
        )

    @api_method
    def schedule_function(self,
                          func,
                          date_rule=None,
                          time_rule=None,
                          half_days=True):
        """
        Schedules a function to be called with some timed rules.
        """
        date_rule = date_rule or DateRuleFactory.every_day()
        time_rule = ((time_rule or TimeRuleFactory.market_open())
                     if self.sim_params.data_frequency == 'minute' else
                     # If we are in daily mode the time_rule is ignored.
                     zipline.utils.events.Always())

        self.add_event(
            make_eventrule(date_rule, time_rule, half_days),
            func,
        )

    @api_method
    def record(self, *args, **kwargs):
        """
        Track and record local variable (i.e. attributes) each day.
        """
        # Make 2 objects both referencing the same iterator
        args = [iter(args)] * 2

        # Zip generates list entries by calling `next` on each iterator it
        # receives.  In this case the two iterators are the same object, so the
        # call to next on args[0] will also advance args[1], resulting in zip
        # returning (a,b) (c,d) (e,f) rather than (a,a) (b,b) (c,c) etc.
        positionals = zip(*args)
        for name, value in chain(positionals, iteritems(kwargs)):
            self._recorded_vars[name] = value

    @api_method
    def symbol(self, symbol_str):
        """
        Default symbol lookup for any source that directly maps the
        symbol to the Asset (e.g. yahoo finance).
        """
        return self.asset_finder.lookup_symbol_resolve_multiple(
            symbol_str,
            as_of_date=self.datetime)

    @api_method
    def symbols(self, *args):
        """
        Default symbols lookup for any source that directly maps the
        symbol to the Asset (e.g. yahoo finance).
        """
        return [self.symbol(identifier) for identifier in args]

    @api_method
    def sid(self, a_sid):
        """
        Default sid lookup for any source that directly maps the integer sid
        to the Asset.
        """
        return self.asset_finder.retrieve_asset(a_sid)

    @api_method
    def future_chain(self, root_symbol, as_of_date=None):
        """ Look up a future chain with the specified parameters.

        Parameters
        ----------
        root_symbol : str
            The root symbol of a future chain.
        as_of_date : datetime.datetime or pandas.Timestamp or str, optional
            Date at which the chain determination is rooted. I.e. the
            existing contract whose notice date is first after this date is
            the primary contract, etc.

        Returns
        -------
        FutureChain
            The future chain matching the specified parameters.

        Raises
        ------
        RootSymbolNotFound
            If a future chain could not be found for the given root symbol.
        """
        return FutureChain(
            asset_finder=self.asset_finder,
            get_datetime=self.get_datetime,
            root_symbol=root_symbol.upper(),
            as_of_date=as_of_date
        )

    def _calculate_order_value_amount(self, asset, value):
        """
        Calculates how many shares/contracts to order based on the type of
        asset being ordered.
        """
        last_price = self.trading_client.current_data[asset].price

        if tolerant_equals(last_price, 0):
            zero_message = "Price of 0 for {psid}; can't infer value".format(
                psid=asset
            )
            if self.logger:
                self.logger.debug(zero_message)
            # Don't place any order
            return 0

        if isinstance(asset, Future):
            value_multiplier = asset.contract_multiplier
        else:
            value_multiplier = 1

        return value / (last_price * value_multiplier)

    @api_method
    def order(self, sid, amount,
              limit_price=None,
              stop_price=None,
              style=None):
        """
        Place an order using the specified parameters.
        """

        def round_if_near_integer(a, epsilon=1e-4):
            """
            Round a to the nearest integer if that integer is within an epsilon
            of a.
            """
            if abs(a - round(a)) <= epsilon:
                return round(a)
            else:
                return a

        # Truncate to the integer share count that's either within .0001 of
        # amount or closer to zero.
        # E.g. 3.9999 -> 4.0; 5.5 -> 5.0; -5.5 -> -5.0
        amount = int(round_if_near_integer(amount))

        # Raises a ZiplineError if invalid parameters are detected.
        self.validate_order_params(sid,
                                   amount,
                                   limit_price,
                                   stop_price,
                                   style)

        # Convert deprecated limit_price and stop_price parameters to use
        # ExecutionStyle objects.
        style = self.__convert_order_params_for_blotter(limit_price,
                                                        stop_price,
                                                        style)
        return self.blotter.order(sid, amount, style)

    def validate_order_params(self,
                              asset,
                              amount,
                              limit_price,
                              stop_price,
                              style):
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

        if not isinstance(asset, Asset):
            raise UnsupportedOrderParameters(
                msg="Passing non-Asset argument to 'order()' is not supported."
                    " Use 'sid()' or 'symbol()' methods to look up an Asset."
            )

        for control in self.trading_controls:
            control.validate(asset,
                             amount,
                             self.updated_portfolio(),
                             self.get_datetime(),
                             self.trading_client.current_data)

    @staticmethod
    def __convert_order_params_for_blotter(limit_price, stop_price, style):
        """
        Helper method for converting deprecated limit_price and stop_price
        arguments into ExecutionStyle instances.

        This function assumes that either style == None or (limit_price,
        stop_price) == (None, None).
        """
        # TODO_SS: DeprecationWarning for usage of limit_price and stop_price.
        if style:
            assert (limit_price, stop_price) == (None, None)
            return style
        if limit_price and stop_price:
            return StopLimitOrder(limit_price, stop_price)
        if limit_price:
            return LimitOrder(limit_price)
        if stop_price:
            return StopOrder(stop_price)
        else:
            return MarketOrder()

    @api_method
    def order_value(self, sid, value,
                    limit_price=None, stop_price=None, style=None):
        """
        Place an order by desired value rather than desired number of shares.
        If the requested sid is found in the universe, the requested value is
        divided by its price to imply the number of shares to transact.
        If the Asset being ordered is a Future, the 'value' calculated
        is actually the exposure, as Futures have no 'value'.

        value > 0 :: Buy/Cover
        value < 0 :: Sell/Short
        Market order:    order(sid, value)
        Limit order:     order(sid, value, limit_price)
        Stop order:      order(sid, value, None, stop_price)
        StopLimit order: order(sid, value, limit_price, stop_price)
        """
        amount = self._calculate_order_value_amount(sid, value)
        return self.order(sid, amount,
                          limit_price=limit_price,
                          stop_price=stop_price,
                          style=style)

    @property
    def recorded_vars(self):
        return copy(self._recorded_vars)

    @property
    def portfolio(self):
        return self.updated_portfolio()

    def updated_portfolio(self):
        if self.portfolio_needs_update:
            self._portfolio = \
                self.perf_tracker.get_portfolio(self.performance_needs_update)
            self.portfolio_needs_update = False
            self.performance_needs_update = False
        return self._portfolio

    @property
    def account(self):
        return self.updated_account()

    def updated_account(self):
        if self.account_needs_update:
            self._account = \
                self.perf_tracker.get_account(self.performance_needs_update)
            self.account_needs_update = False
            self.performance_needs_update = False
        return self._account

    def set_logger(self, logger):
        self.logger = logger

    def on_dt_changed(self, dt):
        """
        Callback triggered by the simulation loop whenever the current dt
        changes.

        Any logic that should happen exactly once at the start of each datetime
        group should happen here.
        """
        assert isinstance(dt, datetime), \
            "Attempt to set algorithm's current time with non-datetime"
        assert dt.tzinfo == pytz.utc, \
            "Algorithm expects a utc datetime"

        self.datetime = dt
        self.perf_tracker.set_date(dt)
        self.blotter.set_date(dt)

    @api_method
    def get_datetime(self, tz=None):
        """
        Returns the simulation datetime.
        """
        dt = self.datetime
        assert dt.tzinfo == pytz.utc, "Algorithm should have a utc datetime"

        if tz is not None:
            # Convert to the given timezone passed as a string or tzinfo.
            if isinstance(tz, string_types):
                tz = pytz.timezone(tz)
            dt = dt.astimezone(tz)

        return dt  # datetime.datetime objects are immutable.

    def set_transact(self, transact):
        """
        Set the method that will be called to create a
        transaction from open orders and trade events.
        """
        self.blotter.transact = transact

    def update_dividends(self, dividend_frame):
        """
        Set DataFrame used to process dividends.  DataFrame columns should
        contain at least the entries in zp.DIVIDEND_FIELDS.
        """
        self.perf_tracker.update_dividends(dividend_frame)

    @api_method
    def set_slippage(self, slippage):
        if not isinstance(slippage, SlippageModel):
            raise UnsupportedSlippageModel()
        if self.initialized:
            raise OverrideSlippagePostInit()
        self.slippage = slippage

    @api_method
    def set_commission(self, commission):
        if not isinstance(commission, (PerShare, PerTrade, PerDollar)):
            raise UnsupportedCommissionModel()

        if self.initialized:
            raise OverrideCommissionPostInit()
        self.commission = commission

    def set_sources(self, sources):
        assert isinstance(sources, list)
        self.sources = sources

    # Remain backwards compatibility
    @property
    def data_frequency(self):
        return self.sim_params.data_frequency

    @data_frequency.setter
    def data_frequency(self, value):
        assert value in ('daily', 'minute')
        self.sim_params.data_frequency = value

    @api_method
    def order_percent(self, sid, percent,
                      limit_price=None, stop_price=None, style=None):
        """
        Place an order in the specified asset corresponding to the given
        percent of the current portfolio value.

        Note that percent must expressed as a decimal (0.50 means 50\%).
        """
        value = self.portfolio.portfolio_value * percent
        return self.order_value(sid, value,
                                limit_price=limit_price,
                                stop_price=stop_price,
                                style=style)

    @api_method
    def order_target(self, sid, target,
                     limit_price=None, stop_price=None, style=None):
        """
        Place an order to adjust a position to a target number of shares. If
        the position doesn't already exist, this is equivalent to placing a new
        order. If the position does exist, this is equivalent to placing an
        order for the difference between the target number of shares and the
        current number of shares.
        """
        if sid in self.portfolio.positions:
            current_position = self.portfolio.positions[sid].amount
            req_shares = target - current_position
            return self.order(sid, req_shares,
                              limit_price=limit_price,
                              stop_price=stop_price,
                              style=style)
        else:
            return self.order(sid, target,
                              limit_price=limit_price,
                              stop_price=stop_price,
                              style=style)

    @api_method
    def order_target_value(self, sid, target,
                           limit_price=None, stop_price=None, style=None):
        """
        Place an order to adjust a position to a target value. If
        the position doesn't already exist, this is equivalent to placing a new
        order. If the position does exist, this is equivalent to placing an
        order for the difference between the target value and the
        current value.
        If the Asset being ordered is a Future, the 'target value' calculated
        is actually the target exposure, as Futures have no 'value'.
        """
        target_amount = self._calculate_order_value_amount(sid, target)
        return self.order_target(sid, target_amount,
                                 limit_price=limit_price,
                                 stop_price=stop_price,
                                 style=style)

    @api_method
    def order_target_percent(self, sid, target,
                             limit_price=None, stop_price=None, style=None):
        """
        Place an order to adjust a position to a target percent of the
        current portfolio value. If the position doesn't already exist, this is
        equivalent to placing a new order. If the position does exist, this is
        equivalent to placing an order for the difference between the target
        percent and the current percent.

        Note that target must expressed as a decimal (0.50 means 50\%).
        """
        target_value = self.portfolio.portfolio_value * target
        return self.order_target_value(sid, target_value,
                                       limit_price=limit_price,
                                       stop_price=stop_price,
                                       style=style)

    @api_method
    def get_open_orders(self, sid=None):
        if sid is None:
            return {
                key: [order.to_api_obj() for order in orders]
                for key, orders in iteritems(self.blotter.open_orders)
                if orders
            }
        if sid in self.blotter.open_orders:
            orders = self.blotter.open_orders[sid]
            return [order.to_api_obj() for order in orders]
        return []

    @api_method
    def get_order(self, order_id):
        if order_id in self.blotter.orders:
            return self.blotter.orders[order_id].to_api_obj()

    @api_method
    def cancel_order(self, order_param):
        order_id = order_param
        if isinstance(order_param, zipline.protocol.Order):
            order_id = order_param.id

        self.blotter.cancel(order_id)

    @api_method
    def add_history(self, bar_count, frequency, field, ffill=True):
        data_frequency = self.sim_params.data_frequency
        history_spec = HistorySpec(bar_count, frequency, field, ffill,
                                   data_frequency=data_frequency)
        self.history_specs[history_spec.key_str] = history_spec
        if self.initialized:
            if self.history_container:
                self.history_container.ensure_spec(
                    history_spec, self.datetime, self._most_recent_data,
                )
            else:
                self.history_container = self.history_container_class(
                    self.history_specs,
                    self.current_universe(),
                    self.sim_params.first_open,
                    self.sim_params.data_frequency,
                )

    def get_history_spec(self, bar_count, frequency, field, ffill):
        spec_key = HistorySpec.spec_key(bar_count, frequency, field, ffill)
        if spec_key not in self.history_specs:
            data_freq = self.sim_params.data_frequency
            spec = HistorySpec(
                bar_count,
                frequency,
                field,
                ffill,
                data_frequency=data_freq,
            )
            self.history_specs[spec_key] = spec
            if not self.history_container:
                self.history_container = self.history_container_class(
                    self.history_specs,
                    self.current_universe(),
                    self.datetime,
                    self.sim_params.data_frequency,
                    bar_data=self._most_recent_data,
                )
            self.history_container.ensure_spec(
                spec, self.datetime, self._most_recent_data,
            )
        return self.history_specs[spec_key]

    @api_method
    def history(self, bar_count, frequency, field, ffill=True):
        history_spec = self.get_history_spec(
            bar_count,
            frequency,
            field,
            ffill,
        )
        return self.history_container.get_history(history_spec, self.datetime)

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
            control.validate(self.updated_portfolio(),
                             self.updated_account(),
                             self.get_datetime(),
                             self.trading_client.current_data)

    @api_method
    def set_max_leverage(self, max_leverage=None):
        """
        Set a limit on the maximum leverage of the algorithm.
        """
        control = MaxLeverage(max_leverage)
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
    def set_max_position_size(self,
                              sid=None,
                              max_shares=None,
                              max_notional=None):
        """
        Set a limit on the number of shares and/or dollar value held for the
        given sid. Limits are treated as absolute values and are enforced at
        the time that the algo attempts to place an order for sid. This means
        that it's possible to end up with more than the max number of shares
        due to splits/dividends, and more than the max notional due to price
        improvement.

        If an algorithm attempts to place an order that would result in
        increasing the absolute value of shares/dollar value exceeding one of
        these limits, raise a TradingControlException.
        """
        control = MaxPositionSize(asset=sid,
                                  max_shares=max_shares,
                                  max_notional=max_notional)
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(self, sid=None, max_shares=None, max_notional=None):
        """
        Set a limit on the number of shares and/or dollar value of any single
        order placed for sid.  Limits are treated as absolute values and are
        enforced at the time that the algo attempts to place an order for sid.

        If an algorithm attempts to place an order that would result in
        exceeding one of these limits, raise a TradingControlException.
        """
        control = MaxOrderSize(asset=sid,
                               max_shares=max_shares,
                               max_notional=max_notional)
        self.register_trading_control(control)

    @api_method
    def set_max_order_count(self, max_count):
        """
        Set a limit on the number of orders that can be placed within the given
        time interval.
        """
        control = MaxOrderCount(max_count)
        self.register_trading_control(control)

    @api_method
    def set_do_not_order_list(self, restricted_list):
        """
        Set a restriction on which sids can be ordered.
        """
        control = RestrictedListOrder(restricted_list)
        self.register_trading_control(control)

    @api_method
    def set_long_only(self):
        """
        Set a rule specifying that this algorithm cannot take short positions.
        """
        self.register_trading_control(LongOnly())

    ###########
    # FFC API #
    ###########
    @api_method
    @require_not_initialized(AddTermPostInit())
    def add_factor(self, factor, name):
        if name in self._factors:
            raise ValueError("Name %r is already a factor!" % name)
        self._factors[name] = factor

    @api_method
    @require_not_initialized(AddTermPostInit())
    def add_filter(self, filter):
        name = "anon_filter_%d" % len(self._filters)
        self._filters[name] = filter

    # Note: add_classifier is not yet implemented since you can't do anything
    # useful with classifiers yet.

    def _all_terms(self):
        # Merge all three dicts.
        return dict(
            chain.from_iterable(
                iteritems(terms)
                for terms in (self._filters, self._factors, self._classifiers)
            )
        )

    def compute_factor_matrix(self, start_date):
        """
        Compute a factor matrix starting at start_date.
        """
        days = self.trading_environment.trading_days
        start_date_loc = days.get_loc(start_date)
        sim_end = self.sim_params.last_close.normalize()
        end_loc = min(start_date_loc + 252, days.get_loc(sim_end))
        end_date = days[end_loc]
        return self.engine.factor_matrix(
            self._all_terms(),
            start_date,
            end_date,
        ), end_date

    def current_universe(self):
        return self._current_universe

    @classmethod
    def all_api_methods(cls):
        """
        Return a list of all the TradingAlgorithm API methods.
        """
        return [
            fn for fn in itervalues(vars(cls))
            if getattr(fn, 'is_api_method', False)
        ]
