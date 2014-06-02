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

import pytz
import pandas as pd
import numpy as np

from datetime import datetime

from itertools import groupby
from six.moves import filter
from six import iteritems, exec_
from operator import attrgetter

from zipline.errors import (
    OverrideCommissionPostInit,
    OverrideSlippagePostInit,
    RegisterTradingControlPostInit,
    UnsupportedCommissionModel,
    UnsupportedOrderParameters,
    UnsupportedSlippageModel,
)

from zipline.finance import trading
from zipline.finance.blotter import Blotter
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance.controls import (
    LongOnly,
    MaxOrderCount,
    MaxOrderSize,
    MaxPositionSize,
)
from zipline.finance.constants import ANNUALIZER
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
from zipline.gens.composites import (
    date_sorted_sources,
    sequential_transforms,
)
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.sources import DataFrameSource, DataPanelSource
from zipline.transforms.utils import StatefulTransform
from zipline.utils.api_support import ZiplineAPI, api_method
from zipline.utils.factory import create_simulation_parameters

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
    from zipline.api import order

    def initialize(context):
        context.sid = 'AAPL'
        context.amount = 100

    def handle_data(self, data):
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
            data_frequency : str (daily, hourly or minutely)
               The duration of the bars.
            annualizer : int <optional>
               Which constant to use for annualizing risk metrics.
               If not provided, will extract from data_frequency.
            capital_base : float <default: 1.0e5>
               How much capital to start with.
            instant_fill : bool <default: False>
               Whether to fill orders immediately or on next bar.
        """
        self.datetime = None

        self.registered_transforms = {}
        self.transforms = []
        self.sources = []

        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        self._recorded_vars = {}
        self.namespace = kwargs.get('namespace', {})

        self.logger = None

        self.benchmark_return_source = None
        self.perf_tracker = None

        # default components for transact
        self.slippage = VolumeShareSlippage()
        self.commission = PerShare()

        if 'data_frequency' in kwargs:
            self.set_data_frequency(kwargs.pop('data_frequency'))
        else:
            self.data_frequency = None

        self.instant_fill = kwargs.pop('instant_fill', False)

        # Override annualizer if set
        if 'annualizer' in kwargs:
            self.annualizer = kwargs['annualizer']

        # set the capital base
        self.capital_base = kwargs.pop('capital_base', DEFAULT_CAPITAL_BASE)

        self.sim_params = kwargs.pop('sim_params', None)
        if self.sim_params:
            if self.data_frequency is None:
                self.data_frequency = self.sim_params.data_frequency
            else:
                self.sim_params.data_frequency = self.data_frequency

            self.perf_tracker = PerformanceTracker(self.sim_params)

        self.blotter = kwargs.pop('blotter', None)
        if not self.blotter:
            self.blotter = Blotter()

        self.portfolio_needs_update = True
        self._portfolio = None

        self.history_container = None
        self.history_specs = {}

        # If string is passed in, execute and get reference to
        # functions.
        self.algoscript = kwargs.pop('script', None)

        self._initialize = None
        self._analyze = None

        if self.algoscript is not None:
            exec_(self.algoscript, self.namespace)
            self._initialize = self.namespace.get('initialize', None)
            if 'handle_data' not in self.namespace:
                raise ValueError('You must define a handle_data function.')
            else:
                self._handle_data = self.namespace['handle_data']

            # Optional analyze function, gets called after run
            self._analyze = self.namespace.get('analyze', None)

        elif kwargs.get('initialize', False) and kwargs.get('handle_data'):
            if self.algoscript is not None:
                raise ValueError('You can not set script and \
                initialize/handle_data.')
            self._initialize = kwargs.pop('initialize')
            self._handle_data = kwargs.pop('handle_data')

        # If method not defined, NOOP
        if self._initialize is None:
            self._initialize = lambda x: None

        # an algorithm subclass needs to set initialized to True when
        # it is fully initialized.
        self.initialized = False
        self.initialize(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        """
        Call self._initialize with `self` made available to Zipline API
        functions.
        """
        with ZiplineAPI(self):
            self._initialize(self)

    def handle_data(self, data):
        if self.history_container:
            self.history_container.update(data, self.datetime)

        self._handle_data(self, data)

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

    def _create_data_generator(self, source_filter, sim_params):
        """
        Create a merged data generator using the sources and
        transforms attached to this algorithm.

        ::source_filter:: is a method that receives events in date
        sorted order, and returns True for those events that should be
        processed by the zipline, and False for those that should be
        skipped.
        """
        if self.benchmark_return_source is None:
            env = trading.environment
            if (self.data_frequency == 'minute'
                    or sim_params.emission_rate == 'minute'):
                update_time = lambda date: env.get_open_and_close(date)[1]
            else:
                update_time = lambda date: date
            benchmark_return_source = [
                Event({'dt': update_time(dt),
                       'returns': ret,
                       'type': zipline.protocol.DATASOURCE_TYPE.BENCHMARK,
                       'source_id': 'benchmarks'})
                for dt, ret in trading.environment.benchmark_returns.iterkv()
                if dt.date() >= sim_params.period_start.date()
                and dt.date() <= sim_params.period_end.date()
            ]
        else:
            benchmark_return_source = self.benchmark_return_source

        date_sorted = date_sorted_sources(*self.sources)

        if source_filter:
            date_sorted = filter(source_filter, date_sorted)

        with_tnfms = sequential_transforms(date_sorted,
                                           *self.transforms)

        with_benchmarks = date_sorted_sources(benchmark_return_source,
                                              with_tnfms)

        # Group together events with the same dt field. This depends on the
        # events already being sorted.
        return groupby(with_benchmarks, attrgetter('dt'))

    def _create_generator(self, sim_params, source_filter=None):
        """
        Create a basic generator setup using the sources and
        transforms attached to this algorithm.

        ::source_filter:: is a method that receives events in date
        sorted order, and returns True for those events that should be
        processed by the zipline, and False for those that should be
        skipped.
        """
        sim_params.data_frequency = self.data_frequency

        # perf_tracker will be instantiated in __init__ if a sim_params
        # is passed to the constructor. If not, we instantiate here.
        if self.perf_tracker is None:
            self.perf_tracker = PerformanceTracker(sim_params)

        self.data_gen = self._create_data_generator(source_filter,
                                                    sim_params)

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
    def run(self, source, sim_params=None, benchmark_return_source=None):
        """Run the algorithm.

        :Arguments:
            source : can be either:
                     - pandas.DataFrame
                     - zipline source
                     - list of zipline sources

               If pandas.DataFrame is provided, it must have the
               following structure:
               * column names must consist of ints representing the
                 different sids
               * index must be DatetimeIndex
               * array contents should be price info.

        :Returns:
            daily_stats : pandas.DataFrame
              Daily performance metrics such as returns, alpha etc.

        """
        if isinstance(source, (list, tuple)):
            assert self.sim_params is not None or sim_params is not None, \
                """When providing a list of sources, \
                sim_params have to be specified as a parameter
                or in the constructor."""
        elif isinstance(source, pd.DataFrame):
            # if DataFrame provided, wrap in DataFrameSource
            source = DataFrameSource(source)
        elif isinstance(source, pd.Panel):
            source = DataPanelSource(source)

        if not isinstance(source, (list, tuple)):
            self.sources = [source]
        else:
            self.sources = source

        # Check for override of sim_params.
        # If it isn't passed to this function,
        # use the default params set with the algorithm.
        # Else, we create simulation parameters using the start and end of the
        # source provided.
        if sim_params is None:
            if self.sim_params is None:
                start = source.start
                end = source.end
                sim_params = create_simulation_parameters(
                    start=start,
                    end=end,
                    capital_base=self.capital_base,
                )
            else:
                sim_params = self.sim_params

        # update sim params to ensure it's set
        self.sim_params = sim_params
        if self.sim_params.sids is None:
            all_sids = [sid for s in self.sources for sid in s.sids]
            self.sim_params.sids = set(all_sids)

        # Create history containers
        if len(self.history_specs) != 0:
            self.history_container = HistoryContainer(
                self.history_specs,
                self.sim_params.sids,
                self.sim_params.first_open)

        # Create transforms by wrapping them into StatefulTransforms
        self.transforms = []
        for namestring, trans_descr in iteritems(self.registered_transforms):
            sf = StatefulTransform(
                trans_descr['class'],
                *trans_descr['args'],
                **trans_descr['kwargs']
            )
            sf.namestring = namestring

            self.transforms.append(sf)

        # force a reset of the performance tracker, in case
        # this is a repeat run of the algorithm.
        self.perf_tracker = None

        # create transforms and zipline
        self.gen = self._create_generator(sim_params)

        with ZiplineAPI(self):
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
                daily_perfs.append(perf['daily_perf'])
            else:
                self.risk_report = perf

        daily_dts = [np.datetime64(perf['period_close'], utc=True)
                     for perf in daily_perfs]
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)

        return daily_stats

    def add_transform(self, transform_class, tag, *args, **kwargs):
        """Add a single-sid, sequential transform to the model.

        :Arguments:
            transform_class : class
                Which transform to use. E.g. mavg.
            tag : str
                How to name the transform. Can later be access via:
                data[sid].tag()

        Extra args and kwargs will be forwarded to the transform
        instantiation.

        """
        self.registered_transforms[tag] = {'class': transform_class,
                                           'args': args,
                                           'kwargs': kwargs}

    @api_method
    def record(self, **kwargs):
        """
        Track and record local variable (i.e. attributes) each day.
        """
        for name, value in kwargs.items():
            self._recorded_vars[name] = value

    @api_method
    def order(self, sid, amount,
              limit_price=None,
              stop_price=None,
              style=None):
        """
        Place an order using the specified parameters.
        """
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
                              sid,
                              amount,
                              limit_price,
                              stop_price,
                              style):
        """
        Helper method for validating parameters to the order API function.

        Raises an UnsupportedOrderParameters if invalid arguments are found.
        """
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
            control.validate(sid,
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

        value > 0 :: Buy/Cover
        value < 0 :: Sell/Short
        Market order:    order(sid, value)
        Limit order:     order(sid, value, limit_price)
        Stop order:      order(sid, value, None, stop_price)
        StopLimit order: order(sid, value, limit_price, stop_price)
        """
        last_price = self.trading_client.current_data[sid].price
        if np.allclose(last_price, 0):
            zero_message = "Price of 0 for {psid}; can't infer value".format(
                psid=sid
            )
            if self.logger:
                self.logger.debug(zero_message)
            # Don't place any order
            return
        else:
            amount = value / last_price
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
            self._portfolio = self.perf_tracker.get_portfolio()
            self.portfolio_needs_update = False
        return self._portfolio

    def set_logger(self, logger):
        self.logger = logger

    def set_datetime(self, dt):
        assert isinstance(dt, datetime), \
            "Attempt to set algorithm's current time with non-datetime"
        assert dt.tzinfo == pytz.utc, \
            "Algorithm expects a utc datetime"
        self.datetime = dt

    @api_method
    def get_datetime(self):
        """
        Returns a copy of the datetime.
        """
        date_copy = copy(self.datetime)
        assert date_copy.tzinfo == pytz.utc, \
            "Algorithm should have a utc datetime"
        return date_copy

    def set_transact(self, transact):
        """
        Set the method that will be called to create a
        transaction from open orders and trade events.
        """
        self.blotter.transact = transact

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

    def set_transforms(self, transforms):
        assert isinstance(transforms, list)
        self.transforms = transforms

    def set_data_frequency(self, data_frequency):
        assert data_frequency in ('daily', 'minute')
        self.data_frequency = data_frequency
        self.annualizer = ANNUALIZER[self.data_frequency]

    @api_method
    def order_percent(self, sid, percent,
                      limit_price=None, stop_price=None, style=None):
        """
        Place an order in the specified security corresponding to the given
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
        """
        last_price = self.trading_client.current_data[sid].price
        if np.allclose(last_price, 0):
            # Don't place an order
            if self.logger:
                zero_message = "Price of 0 for {psid}; can't infer value"
                self.logger.debug(zero_message.format(psid=sid))
            return
        target_amount = target / last_price
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

    def raw_positions(self):
        """
        Returns the current portfolio for the algorithm.

        N.B. this is not done as a property, so that the function can be
        passed and called from within a source.
        """
        # Return the 'internal' positions object, as in the one that is
        # not passed to the algo, and thus should not have tainted keys.
        return self.perf_tracker.cumulative_performance.positions

    def raw_orders(self):
        """
        Returns the current open orders from the blotter.

        N.B. this is not a property, so that the function can be passed
        and called back from within a source.
        """

        return self.blotter.open_orders

    @api_method
    def add_history(self, bar_count, frequency, field,
                    ffill=True):
        history_spec = HistorySpec(bar_count, frequency, field, ffill)
        self.history_specs[history_spec.key_str] = history_spec

    @api_method
    def history(self, bar_count, frequency, field, ffill=True):
        spec_key_str = HistorySpec.spec_key(
            bar_count, frequency, field, ffill)
        history_spec = self.history_specs[spec_key_str]
        return self.history_container.get_history(history_spec, self.datetime)

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
        control = MaxPositionSize(sid=sid,
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
        control = MaxOrderSize(sid=sid,
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
    def set_long_only(self):
        """
        Set a rule specifying that this algorithm cannot take short positions.
        """
        self.register_trading_control(LongOnly())

    @classmethod
    def all_api_methods(cls):
        """
        Return a list of all the TradingAlgorithm API methods.
        """
        return [fn for fn in cls.__dict__.itervalues()
                if getattr(fn, 'is_api_method', False)]
