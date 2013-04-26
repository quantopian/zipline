#
# Copyright 2013 Quantopian, Inc.
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

from itertools import groupby, ifilter
from operator import attrgetter

from zipline.errors import (
    UnsupportedSlippageModel,
    OverrideSlippagePostInit,
    UnsupportedCommissionModel,
    OverrideCommissionPostInit
)
from zipline.sources import DataFrameSource, DataPanelSource
from zipline.utils.factory import create_simulation_parameters
from zipline.transforms.utils import StatefulTransform
from zipline.finance.slippage import (
    VolumeShareSlippage,
    FixedSlippage,
    transact_partial
)
from zipline.finance.commission import PerShare, PerTrade
from zipline.finance.blotter import Blotter
from zipline.finance.constants import ANNUALIZER
import zipline.finance.trading as trading
import zipline.protocol
from zipline.protocol import Event

from zipline.gens.composites import (
    date_sorted_sources,
    sequential_transforms,
    alias_dt
)
from zipline.gens.tradesimulation import AlgorithmSimulator

DEFAULT_CAPITAL_BASE = float("1.0e5")


class TradingAlgorithm(object):
    """Base class for trading algorithms. Inherit and overload
    initialize() and handle_data(data).

    A new algorithm could look like this:
    ```
    class MyAlgo(TradingAlgorithm):
        def initialize(amount):
            self.amount = amount

        def handle_data(data):
            sid = self.sids[0]
            self.order(sid, amount)
    ```
    To then to run this algorithm:

    >>> my_algo = MyAlgo([0], 100) # first argument has to be list of sids
    >>> stats = my_algo.run(data)

    """
    def __init__(self, *args, **kwargs):
        """Initialize sids and other state variables.

        :Arguments:
            data_frequency : str (daily, hourly or minutely)
               The duration of the bars.
            annualizer : int <optional>
               Which constant to use for annualizing risk metrics.
               If not provided, will extract from data_frequency.
            capital_base : float <default: 1.0e5>
               How much capital to start with.
        """
        self._portfolio = None
        self.datetime = None

        self.registered_transforms = {}
        self.transforms = []
        self.sources = []

        self._recorded_vars = {}

        self.logger = None

        # default components for transact
        self.slippage = VolumeShareSlippage()
        self.commission = PerShare()

        if 'data_frequency' in kwargs:
            self.set_data_frequency(kwargs.pop('data_frequency'))
        else:
            self.data_frequency = None

        # Override annualizer if set
        if 'annualizer' in kwargs:
            self.annualizer = kwargs['annualizer']

        # set the capital base
        self.capital_base = kwargs.get('capital_base', DEFAULT_CAPITAL_BASE)

        self.sim_params = kwargs.pop('sim_params', None)

        self.blotter = kwargs.pop('blotter', Blotter())

        # an algorithm subclass needs to set initialized to True when
        # it is fully initialized.
        self.initialized = False

        # call to user-defined constructor method
        self.initialize(*args, **kwargs)

    def _create_data_generator(self, source_filter, sim_params):
        """
        Create a merged data generator using the sources and
        transforms attached to this algorithm.

        ::source_filter:: is a method that receives events in date
        sorted order, and returns True for those events that should be
        processed by the zipline, and False for those that should be
        skipped.
        """
        benchmark_return_source = [
            Event({'dt': ret.date,
                   'returns': ret.returns,
                   'type': zipline.protocol.DATASOURCE_TYPE.BENCHMARK,
                   'source_id': 'benchmarks'})
            for ret in trading.environment.benchmark_returns
            if ret.date.date() >= sim_params.period_start.date()
            and ret.date.date() <= sim_params.period_end.date()
        ]

        date_sorted = date_sorted_sources(*self.sources)

        if source_filter:
            date_sorted = ifilter(source_filter, date_sorted)

        with_tnfms = sequential_transforms(date_sorted,
                                           *self.transforms)
        with_alias_dt = alias_dt(with_tnfms)

        with_benchmarks = date_sorted_sources(benchmark_return_source,
                                              with_alias_dt)

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

    def initialize(self, *args, **kwargs):
        pass

    # TODO: make a new subclass, e.g. BatchAlgorithm, and move
    # the run method to the subclass, and refactor to put the
    # generator creation logic into get_generator.
    def run(self, source, sim_params=None):
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
        if not sim_params:
            if not self.sim_params:
                start = source.start
                end = source.end

                sim_params = create_simulation_parameters(
                    start=start,
                    end=end,
                    capital_base=self.capital_base
                )
            else:
                sim_params = self.sim_params

        # Create transforms by wrapping them into StatefulTransforms
        self.transforms = []
        for namestring, trans_descr in self.registered_transforms.iteritems():
            sf = StatefulTransform(
                trans_descr['class'],
                *trans_descr['args'],
                **trans_descr['kwargs']
            )
            sf.namestring = namestring

            self.transforms.append(sf)

        # create transforms and zipline
        self.gen = self._create_generator(sim_params)

        # loop through simulated_trading, each iteration returns a
        # perf dictionary
        perfs = list(self.gen)

        # convert perf dict to pandas dataframe
        daily_stats = self._create_daily_stats(perfs)

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        cum_perfs = []
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
                cum_perfs.append(perf)

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

    def record(self, **kwargs):
        """
        Track and record local variable (i.e. attributes) each day.
        """
        for name, value in kwargs.items():
            self._recorded_vars[name] = value

    def order(self, sid, amount, limit_price=None, stop_price=None):
        return self.blotter.order(sid, amount, limit_price, stop_price)

    @property
    def recorded_vars(self):
        return copy(self._recorded_vars)

    @property
    def portfolio(self):
        return self._portfolio

    def set_portfolio(self, portfolio):
        self._portfolio = portfolio

    def set_logger(self, logger):
        self.logger = logger

    def set_datetime(self, dt):
        assert isinstance(dt, datetime), \
            "Attempt to set algorithm's current time with non-datetime"
        assert dt.tzinfo == pytz.utc, \
            "Algorithm expects a utc datetime"
        self.datetime = dt

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

    def set_slippage(self, slippage):
        if not isinstance(slippage, (VolumeShareSlippage, FixedSlippage)):
            raise UnsupportedSlippageModel()
        if self.initialized:
            raise OverrideSlippagePostInit()
        self.slippage = slippage

    def set_commission(self, commission):
        if not isinstance(commission, (PerShare, PerTrade)):
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
