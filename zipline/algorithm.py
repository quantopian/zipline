#
# Copyright 2012 Quantopian, Inc.
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

import pandas as pd
import numpy as np

from zipline.gens.tradegens import DataFrameSource
from zipline.utils.factory import create_trading_environment
from zipline.gens.transform import StatefulTransform
from zipline.finance.slippage import (
        VolumeShareSlippage,
        FixedSlippage,
        transact_partial
)
from zipline.finance.commission import PerShare, PerTrade


from zipline.gens.composites import (
    date_sorted_sources,
    sequential_transforms
)
from zipline.gens.tradesimulation import TradeSimulationClient as tsc
from zipline import MESSAGES


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
        """
        Initialize sids and other state variables.
        """
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None

        self.registered_transforms = {}
        self.transforms = []
        self.sources = []

        self.logger = None

        # default components for transact
        self.slippage = VolumeShareSlippage()
        self.commission = PerShare()

        # an algorithm subclass needs to set initialized to True
        # when it is fully initialized.
        self.initialized = False

        # call to user-defined constructor method
        self.initialize(*args, **kwargs)

    def _create_generator(self, environment):
        """
        Create a basic generator setup using the sources and
        transforms attached to this algorithm.
        """

        self.date_sorted = date_sorted_sources(*self.sources)
        self.with_tnfms = sequential_transforms(self.date_sorted,
                                                *self.transforms)
        self.trading_client = tsc(self, environment)

        transact_method = transact_partial(self.slippage, self.commission)
        self.set_transact(transact_method)

        return self.trading_client.simulate(self.with_tnfms)

    def get_generator(self, environment):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(environment)

    def initialize(self, *args, **kwargs):
        pass

    # TODO: make a new subclass, e.g. BatchAlgorithm, and move
    # the run method to the subclass, and refactor to put the
    # generator creation logic into get_generator.
    def run(self, source, start=None, end=None):
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
            assert start is not None and end is not None, \
            """When providing a list of sources, \
start and end date have to be specified."""
        elif isinstance(source, pd.DataFrame):
            assert isinstance(source.index, pd.tseries.index.DatetimeIndex)
            # if DataFrame provided, wrap in DataFrameSource
            source = DataFrameSource(source)

        # If values not set, try to extract from source.
        if start is None:
            start = source.start
        if end is None:
            end = source.end

        if not isinstance(source, (list, tuple)):
            self.sources = [source]
        else:
            self.sources = source

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

        environment = create_trading_environment(start=start, end=end)

        # create transforms and zipline
        self.gen = self._create_generator(environment)

        # loop through simulated_trading, each iteration returns a
        # perf ndict
        perfs = list(self.gen)

        # convert perf ndict to pandas dataframe
        daily_stats = self._create_daily_stats(perfs)

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        cum_perfs = []
        for perf in perfs:
            if 'daily_perf' in perf:
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

    def set_portfolio(self, portfolio):
        self.portfolio = portfolio

    def set_order(self, order_callable):
        self.order = order_callable

    def set_logger(self, logger):
        self.logger = logger

    def init(self, *args, **kwargs):
        """Called from constructor."""
        pass

    def set_transact(self, transact):
        """
        Set the method that will be called to create a
        transaction from open orders and trade events.
        """
        self.trading_client.ordering_client.transact = transact

    def set_slippage(self, slippage):
        assert isinstance(slippage, (VolumeShareSlippage, FixedSlippage)), \
                MESSAGES.ERRORS.UNSUPPORTED_SLIPPAGE_MODEL
        if self.initialized:
            raise Exception(MESSAGES.ERRORS.OVERRIDE_SLIPPAGE_POST_INIT)
        self.slippage = slippage

    def set_commission(self, commission):
        assert isinstance(commission, (PerShare, PerTrade)), \
                MESSAGES.ERRORS.UNSUPPORTED_COMMISSION_MODEL

        if self.initialized:
            raise Exception(MESSAGES.ERRORS.OVERRIDE_COMMISSION_POST_INIT)
        self.commission = commission

    def set_sources(self, sources):
        assert isinstance(sources, list)
        self.sources = sources

    def set_transforms(self, transforms):
        assert isinstance(transforms, list)
        self.transforms = transforms
