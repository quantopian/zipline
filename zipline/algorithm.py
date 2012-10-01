import pandas as pd
import numpy as np

from zipline.gens.tradegens import DataFrameSource
from zipline.utils.factory import create_trading_environment
from zipline.gens.transform import StatefulTransform
from zipline.lines import SimulatedTrading
from zipline.finance.slippage import FixedSlippage, transact_partial
from zipline.finance.commission import PerShare


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
    def __init__(self, sids, *args, **kwargs):
        """
        Initialize sids and other state variables.

        Calls user-defined initialize() forwarding *args and **kwargs.
        """
        self.sids = sids
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None

        self.registered_transforms = {}

        # call to user-defined initialize method
        self.initialize(*args, **kwargs)

        self.initialized = True

    def _create_simulator(self, start, end):
        """
        Create trading environment, transforms and SimulatedTrading object.

        Gets called by self.run().
        """
        environment = create_trading_environment(start=start, end=end)

        # Create transforms by wrapping them into StatefulTransforms
        transforms = []
        for namestring, trans_descr in self.registered_transforms.iteritems():
            sf = StatefulTransform(
                trans_descr['class'],
                *trans_descr['args'],
                **trans_descr['kwargs']
            )
            sf.namestring = namestring

            transforms.append(sf)

        # SimulatedTrading is the main class handling data streaming,
        # application of transforms and calling of the user algo.
        return SimulatedTrading(
            self.sources,
            transforms,
            self,
            environment,
            transact_partial(FixedSlippage(), PerShare(0.0))
        )

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
            "When providing a list of sources, start and end date have to be specified."
        elif isinstance(source, pd.DataFrame):
            assert isinstance(source.index, pd.tseries.index.DatetimeIndex)
            # if DataFrame provided, wrap in DataFrameSource
            source = DataFrameSource(source, sids=self.sids)

        # If values not set, try to extract from source.
        if start is None:
            start = source.start
        if end is None:
            end = source.end

        if not isinstance(source, (list, tuple)):
            self.sources = [source]
        else:
            self.sources = source

        # create transforms and zipline
        self.simulated_trading = self._create_simulator(start=start, end=end)

        # loop through simulated_trading, each iteration returns a
        # perf ndict
        perfs = list(self.simulated_trading)

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

        daily_dts = [np.datetime64(perf['period_close'], utc=True) for perf in daily_perfs]
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

    def get_sid_filter(self):
        return self.sids

    def set_logger(self, logger):
        self.logger = logger

    def initialize(self, *args, **kwargs):
        pass

    def set_transact_setter(self, transact_setter):
        pass
