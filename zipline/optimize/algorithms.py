import pandas as pd
import numpy as np

from datetime import datetime
from zipline.gens.tradegens import DataFrameSource
from zipline import ndict
from zipline.utils.factory import create_trading_environment
from zipline.gens.transform import StatefulTransform
from zipline.lines import SimulatedTrading
from zipline.finance.slippage import FixedSlippage

from logbook import Logger

logger = Logger('Algo')

class BuySellAlgorithm(object):
    """Algorithm that buys and sells alternatingly. The amount for
    each order can be specified. In addition, an offset that will
    quadratically reduce the amount that will be bought can be
    specified.

    This algorithm is used to test the parameter optimization
    framework. If combined with the UpDown trade source, an offset of
    0 will produce maximum returns.

    """

    def __init__(self, sid, amount, offset):
        self.sid = sid
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.buy_or_sell = -1
        self.offset = offset
        self.orders = []
        self.prices = []

    def initialize(self):
        pass

    def set_order(self, order_callable):
        self.order = order_callable

    def set_portfolio(self, portfolio):
        self.portfolio = portfolio

    def handle_data(self, frame):
        print frame.sid
        order_size = self.buy_or_sell * (self.amount - (self.offset**2))
        self.order(self.sid, order_size)

        #sell next time around.
        self.buy_or_sell *= -1

        self.orders.append(order_size)

        self.frame_count += 1
        self.incr += 1

    def get_sid_filter(self):
        return [self.sid]


class TradingAlgorithm(object):
    """
    Base class for trading algorithms. Inherit and overload handle_data(data).

    A new algorithm could look like this:
    ```
    class MyAlgo(TradingAlgorithm):
        def initialize(amount):
            self.amount = amount

        def handle_data(data):
            sid = self.sids[0]
            self.order(sid, amount)
    ```
    To then run this algorithm:

    >>> my_algo = MyAlgo(100)
    >>> stats = my_algo.run(data)

    """
    def __init__(self, sids, *args, **kwargs):
        """
        Initialize sids and other state variables.

        Calls user-defined initialize and forwarding *args and **kwargs.
        """
        self.sids = sids
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None

        self.registered_transforms = {}

        # call to user-defined initialize method
        self.initialize(*args, **kwargs)

    def _create_simulator(self, source):
        """
        Create trading environment, transforms and SimulatedTrading object.

        Gets called by self.run().
        """
        environment = create_trading_environment(start=source.data.index[0], end=source.data.index[-1])

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
            [source],
            transforms,
            self,
            environment,
            FixedSlippage()
        )

    def run(self, data):
        """
        Run the algorithm.

        :Arguments:
            data : pandas.DataFrame
               * columns must consist of ints representing the different sids
               * index must be TimeStamps
               * array contents should be price

        :Returns:
            daily_stats : pandas.DataFrame
              Daily performance metrics such as returns, alpha etc.

        """
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.tseries.index.DatetimeIndex)

        source = DataFrameSource(data, sids=self.sids)

        # create transforms and zipline
        simulated_trading = self._create_simulator(source)

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

    def set_slippage_override(self, slippage_callable):
        pass



class BuySellAlgorithmNew(TradingAlgorithm):
    """Algorithm that buys and sells alternatingly. The amount for
    each order can be specified. In addition, an offset that will
    quadratically reduce the amount that will be bought can be
    specified.

    This algorithm is used to test the parameter optimization
    framework. If combined with the UpDown trade source, an offset of
    0 will produce maximum returns.

    """

    def __init__(self, sids, amount, offset):
        self.sids = sids
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.buy_or_sell = -1
        self.offset = offset
        self.orders = []
        self.prices = []

    def handle_data(self, data):
        order_size = self.buy_or_sell * (self.amount - (self.offset**2))
        self.order(self.sids[0], order_size)
        logger.debug("ordering" + str(order_size))

        #sell next time around.
        self.buy_or_sell *= -1

        self.orders.append(order_size)

        self.frame_count += 1
        self.incr += 1

