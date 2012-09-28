"""
Ziplines are composed of multiple components connected by asynchronous
messaging. All ziplines follow a general topology of parallel sources,
datetimestamp serialization, parallel transformations, and finally sinks.
Furthermore, many ziplines have common needs. For example, all trade
simulations require a
:py:class:`~zipline.finance.trading.TradeSimulationClient`.

To establish best practices and minimize code replication, the lines module
provides complete zipline topologies. You can extend any zipline without
the need to extend the class. Simply instantiate any additional components
that you would like included in the zipline, and add them to the zipline
before invoking simulate.


        Here is a diagram of the SimulatedTrading zipline:


              +----------------------+  +------------------------+
              |    Trade History     |  |    (DataSource added   |
              |                      |  |     via add_source)    |
              |                      |  |                        |
              +--------------------+-+  +-+----------------------+
                                   |      |
                                   |      |
                                   v      v
                                  +---------+
                                  |   Feed  |  (ensures events are serialized
                                  +-+------++   in chronological order)
                                    |      |
                                    |      |
                                    v      v
               +----------------------+   +----------------------+
               | (Transforms added    |   |  (Transforms added   |
               |  via add_transform)  |   |   via add_transform) |
               +-------------------+--+   +-+--------------------+
                                   |        |
                                   |        |
                                   v        v
                                 +------------+
                                 |    Merge   | (combines original event and
                                 +------+-----+  transforms into one vector)
                                        |
                                        |
                                        V
    +---------------+     +--------------------------------+
    | Risk and Perf |     |                                |
    | Tracker       |     |     TradingSimulationClient    |
    +---------------+     |     tracks performance and     |
       ^  Trades and      |     provides API to algorithm. |
       |  simulated       |                                |
       |  transactions    +--+------------------+----------+
       |                     |      ^           |
       +---------------------+      | orders    |  frames
                                    |           |
                                    |           v
                          +---------------------------------+
                          |      Algorithm added via        |
                          |      __init__.                  |
                          +---------------------------------+
"""

from zipline.utils import factory

from zipline.gens.composites import (
    date_sorted_sources,
    sequential_transforms
)
from zipline.gens.tradesimulation import TradeSimulationClient as tsc
from zipline.finance.slippage import FixedSlippage

from logbook import Logger

log = Logger('Lines')


class SimulatedTrading(object):

    def __init__(self,
            sources,
            transforms,
            algorithm,
            environment,
            slippage):
        """
        @sources - an iterable of iterables
        These iterables must yield ndicts that contain:
        - type :: a ziplines.protocol.DATASOURCE_TYPE
        - dt :: a milliseconds since epoch timestamp in UTC

        @transforms - An iterable of instances of StatefulTransform.

        @algorithm - An object that implements:
        `def initialize(self)`
        `def handle_data(self, data)`
        `def get_sid_filter(self)`
        `def set_logger(self, logger)`
        `def set_order(self, order_callable)`

        @environment - An instance of finance.trading.TradingEnvironment

        @slippage - an object with a simulate method that takes a
        trade event and returns a transaction
        """

        self.date_sorted = date_sorted_sources(*sources)
        self.transforms = transforms
        # Formerly merged_transforms.
        self.with_tnfms = sequential_transforms(self.date_sorted,
                                                *self.transforms)
        self.trading_client = tsc(algorithm, environment, slippage)
        self.gen = self.trading_client.simulate(self.with_tnfms)

    def __iter__(self):
        return self

    def next(self):
        return self.gen.next()

    @staticmethod
    def create_test_zipline(**config):
        """
        :param config: A configuration object that is a dict with:

            - environment - a \
              :py:class:`zipline.finance.trading.TradingEnvironment`
            - sid - an integer, which will be used as the security ID.
            - order_count - the number of orders the test algo will place,
              defaults to 100
            - order_amount - the number of shares per order, defaults to 100
            - trade_count - the number of trades to simulate, defaults to 101
              to ensure all orders are processed.
            - algorithm - optional parameter providing an algorithm. defaults
              to :py:class:`zipline.test.algorithms.TestAlgorithm`
            - trade_source - optional parameter to specify trades, if present.
              If not present :py:class:`zipline.sources.SpecificEquityTrades`
              is the source, with daily frequency in trades.
            - slippage: optional parameter that configures the
              :py:class:`zipline.gens.tradingsimulation.TransactionSimulator`. Expects
              an object with a simulate mehod, such as
              :py:class:`zipline.gens.tradingsimulation.FixedSlippage`.
              :py:mod:`zipline.finance.trading`
            - transforms: optional parameter that provides a list
              of StatefulTransform objects.
        """
        from zipline.test_algorithms import TestAlgorithm

        assert isinstance(config, dict)
        sid_list = config.get('sid_list')
        if not sid_list:
            sid = config.get('sid')
            sid_list = [sid]

        concurrent_trades = config.get('concurrent_trades', False)

        #--------------------
        # Trading Environment
        #--------------------
        if 'environment' in config:
            trading_environment = config['environment']
        else:
            trading_environment = factory.create_trading_environment()

        if 'order_count' in config:
            order_count = config['order_count']
        else:
            order_count = 100

        if 'order_amount' in config:
            order_amount = config['order_amount']
        else:
            order_amount = 100

        if 'trade_count' in config:
            trade_count = config['trade_count']
        else:
            # to ensure all orders are filled, we provide one more
            # trade than order
            trade_count = 101

        slippage = config.get('slippage', FixedSlippage())

        #-------------------
        # Trade Source
        #-------------------
        if 'trade_source' in config:
            trade_source = config['trade_source']
        else:
            trade_source = factory.create_daily_trade_source(
                sid_list,
                trade_count,
                trading_environment,
                concurrent=concurrent_trades
            )

        #-------------------
        # Transforms
        #-------------------
        transforms = config.get('transforms', [])

        #-------------------
        # Create the Algo
        #-------------------
        if 'algorithm' in config:
            test_algo = config['algorithm']
        else:
            test_algo = TestAlgorithm(
                sid,
                order_amount,
                order_count
            )

        #-------------------
        # Simulation
        #-------------------

        sim = SimulatedTrading(
                [trade_source],
                transforms,
                test_algo,
                trading_environment,
                slippage,
                )
        #-------------------

        return sim
