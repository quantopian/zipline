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

import logbook

from zipline.test_algorithms import TestAlgorithm
from zipline.finance.trading import SIMULATION_STYLE
from zipline.utils import factory
import pytz

from pprint import pprint as pp
from datetime import datetime, timedelta

from zipline.utils.factory import create_trading_environment
from zipline.test_algorithms import TestAlgorithm

from zipline.gens.composites import SourceBundle, TransformBundle, \
    date_sorted_sources, merged_transforms
from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import MovingAverage, Passthrough, StatefulTransform
from zipline.gens.tradesimulation import TradeSimulationClient as tsc

import zipline.protocol as zp


log = logbook.Logger('Lines')

class SimulatedTrading(object):

    @staticmethod
    def create_simulation(sources, transforms, algorithm, environment, style):

        sorted = date_sorted_sources(*sources)
        passthrough = StatefulTransform(Passthrough)

        merged = merged_transforms(sorted, passthrough, *transforms)
        trading_client = tsc(algorithm, environment, style)
        return trading_client.simluate(merged)


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
            - simulation_style: optional parameter that configures the
              :py:class:`zipline.finance.trading.TransactionSimulator`. Expects
              a SIMULATION_STYLE as defined in :py:mod:`zipline.finance.trading`
            - transforms: optional parameter that provides a list
              of StatefulTransform objects.
        """
        assert isinstance(config, dict)

        sid = config['sid']

        #--------------------
        # Trading Environment
        #--------------------
        if config.has_key('environment'):
            trading_environment = config['environment']
        else:
            trading_environment = factory.create_trading_environment()

        if config.has_key('order_count'):
            order_count = config['order_count']
        else:
            order_count = 100

        if config.has_key('order_amount'):
            order_amount = config['order_amount']
        else:
            order_amount = 100

        if config.has_key('trade_count'):
            trade_count = config['trade_count']
        else:
            # to ensure all orders are filled, we provide one more
            # trade than order
            trade_count = 101

        simulation_style = config.get('simulation_style')
        if not simulation_style:
            simulation_style = SIMULATION_STYLE.FIXED_SLIPPAGE

        #-------------------
        # Trade Source
        #-------------------
        sids = [sid]
        #-------------------
        if config.has_key('trade_source'):
            trade_source = config['trade_source']
        else:
            trade_source = factory.create_daily_trade_source(
                sids,
                trade_count,
                trading_environment
            )

        #-------------------
        # Transforms
        #-------------------
        transforms = config.get('transforms', [])

        #-------------------
        # Create the Algo
        #-------------------
        if config.has_key('algorithm'):
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

        sim = SimulatedTrading.create_simulation(
                [trade_source],
                transforms,
                test_algo,
                trading_environment,
                simulation_style)
        #-------------------

        return sim


class ZiplineException(Exception):
    def __init__(self, zipline_name, msg):
        self.name = zipline_name
        self.message = msg

    def __str__(self):
        return "Unexpected exception {line}: {msg}".format(
            line=self.name,
            msg=self.message
        )
