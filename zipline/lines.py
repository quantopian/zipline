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

import zipline.utils.factory as factory

from zipline.components import DataSource
from zipline.transforms import BaseTransform

from zipline.test_algorithms import TestAlgorithm
from zipline.components import TradeSimulationClient
from zipline.core.devsimulator import Simulator
from zipline.core.monitor import Controller
from zipline.finance.trading import SIMULATION_STYLE

class SimulatedTrading(object):
    """
    Zipline with::

    - _no_ data sources.
    - Trade simulation client, which is available to send callbacks on
    events and also accept orders to be simulated.
    - An order data source, which will receive orders from the trade
    simulation client, and feed them into the event stream to be
    serialized and order alongside all other data source events.
    - transaction simulation transformation, which receives the order
    events and estimates a theoretical execution price and volume.

    All components in this zipline are subject to heartbeat checks and
    a control monitor, which can kill the entire zipline in the event of
    exceptions in one of the components or an external request to end the
    simulation.
    """

    def __init__(self, **config):
        """
        :param config: a dict with the following required properties::

        - algorithm: a class that follows the algorithm protocol. See
        :py:meth:`zipline.finance.trading.TradingSimulationClient.add_algorithm
        for details.
        - trading_environment: an instance of
        :py:class:`zipline.trading.TradingEnvironment`
        - allocator: an instance of
        :py:class:`zipline.simulator.AddressAllocator`
        - simulator_class: a :py:class:`zipline.core.host.ComponentHost`
        subclass (not an instance)
        - simulation_style: optional parameter that configures the
        :py:class:`zipline.finance.trading.TransactionSimulator`. Expects
        a SIMULATION_STYLE as defined in :py:mod:`zipline.finance.trading`
        """
        assert isinstance(config, dict)
        self.algorithm = config['algorithm']
        self.allocator = config['allocator']
        self.trading_environment = config['trading_environment']
        self.sim_style = config.get('simulation_style')

        self.leased_sockets = []
        self.sim_context = None

        sockets = self.allocate_sockets(8)
        addresses = {
            'sync_address'   : sockets[0],
            'data_address'   : sockets[1],
            'feed_address'   : sockets[2],
            'merge_address'  : sockets[3],
            'result_address' : sockets[4],
            'order_address'  : sockets[5]
        }

        self.con = Controller(
            sockets[6],
            sockets[7],
        )

        self.con.cancel_socket = self.allocator.lease(1)[0]

        # TODO: Not freeform
        self.con.manage(
            'freeform'
        )

        self.started = False

        self.sim = config['simulator_class'](addresses)

        self.clients = {}
        self.trading_client = TradeSimulationClient(
            self.trading_environment,
            self.sim_style
        )
        self.add_client(self.trading_client)

        # setup all sources
        self.sources = {}
        #self.order_source = OrderDataSource()
        #self.add_source(self.order_source)

        #setup transforms
        #self.transaction_sim = TransactionSimulator(self.sim_style)
        self.transforms = {}
        #self.add_transform(self.transaction_sim)

        self.sim.register_controller( self.con )

        self.trading_client.set_algorithm(self.algorithm)

    @staticmethod
    def create_test_zipline(**config):
        """
        :param config: A configuration object that is a dict with:

            - environment - a \
              :py:class:`zipline.finance.trading.TradingEnvironment`
            - allocator - a :py:class:`zipline.simulator.AddressAllocator`
            - sid - an integer, which will be used as the security ID.
            - order_count - the number of orders the test algo will place,
              defaults to 100
            - order_amount - the number of shares per order, defaults to 100
            - trade_count - the number of trades to simulate, defaults to 101
              to ensure all orders are processed.
            - simulator_class - optional parameter that provides an alternative
              subclass of ComponentHost to hold the whole zipline. Defaults to
              :py:class:`zipline.simulator.Simulator`
            - algorithm - optional parameter providing an algorithm. defaults
              to :py:class:`zipline.test.algorithms.TestAlgorithm`
            - trade_source - optional parameter to specify trades, if present.
              If not present :py:class:`zipline.sources.SpecificEquityTrades`
              is the source, with daily frequency in trades.
            - simulation_style: optional parameter that configures the
              :py:class:`zipline.finance.trading.TransactionSimulator`. Expects
              a SIMULATION_STYLE as defined in :py:mod:`zipline.finance.trading`
        """
        assert isinstance(config, dict)

        allocator = config['allocator']
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

        if config.has_key('simulator_class'):
            simulator_class = config['simulator_class']
        else:
            simulator_class = Simulator

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
        zipline = SimulatedTrading(**{
            'algorithm'           : test_algo,
            'trading_environment' : trading_environment,
            'allocator'           : allocator,
            'simulator_class'     : simulator_class,
            'simulation_style'    : simulation_style
        })
        #-------------------

        zipline.add_source(trade_source)

        return zipline

    def add_source(self, source):
        """
        Adds the source to the zipline, sets the sid filter of the
        source to the algorithm's sid filter.
        """
        assert isinstance(source, DataSource)
        self.check_started()
        source.set_filter('sid', self.algorithm.get_sid_filter())
        self.sim.register_components([source])

        # ``id`` is name of source_id, ``get_id`` is the class name
        self.sources[source.source_id] = source

    def add_transform(self, transform):
        assert isinstance(transform, BaseTransform)
        self.check_started()
        self.sim.register_components([transform])
        self.transforms[transform.get_id] = transform

    def add_client(self, client):
        assert isinstance(client, TradeSimulationClient)
        self.check_started()
        self.sim.register_components([client])
        self.clients[client.get_id] = client

    def check_started(self):
        if self.started:
            raise ZiplineException("TradeSimulation", "You cannot add \
            components after the simulation has begun.")

    def get_cumulative_performance(self):
        return self.trading_client.perf.cumulative_performance.to_dict()

    def publish_to(self, result_socket):
        self.trading_client.perf.publish_to(result_socket)

    def allocate_sockets(self, n):
        """
        Allocate sockets local to this line, track them so
        we can gc after test run.
        """

        assert isinstance(n, int)
        assert n > 0

        leased = self.allocator.lease(n)
        self.leased_sockets.extend(leased)

        return leased

    def simulate(self, blocking=False):
        self.started = True
        self.sim_context = self.sim.simulate()

        if blocking:
            self.sim_context.join()

    @property
    def is_success(self):
        return self.sim.ready() and not self.sim.exception

    #--------------------------------
    # Component property accessors
    #--------------------------------

    def get_positions(self):
        """
        returns current positions as a dict. draws from the cumulative
        performance period in the performance tracker.
        """
        perf = self.trading_client.perf.cumulative_performance
        positions = perf.get_positions()
        return positions

class ZiplineException(Exception):
    def __init__(self, zipline_name, msg):
        self.name = zipline_name
        self.message = msg

    def __str__(self):
        return "Unexpected exception {line}: {msg}".format(
            line=self.name,
            msg=self.message
        )
