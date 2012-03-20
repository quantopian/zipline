"""
Ziplines are composed of multiple components connected by asynchronous 
messaging. All ziplines follow a general topology of parallel sources, 
datetimestamp serialization, parallel transformations, and finally sinks. 
Furthermore, many ziplines have common needs. For example, all trade 
simulations require a 
:py:class:`~zipline.finance.trading.TradeSimulationClient`, an
:py:class:`~zipline.finance.trading.OrderSource`, and a 
:py:class:`~zipline.finance.trading.TransactionSimulator` (a transform).

To establish best practices and minimize code replication, the lines module 
provides complete zipline topologies. You can extend any zipline without
the need to extend the class. Simply instantiate any additional components
that you would like included in the zipline, and add them to the zipline 
before invoking simulate. 

        
        Here is a diagram of the SimulatedTrading zipline:
        
        
            +----------------------+  +------------------------+
        +-->|  Orders DataSource   |  |    (DataSource added   |
        |   |  Integrates algo     |  |     via add_source)    |
        |   |  orders into history |  |                        |
        |   +--------------------+-+  +-+----------------------+
        |                        |      |
        |                        |      |
        |                        v      v
        |                       +---------+
        |                       |   Feed  |
        |                       +-+------++
        |                         |      |
        |                         |      |    
        |                         v      v
        |    +----------------------+   +----------------------+
        |    | Transaction          |   |                      |
        |    | Transform simulates  |   |  (Transforms added   |
        |    | trades based on      |   |   via add_transform) |
        |    | orders from algo.    |   |                      |
        |    +-------------------+--+   +-+--------------------+
        |                        |        |
        |                        |        |
        |                        v        v
        |                      +------------+
        |                      |    Merge   |
        |                      +------+-----+
        |                             |
        |                             |
        |                             V
        |               +--------------------------------+
        |               |                                |
        |               |     TradingSimulationClient    |
        |  orders       |     tracks performance and     |
        +---------------+     provides API to algorithm. |
                        |                                |
                        +---------------------+----------+
                                  ^           |
                                  | orders    |  frames
                                  |           |
                                  |           v
                        +---------+-----------------------+
                        |                                 |
                        |  Algorithm added via            |
                        |  __init__.                      |
                        |                                 |
                        |                                 |
                        |                                 |
                        +---------------------------------+

"""

import mock
import pytz

from datetime import datetime, timedelta
from collections import defaultdict

from nose.tools import timed

import zipline.test.factory as factory
import zipline.util as qutil
import zipline.finance.risk as risk
import zipline.protocol as zp
import zipline.finance.performance as perf
import zipline.messaging as zmsg

from zipline.test.client import TestAlgorithm
from zipline.sources import SpecificEquityTrades
from zipline.finance.trading import TransactionSimulator, OrderDataSource, \
TradeSimulationClient
from zipline.simulator import AddressAllocator, Simulator
from zipline.monitor import Controller



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
            - algorithm: a class that follows the algorithm protocol. Must
            have a handle_frame method that accepts a pandas.Dataframe of the 
            current state of the simulation universe. Must have an order 
            property which can be set equal to the order method of 
            trading_client. (TODO: where should this protocol be documented?)
            - trading_environment: an instance of
            :py:class:`zipline.trading.TradingEnvironment`
            - allocator: an instance of 
            :py:class:`zipline.simulator.AddressAllocator`
            - simulator_class: a :py:class:`zipline.messaging.ComponentHost` 
            subclass (not an instance)
        """
        assert isinstance(config, dict)
        self.algorithm = config['algorithm']
        self.allocator = config['allocator']
        self.trading_environment = config['trading_environment']
        
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
            logging = qutil.LOGGER
        )

        
        self.sim = config['simulator_class'](addresses)
            
        self.clients = {}
        self.trading_client = TradeSimulationClient(self.trading_environment)
        self.clients[self.trading_client.get_id] = self.trading_client
        
        # setup all sources
        self.sources = {}
        self.order_source = OrderDataSource()
        self.sources[self.order_source.get_id] = self.order_source
        
        #setup transforms
        self.transaction_sim = TransactionSimulator()
        self.transforms = {}
        self.transforms[self.transaction_sim.get_id] = self.transaction_sim
        
        #register all components
        self.sim.register_components([
            self.trading_client, 
            self.order_source, 
            self.transaction_sim 
            ])
            
        self.sim.register_controller( self.con )
        self.sim.on_done = self.shutdown()
        self.started = False
        
        ##################################################################
        #TODO: the next two lines of code need refactoring from RealDiehl
        ##################################################################
        #wire up a callback inside the algorithm to receive frames from the
        #trading client
        self.trading_client.add_event_callback(self.algorithm.handle_frame)
        #register the trading_client's order method with the algorithm
        self.algorithm.set_order(self.trading_client.order)
    
    @staticmethod
    def create_test_zipline(**config):
        """
        :param config: A configuration object that is a dict with::
            - environment - a \
            :py:class:`zipline.finance.trading.TradeSimulationClient`
            - allocator - a :py:class:`zipline.simulator.AddressAllocator`
            - sid - an integer, which will be used as the security ID. 
            - order_count - the number of orders the test algo will place,
            defaults to 100
            - trade_count - the number of trades to simulate, defaults to 100
            - simulator_class - optional parameter that provides an alternative 
            subclass of ComponentHost to hold the whole zipline. Defaults to
            :py:class:`zipline.simulator.Simulator`   
        """
        assert isinstance(config, dict)
        trading_environment = config['environment']
        allocator = config['allocator']
        sid = config['sid']
        if config.has_key('order_count'):
            order_count = config['order_count']
        else:
            order_count = 100
            
        if config.has_key('trade_count'):
            trade_count = config['trade_count']
        else:
            trade_count = 100
            
        if config.has_key('simulator_class'):
            simulator_class = config['simulator_class']
        else:
            simulator_class = Simulator
            
        #-------------------
        # Trade Source
        #-------------------
        sids = [sid]
        #-------------------
        trade_source = factory.create_daily_trade_source(
            sids,
            trade_count,
            trading_environment
        )
        #-------------------
        # Create the Algo
        #-------------------
        order_amount = 100
        #-------------------
        test_algo = TestAlgorithm(
            sid,
            order_amount,
            order_count
        )
        #-------------------
        # Simulation
        #-------------------
        zipline = SimulatedTrading(**{
            'algorithm':test_algo,
            'trading_environment':trading_environment,
            'allocator':allocator,
            'simulator_class':simulator_class
        })
        #-------------------

        zipline.add_source(trade_source)

        return zipline
        
    def add_source(self, source):
        assert isinstance(source, zmsg.DataSource)
        self.check_started()    
        self.sim.register_components([source])
        self.sources[source.get_id] = source
        
    
    def add_transform(self, transform):
        assert isinstance(transform, zmsg.BaseTransform)
        self.check_started()
        self.sim.register_components([transform])
        self.sources[transform.get_id] = transform
    
    def check_started(self):
        if self.started:
            raise ZiplineException("You cannot add sources after the \
            simulation has begun.")
    
    def get_cumulative_performance(self):
        self.trading_client.perf.cumulative_performance.to_dict()
    
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

    def shutdown(self):
        self.allocator.reaquire(*self.leased_sockets)
        
    #--------------------------------#
    # Component property accessors   #
    #--------------------------------#
    
    def get_positions(self):
        """
        returns current positions as a dict. draws from the cumulative
        performance period in the performance tracker.
        """
        perf = self.trading_client.perf.cumulative_performance
        positions = perf.get_positions()
        return positions
        
class ZiplineException(Exception):
    def __init__(msg):
        Exception.__init__(msg)
        