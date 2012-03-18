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
    
    def __init__(self, trading_environment, allocator):
        self.allocator = allocator
        self.leased_sockets = []
        self.trading_environment = trading_environment
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

        self.sim = Simulator(addresses)
            
        self.trading_environment.frame_index = ['sid', 'volume', 'dt', \
        'price', 'changed']
        
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
        