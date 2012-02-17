"""
Test suite for the messaging infrastructure of QSim.
"""
#don't worry about excessive public methods pylint: disable=R0904  

import unittest2 as unittest
import multiprocessing
import time

from zipline.simulator import ThreadSimulator, ProcessSimulator
from zipline.transforms.technical import MovingAverage
from zipline.sources import RandomEquityTrades
import zipline.util as qutil
import zipline.messaging as qmsg

from zipline.test.client import TestClient

qutil.configure_logging()

class ThreadSimulatorTestCase(unittest.TestCase):  
    """Tests the message passing: datasources -> feed -> transforms -> merge -> client"""

    def setUp(self):
        """generate some config objects for the datafeed, sources, and transforms."""
        self.addresses              = {'sync_address'           : "tcp://127.0.0.1:10100",
                                       'data_address'           : "tcp://127.0.0.1:10101",
                                       'feed_address'           : "tcp://127.0.0.1:10102",
                                       'merge_address'          : "tcp://127.0.0.1:10103",
                                       'result_address'         : "tcp://127.0.0.1:10104"
                                      }
                                      
        self.addressesblarg = "test"
    
    def get_simulator(self):
        return ThreadSimulator(self.addresses)


    def test_sources_only(self):
        """streams events from two data sources, no transforms."""
        sim = self.get_simulator()
        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        client = TestClient(self, expected_msg_count=800)
        sim.register_components([ret1, ret2, client])
        sim.simulate()
              
        self.assertEqual(sim.feed.pending_messages(), 0, 
                        "The feed should be drained of all messages, found {n} remaining."
                            .format(n=sim.feed.pending_messages()))
    
    
    def test_transforms(self):
        """
        2 datasources -> feed -> 2 moving average transforms -> transform merge -> testclient
        verify message count at client.
        """
        sim = self.get_simulator()
        ret1 = RandomEquityTrades(133, "ret1", 5000)
        ret2 = RandomEquityTrades(134, "ret2", 5000)
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        client = TestClient(self, expected_msg_count=10000)
        sim.register_components([ret1, ret2, mavg1, mavg2, client])
        sim.simulate()
        
        self.assertEqual(sim.feed.pending_messages(), 0, "The feed should be drained of all messages.")
        
    def dtest_error_in_feed(self):
        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        sources = {"ret1":ret1, "ret2":ret2}
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        transforms = {"mavg1":mavg1, "mavg2":mavg2}
        client = TestClient(self, expected_msg_count=0)
        sim = self.get_simulator(sources, transforms, client)
        sim.feed = DataFeedErr(sources.keys(), sim.data_address, sim.feed_address, sim.performance_address, qmsg.Sync(sim, "DataFeedErrorGenerator"))
        sim.simulate()
        
    
class ProcessSimulatorTestCase(ThreadSimulatorTestCase):
    
    def get_simulator(self):
        return ProcessSimulator(self.addresses)

