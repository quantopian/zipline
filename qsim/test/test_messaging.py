"""
Test suite for the messaging infrastructure of QSim.
"""
#don't worry about excessive public methods pylint: disable=R0904  

import unittest2 as unittest
import multiprocessing
import time

from qsim.core import ThreadSimulator, DataFeed
from qsim.transforms.technical import MovingAverage
from qsim.sources import RandomEquityTrades
import qsim.util as qutil
import qsim.messaging as qmsg

from qsim.test.client import TestClient


class MessagingTestCase(unittest.TestCase):  
    """Tests the message passing: datasources -> feed -> transforms -> merge -> client"""

    def setUp(self):
        """generate some config objects for the datafeed, sources, and transforms."""
        qutil.configure_logging() 

    def get_simulator(self, sources, transforms, client, feed=None, merge=None):
        return ThreadSimulator(sources, transforms, client, feed=feed, merge=merge)

    def test_sources_only(self):
        """streams events from two data sources, no transforms."""

        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        sources = {"ret1":ret1, "ret2":ret2}
        client = TestClient(self, expected_msg_count=800)
        sim = self.get_simulator(sources, {}, client)
        sim.simulate()
              
        self.assertEqual(sim.feed.data_buffer.pending_messages(), 0, 
                        "The feed should be drained of all messages, found {n} remaining."
                            .format(n=sim.feed.data_buffer.pending_messages()))
    
    
    def test_merged_to_client(self):
        """
        2 datasources -> feed -> 2 moving average transforms -> transform merge -> testclient
        verify message count at client.
        """
        
        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        sources = {"ret1":ret1, "ret2":ret2}
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        transforms = {"mavg1":mavg1, "mavg2":mavg2}
        client = TestClient(self, expected_msg_count=800)
        sim = self.get_simulator(sources, transforms, client)
        sim.simulate()
        
        
        self.assertEqual(sim.feed.data_buffer.pending_messages(), 0, "The feed should be drained of all messages.")
        
    def test_zerror_in_feed(self):
        ret1 = RandomEquityTrades(133, "ret1", 400)
        ret2 = RandomEquityTrades(134, "ret2", 400)
        sources = {"ret1":ret1, "ret2":ret2}
        mavg1 = MovingAverage("mavg1", 30)
        mavg2 = MovingAverage("mavg2", 60)
        transforms = {"mavg1":mavg1, "mavg2":mavg2}
        client = TestClient(self, expected_msg_count=0)
        sim = self.get_simulator(sources, transforms, client)
        sim.feed = DataFeedErr(sources.keys(), sim.data_address, sim.feed_address, qmsg.Sync(sim, "DataFeedErrorGenerator"))
        sim.simulate()
        
class DataFeedErr(DataFeed):
    """Helper class for testing, simulates exceptions inside the DataFeed"""
    
    def __init__(self, source_list, data_address, feed_address, sync):
        DataFeed.__init__(self, source_list, data_address, feed_address, sync)
    
    def handle_all(self):
        #time.sleep(1000)
        raise Exception("simulated error in data feed from test helper")
    
        
