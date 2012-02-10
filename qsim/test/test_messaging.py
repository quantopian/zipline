"""
Test suite for the messaging infrastructure of QSim.
"""
#don't worry about excessive public methods pylint: disable=R0904  

import unittest2 as unittest
import multiprocessing

from qsim.simulator.feed import DataFeed
from qsim.transforms.merge import MergedTransformsFeed
from qsim.transforms.technical import MovingAverage
import qsim.util as qutil

from qsim.test.client import TestClient


class MessagingTestCase(unittest.TestCase):  
    """Tests the message passing: datasources -> feed -> transforms -> merge -> client"""

    def setUp(self):
        """generate some config objects for the datafeed, sources, and transforms."""
        
        qutil.configure_logging()
        qutil.logger.info("testing...")
        self.total_data_count = 800
        self.feed_config = {'emt1':{'sid':133, 'class':'RandomEquityTrades', 'count':400},
                            'emt2':{'sid':134, 'class':'RandomEquityTrades', 'count':400}}
        self.feed = DataFeed(self.feed_config) 
        self.feed_proc = multiprocessing.Process(target=self.feed.run)
        
        self.config = {}
        self.config['name'] = '**merged feed**'
        self.config['transforms'] = [{'name':'mavg1', 'class':'MovingAverage', 'hours':1},
                                    {'name':'mavg2', 'class':'MovingAverage', 'hours':2}]  

    def test_client(self):
        """directly connect the test client to the feed, using two random data sources"""
          
        #subscribe a client to the multiplexed feed
        client = TestClient(self.feed, self.feed.feed_address)
    
        feed_proc = multiprocessing.Process(target=self.feed.run)
        feed_proc.start()
        
        
        client.run()
        self.assertEqual(self.feed.data_buffer.pending_messages(), 0, 
                        "The feed should be drained of all messages, found {n} remaining."
                            .format(n=self.feed.data_buffer.pending_messages()))
        self.assertEqual(self.total_data_count, client.received_count, 
                        "The client should have received ({n}) the same number of messages as the feed sent ({m})."
                            .format(n=client.received_count, m=self.total_data_count))
    
    
    def dtest_moving_average_to_client(self):
        """2 datasources -> feed -> moving average transform -> testclient
        verify message count at client."""
        
        mavg = MovingAverage(self.feed, self.config['transforms'][0], result_address="tcp://127.0.0.1:20202")
        mavg_proc = multiprocessing.Process(target=mavg.run)
        mavg_proc.start()
    
        client = TestClient(self.feed, mavg.result_address, bind=True)
    
        feed_proc = multiprocessing.Process(target=self.feed.run)
        feed_proc.start()
    
        client.run()
        self.assertEqual(self.feed.data_buffer.pending_messages(), 0, "The feed should be drained of all messages.")
        self.assertEqual(self.total_data_count, client.received_count, 
                        "The client should have received the same number of messages as the feed sent.")
        
    def dtest_merged_to_client(self):
        """
        2 datasources -> feed -> 2 moving average transforms -> transform merge -> testclient
        verify message count at client.
        """
        merger = MergedTransformsFeed(self.feed, self.config)
        merger_proc = multiprocessing.Process(target=merger.run)
        merger_proc.start() 
    
        client = TestClient(self.feed, merger.result_address)
    
        feed_proc = multiprocessing.Process(target=self.feed.run)
        feed_proc.start()
    
        client.run()
        self.assertEqual(self.feed.data_buffer.pending_messages(), 0, "The feed should be drained of all messages.")
        self.assertEqual(self.total_data_count, client.received_count, 
                        "The client should have received the same number of messages as the feed sent.")
        
