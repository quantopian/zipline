import unittest2 as unittest
import zmq
import logging
import tornado
from simulator.data.sources.equity import *
from simulator.data.feed import *
from transforms.transforms import MergedTransformsFeed, MovingAverage

from simulator.qbt_client import TestClient


class MessagingTestCase(unittest.TestCase):

    def setUp(self):
        self.total_data_count = 800
        self.feed_config = {'emt1':{'sid':133, 'class':'RandomEquityTrades', 'count':400},
                            'emt2':{'sid':134, 'class':'RandomEquityTrades', 'count':400}}
        self.feed = DataFeed(self.feed_config) 
        self.feed_proc = multiprocessing.Process(target=self.feed.run)
        
        self.config = {}
        self.config['name'] = '**merged feed**'
        self.config['transforms'] = [{'name':'mavg1', 'class':'MovingAverage', 'hours':1},{'name':'mavg2', 'class':'MovingAverage', 'hours':2}]  

    def test_client(self):  
        #subscribe a client to the transformed feed
        client = TestClient(self.feed, self.feed.feed_address)
    
        feed_proc = multiprocessing.Process(target=self.feed.run)
        feed_proc.start()
        
        
        client.run()
        self.assertEqual(self.feed.data_buffer.pending_messages(), 0, "The feed should be drained of all messages, found {n} remaining.".format(n=self.feed.data_buffer.pending_messages()))
        self.assertEqual(self.total_data_count, client.received_count, "The client should have received ({n}) the same number of messages as the feed sent ({m}).".format(n=client.received_count, m=self.total_data_count))
    
    
    def dtest_moving_average_to_client(self):
        mavg = MovingAverage(self.feed, self.config['transforms'][0])
        mavg_proc = multiprocessing.Process(target=mavg.run)
        mavg_proc.start()
    
        client = TestClient(self.feed, mavg.result_address, bind=True)
    
        feed_proc = multiprocessing.Process(target=self.feed.run)
        feed_proc.start()
    
        client.run()
        self.assertEqual(self.feed.data_buffer.pending_messages(), 0, "The feed should be drained of all messages.")
        self.assertEqual(self.total_data_count, client.received_count, "The client should have received the same number of messages as the feed sent.")
        
    def dtest_merged_to_client(self):
        merger = MergedTransformsFeed(self.feed, self.config)
        merger_proc = multiprocessing.Process(target=merger.run)
        merger_proc.start() 
    
        client = TestClient(self.feed, merger.result_address)
    
        feed_proc = multiprocessing.Process(target=self.feed.run)
        feed_proc.start()
    
        client.run()
        self.assertEqual(self.feed.data_buffer.pending_messages(), 0, "The feed should be drained of all messages.")
        self.assertEqual(self.total_data_count, client.received_count, "The client should have received the same number of messages as the feed sent.")
        
