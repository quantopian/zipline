import copy
import multiprocessing
import zmq
import logging

from backtest.util import *

class BacktestClient(object):
    
    def __init__(self,feed_address, sync_address):
        self.context = zmq.Context()
        self.logger = logging.getLogger()
        self.feed_address = feed_address
        self.sync_address = sync_address
        
    def run(self):
        self.logger.info("running the client")
        self.data_feed = self.context.socket(zmq.PULL)
        self.data_feed.connect(self.feed_address)
        
        #synchronize with feed
        sync_socket = self.context.socket(zmq.REQ)
        sync_socket.connect(self.sync_address)
        # send a synchronization request to the feed
        sync_socket.send('')
        # wait for synchronization reply from the feed
        sync_socket.recv()
        sync_socket.close()
        
        counter = 0
        while True:
            counter += 1
            msg = self.data_feed.recv()
            self.logger.info("received {n} messages".format(n=counter))
            