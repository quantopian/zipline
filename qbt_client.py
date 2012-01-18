import copy
import multiprocessing
import zmq
import logging
import json

from backtest.util import *

class TestClient(object):
    
    def __init__(self,feed_address, sync_address, bind=False):
        self.logger = logging.getLogger()
        self.feed_address = feed_address
        self.sync_address = sync_address
        self.bind = bind
        
    def run(self):
        
        self.logger.info("running the client")
        self.context = zmq.Context()
        
        self.data_feed = self.context.socket(zmq.PULL)

        if(self.bind):
            self.logger.info("binding to {feed_address}".format(feed_address=self.feed_address))
            self.data_feed.bind(self.feed_address)
        else:
            self.data_feed.connect(self.feed_address)
        
        self.logger.info("synchronizing with data feed")
        
        #synchronize with feed
        sync_socket = self.context.socket(zmq.REQ)
        sync_socket.connect(self.sync_address)
        # send a synchronization request to the feed
        sync_socket.send('')
        # wait for synchronization reply from the feed
        sync_socket.recv()
        sync_socket.close()
        
        self.logger.info("Starting the client loop")

        counter = 0
        prev_dt = None
        while True:
            msg = self.data_feed.recv()
            counter += 1
            if(msg == "DONE"):
                self.logger.info("DONE!")
                break
            event = json.loads(msg)
            if(prev_dt != None):
                if(not event['dt'] >= prev_dt):
                    raise Exception("message arrived out of order: {date} after {prev}".format(date=event['dt'], prev=prev_dt))
            
            prev_dt = event['dt']
            if(counter % 100 == 0):
                self.logger.info("received {n} messages".format(n=counter))
            
        self.logger.info("received {n} messages".format(n=counter))
        self.data_feed.close()
        self.context.term()