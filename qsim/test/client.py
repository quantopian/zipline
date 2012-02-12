import copy
import multiprocessing
import zmq
import logging
import json

import qsim.util as qutil
import qsim.messaging as qmsg

class TestClient(object):
    
    def __init__(self):
        self.address        = None
        self.sync           = None
        self.received_count = 0
        
    def run(self):
        
        qutil.LOGGER.info("running the client")
        self.context = zmq.Context()
        
        self.data_feed = self.context.socket(zmq.PULL)

        qutil.LOGGER.info("connecting to {address}".format(address=self.address))
        self.data_feed.connect(self.address)
        
        self.sync.confirm()
        
        qutil.LOGGER.info("Starting the client loop")

        prev_dt = None
        while True:
            msg = self.data_feed.recv()
            if(msg == "DONE"):
                qutil.LOGGER.info("DONE!")
                break
            self.received_count += 1
            event = json.loads(msg)
            if(prev_dt != None):
                if(not event['dt'] >= prev_dt):
                    raise Exception("message arrived out of order: {date} after {prev}".format(date=event['dt'], prev=prev_dt))
            
            prev_dt = event['dt']
            if(self.received_count % 100 == 0):
                qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
            
        qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
        self.data_feed.close()
        self.context.term()