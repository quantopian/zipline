import copy
import multiprocessing
import zmq

from backtest.util import *

class BacktestClient(object):
    
    def __init__(self,feed_port, logger):
        self.context = zmq.Context()
        
        self.data_feed = self.context.socket(zmq.PULL)
        self.data_feed.connect("tcp://localhost:{port}".format(port=feed_port))
        self.logger = logger
        
    def run(self):
        self.logger.info("running the client")
        while True:
            msg = self.data_feed.recv()
            self.logger.info(msg)