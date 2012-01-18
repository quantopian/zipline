
from data.sources.equity import *
from backtest.util import *
import time
import logging

class DataFeed(object):
    
    def __init__(self, db, subscriber_count):
        self.logger = logging.getLogger()
        
        self.data_address = "tcp://127.0.0.1:{port}".format(port=10101)
        self.sync_address = "tcp://127.0.0.1:{port}".format(port=10102)
        self.feed_address = "tcp://127.0.0.1:{port}".format(port=10103)
        
        self.db = db
        self.data_workers = {}
        emt1 = EquityMinuteTrades(133, self.db, self.data_address, self.sync_address, 1)
        self.data_workers[1] = emt1
        emt2 = EquityMinuteTrades(134, self.db, self.data_address, self.sync_address, 2)
        self.data_workers[2] = emt2
        
        self.data_buffer = ParallelBuffer(self.data_workers.keys())
        self.sync_count = subscriber_count + len(self.data_workers)
        
        
    def start_data_workers(self):
        """Start a sub-process for each datasource.""" 
        for source_id, source in self.data_workers.iteritems():
            self.logger.info("starting {id}".format(id=source_id))
            source.start()
        self.logger.info("ds processes launched")
        
    def sync_clients(self):
        # Socket to receive signals
        self.logger.info("waiting for all datasources and clients to be ready")
        self.syncservice = self.context.socket(zmq.REP)
        self.syncservice.bind(self.sync_address) 
        
        
        subscribers = 1
        while subscribers <= self.sync_count:
            self.logger.info("sync'ing {count} of {total}".format(count=subscribers, total=self.sync_count))
            # wait for synchronization request
            msg = self.syncservice.recv()
            # send synchronization reply
            self.syncservice.send('')
            subscribers += 1
        
        self.syncservice.close()
        self.logger.info("sync'd all datasources and clients")
       
    def run(self):   
        # Prepare our context and sockets
        self.context = zmq.Context()
        
        ds_finished_counter = 0
             
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        #see: http://zguide.zeromq.org/py:taskwork2
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind(self.data_address)
        
        #create the feed
        self.feed_socket = self.context.socket(zmq.PUSH)
        self.feed_socket.bind(self.feed_address)
        
        self.data_buffer.out_socket = self.feed_socket
        
        #start the data source workers
        self.start_data_workers()
        
        #wait for all feed subscribers
        self.sync_clients()
        
        self.logger.info("entering feed loop on {addr}".format(addr=self.data_address))
        
        while True:
            message = self.data_socket.recv()
            event = json.loads(message)
            if(event["type"] == "DONE"):
                ds_finished_counter += 1
                if(len(self.data_workers) == ds_finished_counter):
                    break
            else:
                self.data_buffer.append(event[u's'], event)
                self.data_buffer.send_next()
            
                
        #drain any remaining messages in the buffer
        self.data_buffer.drain()
        
        #send the DONE message
        self.feed_socket.send("DONE")
        
        self.logger.info("received {n} messages, sent {m} messages".format(n=self.data_buffer.received_count, m=self.data_buffer.sent_count))
        self.data_socket.close()
        self.feed_socket.close()
        self.context.term()
        
            
       
        
        

        
            