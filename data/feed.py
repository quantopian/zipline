
from data.sources.equity import *
import time
import logging

class DataFeed(object):
    
    def __init__(self, db, subscriber_count):
        self.logger = logging.getLogger()
        self.db = db
        self.data_workers = {}
        self.data_address = "tcp://127.0.0.1:{port}".format(port=10101)
        self.sync_address = "tcp://127.0.0.1:{port}".format(port=10102)
        self.feed_address = "tcp://127.0.0.1:{port}".format(port=10103)
        self.data_buffer = {} #source_id -> []
        self.subscriber_count = subscriber_count
        
        self.received_count = 0
        self.sent_count = 0
        
    def start_data_workers(self):
        """Start a sub-process for each datasource."""
        
        emt1 = EquityMinuteTrades(133, self.db, self.data_address, self.sync_address, 1)
        self.data_workers[1] = emt1
        self.data_buffer[1] = []
        emt1.start()
        emt2 = EquityMinuteTrades(134, self.db, self.data_address, self.sync_address, 2)
        self.data_workers[2] = emt1
        self.data_buffer[2] = []
        emt2.start()
        
        self.logger.info("ds processes launched")
        
    def sync_clients(self):
        # Socket to receive signals
        self.logger.info("waiting for all datasources and clients to be ready")
        self.syncservice = self.context.socket(zmq.REP)
        self.syncservice.bind(self.sync_address) 
        
        subscribers = 1
        total =  self.subscriber_count + len(self.data_workers)
        while subscribers <= total:
            self.logger.info("sync'ing {count} of {total}".format(count=subscribers, total=total))
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
                self.data_buffer[event[u's']].append(event)
                self.received_count = self.received_count + 1
                self.send_next()
            
                
        #drain any remaining messages in the buffer
        while(self.pending_messages() > 0):
            self.send_next(drain=True)
        
        #send the DONE message
        self.feed_socket.send("DONE")
        
        self.logger.info("received {n} messages, sent {m} messages".format(n=self.received_count, m=self.sent_count))
        self.data_socket.close()
        self.feed_socket.close()
        self.context.term()
        
            
    def send_next(self, drain=False):
        if(not(self.buffers_full() or drain)):
            return
            
        cur = None
        earliest = None
        for source, events in self.data_buffer.iteritems():
            if len(events) == 0:
                continue
            cur = events
            if(earliest == None) or (cur[0]['dt'] <= earliest[0]['dt']):
                earliest = cur
        
        if(earliest != None):
            event = earliest.pop(0)
            self.feed_socket.send(json.dumps(event))
            self.sent_count += 1      
        
        
    def buffers_full(self):
        for source, events in self.data_buffer.iteritems():
            if (len(events) == 0):
                return False
        return True
    
    def pending_messages(self):
        total = 0
        for source, events in self.data_buffer.iteritems():
            total += len(events)
        return total
        
    
                