
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
        
        subscribers = 0
        while subscribers < (self.subscriber_count + len(self.data_workers)):
            self.logger.info("sync'ing {count}".format(count=subscribers))
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
        
        counter = 0
        self.start_data_workers()       
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        #see: http://zguide.zeromq.org/py:taskwork2
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind(self.data_address)
        
        #create the feed
        self.feed_socket = self.context.socket(zmq.PUSH)
        self.feed_socket.bind(self.feed_address)
        
        #wait for all feed subscribers
        self.sync_clients()
        
        self.logger.info("entering feed loop on {addr}".format(addr=self.data_address))
        
        while True:
            message = self.data_socket.recv()
            event = json.loads(message)
            #self.logger.info(" count " + str(counter) + " - " + str(event['dt']))
            if(event["type"] == "DONE"):
                self.logger.info("DONE")
                source = event[u's']
                if(self.data_workers.has_key(source)):
                    del(self.data_workers[source])
                if(len(self.data_workers) == 0):
                    break
            else:
                self.data_buffer[event[u's']].append(event)
                counter = counter + 1
                self.send_earliest_event()
            
                
        #drain any remaining messages in the buffer
        self.send_earliest_event(drain=True)
        
        self.logger.info("Collected {n} messages".format(n=counter))
        self.data_socket.close()
        self.feed_socket.close()
        self.context.term()
        
    def send_earliest_event(self, drain=False):
        earliest = None
        next_source = None
        while True: #send messages as long as we have >0 messages from each source
            for source, events in self.data_buffer.iteritems():
                if(not drain and len(events) == 0 and self.data_workers.has_key(source)):
                    #there's no way to know that we have the next message
                    return 
                if(len(events) > 0 and (earliest == None or earliest > events[0])):
                    earliest = events[0]['dt']
                    next_source = source
                    
        
            event = self.data_buffer[next_source].pop(0)
            self.feed_socket.send(json.dumps(event))