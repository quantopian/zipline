
import qsim.data.equity as qequity
import qsim.util as qutil
import zmq
import time
import logging
import json

class DataFeed(object):
    
    def __init__(self, config):
        self.logger = qutil.logger
        
        self.data_address = "tcp://127.0.0.1:{port}".format(port=10101)
        self.sync_address = "tcp://127.0.0.1:{port}".format(port=10102)
        self.feed_address = "tcp://127.0.0.1:{port}".format(port=10103)
        
        self.client_register = {}
        
        self.data_workers = {}
        self.config = config
        for name, info in config.iteritems():
            if(info['class'] == "EquityMinuteTrades"):
                emt = EquityMinuteTrades(info['sid'], self, name)
                self.data_workers[name] = emt
            elif(info['class'] == "RandomEquityTrades"):
                ret = qequity.RandomEquityTrades(info['sid'], self, name, info['count'])
                self.data_workers[name] = ret
                
        self.data_buffer = qutil.ParallelBuffer(self.data_workers.keys())
             
    def start_data_workers(self):
        """Start a sub-process for each datasource.""" 
        for source_id, source in self.data_workers.iteritems():
            self.logger.info("starting {id}".format(id=source_id))
            source.start()
        self.logger.info("ds processes launched")
        
    def register_sync(self, sync_id):
        self.client_register[sync_id] = "UNCONFIRMED"
        
    def registration_complete(self):
        for sync_id, status in self.client_register.iteritems():
            if status == "UNCONFIRMED":
                return False
        
        return True
    
    def sync_clients(self):
        # Socket to receive signals
        self.logger.info("waiting for all datasources and clients to be ready")
        self.syncservice = self.context.socket(zmq.REP)
        self.syncservice.bind(self.sync_address) 
        
        while not self.registration_complete():
            # wait for synchronization request
            msg = self.syncservice.recv()
            self.client_register[msg] = "CONFIRMED"
            #self.logger.info("confirmed {id}".format(id=msg))
            # send synchronization reply
            self.syncservice.send('CONFIRMED')
        
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
        self.feed_socket = self.context.socket(zmq.PUB)
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
        
            
       
        
        

        
            
