"""
Provides simulated data feed services. 
"""

import qsim.sources as sources
import qsim.util as qutil
import qsim.messaging as qmsg
import zmq
import time
import logging
import json

class Simulator(object):
    """
    Simulator is the heart of QSim. The beating heart...
    """
    
    def __init__(self, config):
        """
        :config: a qsim.config.Config object that contains configuration information for all datasources, all transforms, and all 
        client algorithms that simulator should create.
        """
        self.config = config
        
    def launch(self):
        """
        Create all components specified in config. 
        """
        pass


class DataFeed(object):
    """DataFeed is the heart of a simulation. It is initialized with a configuration for """
    
    def __init__(self, config):
        qutil.LOGGER = qutil.LOGGER
        
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
                ret = sources.RandomEquityTrades(info['sid'], self, name, info['count'])
                self.data_workers[name] = ret
                
        self.data_buffer = qmsg.ParallelBuffer(self.data_workers.keys())
             
    def start_data_workers(self):
        """Start a sub-process for each datasource.""" 
        for source_id, source in self.data_workers.iteritems():
            qutil.LOGGER.info("starting {id}".format(id=source_id))
            source.start()
        qutil.LOGGER.info("ds processes launched")
        
    def register_sync(self, sync_id):
        self.client_register[sync_id] = "UNCONFIRMED"
        
    def registration_complete(self):
        for sync_id, status in self.client_register.iteritems():
            if status == "UNCONFIRMED":
                return False
        
        return True
    
    def sync_clients(self):
        # Socket to receive signals
        qutil.LOGGER.info("waiting for all datasources and clients to be ready")
        self.syncservice = self.context.socket(zmq.REP)
        self.syncservice.bind(self.sync_address) 
        
        while not self.registration_complete():
            # wait for synchronization request
            msg = self.syncservice.recv()
            self.client_register[msg] = "CONFIRMED"
            #qutil.LOGGER.info("confirmed {id}".format(id=msg))
            # send synchronization reply
            self.syncservice.send('CONFIRMED')
        
        self.syncservice.close()
        qutil.LOGGER.info("sync'd all datasources and clients")
       
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
        
        qutil.LOGGER.info("entering feed loop on {addr}".format(addr=self.data_address))
        
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
        qutil.LOGGER.info("received {n} messages, sent {m} messages".format(n=self.data_buffer.received_count, 
                                                                            m=self.data_buffer.sent_count))
        self.data_socket.close()
        self.feed_socket.close()
        self.context.term()
        
            



        

        
            
