"""
Provides simulated data feed services. 
"""

import qsim.sources as sources
import qsim.util as qutil
import qsim.messaging as qmsg
import qsim.transforms.technical as ta

import zmq
import time
import logging
import json

class Simulator(object):
    """
    Simulator translates configuration data into running source, feed, transform, and merge components.
    """
    
    def __init__(self, config):
        """
        :config: a qsim.config.Config object that contains configuration information for all datasources, all transforms, and all 
        client algorithms that simulator should create.
        """
        self.config = config
        self.data_workers = {}
        
        
    def launch(self):
        """
        Create all components specified in config...
        """
        self.feed = DataFeed(self.config.sources.keys())
        self.start_data_sources(self.config.sources)
        self.create_transforms(self.config.transforms)
        
    def start_data_sources(self, configs):
        """
        :configs: array of dicts with properties
        """
        for name, info in configs.iteritems():
            if(info['class'] == "EquityMinuteTrades"):
                emt = EquityMinuteTrades(info['sid'], self.feed, name)
                self.data_workers[name] = emt
            elif(info['class'] == "RandomEquityTrades"):
                ret = sources.RandomEquityTrades(info['sid'], self.feed, name, info['count'])
                self.data_workers[name] = ret
               
            qutil.LOGGER.info("starting {id}".format(id=source_id))
            self.data_workers[name].start()
            
        qutil.LOGGER.info("datasources processes launched")
        
    def start_transforms(self, configs):
        """
        :configs: Must be an array of dicts holding properties needed for each transform. See the classes in :py:module:`qsim.transforms`
        Create transforms based on configs, set each transform's result address to
        transforms_address. Each transform will connect to transforms_address that all transformed events will be PUSH'd
        to this object.
        """
        self.transforms = {}
        for props in configs:
            class_name = props['class']
            if(class_name == 'MovingAverage'):
                mavg = ta.MovingAverage(self.feed, props, self.transform_address)
                self.transforms[mavg.config.name] = mavg

        keys = copy.copy(self.transforms.keys())
        keys.append("feed") #for the raw feed
        self.data_buffer = qmsg.MergedParallelBuffer(keys) 

        self.buffers = {}
        for name, transform in self.transforms.iteritems():
            self.buffers[name] = []
            qutil.LOGGER.info("starting {name}".format(name=name))
            proc = multiprocessing.Process(target=transform.run)
            proc.start()


class DataFeed(object):
    
    def __init__(self, source_list):
        """
        :source_list: list of source IDs
        """
        
        self.data_address = "tcp://127.0.0.1:{port}".format(port=10101)
        self.sync_address = "tcp://127.0.0.1:{port}".format(port=10102)
        self.feed_address = "tcp://127.0.0.1:{port}".format(port=10103)
        
        self.client_register = {}
                
        self.data_buffer = qmsg.ParallelBuffer(source_id_list)
        
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
        
            



        

        
            
