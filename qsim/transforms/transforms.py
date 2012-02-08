import zmq
import logging
import datetime
import json
import copy
import multiprocessing
import qsim.simulator.backtest.util as qutil
import qsim.simulator.config as config
class Transform(object):
    """Parent class for feed transforms. Subclass to create a new derived value from the combined feed."""
    
    def __init__(self, feed, config_dict, result_address):
        """
            feed_address    - zmq socket address, Transform will CONNECT a PULL socket and receive messages until "DONE" is received.
            result_address  - zmq socket address, Transform will CONNECT a PUSH socket and send messaes until feed_socket receives "DONE"
            sync_address    - zmq socket address, Transform will CONNECT a REQ socket and send/receive one message before entering feed loop
            config          - must be a dict that can be wrapped in a config.Config object with at least an entry for 'name':string value
            server          - if True, transform will bind to the result address (and act as a server), if False it will connect. The
                              the last transform in a series should be server=True so that clients can connect.
        """
        self.logger             = qutil.logger
        self.feed               = feed
        self.feed_address       = feed.feed_address
        self.result_address     = result_address
        self.config             = config.Config(config_dict)
        self.name               = self.config.get_string('name')
        self.sync               = FeedSync(feed, self.name)
        self.state              = {}
        self.state['name']      = self.name
        self.received_count     = 0
        self.sent_count         = 0 
     
    def run(self):
        self.open()
        self.process_all()
        self.close()
     
    def open(self): 
        self.context = zmq.Context()
        
        self.logger.info("starting {name} transform".format(name = self.name))
        #create the feed SUB. 
        self.feed_socket = self.context.socket(zmq.SUB)
        self.feed_socket.connect(self.feed_address)
        self.feed_socket.setsockopt(zmq.SUBSCRIBE,'')
        
        #create the result PUSH
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(self.result_address)
        
    def process_all(self):
        self.logger.info("starting {name} event loop".format(name = self.name))
        self.sync.confirm()
        
        while True:
            message = self.feed_socket.recv()
            if(message == "DONE"):
                self.logger.info("{name} received the Done message from the feed".format(name=self.name))
                self.result_socket.send("DONE")
                break;
            self.received_count += 1
            event = json.loads(message)
            cur_state = self.transform(event)
            cur_state['dt'] = event['dt']
            cur_state['name'] = self.name
            self.result_socket.send(json.dumps(cur_state))
            self.sent_count += 1
    
    def close(self):
        self.logger.info("Transform {name} recieved {r} and sent {s}".format(name=self.name, r=self.received_count, s=self.sent_count))
            
        self.feed_socket.close()
        self.result_socket.close()
        self.context.term()
        
    def transform(self, event):
        return {}
                    
        
class MovingAverage(Transform):
    
    def __init__(self, feed, props, result_address): 
        Transform.__init__(self, feed, props, result_address)
        self.events = []
        
        self.window = datetime.timedelta(days           = self.config.get_integer('days'), 
                                        seconds         = self.config.get_integer('seconds'), 
                                        microseconds    = self.config.get_integer('microseconds'), 
                                        milliseconds    = self.config.get_integer('milliseconds'),
                                        minutes         = self.config.get_integer('minutes'),
                                        hours           = self.config.get_integer('hours'),
                                        weeks           = self.config.get_integer('weeks'))
    
        
  
        
    def transform(self, event):
        self.events.append(event)
        
        #filter the event list to the window length.
        self.events = [x for x in self.events if (qutil.parse_date(x['dt']) - qutil.parse_date(event['dt'])) <= self.window]
        
        if(len(self.events) == 0):
            return 0.0
            
        total = 0.0
        for event in self.events:
            total += event['price']
        
        self.average = total/len(self.events)
        
        self.state['value'] = self.average
        
        return self.state
        
        
class MergedTransformsFeed(Transform):
    """ Merge data feed and array of transform feeds into a single result vector.
        PULL from feed
        PULL from child transforms
        PUSH merged message to client
    
    """        

    def __init__(self, feed, props):
        """
            config - must have an entry for 'transforms':array of dicts, which are convertedto configs.
        """
        Transform.__init__(self, feed, props, "tcp://127.0.0.1:20202")
        self.transform_address  = "tcp://127.0.0.1:{port}".format(port=10104)
        self.transform_socket   = None
        self.create_transforms(self.config.transforms)
        
        
    def create_transforms(self, configs):
        self.transforms = {}
        for props in configs:
            class_name = props['class']
            if(class_name == 'MovingAverage'):
                mavg = MovingAverage(self.feed, props, self.transform_address)
                self.transforms[mavg.name] = mavg
        
        keys = copy.copy(self.transforms.keys())
        keys.append("feed") #for the raw feed
        self.data_buffer = MergedParallelBuffer(keys) 
            
        self.buffers = {}
        for name, transform in self.transforms.iteritems():
            self.buffers[name] = []
            
    def open(self):
        self.context = zmq.Context()
        
        self.logger.info("starting {name} transform".format(name = self.name))
        #create the feed SUB. 
        self.feed_socket = self.context.socket(zmq.SUB)
        self.feed_socket.connect(self.feed_address)
        self.feed_socket.setsockopt(zmq.SUBSCRIBE,'')
        
        #create the result PUSH
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.bind(self.result_address)
        
        #create the transform PULL. 
        self.transform_socket = self.context.socket(zmq.PULL)
        self.transform_socket.bind(self.transform_address)
        self.data_buffer.out_socket = self.result_socket
        
        # Initialize poll set
        self.poller = zmq.Poller()
        self.poller.register(self.feed_socket, zmq.POLLIN)
        self.poller.register(self.transform_socket, zmq.POLLIN)
        
        for name, transform in self.transforms.iteritems():
            self.logger.info("starting {name}".format(name=name))
            proc = multiprocessing.Process(target=transform.run)
            proc.start()
            
        self.sync.confirm()
        
    def close(self):
        self.transform_socket.close()
        Transform.close(self)
        
    def process_all(self):
        
        done_count = 0
        while True:
            socks = dict(self.poller.poll())
            
            if self.feed_socket in socks and socks[self.feed_socket] == zmq.POLLIN:
                message = self.feed_socket.recv()
                if(message == "DONE"):
                    self.logger.info("finished receiving feed to merge")
                    done_count += 1
                else:
                    self.received_count += 1
                    event = json.loads(message)
                    self.data_buffer.append("feed",event)
                
            if self.transform_socket in socks and socks[self.transform_socket] == zmq.POLLIN:
                t_message = self.transform_socket.recv()
                if(t_message == "DONE"):
                    self.logger.info("finished receiving a transform to merge")
                    done_count += 1
                else:
                    self.received_count += 1
                    t_event = json.loads(t_message)
                    self.data_buffer.append(t_event['name'], t_event)
                
            if(done_count >= len(self.data_buffer)):
                break #done!
            
            self.data_buffer.send_next()
            
        self.logger.info("Transform {name} received {r} and sent {s}".format(name=self.name, r=self.data_buffer.received_count, s=self.data_buffer.sent_count))  
        self.logger.info("about to drain {n} messages from merger's buffer".format(n=self.data_buffer.pending_messages()))
        
        #drain any remaining messages in the buffer
        self.data_buffer.drain()
        
        #signal to client that we're done
        self.result_socket.send("DONE")
        self.logger.info("Transform {name} received {r} and sent {s}".format(name=self.name, r=self.data_buffer.received_count, s=self.data_buffer.sent_count))  
                
