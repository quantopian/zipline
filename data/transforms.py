import zmq
import logging
import datetime
import json
import config
import multiprocessing
from backtest import util

class Transform(object):
    """Parent class for feed transforms. Subclass to create a new derived value from the combined feed."""
    
    def __init__(self, feed_address, result_address, sync_address, config_dict):
        """
            feed_address    - zmq socket address, Transform will CONNECT a PULL socket and receive messages until "DONE" is received.
            result_address  - zmq socket address, Transform will CONNECT a PUSH socket and send messaes until feed_socket receives "DONE"
            sync_address    - zmq socket address, Transform will CONNECT a REQ socket and send/receive one message before entering feed loop
            config - must be a config.Config object with at least an entry for 'name':string value
        """
        self.logger = logging.getLogger()
        self.feed_address       = feed_address
        self.result_address     = result_address
        self.sync_address       = sync_address
        self.config             = config.Config(config_dict)
        self.name               = self.config.get_string('name')
        self.state              = {}
        self.state['name']      = self.name
        self.received_count     = 0
        self.sent_count         = 0 
        
    def run(self):
        self.context = zmq.Context()
        
        self.logger.info("starting {name} transform".format(name = self.name))
        #create the feed PULL. 
        self.feed_socket = self.context.socket(zmq.PULL)
        self.feed_socket.connect(self.feed_address)
        
        #create the result PUSH
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(self.result_address)
        
        self.logger.info("sync'ing feed from {name}".format(name = self.name))
        #synchronize with feed
        sync_socket = self.context.socket(zmq.REQ)
        sync_socket.connect(self.sync_address)
        # send a synchronization request to the feed
        sync_socket.send('')
        # wait for synchronization reply from the feed
        sync_socket.recv()
        sync_socket.close()
        
        self.logger.info("starting {name} event loop".format(name = self.name))
        
        while True:
            message = self.feed_socket.recv()
            self.received_count += 1
            if(message == "DONE"):
                break;
            event = json.loads(message)
            cur_state = self.update(event)
            self.result_socket.send(json.dumps(cur_state))
            self.sent_count += 1
        
        self.logger.info("Transform {name} recieved {r} and sent {s}".format(name=self.name, r=self.received_count, s=self.sent_count))
            
        self.feed_socket.close()
        self.result_socket.close()
        self.context.term()
        
    def update(self, event):
        return {}
        
class Merge(Transform):
    """ Merge data feed and array of transform feeds into a single result vector.
        PULL from feed
        PULL from child transforms
        PUSH to client
    
    """        

    def __init__(self, feed_address, result_address, sync_address, props):
        """
            config - must have an entry for 'transforms':array of dicts, which are convertedto configs.
        """
        Transform.__init__(self, feed_address, result_address, sync_address, props)
        self.transform_address  = "tcp://127.0.0.1:{port}".format(port=10104)
        self.transform_socket   = None
        self.create_transforms(self.config.transforms)
        
        
    def create_transforms(self, configs):
        self.transforms = {}
        for props in configs:
            class_name = props['class']
            if(class_name == 'MovingAverage'):
                mavg = MovingAverage(self.feed_address, self.transform_address, self.sync_address, props)
                self.transforms[mavg.name] = mavg
        
        for name, transform in self.transforms.iteritems():
            self.logger.info("starting {name}".format(name=name))
            proc = multiprocessing.Process(target=transform.run)
            proc.start()
            
    def get_socket(self):
        
        if(self.transform_socket == None):
            #create the feed PULL. 
            self.transform_socket = self.context.socket(zmq.PULL)
            self.transform_socket.connect(self.transform_address)
    
    def update(self, event):
        
        state = {}
        state['feed'] = event
        
        count = 0
        while count < len(transforms):
            message = get_socket().recv
            data = json.loads(message)
            state[data['name']] = data
            
        return state
            
            
        
class MovingAverage(Transform):
    
    def __init__(self, feed_address, result_address, sync_address, props): 
        Transform.__init__(self, feed_address, result_address, sync_address, props)
        self.events = []
        
        self.window = datetime.timedelta(days           = self.config.get_integer('days'), 
                                        seconds         = self.config.get_integer('seconds'), 
                                        microseconds    = self.config.get_integer('microseconds'), 
                                        milliseconds    = self.config.get_integer('milliseconds'),
                                        minutes         = self.config.get_integer('minutes'),
                                        hours           = self.config.get_integer('hours'),
                                        weeks           = self.config.get_integer('weeks'))
    
        
  
        
    def update(self, event):
        self.events.append(event)
        
        #filter the event list to the window length.
        self.events = [x for x in self.events if (util.parse_date(x['dt']) - util.parse_date(event['dt'])) <= self.window]
        
        if(len(self.events) == 0):
            return 0.0
            
        total = 0.0
        for event in self.events:
            total += event['price']
        
        self.average = total/len(self.events)
        
        self.state['avg'] = self.average