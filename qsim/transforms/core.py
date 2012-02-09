"""
Transforms
==========

Transforms provide re-useable components for stream processing. All
Transforms expect to receive data events from qsim.simulator.feed.DataFeed
asynchronously via zeromq. Each transform is designed to run in independent 
process space, independently of all other transforms, to allow for parallel
computation. 

Each transform must maintain the state necessary to calculate the transform of 
each new feed events. 

To simplify the consumption of feed and transform data events, this module
also provides the TransformsMerge class. TransformsMerge initializes as set of 
transforms and subscribes to their output. Each feed event is then combined with
all the transforms of that event into a single new message.
  
"""
import zmq
import json
import qsim.util as qutil
import qsim.simulator.config as config


class BaseTransform(object):
    """Parent class for feed transforms. Subclass and override transform 
    method to create a new derived value from the combined feed."""
    
    def __init__(self, feed, config_dict, result_address):
        """
            feed_address    - zmq socket address, Transform will CONNECT a PULL socket and receive messages until "DONE" is received.
            result_address  - zmq socket address, Transform will CONNECT a PUSH socket and send messaes until feed_socket receives "DONE"
            sync_address    - zmq socket address, Transform will CONNECT a REQ socket and send/receive one message before entering feed loop
            config          - must be a dict that can be wrapped in a config.Config object with at least an entry for 'name':string value
            server          - if True, transform will bind to the result address (and act as a server), if False it will connect. The
                              the last transform in a series should be server=True so that clients can connect.
        """

        self.feed               = feed
        self.result_address     = result_address
        self.config             = config.Config(config_dict)
        self.state              = {}
        self.state['name']      = self.config.name
        self.sync               = qutil.FeedSync(feed, self.state['name'])
        self.received_count     = 0
        self.sent_count         = 0 
        self.context            = None
     
    def run(self):
        """Top level execution entry point for the transform::
        
                - connects to the feed socket to subscribe to events
                - connets to the result socket (most oftened bound by a TransformsMerge) to PUSH transforms
                - processes all messages received from feed, until DONE message received
                - pushes all transforms
                - sends DONE to result socket, closes all sockets and context"""
        self.open()
        self.process_all()
        self.close()
     
    def open(self): 
        """
        Establishes zmq connections.
        """
        self.context = zmq.Context()
        
        qutil.logger.info("starting {name} transform".
                        format(name = self.state['name']))
        #create the feed SUB. 
        self.feed_socket = self.context.socket(zmq.SUB)
        self.feed_socket.connect(self.feed.feed_address)
        self.feed_socket.setsockopt(zmq.SUBSCRIBE,'')
        
        #create the result PUSH
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(self.result_address)
        
    def process_all(self):
        """
        Loops until feed's DONE message is received:
            - receive an event from the data feed 
            - call transform (subclass' method) on event
            - send the transformed event
        """
        qutil.logger.info("starting {name} event loop".format(name = self.state['name']))
        self.sync.confirm()
        
        while True:
            message = self.feed_socket.recv()
            if(message == "DONE"):
                qutil.logger.info("{name} received the Done message from the feed".format(name=self.state['name']))
                self.result_socket.send("DONE")
                break;
            self.received_count += 1
            event = json.loads(message)
            cur_state = self.transform(event)
            cur_state['dt'] = event['dt']
            cur_state['name'] = self.state['name']
            self.result_socket.send(json.dumps(cur_state))
            self.sent_count += 1
    
    def close(self):
        """
        Shut down zmq resources.
        """
        qutil.logger.info("Transform {name} recieved {r} and sent {s}".format(name=self.state['name'], r=self.received_count, s=self.sent_count))
            
        self.feed_socket.close()
        self.result_socket.close()
        self.context.term()
        
    def transform(self, event):
        """ Must return the transformed value as a map with {name:"name of new transform", value: "value of new field"}
            Transforms run in parallel and results are merged into a single map, so transform names must be unique. 
            Best practice is to use the self.state object initialized from the transform configuration, and only set the
            transformed value:
                self.state['value'] = transformed_value
        """
        return {}
                    