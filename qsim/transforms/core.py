"""
"""
import zmq
import json
import copy
import multiprocessing
import zmq

import qsim.util as qutil
import qsim.messaging as qmsg
import qsim.config as config

class BaseTransform(object):
    """Parent class for feed transforms. Subclass and override transform 
    method to create a new derived value from the combined feed."""
    
    def __init__(self, feed, config_dict, result_address):
        """
            :feed_address:    zmq socket address, Transform will CONNECT a PULL socket and receive messages until "DONE" is received.
            :result_address:  zmq socket address, Transform will CONNECT a PUSH socket and send messaes until feed_socket receives "DONE"
            :sync_address:    zmq socket address, Transform will CONNECT a REQ socket and send/receive one message before entering feed loop
            :config:          must be a dict that can be wrapped in a config.Config object with at least an entry for 'name':string value
            :server:          if True, transform will bind to the result address (and act as a server), if False it will connect. The
                              the last transform in a series should be server=True so that clients can connect.
        """

        self.feed               = feed
        self.result_address     = result_address
        self.config             = config.Config(config_dict)
        self.state              = {}
        self.state['name']      = self.config.name
        self.sync               = qmsg.FeedSync(feed, self.state['name'])
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
        
        qutil.LOGGER.info("starting {name} transform".
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
        qutil.LOGGER.info("starting {name} event loop".format(name = self.state['name']))
        self.sync.confirm()
        
        while True:
            message = self.feed_socket.recv()
            if(message == "DONE"):
                qutil.LOGGER.info("{name} received the Done message from the feed".format(name=self.state['name']))
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
        qutil.LOGGER.info("Transform {name} recieved {r} and sent {s}".format(name=self.state['name'], r=self.received_count, s=self.sent_count))
            
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
              
class MergedTransformsFeed(BaseTransform):
    """ Merge data feed and array of transform feeds into a single result vector.
        PULL from feed
        PULL from child transforms
        PUSH merged message to client

    """        

    def __init__(self, feed, props):
        """
            :props: - must have an entry for 'transforms':array of dicts, which are 
            converted to configs.
        """
        BaseTransform.__init__(self, feed, props, "tcp://127.0.0.1:20202")
        self.transform_address  = "tcp://127.0.0.1:{port}".format(port=10104)
        self.transform_socket   = None
        self.create_transforms(self.config.transforms)


    def create_transforms(self, configs):
        """
        :configs: an array of config objects with a class property. Each type of transform needs 
        Create transforms based on configs, set each transform's result address to
        this object's transform_address, so that all transformed events will be delivered
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

    def open(self):
        """Establish zmq context, feed socket, result socket for client, and transform 
        socket to receive transformed events. Create and launch transforms. Will confirm 
        ready with the DataFeed at the conclusion."""
        self.context = zmq.Context()

        qutil.LOGGER.info("starting {name} transform".format(name = self.state['name']))
        #create the feed SUB. 
        self.feed_socket = self.context.socket(zmq.SUB)
        self.feed_socket.connect(self.feed.feed_address)
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

        self.sync.confirm()

    def close(self):
        """
        Close all zmq sockets and context.
        """
        self.transform_socket.close()
        BaseTransform.close(self)

    def process_all(self):
        """
        Uses a Poller to receive messages from all transforms and the feed.
        All transforms corresponding to the same event are merged with each other
        and the original feed event into a single message. That message is then
        sent to the result socket.
        """
        done_count = 0
        while True:
            socks = dict(self.poller.poll())

            if self.feed_socket in socks and socks[self.feed_socket] == zmq.POLLIN:
                message = self.feed_socket.recv()
                if(message == "DONE"):
                    qutil.LOGGER.info("finished receiving feed to merge")
                    done_count += 1
                else:
                    self.received_count += 1
                    event = json.loads(message)
                    self.data_buffer.append("feed",event)

            if self.transform_socket in socks and socks[self.transform_socket] == zmq.POLLIN:
                t_message = self.transform_socket.recv()
                if(t_message == "DONE"):
                    qutil.LOGGER.info("finished receiving a transform to merge")
                    done_count += 1
                else:
                    self.received_count += 1
                    t_event = json.loads(t_message)
                    self.data_buffer.append(t_event['name'], t_event)

            if(done_count >= len(self.data_buffer)):
                break #done!

            self.data_buffer.send_next()

        qutil.LOGGER.info("Transform {name} received {r} and sent {s}".format(name=self.state['name'], r=self.data_buffer.received_count, s=self.data_buffer.sent_count))  
        qutil.LOGGER.info("about to drain {n} messages from merger's buffer".format(n=self.data_buffer.pending_messages()))

        #drain any remaining messages in the buffer
        self.data_buffer.drain()

        #signal to client that we're done
        self.result_socket.send("DONE")
        qutil.LOGGER.info("Transform {name} received {r} and sent {s}".format(name=self.state['name'], r=self.data_buffer.received_count, s=self.data_buffer.sent_count))  


            