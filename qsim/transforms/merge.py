import copy
import multiprocessing
import zmq

import technical as ta
from core import BaseTransform
import qsim.util as qutil
import qsim.messaging as qmsg
        
class MergedTransformsFeed(BaseTransform):
    """ Merge data feed and array of transform feeds into a single result vector.
        PULL from feed
        PULL from child transforms
        PUSH merged message to client
    
    """        

    def __init__(self, feed, props):
        """
            config - must have an entry for 'transforms':array of dicts, which are 
            convertedto configs.
        """
        BaseTransform.__init__(self, feed, props, "tcp://127.0.0.1:20202")
        self.transform_address  = "tcp://127.0.0.1:{port}".format(port=10104)
        self.transform_socket   = None
        self.create_transforms(self.config.transforms)
        
        
    def create_transforms(self, configs):
        """
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
        
        for name, transform in self.transforms.iteritems():
            qutil.LOGGER.info("starting {name}".format(name=name))
            proc = multiprocessing.Process(target=transform.run)
            proc.start()
            
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
                
