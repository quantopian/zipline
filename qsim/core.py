"""
Provides simulated data feed services. 
"""
import multiprocessing
import zmq
import json
import copy
import threading
import datetime

import qsim.util as qutil
import qsim.messaging as qmsg

class SimulatorBase(object):
    """
    Simulator coordinates the launch and communication of source, feed, transform, and merge components.
    """
    
    def __init__(self, sources, transforms, client, feed=None, merge=None):
        """
        """
        self.sources        = sources
        self.transforms     = transforms
        self.client         = client
        self.merge          = None
        self.feed           = None
        self.context        = None
        self.sync_context   = None
        self.sync_socket    = None      
        self.sync_register  = {}
        self.sync_address   = "tcp://127.0.0.1:{port}".format(port=10100)     
        self.data_address   = "tcp://127.0.0.1:{port}".format(port=10101)
        self.feed_address   = "tcp://127.0.0.1:{port}".format(port=10102)
        self.merge_address  = "tcp://127.0.0.1:{port}".format(port=10103)
        self.result_address = "tcp://127.0.0.1:{port}".format(port=10104)
        
        self.timeout        = datetime.timedelta(seconds=1)
        
        #workaround for defect in threaded use of strptime: http://bugs.python.org/issue11108
        qutil.parse_date("2012/02/13-10:04:28.114")
        
        if(feed == None):
            self.feed = DataFeed(self.sources.keys(), self.data_address, self.feed_address, qmsg.Sync(self,"DataFeed"))
        else:
            self.feed = feed
            
        if(merge == None):
            #connect merge to feed, set expected transforms
            self.merge = TransformsMerge(self.feed_address, 
                                         self.merge_address, 
                                         self.result_address, 
                                         qmsg.Sync(self,"TransformsMerge"), 
                                         self.transforms.keys())
        else:
            self.merge = merge 
            
    def simulate(self):
        #launch the feed
        self.launch_component("DataFeed", self.feed)
        
        #launch the data sources
        for name, data_source in self.sources.iteritems():
            data_source.data_address = self.data_address
            data_source.sync = qmsg.Sync(self, str(data_source.source_id))
            self.launch_component(name, data_source)
        qutil.LOGGER.info("datasources processes launched")
                            
        #connect all the transforms to the feed and merge, launch each         
        for name, transform in self.transforms.iteritems():
            transform.feed_address  = self.feed_address #connect transform to receive feed.
            transform.merge_address = self.merge_address #connect transform to push results to merge
            transform.sync  = qmsg.Sync(self, name) #synchronize the transform against this simulation.
            self.launch_component(name, transform) #start transforms            
        
        #launch merge
        self.launch_component("transforms merge", self.merge)
        qutil.LOGGER.info("transform processes launched")
        
        #connect client to merged feed
        self.client.address = self.result_address
        self.client.sync = qmsg.Sync(self,"Client")   
        client_proc = self.launch_component("client", self.client)           
        qutil.LOGGER.info("client process launched")
        
        self.sync_components()
        #client_proc.join() #wait for client to complete processing
    
    def launch_component(self, name, component):
        raise NotImplementedError
        
    def register_sync(self, sync_id):
        self.sync_register[sync_id] = datetime.datetime.utcnow()
    
    def unregister_sync(self, sync_id):
        del(self.sync_register[sync_id])
    
    def is_timed_out(self):
        cur_time = datetime.datetime.utcnow()
        if(len(self.sync_register) == 0):
            qutil.LOGGER.info("**********Simulator sync register is empty.")
            return True
        for source, last_dt in self.sync_register.iteritems():
            if((cur_time - last_dt) > self.timeout):
                qutil.LOGGER.info("Time out for {source}".format(source=source))
                return True
        return False

    def sync_components(self):
        # Socket to receive signals
        self.context = zmq.Context()
        qutil.LOGGER.info("waiting for all datasources and clients to be ready")
        self.sync_socket = self.context.socket(zmq.REP)
        self.sync_socket.bind(self.sync_address) 
        self.sync_socket.setsockopt(zmq.LINGER,0)
        self.poller = zmq.Poller()
        self.poller.register(self.sync_socket, zmq.POLLIN)

        while not self.is_timed_out():
            # wait for synchronization request
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

            if self.sync_socket in socks and socks[self.sync_socket] == zmq.POLLIN:
                try:
                    msg = self.sync_socket.recv()
                    parts = msg.split(':')
                    sync_id = parts[0]
                    status  = parts[1]
                    if(status == "DONE"):
                        self.unregister_sync(sync_id)
                    else:        
                        self.sync_register[sync_id] = datetime.datetime.utcnow()
                    #qutil.LOGGER.info("confirmed {id}".format(id=msg))
                    # send synchronization reply
                    self.sync_socket.send('ack')
                except:
                    continue

        self.sync_socket.close()
        qutil.LOGGER.info("simulator heartbeat stopped.")

class ThreadSimulator(SimulatorBase):
    
    def __init__(self, sources, transforms, client, feed=None, merge=None):
        SimulatorBase.__init__(self, sources, transforms, client, feed, merge)
        
    def launch_component(self, name, component):
        qutil.LOGGER.info("starting {name}".format(name=name))
        thread = threading.Thread(target=component.run)
        thread.start()
        return thread
    
class ProcessSimulator(SimulatorBase):        
    
    def __init__(self, sources, transforms, client, feed=None, merge=None):
        SimlulatorBase.__init__(self, sources, transforms, client, feed, merge)
    
    def launch_component(self, name, component):
        qutil.LOGGER.info("starting {name}".format(name=name))
        proc = multiprocessing.Process(target=component.run)
        proc.start()
        return proc
        
    
class DataFeed(object):
    
    def __init__(self, source_list, data_address, feed_address, sync):
        """
        :source_list: list of data source IDs
        """
        self.feed_address       = feed_address
        self.data_address       = data_address     
        self.data_buffer        = qmsg.ParallelBuffer(source_list)
        self.sync               = sync
        self.feed_socket        = None
        self.data_socket        = None
        self.context            = None
        self.poller             = None
        
    def open(self):
        # Prepare our context and sockets
        self.context = zmq.Context()
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        #see: http://zguide.zeromq.org/py:taskwork2
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind(self.data_address)
    
        #create the feed
        self.feed_socket = self.context.socket(zmq.PUB)
        self.feed_socket.bind(self.feed_address)
        self.feed_socket.setsockopt(zmq.LINGER,0)
        
        self.data_buffer.out_socket = self.feed_socket
        self.poller = zmq.Poller()
        self.poller.register(self.data_socket, zmq.POLLIN)
        
        self.sync.open()
        
    def close(self):
        self.data_socket.close()
        self.feed_socket.close()
        self.sync.close()
        self.context.term()
        
    def handle_all(self):   
        qutil.LOGGER.info("entering feed loop on {addr}".format(addr=self.data_address))
        ds_finished_counter = 0
        while self.sync.confirm():
            # wait for synchronization reply from the host
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

            if self.data_socket in socks and socks[self.data_socket] == zmq.POLLIN:
                message = self.data_socket.recv()
            
                event = json.loads(message)
                if(event["type"] == "DONE"):
                    ds_finished_counter += 1
                    if(len(self.data_buffer) == ds_finished_counter):
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

    
    def run(self):
        try:
            self.open()
            self.handle_all()
        except:
            qutil.LOGGER.exception("Exception in Feed, attempting to close.")
        finally:
            self.close()
        
            

class BaseTransform(object):
    """Parent class for feed transforms. Subclass and override transform 
    method to create a new derived value from the combined feed."""

    def __init__(self, name):
        """     
        """

        self.feed_address       = None
        self.merge_address      = None
        self.state              = {}
        self.state['name']      = name
        self.sync               = None
        self.received_count     = 0
        self.sent_count         = 0 
        self.context            = None
        self.feed_socket        = None
        self.result_socket      = None

    def run(self):
        """Top level execution entry point for the transform::

                - connects to the feed socket to subscribe to events
                - connets to the result socket (most oftened bound by a TransformsMerge) to PUSH transforms
                - processes all messages received from feed, until DONE message received
                - pushes all transforms
                - sends DONE to result socket, closes all sockets and context"""
        try:
            self.open()
            self.process_all()
        except:
            qutil.LOGGER.exception("Exception during transform processing, attempting to close merge.")
        finally:
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
        self.feed_socket.connect(self.feed_address)
        self.feed_socket.setsockopt(zmq.SUBSCRIBE,'')

        self.poller = zmq.Poller()
        self.poller.register(self.feed_socket, zmq.POLLIN)

        #create the result PUSH
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(self.merge_address)
        self.result_socket.setsockopt(zmq.LINGER,0)
        
        self.sync.open()

    def process_all(self):
        """
        Loops until feed's DONE message is received:
            - receive an event from the data feed 
            - call transform (subclass' method) on event
            - send the transformed event
        """
        qutil.LOGGER.info("starting {name} event loop".format(name = self.state['name']))

        while self.sync.confirm():
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.
            if self.feed_socket in socks and socks[self.feed_socket] == zmq.POLLIN:
                message = self.feed_socket.recv()
                if(message == "DONE"):
                    qutil.LOGGER.info("{name} received the Done message from the feed".format(name=self.state['name']))
                    self.result_socket.send("DONE")
                    break
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
        qutil.LOGGER.info("Transform {name} recieved {r} and sent {s}".format(
                                                                        name=self.state['name'], 
                                                                        r=self.received_count, 
                                                                        s=self.sent_count))

        self.feed_socket.close()
        self.result_socket.close()
        self.sync.close()
        self.context.term()

    def transform(self, event):
        """ Must return the transformed value as a map with {name:"name of new transform", value: "value of new field"}
            Transforms run in parallel and results are merged into a single map, so transform names must be unique. 
            Best practice is to use the self.state object initialized from the transform configuration, and only set the
            transformed value:
                self.state['value'] = transformed_value
        """
        return {}

class TransformsMerge(object):
    """ Merge data feed and array of transform feeds into a single result vector.
        PULL from feed
        PULL from child transforms
        PUSH merged message to client

    """        

    def __init__(self, feed_address, transform_address, result_address, sync, transform_list):
        """
        """
        self.sync               = sync
        self.feed_address       = feed_address
        self.transform_address  = transform_address
        self.result_address     = result_address
        buffer_list             = copy.copy(transform_list)
        buffer_list.append("feed") #for the raw feed
        self.data_buffer        = qmsg.MergedParallelBuffer(buffer_list)
        self.feed_socket        = None
        self.result_socket      = None
        self.poller             = None
        self.context            = None
        self.transform_socket   = None

    def run(self):
        """"""
        try:
            self.open()
            self.process_all()
        except:
            qutil.LOGGER.exception("Exception during merge processing, attempting to close merge.")
        finally:
            self.close()

    def open(self):
        """Establish zmq context, feed socket, result socket for client, and transform 
        socket to receive transformed events. Create and launch transforms. Will confirm 
        ready with the DataFeed at the conclusion."""
        self.context = zmq.Context()

        qutil.LOGGER.info("starting transforms merge")
        #create the feed SUB. 
        self.feed_socket = self.context.socket(zmq.SUB)
        self.feed_socket.connect(self.feed_address)
        self.feed_socket.setsockopt(zmq.SUBSCRIBE,'')

        #create the result PUSH
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.bind(self.result_address)
        self.result_socket.setsockopt(zmq.LINGER,0)

        #create the transform PULL. 
        self.transform_socket = self.context.socket(zmq.PULL)
        self.transform_socket.bind(self.transform_address)
        self.data_buffer.out_socket = self.result_socket

        # Initialize poll set
        self.poller = zmq.Poller()
        self.poller.register(self.feed_socket, zmq.POLLIN)
        self.poller.register(self.transform_socket, zmq.POLLIN)

        self.sync.open()

    def close(self):
        """
        Close all zmq sockets and context.
        """
        self.transform_socket.close()
        self.feed_socket.close()
        self.result_socket.close()
        self.context.term()

    def process_all(self):
        """
        Uses a Poller to receive messages from all transforms and the feed.
        All transforms corresponding to the same event are merged with each other
        and the original feed event into a single message. That message is then
        sent to the result socket.
        """
        done_count = 0
        while self.sync.confirm():
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

            if self.feed_socket in socks and socks[self.feed_socket] == zmq.POLLIN:
                message = self.feed_socket.recv()
                if(message == "DONE"):
                    qutil.LOGGER.info("finished receiving feed to merge")
                    done_count += 1
                else:
                    event = json.loads(message)
                    self.data_buffer.append("feed", event)

            if self.transform_socket in socks and socks[self.transform_socket] == zmq.POLLIN:
                t_message = self.transform_socket.recv()
                if(t_message == "DONE"):
                    qutil.LOGGER.info("finished receiving a transform to merge")
                    done_count += 1
                else:
                    t_event = json.loads(t_message)
                    self.data_buffer.append(t_event['name'], t_event)

            if(done_count >= len(self.data_buffer)):
                break #done!

            self.data_buffer.send_next()

        qutil.LOGGER.info("about to drain {n} messages from merger's buffer".format(n=self.data_buffer.pending_messages()))

        #drain any remaining messages in the buffer
        self.data_buffer.drain()

        #signal to client that we're done
        self.result_socket.send("DONE")






