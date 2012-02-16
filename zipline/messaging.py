"""
Commonly used messaging components.
"""
import json
import uuid
from gevent_zeromq import zmq
import zipline.util as qutil

class Component(object):
    
    def __init__(self, addresses):
        """
        :addresses: a dict of name_string -> zmq port address strings. Must have the following entries::
        
            - sync_address: socket address used for synchronizing the start of all workers, heartbeating, and exit notification
                            will be used in REP/REQ sockets. Bind is always on the REP side.
            - data_address: socket address used for data sources to stream their records. 
                            will be used in PUSH/PULL sockets between data sources and a ParallelBuffer (aka the Feed). Bind
                            will always be on the PULL side (we always have N producers and 1 consumer)
            - feed_address: socket address used to publish consolidated feed from serialization of data sources
                            will be used in PUB/SUB sockets between Feed and Transforms. Bind is always on the PUB side.
            - merge_address: socket address used to publish transformed values.
                            will be used in PUSH/PULL from many transforms to one MergedParallelBuffer (aka the Merge). Bind
                            will always be on the PULL side (we always have N producers and 1 consumer)
            - result_address: socket address used to publish merged data source feed and transforms to clients
                            will be used in PUB/SUB from one Merge to one or many clients. Bind is always on the PUB side.
        
        Bind/Connect methods will return the correct socket type for each address. Any sockets on which recv is expected to be called
        will also return a Poller.
        
        """
        self.context    = zmq.Context()
        self.addresses  = addresses
        self.sockets    = []
      
    def get_id(self):
        raise NotImplemented
        
    def open(self):
        raise NotImplemented
        
    def do_work(self):
        raise NotImplemented
          
    def run(self):
        try:
            self.open()
            self.connect_sync()
            while self.confirm():
                self.do_work()
            #notify host we're done
            self.sync_socket.send(self.sync_id + ":DONE") 
            #close all the sockets
            for sock in self.sockets:
                sock.close()
        finally:
            self.context.destroy()

    def confirm(self):
        try:
            # send a synchronization request to the host
            self.sync_socket.send(self.sync_id + ":RUNNING")
            # wait for synchronization reply from the host
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

            if self.sync_socket in socks and socks[self.sync_socket] == zmq.POLLIN:
                message = self.sync_socket.recv()

            return True
        except:
            qutil.LOGGER.exception("exception in confirmation for {source}. Exiting.".format(source=self.sync_id))
            return False
            
    def bind_data(self):
        return self.bind_pull_socket(self, self.addresses['data_address'])
        
    def connect_data(self):
        return self.connect_push_socket(self, self.addresses['data_address'])
        
    def bind_feed(self):
        return self.bind_pub_socket(self, self.addresses['feed_address'])
    
    def connect_feed(self):
        return self.bind_sub_socket(self, self.addresses['feed_address'])
        
    def bind_merge(self):
        return self.bind_pull_socket(self, self.addresses['merge_address'])
        
    def connect_merge(self):
        return self.connect_push_socket(self, self.addresses['merge_address'])
        
    def bind_result(self):
        return self.bind_pub_socket(self, self.addresses['result_address'])

    def connect_result(self):
        return self.bind_sub_socket(self, self.addresses['result_address'])
            
    def bind_pull_socket(self, addr):
        pull_socket = self.context.socket(zmq.PULL)
        pull_socket.bind(addr)
        poller = zmq.Poller()
        poller.register(self.pull_socket, zmq.POLLIN)
        self.sockets.append(pull_socket)
        return pull_socket, poller
        
    def connect_push_socket(self, addr):
        push_socket = self.context.socket(zmq.PUSH)
        push_socket.connect(self.merge_address)
        #push_socket.setsockopt(zmq.LINGER,0)
        self.sockets.append(push_socket)
        return push_socket
    
    def bind_pub_socket(self, addr):
        pub_socket = self.context.socket(zmq.PUB)
        pub_socket.bind(self.pub_address)
        #pub_socket.setsockopt(zmq.LINGER,0)
        poller = zmq.Poller()
        poller.register(self.pub_socket, zmq.POLLIN)
        self.sockets.append(pub_socket)
        return pub_socket, poller
        
    def connect_sub_socket(self, addr):
        sub_socket = self.context.socket(zmq.SUB)
        sub_socket.connect(self.feed_address)
        sub_socket.setsockopt(zmq.SUBSCRIBE,'')
        self.sockets.append(sub_socket)
        return sub_socket
        
    def bind_sync(self):
        sync_socket = self.context.socket(zmq.REP)
        sync_socket.bind(self.addresses['sync_address'])
        poller = zmq.Poller()
        poller.register(self.sync_socket, zmq.POLLIN)
        self.sockets.append(sync_socket)
        return sync_socket, poller
        
    def connect_sync(self):
        self.sync_socket = self.context.socket(zmq.REQ)
        self.sync_socket.connect(self.addresses['sync_address'])
        self.sync_socket.setsockopt(zmq.LINGER,0)
        self.poller = zmq.Poller()
        self.poller.register(self.sync_socket, zmq.POLLIN)
        self.sockets.append(sync_socket)
        
class ComponentHost(Component):
    """Component that can launch multiple sub-components, synchronize their start, and then wait for all
    components to be finished."""
    def __init__(self, addresses):
        Component.__init__(self, addresses)
        self.components = {}
        self.timeout    = datetime.timedelta(seconds=5)
    
    def register_components(self, component_list):
        for component in component_list:
            self.components[component.get_id()] = component
    
    def unregister_component(self, component_id):
        del(self.components[component_id])
    
    def open(self):
        self.sync_socket, self.poller = self.bind_sync()
        for component in self.components.values():
            self.launch_component(component)
    
    def is_timed_out(self):
        cur_time = datetime.datetime.utcnow()
        if(len(self.sync_register) == 0):
            qutil.LOGGER.info("Component register is empty.")
            return True
        for source, last_dt in self.sync_register.iteritems():
            if((cur_time - last_dt) > self.timeout):
                qutil.LOGGER.info("Time out for {source}. Current registery: {reg}".format(source=source, reg=self.sync_register))
                return True
        return False
        
    def run(self):
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
                        self.unregister_component(sync_id)
                    else:        
                        self.sync_register[sync_id] = datetime.datetime.utcnow()
                    #qutil.LOGGER.info("confirmed {id}".format(id=msg))
                    # send synchronization reply
                    self.sync_socket.send('ack', zmq.NOBLOCK)
                except:
                    qutil.LOGGER.exception("Exception in sync components loop")
                    
    def luanch_component(self, component):
        raise NotImplemented
    
class ParallelBuffer(Component):
    """Connects to N PULL sockets, publishing all messages received to a PUB socket.
     Published messages are guaranteed to be in chronological order based on message property dt.
     Expects to be instantiated in one execution context (thread, process, etc) and run in another."""
     
    def __init__(self):
        self.sent_count             = 0
        self.received_count         = 0
        self.draining               = False
        self.data_buffer            = {}
        self.ds_finished_counter    = 0
        
    
    def get_id(self):
        return "FEED"
    
    def add_source(self, source_id):
        self.data_buffer[source_id] = []
        
    def open(self):
        self.pull_socket, self.poller   = self.bind_data()
        self.feed_socket                 = self.bind_feed() 

    def do_work(self):   
        # wait for synchronization reply from the host
        socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

        if self.pull_socket in socks and socks[self.pull_socket] == zmq.POLLIN:
            message = self.pull_socket.recv()

            event = json.loads(message)
            if(event["type"] == "DONE"):
                ds_finished_counter += 1
                if(len(self.data_buffer) == ds_finished_counter):
                     #drain any remaining messages in the buffer
                    self.drain()
                    #send the DONE message
                    self.feed_socket.send("DONE", zmq.NOBLOCK)
                    qutil.LOGGER.info("received {n} messages, sent {m} messages".format(n=self.received_count, 
                                                                                        m=self.sent_count))
            else:
                self.append(event[u's'], event)
                self.send_next()

    def __len__(self):
        """buffer's length is same as internal map holding separate sorted arrays of events keyed by source id"""
        return len(self.data_buffer)        
    
    def append(self, source_id, value):
        """add an event to the buffer for the source specified by source_id"""
        self.data_buffer[source_id].append(value)
        self.received_count += 1
    
    def next(self):
        """Get the next message in chronological order"""
        if(not(self.is_full() or self.draining)):
            return
            
        cur = None
        earliest = None
        for events in self.data_buffer.values():
            if len(events) == 0:
                continue
            cur = events
            if(earliest == None) or (cur[0]['dt'] <= earliest[0]['dt']):
                earliest = cur
        
        if(earliest != None):
            return earliest.pop(0)
        
    def is_full(self):
        """indicates whether the buffer has messages in buffer for all un-DONE sources"""
        for events in self.data_buffer.values():
            if (len(events) == 0):
                return False
        return True
    
    def pending_messages(self):
        """returns the count of all events from all sources in the buffer"""
        total = 0
        for events in self.data_buffer.values():
            total += len(events)
        return total
        
    def drain(self):
        """send all messages in the buffer"""
        self.draining = True
        while(self.pending_messages() > 0):
            self.send_next()
            
    def send_next(self):
        """send the (chronologically) next message in the buffer."""
        if(not(self.is_full() or self.draining)):
            return
  
        event = self.next()
        if(event != None):
            self.feed_socket.send(json.dumps(event), zmq.NOBLOCK)
            self.sent_count += 1   
    
    
class MergedParallelBuffer(ParallelBuffer):
    """
    Merges multiple streams of events into single messages.
    """
    
    def __init__(self):
        ParallelBuffer.__init__(self)
    
    def next(self):
        """Get the next merged message from the feed buffer."""
        if(not(self.is_full() or self.draining)):
            return
        
        result = self.feed.pop(0)
        for source, events in self.data_buffer.iteritems():
            if(source == "feed"):
                continue
            if(len(events) > 0):
                cur = events.pop(0)
                result[source] = cur['value']
        return result
        

class BaseTransform(Component):
    """Top level execution entry point for the transform::

            - connects to the feed socket to subscribe to events
            - connets to the result socket (most oftened bound by a TransformsMerge) to PUSH transforms
            - processes all messages received from feed, until DONE message received
            - pushes all transforms
            - sends DONE to result socket, closes all sockets and context
            
    Parent class for feed transforms. Subclass and override transform 
    method to create a new derived value from the combined feed."""

    def __init__(self, name):
        self.state              = {}
        self.state['name']      = name

    def get_id(self):
        return self.state['name']

    def open(self): 
        """
        Establishes zmq connections.
        """    
        #create the feed. 
        self.feed_socket, self.poller = self.connect_feed()

        #create the result PUSH
        self.result_socket = self.connect_merge()
        

    def do_work(self):
        """
        Loops until feed's DONE message is received:
            - receive an event from the data feed 
            - call transform (subclass' method) on event
            - send the transformed event
        """
        qutil.LOGGER.info("starting {name} event loop".format(name = self.state['name']))
        socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.
        if self.feed_socket in socks and socks[self.feed_socket] == zmq.POLLIN:
            message = self.feed_socket.recv()
            if(message == "DONE"):
                qutil.LOGGER.info("{name} received the Done message from the feed".format(name=self.state['name']))
                self.result_socket.send("DONE", zmq.NOBLOCK)
                return
            self.received_count += 1
            event = json.loads(message)
            cur_state = self.transform(event)
            cur_state['dt'] = event['dt']
            cur_state['name'] = self.state['name']
            self.result_socket.send(json.dumps(cur_state), zmq.NOBLOCK)
            self.sent_count += 1

    def transform(self, event):
        """ Must return the transformed value as a map with {name:"name of new transform", value: "value of new field"}
            Transforms run in parallel and results are merged into a single map, so transform names must be unique. 
            Best practice is to use the self.state object initialized from the transform configuration, and only set the
            transformed value:
                self.state['value'] = transformed_value
        """
        return {'value'=event}
        