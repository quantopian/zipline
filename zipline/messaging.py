"""
Commonly used messaging components.
"""
import json
import uuid
import datetime
import zipline.util as qutil

class Component(object):
    
    def __init__(self):
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
        self.zmq            = None
        self.context        = None
        self.addresses      = None
        self.out_socket     = None
        self.gevent_needed  = False
        
    def get_id(self):
        NotImplemented
        
    def open(self):
        NotImplemented
        
    def do_work(self):
        NotImplemented
         
    def run(self):
        try:
            #TODO: can't initialize these values in the __init__?
            self.done       = False
            self.sockets    = []
            if self.gevent_needed:
                qutil.LOGGER.info("Loading gevent specific zmq for {id}".format(id=self.get_id()))
                import gevent_zeromq
                self.zmq = gevent_zeromq.zmq
            else:
                import zmq
                self.zmq = zmq
            self.context = self.zmq.Context()
            self.open()
            self.setup_sync()
            self.loop()
            #close all the sockets
            for sock in self.sockets:
                sock.close()
        except Exception as e:
            qutil.LOGGER.exception("Unexpected error in run for {id}.".format(id=self.get_id()))
            raise e
        finally:
            if(self.context != None):
                self.context.destroy()
                
    def loop(self):
        while not self.done:
            self.confirm()
            self.do_work()
                  
    def signal_done(self):
        #notify down stream components that we're done
        if(self.out_socket != None):
            self.out_socket.send("DONE")
        #notify host we're done
        self.sync_socket.send(self.get_id() + ":DONE")
        self.receive_sync_ack()
        #notify internal work look that we're done
        self.done = True
    
    def is_done_message(self, message):
        return message == "DONE"
        
    def confirm(self):  
        # send a synchronization request to the host
        self.sync_socket.send(self.get_id() + ":RUN")
        self.receive_sync_ack()
        
    def receive_sync_ack(self):
        # wait for synchronization reply from the host
        socks = dict(self.sync_poller.poll(2000)) #timeout after 2 seconds.
        if self.sync_socket in socks and socks[self.sync_socket] == self.zmq.POLLIN:
            message = self.sync_socket.recv()
        else:
            raise Exception("Sync ack timed out on response for {id}".format(id=self.get_id()))
            
            
    def bind_data(self):
        return self.bind_pull_socket(self.addresses['data_address'])
        
    def connect_data(self):
        return self.connect_push_socket(self.addresses['data_address'])
        
    def bind_feed(self):
        return self.bind_pub_socket(self.addresses['feed_address'])
    
    def connect_feed(self):
        return self.connect_sub_socket(self.addresses['feed_address'])
        
    def bind_merge(self):
        return self.bind_pull_socket(self.addresses['merge_address'])
        
    def connect_merge(self):
        return self.connect_push_socket(self.addresses['merge_address'])
        
    def bind_result(self):
        return self.bind_pub_socket(self.addresses['result_address'])

    def connect_result(self):
        return self.connect_sub_socket(self.addresses['result_address'])
            
    def bind_pull_socket(self, addr):
        pull_socket = self.context.socket(self.zmq.PULL)
        pull_socket.bind(addr)
        poller = self.zmq.Poller()
        poller.register(pull_socket, self.zmq.POLLIN)
        self.sockets.append(pull_socket)
        return pull_socket, poller
        
    def connect_push_socket(self, addr):
        push_socket = self.context.socket(self.zmq.PUSH)
        push_socket.connect(addr)
        #push_socket.setsockopt(self.zmq.LINGER,0)
        self.sockets.append(push_socket)
        self.out_socket = push_socket
        return push_socket
    
    def bind_pub_socket(self, addr):
        pub_socket = self.context.socket(self.zmq.PUB)
        pub_socket.bind(addr)
        #pub_socket.setsockopt(self.zmq.LINGER,0)
        self.out_socket = pub_socket
        return pub_socket
        
    def connect_sub_socket(self, addr):
        sub_socket = self.context.socket(self.zmq.SUB)
        sub_socket.connect(addr)
        sub_socket.setsockopt(self.zmq.SUBSCRIBE,'')
        poller = self.zmq.Poller()
        poller.register(sub_socket, self.zmq.POLLIN)
        self.sockets.append(sub_socket)
        return sub_socket, poller
            
    def setup_sync(self):
        qutil.LOGGER.debug("Connecting sync client for {id}".format(id=self.get_id()))
        self.sync_socket = self.context.socket(self.zmq.REQ)
        self.sync_socket.connect(self.addresses['sync_address'])
        #self.sync_socket.setsockopt(self.zmq.LINGER,0)
        self.sync_poller = self.zmq.Poller()
        self.sync_poller.register(self.sync_socket, self.zmq.POLLIN)
        self.sockets.append(self.sync_socket)
        
class ComponentHost(Component):
    """Component that can launch multiple sub-components, synchronize their start, and then wait for all
    components to be finished."""
    def __init__(self, addresses, gevent_needed=False):
        Component.__init__(self)
        self.addresses = addresses
        #workaround for defect in threaded use of strptime: http://bugs.python.org/issue11108
        qutil.parse_date("2012/02/13-10:04:28.114")
        self.components     = {}
        self.sync_register  = {}
        self.timeout        = datetime.timedelta(seconds=5)
        self.feed           = ParallelBuffer()
        self.merge          = MergedParallelBuffer()
        self.passthrough    = PassthroughTransform()
        self.gevent_needed  = gevent_needed
        
        #register the feed and the merge
        self.register_components([self.feed, self.merge, self.passthrough])
    
    def register_components(self, component_list):
        for component in component_list:
            component.gevent_needed = self.gevent_needed
            component.addresses = self.addresses
            self.components[component.get_id()] = component
            self.sync_register[component.get_id()] = datetime.datetime.utcnow()
            if(isinstance(component, DataSource)):
                self.feed.add_source(component.get_id())
            if(isinstance(component, BaseTransform)):
                self.merge.add_source(component.get_id())
    
    def unregister_component(self, component_id):
        del(self.components[component_id])
        del(self.sync_register[component_id])
    
    def setup_sync(self):
        """Start the sync server."""
        qutil.LOGGER.debug("Connecting sync server.")
        self.sync_socket = self.context.socket(self.zmq.REP)
        self.sync_socket.bind(self.addresses['sync_address'])
        self.poller = self.zmq.Poller()
        self.poller.register(self.sync_socket, self.zmq.POLLIN)
        self.sockets.append(self.sync_socket)
    
    def open(self):
        for component in self.components.values():
            self.launch_component(component)
    
    def is_timed_out(self):
        cur_time = datetime.datetime.utcnow()
        if(len(self.components) == 0):
            qutil.LOGGER.info("Component register is empty.")
            return True
        for source, last_dt in self.sync_register.iteritems():
            if((cur_time - last_dt) > self.timeout):
                qutil.LOGGER.info("Time out for {source}. Current component registery: {reg}".format(source=source, reg=self.components))
                return True
        return False
        
    def loop(self):
        while not self.is_timed_out():
            # wait for synchronization request
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

            if self.sync_socket in socks and socks[self.sync_socket] == self.zmq.POLLIN:
                msg = self.sync_socket.recv()
                parts = msg.split(':')
                if(len(parts) < 2):
                    qutil.LOGGER.info("got bad confirm: {msg}".format(msg=msg))
                sync_id = parts[0]
                status  = parts[1]
                if(self.is_done_message(status)):
                    qutil.LOGGER.info("{id} is DONE".format(id=sync_id))
                    self.unregister_component(sync_id)
                else:        
                    self.sync_register[sync_id] = datetime.datetime.utcnow()
                #qutil.LOGGER.info("confirmed {id}".format(id=msg))
                # send synchronization reply
                self.sync_socket.send('ack', self.zmq.NOBLOCK)
                    
    def launch_component(self, component):
        NotImplemented
    
class ParallelBuffer(Component):
    """Connects to N PULL sockets, publishing all messages received to a PUB socket.
     Published messages are guaranteed to be in chronological order based on message property dt.
     Expects to be instantiated in one execution context (thread, process, etc) and run in another."""
     
    def __init__(self):
        Component.__init__(self)
        self.sent_count             = 0
        self.received_count         = 0
        self.draining               = False
        #data source component ID -> List of messages
        self.data_buffer            = {}
        self.ds_finished_counter    = 0
        
    
    def get_id(self):
        return "FEED"
    
    def add_source(self, source_id):
        self.data_buffer[source_id] = []
        
    def open(self):
        self.pull_socket, self.poller   = self.bind_data()
        self.feed_socket                = self.bind_feed() 

    def do_work(self):   
        # wait for synchronization reply from the host
        socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

        if self.pull_socket in socks and socks[self.pull_socket] == self.zmq.POLLIN:
            message = self.pull_socket.recv()
            if(self.is_done_message(message)):
                self.ds_finished_counter += 1
                if(len(self.data_buffer) == self.ds_finished_counter):
                     #drain any remaining messages in the buffer
                    self.drain()
                    self.signal_done()
            else:
                event = json.loads(message)
                self.append(event[u'id'], event)
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
            self.feed_socket.send(json.dumps(event), self.zmq.NOBLOCK)
            self.sent_count += 1   
    
    
class MergedParallelBuffer(ParallelBuffer):
    """
    Merges multiple streams of events into single messages.
    """
    
    def __init__(self):
        ParallelBuffer.__init__(self)
        
    def open(self):
        self.pull_socket, self.poller   = self.bind_merge()
        self.feed_socket                = self.bind_result()
    
    def next(self):
        """Get the next merged message from the feed buffer."""
        if(not(self.is_full() or self.draining)):
            return
        
        #get the raw event from the passthrough transform.
        result = self.data_buffer["PASSTHROUGH"].pop(0)['value']
        for source, events in self.data_buffer.iteritems():
            if(source == "PASSTHROUGH"):
                continue
            if(len(events) > 0):
                cur = events.pop(0)
                result[source] = cur['value']
        return result
        
    def get_id(self):
        return "MERGE"
        

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
        Component.__init__(self)
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
        socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.
        if self.feed_socket in socks and socks[self.feed_socket] == self.zmq.POLLIN:
            message = self.feed_socket.recv()
            if(self.is_done_message(message)):
                self.signal_done()
                return
            event = json.loads(message)
            cur_state = self.transform(event)
            #TODO: do we want to relay the datetime again? maybe drop this?
            #cur_state['dt'] = event['dt']
            cur_state['id'] = self.state['name']
            self.result_socket.send(json.dumps(cur_state), self.zmq.NOBLOCK)

    def transform(self, event):
        """ Must return the transformed value as a map with {name:"name of new transform", value: "value of new field"}
            Transforms run in parallel and results are merged into a single map, so transform names must be unique. 
            Best practice is to use the self.state object initialized from the transform configuration, and only set the
            transformed value:
                self.state['value'] = transformed_value
        """
        NotImplemented
        
class PassthroughTransform(BaseTransform):
    
    def __init__(self):
        BaseTransform.__init__(self, "PASSTHROUGH")

    def transform(self, event):    
        return {'value':event}
        
class DataSource(Component):
    """
    Baseclass for data sources. Subclass and implement send_all - usually this 
    means looping through all records in a store, converting to a dict, and
    calling send(map).
    """
    def __init__(self, source_id):
        Component.__init__(self)
        self.id                     = source_id
        self.cur_event              = None

    def get_id(self):
        return self.id

    def open(self):    
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_socket = self.connect_data()

    def send(self, event):
        """
            event is expected to be a dict
            sets id and type properties in the dict
            sends to the data_socket.
        """
        event['id'] = self.id             
        event['type'] = self.get_type()
        self.data_socket.send(json.dumps(event))

    def get_type(self):
        raise NotImplemented

    