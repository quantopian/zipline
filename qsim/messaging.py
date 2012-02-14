"""
Commonly used messaging components.
"""
import json
import uuid
import zmq

import qsim.util as qutil

class ParallelBuffer(object):
    """ holds several queues of events by key, allows retrieval in date order 
    or by merging"""
    def __init__(self, key_list):
        self.out_socket = None
        self.sent_count = 0
        self.received_count = 0
        self.draining = False
        self.data_buffer = {}
        for key in key_list:
            self.data_buffer[key] = []
    
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
            self.out_socket.send(json.dumps(event))
            self.sent_count += 1   
    
    
class MergedParallelBuffer(ParallelBuffer):
    """
    Merges multiple streams of events into single messages.
    """
    
    def __init__(self, keys):
        ParallelBuffer.__init__(self, keys)
        self.feed = []
        self.data_buffer["feed"] = self.feed
    
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
        
        
class Sync(object):
    """Sync instances register themselves with a Host. Once the Sync
    is created, the Host is guaranteed to block until confirm is called on this
    instance (and all others registered with the host). Components can use instances
    to delay the start of the host until initial setup is complete."""
    
    def __init__(self, host, name):
        self.host           = host
        self.sync_id        = "{name}-{id}".format(name=name, id=uuid.uuid1())
        self.context        = None
        self.sync_socket    = None
        self.poller         = None
        self.host.register_sync(self.sync_id)
        
        #qutil.LOGGER.info("registered {id} with host".format(id=self.sync_id))
        
    def open(self):
        self.context = zmq.Context()
        #synchronize with host
        self.sync_socket = self.context.socket(zmq.REQ)
        self.sync_socket.connect(self.host.sync_address)
        self.sync_socket.setsockopt(zmq.LINGER,0)
        self.poller = zmq.Poller()
        self.poller.register(self.sync_socket, zmq.POLLIN)
        
    def confirm(self):
        """Confirm readiness with the Host."""
        try:
            # send a synchronization request to the host
            self.sync_socket.send(self.sync_id + ":RUNNING", zmq.NOBLOCK)
            # wait for synchronization reply from the host
            socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

            if self.sync_socket in socks and socks[self.sync_socket] == zmq.POLLIN:
                message = self.sync_socket.recv()
            return True
        except:
            qutil.LOGGER.exception("exception in confirmation for {source}. Exiting.".format(source=self.sync_id))
            return False
        
    def close(self):
        try:
            self.sync_socket.send(self.sync_id + ":DONE", zmq.NOBLOCK) 
            self.sync_socket.close()
            self.context.term()
        except:
            pass #just don't want to error out on closing
        