import json
import uuid
import zmq

import util as qutil

class ParallelBuffer(object):
    """ holds several queues of events by key, allows retrieval in date order or by merging"""
    def __init__(self, key_list):
        self.out_socket = None
        self.sent_count = 0
        self.received_count = 0
        self.draining = False
        self.data_buffer = {}
        for key in key_list:
            self.data_buffer[key] = []
    
    def __len__(self):
        return len(self.data_buffer)        
    
    def append(self, key, value):
        self.data_buffer[key].append(value)
        self.received_count += 1
    
    def next(self):
        if(not(self.is_full() or self.draining)):
            return
            
        cur = None
        earliest = None
        for source, events in self.data_buffer.iteritems():
            if len(events) == 0:
                continue
            cur = events
            if(earliest == None) or (cur[0]['dt'] <= earliest[0]['dt']):
                earliest = cur
        
        if(earliest != None):
            return earliest.pop(0)
        
    def is_full(self):
        for source, events in self.data_buffer.iteritems():
            if (len(events) == 0):
                return False
        return True
    
    def pending_messages(self):
        total = 0
        for source, events in self.data_buffer.iteritems():
            total += len(events)
        return total       
        
    def drain(self):
        self.draining = True
        while(self.pending_messages() > 0):
            self.send_next()
            
    def send_next(self):
        if(not(self.is_full() or self.draining)):
            return
            
        event = self.next()
        if(event != None):        
            self.out_socket.send(json.dumps(event))
            self.sent_count += 1   
    
    
class MergedParallelBuffer(ParallelBuffer):
    
    def __init__(self, keys):
        ParallelBuffer.__init__(self, keys)
        self.feed = []
        self.data_buffer["feed"] = self.feed
    
    def next(self):
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
        
        
class FeedSync(object):
    
    def __init__(self, feed, name):
        self.feed = feed
        self.id = "{name}-{id}".format(name=name, id=uuid.uuid1())
        self.feed.register_sync(self.id)
        #qutil.logger.info("registered {id} with feed".format(id=self.id))
        
    def confirm(self):
        context = zmq.Context()
        #synchronize with feed
        sync_socket = context.socket(zmq.REQ)
        sync_socket.connect(self.feed.sync_address)
        # send a synchronization request to the feed
        sync_socket.send(self.id)
        # wait for synchronization reply from the feed
        sync_socket.recv()
        sync_socket.close()
        context.term()
        qutil.logger.info("sync'd feed from {id}".format(id = self.id))
   