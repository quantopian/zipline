import json
import datetime

from zmq.core.poll import select

import zipline.messaging as qmsg
import zipline.util as qutil
import zipline.protocol as zp

class TradeSimulationClient(qmsg.Component):
    
    def __init__(self):
        qmsg.Component.__init__(self)
        self.received_count     = 0
        self.prev_dt            = None
        self.event_queue        = []
    
    @property
    def get_id(self):
        return "TRADING_CLIENT"
    
    def open(self):
        self.result_feed = self.connect_result()
        self.order_socket = self.connect_order()
    
    def do_work(self):
        #next feed event
        (rlist, wlist, xlist) = select([self.result_feed],
                                                  [],
                                                  [self.result_feed],
                                                  timeout=self.heartbeat_timeout/1000) #select timeout is in sec
        #
        #no more orders, should be an error condition
        if len(rlist) == 0 or len(xlist) > 0: 
            raise Exception("unexpected end of feed stream")
        message = rlist[0].recv()    
        if message == str(zp.CONTROL_PROTOCOL.DONE):
            self.signal_done()
            return #leave open orders hanging? client requests for orders?
            
        event = zp.MERGE_UNFRAME(message)
        self._handle_event(event)
    
    def connect_order(self):
        return self.connect_push_socket(self.addresses['order_address'])
    
    def _handle_event(self, event):
        self.event_queue.append(event)
        if event.ALGO_TIME <= event.dt:
            #event occurred in the present, send the queue to be processed
            self.handle_events(self.event_queue)
        self.order_socket.send(str(zp.CONTROL_PROTOCOL.DONE))
    
    def handle_events(self, event_queue):
        raise NotImplementedError    
    
    def order(self, sid, amount):
        self.order_socket.send(zp.ORDER_FRAME(sid, amount))
        
    

class TradeSimulator(qmsg.BaseTransform):
    
    def __init__(self, expected_orders): 
        qmsg.BaseTransform.__init__(self, "")
        self.open_orders            = {}
        self.algo_time              = None
        self.event_start            = None
        self.last_event_time        = None
        self.last_iteration_duration    = None
        self.expected_orders        = expected_orders
        self.order_count            = 0
        self.trade_count            = 0
        
    @property
    def get_id(self):
        return "ALGO_TIME"    
    
    def open(self):
        qmsg.BaseTransform.open(self)
        self.order_socket = self.bind_order()
        
    def bind_order(self):
        return self.bind_pull_socket(self.addresses['order_address'])

    def do_work(self):
        """
        Pulls one message from the event feed, then
        loops on orders until client sends DONE message.
        """
        
        #next feed event
        (rlist, wlist, xlist) = select([self.feed_socket],
                                                  [],
                                                  [self.feed_socket],
                                                  timeout=self.heartbeat_timeout/1000) #select timeout is in sec
        self.trade_count += 1
        #no more orders, should be an error condition
        if len(rlist) == 0 or len(xlist) > 0: 
            raise Exception("unexpected end of feed stream")
        message = rlist[0].recv()    
        if message == str(zp.CONTROL_PROTOCOL.DONE):
            self.signal_done()
            if(self.expected_orders > 0):
                assert self.expected_orders == self.order_count
            return #leave open orders hanging? client requests for orders?
            
        event = zp.FEED_UNFRAME(message)
        
        if self.last_iteration_duration != None:
            self.algo_time = self.last_event_time + self.last_iteration_duration
        else:
            self.algo_time = event.dt #base case, first event we're transporting.
        
        self.last_event_time = event.dt
        
        if self.algo_time < self.last_event_time:
            #compress time, move algo's clock to the time of this event
            self.algo_time = self.last_event_time
            
        #self.process_orders(event)

        #mark the start time for client's processing of this event.
        self.event_start = datetime.datetime.utcnow()
        self.result_socket.send(zp.TRANSFORM_FRAME('ALGO_TIME', self.algo_time), self.zmq.NOBLOCK)
        
            
        while True: #this loop should also poll for portfolio state req/rep
            (rlist, wlist, xlist) = select([self.order_socket],
                                           [],
                                           [self.order_socket],
                                           timeout=self.heartbeat_timeout/1000) #select timeout is in sec
            
            #no more orders, should this be an error condition?
            if len(rlist) == 0 or len(xlist) > 0: 
                continue
                
            order_msg = rlist[0].recv()
            if order_msg == str(zp.CONTROL_PROTOCOL.DONE):
                qutil.LOGGER.info("order loop finished")
                break

            sid, amount = zp.ORDER_UNFRAME(order_msg)
            self.add_open_order(sid, amount)
            
        #end of order processing loop
        self.last_iteration_duration = datetime.datetime.utcnow() - self.event_start
        
            
    def add_open_order(self, sid, amount):
        self.order_count = self.order_count + 1
        
    def process_orders(self, event):
        #TODO put real fill logic here, return a list of fills
        return [{'sid':133, 'amount':-100}]