import json
import datetime
import zipline.messaging as qmsg
import zipline.util as qutil

class TradeSimulationClient(qmsg.Component):
    
    def __init__(self):
        qmsg.Component.__init__(self)
        self.received_count     = 0
        self.prev_dt            = None
    
    def get_id(self):
        return "TRADING_CLIENT"
    
    def open(self):
        self.result_feed = self.connect_result()
        self.order_socket = self.connect_order()
    
    def do_work(self):
        #next feed event
        (rlist, wlist, xlist) = self.poller.selec([self.result_feed],
                                                  [],
                                                  [self.result_feed],
                                                  timeout=self.heartbeat_timeout/1000) #select timeout is in sec
        #
        #no more orders, should be an error condition
        if len(rlist) == 0 or len(xlist) > 0: 
            raise Exception("unexpected end of feed stream")
        message = rlist[0].recv()    
        if message == str(CONTROL_PROTOCOL.DONE):
            self.signal_done()
            return #leave open orders hanging? client requests for orders?
            
        event = json.loads(message)
        self._handle_event(event)
    
    def connect_order(self):
        return self.connect_push_socket(self.addresses['order_address'])
    
    def _handle_event(self, event):
        self.event_queue.append(event)
        if event['TRADE_SIM']['ALGO_TIME'] <= event['dt']:
            del(event['TRADE_SIM'])
            event['dt'] = qutil.parse_date(event['dt'])
            #event occurred in the present, send the queue to be processed
            self.handle_events(self.event_queue)
    
    def handle_events(self, event_queue):
        raise NotImplementedError    
    
    def order(self, sid, volume):
        order = {'sid':sid, 'volume':volume}
        self.order_feed.send(json.dumps(order))
    

class TradeSimulator(qmsg.BaseTransform):
    
    def __init__(self): 
        qmsq.BaseTransform.__init__(self, "")
        self.open_orders            = {}
        self.algo_time              = None
        self.event_start            = None
        self.last_event_time        = None
        self.last_iteration_duration    = None
    
    def get_id(self):
        return "TRADE_SIM"    
    
    def open():
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
        (rlist, wlist, xlist) = self.poller.selec([self.feed_socket],
                                                  [],
                                                  [self.feed_socket],
                                                  timeout=self.heartbeat_timeout/1000) #select timeout is in sec
        #
        #no more orders, should be an error condition
        if len(rlist) == 0 or len(xlist) > 0: 
            raise Exception("unexpected end of feed stream")
        message = rlist[0].recv()    
        if message == str(CONTROL_PROTOCOL.DONE):
            self.signal_done()
            return #leave open orders hanging? client requests for orders?
            
        event = json.loads(message)
        
        if self.last_iteration_duration != None:
            self.algo_time = self.last_event_time + self.last_iteration_duration
        
        self.last_event_time = qutil.parse_date(event['dt'])
        
        if self.algo_time < self.last_event_time:
            #compress time, move algo's clock to the time of this event
            self.algo_time = self.last_event_time
            
        fill  = self.process_orders(event)
        
        #TODO: decide what this transform should send downstream, maybe fills? effective algo time?
        self.state['value'] = {'FILL':fill,
                               'ALGO_TIME':qutil.format_date(self.algo_time)}
        
        #mark the start time for client's processing of this event.
        self.event_start = datetime.datetime.utcnow()
        self.result_socket.send(json.dumps(cur_state), self.zmq.NOBLOCK)
        
            
        while True: #this loop should also poll for portfolio state req/rep
            (rlist, wlist, xlist) = self.poller.selec([self.order_socket],
                                                      [],
                                                      [self.order_socket],
                                                      timeout=self.heartbeat_timeout/1000) #select timeout is in sec
            
            
            #no more orders, should this be an error condition?
            if len(rlist) == 0 or len(xlist) > 0: 
                break
                
            order_msg = rlist[0].recv()
            if order_msg == str(CONTROL_PROTOCOL.DONE):
                break

            order = json.loads(order_msg)
            self.add_open_order(order)
            
        #end of order processing loop
        self.last_iteration_duration = datetime.datetime.utcnow() - self.event_start
        
            
    def add_open_order(self, order):
        self.open_orders[order['sid']] = order
        