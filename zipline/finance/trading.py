import datetime
import pytz
import math
import pandas

from zmq.core.poll import select

import zipline.messaging as qmsg
import zipline.util as qutil
import zipline.protocol as zp
import zipline.finance.performance as perf

class TradeSimulationClient(qmsg.Component):
    
    def __init__(self, trading_environment):
        qmsg.Component.__init__(self)
        self.received_count         = 0
        self.prev_dt                = None
        self.event_queue            = None
        self.event_callbacks        = []
        self.txn_count              = 0
        self.trading_environment    = trading_environment
        self.current_dt             = trading_environment.period_start
        self.last_iteration_dur     = datetime.timedelta(seconds=0)
        
        assert self.trading_environment.frame_index != None
        self.event_frame = pandas.DataFrame(
            index=self.trading_environment.frame_index
        )
        
        self.perf = perf.PerformanceTracker(self.trading_environment)
        self.on_done = self.perf.handle_simulation_end
        
    
    @property
    def get_id(self):
        return str(zp.FINANCE_COMPONENT.TRADING_CLIENT)
    
    def add_event_callback(self, callback):
        """
        :param callable callback: must be a function with the signature
        f(frame).
        """
        self.event_callbacks.append(callback)
    
    def open(self):
        self.result_feed = self.connect_result()
        self.order_socket = self.connect_order()
    
    def do_work(self):
        #next feed event
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        if self.result_feed in socks and \
            socks[self.result_feed] == self.zmq.POLLIN:   
            
            msg = self.result_feed.recv()

            if msg == str(zp.CONTROL_PROTOCOL.DONE):
                qutil.LOGGER.info("Client is DONE!")
                self.run_callbacks()
                self.signal_done()
                return
            
            event = zp.MERGE_UNFRAME(msg)
            
            if(event.TRANSACTION != None):
                self.txn_count += 1
            
            #filter order flow out of the events sent to callbacks
            if event.source_id != zp.FINANCE_COMPONENT.ORDER_SOURCE:
                #mark the start time for client's processing of this event.
                event_start = datetime.datetime.utcnow()
                self.queue_event(event)
                
                if event.dt >= self.current_dt:
                    self.run_callbacks()
                
                #update time based on receipt of the order
                self.last_iteration_dur = datetime.datetime.utcnow() - event_start
                    
                self.current_dt = self.current_dt + self.last_iteration_dur
            
            #signal done to order source.
            self.order_socket.send(str(zp.ORDER_PROTOCOL.BREAK))
            
    def run_callbacks(self):
        frame = self.get_frame()
        for cb in self.event_callbacks:
            cb(frame)
    
    def connect_order(self):
        return self.connect_push_socket(self.addresses['order_address'])
    
    def order(self, sid, amount):
        order = zp.namedict({
            'dt':self.current_dt,
            'sid':sid,
            'amount':amount
        })
        
        self.order_socket.send(zp.ORDER_FRAME(order))
        
    def signal_order_done(self):
        self.order_socket.send(str(zp.ORDER_PROTOCOL.DONE))
        
    def queue_event(self, event):
        self.perf.process_event(event)
        if self.event_queue == None:
            self.event_queue = []
        series = event.as_series()
        self.event_queue.append(series)
    
    def get_frame(self):
        for event in self.event_queue:
            self.event_frame[event['sid']] = event
        return self.event_frame
        
class OrderDataSource(qmsg.DataSource):
    """DataSource that relays orders from the client"""

    def __init__(self):
        """
        :param simulation_time: datetime in UTC timezone, sets the start 
        time of simulation. orders
            will be timestamped relative to this datetime.
                event = {
                    'sid'    : an integer for security id,
                    'dt'     : datetime object,
                    'price'  : float for price,
                    'volume' : integer for volume
                }
        """
        qmsg.DataSource.__init__(self, zp.FINANCE_COMPONENT.ORDER_SOURCE)
        self.sent_count         = 0

    @property
    def get_type(self):
        return zp.DATASOURCE_TYPE.ORDER
        
    def open(self):
        qmsg.DataSource.open(self)
        self.order_socket = self.bind_order()
        
    def bind_order(self):
        return self.bind_pull_socket(self.addresses['order_address'])

    def do_work(self):    
        
        #TODO: if this is the first iteration, break deadlock by sending a dummy order
        if(self.sent_count == 0):
            self.send(zp.namedict({}))
        
        #pull all orders from client.
        orders = []
        count = 0
        while True:
            
            (rlist, wlist, xlist) = select(
                [self.order_socket],
                [],
                [self.order_socket],
                #allow half the time of a heartbeat for the order
                #timeout, so we have time to signal we are done.
                timeout=self.heartbeat_timeout/2000
            ) 
        
            
            #no more orders, should this be an error condition?
            if len(rlist) == 0 or len(xlist) > 0: 
                #no order message means there was a timeout above, 
                #and the client is done sending orders (but isn't
                #telling us himself!).
                self.signal_done()
                return
                
            order_msg = rlist[0].recv()
            
            if order_msg == str(zp.ORDER_PROTOCOL.DONE):
                self.signal_done()
                return
                
            if order_msg == str(zp.ORDER_PROTOCOL.BREAK):
                break

            order = zp.ORDER_UNFRAME(order_msg)
            #send the order along
            self.send(order)
            count += 1
            self.sent_count += 1
    
        #TODO: we have to send at least one dummy order per do_work iteration 
        # or the feed will block waiting for our messages.
        if(count == 0):
            self.send(zp.namedict({}))
    
    

class TransactionSimulator(qmsg.BaseTransform):
    
    def __init__(self): 
        qmsg.BaseTransform.__init__(self, zp.TRANSFORM_TYPE.TRANSACTION)
        self.open_orders                = {}
        self.order_count                = 0
        self.txn_count                  = 0
        self.trade_window                = datetime.timedelta(seconds=30)
        self.orderTTL                   = datetime.timedelta(days=1)
        self.volume_share               = 0.05
        self.commission                 = 0.03
            
    def transform(self, event):
        """
        Pulls one message from the event feed, then
        loops on orders until client sends DONE message.
        """
        if(event.type == zp.DATASOURCE_TYPE.ORDER):
            self.add_open_order(event)
            self.state['value'] = None
        elif(event.type == zp.DATASOURCE_TYPE.TRADE):
            txn = self.apply_trade_to_open_orders(event)
            self.state['value'] = txn
        else:
            self.state['value'] = None
            qutil.LOGGER.info("unexpected event type in transform: {etype}".format(etype=event.type))
        #TODO: what to do if we get another kind of datasource event.type?
        
        return self.state
            
    def add_open_order(self, event):
        """Orders are captured in a buffer by sid. No calculations are done here.
            Amount is explicitly converted to an int.
            Orders of amount zero are ignored.
        """
        event.amount = int(event.amount)
        if event.amount == 0:
            qutil.LOGGER.debug("requested to trade zero shares of {sid}".format(sid=event.sid))
            return
            
        self.order_count += 1
        
        if(not self.open_orders.has_key(event.sid)):
            self.open_orders[event.sid] = []
        self.open_orders[event.sid].append(event)
     
    def apply_trade_to_open_orders(self, event):
        
        if(event.volume == 0):
            #there are zero volume events bc some stocks trade 
            #less frequently than once per minute.
            return self.create_dummy_txn(event.dt)
            
        if self.open_orders.has_key(event.sid):
            orders = self.open_orders[event.sid] 
        else:
            return None
            
        remaining_orders = []
        total_order = 0        
        dt = event.dt
    
        for order in orders:
            #we're using minute bars, so allow orders within 
            #30 seconds of the trade
            if((order.dt - event.dt) < self.trade_window):
                total_order += order.amount
                if(order.dt > dt):
                    dt = order.dt
            #if the order still has time to live (TTL) keep track
            elif((self.algo_time - order.dt) < self.orderTTL):
                remaining_orders.append(order)
    
        self.open_orders[event.sid] = remaining_orders
    
        if(total_order != 0):
            direction = total_order / math.fabs(total_order)
        else:
            direction = 1
            
        volume_share = (direction * total_order) / event.volume
        if volume_share > .25:
            volume_share = .25
        amount = volume_share * event.volume * direction
        impact = (volume_share)**2 * .1 * direction * event.price
        return self.create_transaction(
            event.sid, 
            amount, 
            event.price + impact, 
            dt.replace(tzinfo = pytz.utc), 
            direction
        )
    
        
    def create_transaction(self, sid, amount, price, dt, direction):   
        self.txn_count += 1             
        txn = {'sid'                : sid, 
                'amount'             : int(amount), 
                'dt'                 : dt, 
                'price'              : price, 
                'commission'          : self.commission * amount * direction,
                'source_id'          : zp.FINANCE_COMPONENT.TRANSACTION_SIM
                }
        return zp.namedict(txn) 
                
                
                

