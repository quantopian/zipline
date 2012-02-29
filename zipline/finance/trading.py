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
                                                  timeout=self.heartbeat_timeout/100) #select timeout is in sec, use 10x
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
        if event.SIM_DT <= event.dt:
            #event occurred in the present, send the queue to be processed
            self.handle_events(self.event_queue)
        self.order_socket.send(str(zp.CONTROL_PROTOCOL.DONE))
    
    def handle_events(self, event_queue):
        raise NotImplementedError    
    
    def order(self, sid, amount):
        self.order_socket.send(zp.ORDER_FRAME(sid, amount))
        
class OrderDataSource(qmsg.DataSource):
    """DataSource that relays orders from the client"""

    def __init__(self, simulation_dt):
        """
        :param simulation_time: datetime in UTC timezone, sets the start time of simulation. orders
            will be timestamped relative to this datetime.
                event = {
                    'sid'    : an integer for security id,
                    'dt'     : datetime object,
                    'price'  : float for price,
                    'volume' : integer for volume
                }
        """
        zm.DataSource.__init__(self, str(zp.FINANCE_PROTOCOL.ORDER))
        self.simulation_dt = simulation_dt
        self.last_iteration_duration = datetime.timedelta(seconds=0)

    def get_type(self):
        return str(zp.FINANCE_PROTOCOL.ORDER)
        
    def open(self):
        qmsg.DataSource.open(self)
        self.order_socket = self.bind_order()
        
    def bind_order(self):
        return self.bind_pull_socket(self.addresses['order_address'])

    def do_work(self):
        #mark the start time for client's processing of this event.
        self.event_start = datetime.datetime.utcnow()
        self.result_socket.send(zp.TRANSFORM_FRAME(str(zp.FINANCE_PROTOCOL.ORDER), self.simulation_dt), self.zmq.NOBLOCK)
        
        self.simulation_dt = self.simulation_dt + self.last_iteration_duration
        
        #pull all orders from client.
        orders = []
        order_dt = None
        while True:
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
            #send the order along
            
            self.last_iteration_duration = datetime.datetime.utcnow() - self.event_start
            dt = self.simulation_dt + self.last_iteration_duration
            order_event = zp.namedict({"sid":sid, "amount":amount, "dt":dt, source_id=self.get_id})
            
            message = zp.DATASOURCE_FRAME(event)
            self.data_socket.send(message)
    
    
    

class TransactionSimulator(qmsg.BaseTransform):
    
    def __init__(self): 
        qmsg.BaseTransform.__init__(self, "TRANSACTION_SIM")
        self.open_orders                = {}
        self.order_count                = 0
        self.tradeWindow                = datetime.timedelta(seconds=30)
        self.orderTTL                   = datetime.timedelta(days=1)
        self.volume_share               = 0.05
        self.commission                 = 0.03
            
    def transform(self, event):
        """
        Pulls one message from the event feed, then
        loops on orders until client sends DONE message.
        """
        if(event.type == zp.FINANCE_PROTOCOL.ORDER):
            self.add_open_order(event.sid, event.amount)
            self.state['value'] = self.average
        elif(event.type == zp.FINANCE_PROTOCOL.TRADE):
            txn = apply_trade_to_open_orders(event)
            self.state['value'] = txn
        
        return self.state
            
    def add_open_order(self, sid, amount):
        """Orders are captured in a buffer by sid. No calculations are done here.
            Amount is explicitly converted to an int.
            Orders of amount zero are ignored.
        """
        amount = int(amount)
        if amount == 0:
            qutil.LOGGER.debug("{title}:{id} requested to trade zero shares of {sid}".format(sid=sid,
                                                                                            title=self.hostedAlgo.algo.title,
                                                                                            id=self.hostedAlgo.algo.id))
            return
            
        self.order_count += 1
        order = zp.namedict({'sid' : sid, 
                 'amount' : amount, 
                 'dt' : self.algo_time},
                 'filled': 0,
                 'direction': math.fabs(amount) / amount)
        
        if(not self.open_orders.has_key(sid)):
            self.open_orders[sid] = []
        self.open_orders[sid].append(order)
     
    def apply_trade_to_open_orders(self, event):
        
        if(event.volume == 0):
            #there are zero volume events bc some stocks trade less frequently than once per minute.
            continue 
        if self.open_orders.has_key(event.sid):
            orders = self.open_orders[event.sid] 
            remaining_orders = []
            total_order = 0        
            dt = event.dt
        
            for order in orders:
                #we're using minute bars, so allow orders within 30 seconds of the trade
                if((order.dt - event.dt) < self.tradeWindow):
                    total_order += order.amount
                    if(order.dt > dt):
                        dt = order.dt
                #if the order still has time to live (TTL) keep track
                elif((self.algo_time - order.dt) < self.orderTTL):
                    remaining_orders.append(order)
        
            self.open_orders[event.sid] = remaining_orders
        
            if(total_order != 0):
                direction = total_order / math.fabs(total_order)
                volume_share = (direction * total_order) / event.volume
                if volume_share > .25:
                    volume_share = .25
                amount = volume_share * event.volume * direction
                impact = (volShare)**2 * .1 * direction * event.price
                return self.create_transaction(event.sid, amount, event.price + impact, dt.replace(tzinfo = pytz.utc), direction)
        
    
        
    def create_transaction(self, sid, amount, price, dt, direction):                
        if(amount != 0):
            txn = {'sid'                : sid, 
                   'amount'             : amount, 
                   'dt'                 : dt, 
                   'price'              : price, 
                   'back_test_run_id'   : self.btRun.id,
                   'transaction_cost'   : -1*(price * amount),
                   'commision'          : self.commission * amount * direction}
            
            return namedict(txn) 
                
                
                

