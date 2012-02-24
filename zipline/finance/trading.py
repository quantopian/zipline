import json
import zipline.messaging as qmsg

class TradingClient(qmsg.Component):

    def __init__(self):
        qmsg.Component.__init__(self)
        self.received_count     = 0
        self.prev_dt            = None
        
    def get_id(self):
        return "TRADING_CLIENT"

    def open(self):
        self.data_feed, self.poller = self.connect_result()
        self.order_feed = self.connect_order()

    def do_work(self):
        socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.
        if self.data_feed in socks and socks[self.data_feed] == self.zmq.POLLIN:   
            msg = self.data_feed.recv()
            if(self.is_done_message(msg)):
                qutil.LOGGER.info("Client is DONE!")
                self.signal_done()
                return

            self.received_count += 1
            event = json.loads(msg)
            self.handle_event(event)
    
    def connect_order(self):
        return self.connect_push_socket(self.addresses['order_address'])
           
    def handle_event(self, event):
        NotImplemented
           
    def order(self, sid, volume):
        order = {'sid':sid, 'volume':volume}
        self.order_feed.send(json.dumps(order))
        
        

class TradeSimulator(qmsg.BaseTransform):
    
    def __init__(self): 
        qmsq.BaseTransform.__init__(self, "")
        self.open_orders    = {}
        self.algo_time      = None
    
    def get_id(self):
        return "EQUITY_TRADE_SIM"    
    
    def open():
        qmsg.BaseTransform.open(self)
        self.order_socket, self.order_poller = self.bind_order()
        
    def bind_order(self):
        return self.bind_pull_socket(self.addresses['order_address'])

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
            
            #receive all orders.
            while True:
                message = self.order_socket.recv()
                if(self.is_done_message(message)):
                    break; #no more orders on this tick
                self.add_open_order(json.loads(order))
                
            cur_state['id'] = self.state['name']
            self.result_socket.send(json.dumps(cur_state), self.zmq.NOBLOCK)
            
    def add_open_order(self, order):
        self.open_orders[order['sid']] = order
        