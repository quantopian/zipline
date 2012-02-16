"""
Provides data handlers that can push messages to a zipline.core.DataFeed
"""
import datetime
from gevent_zeromq import zmq
import json
import random

import zipline.util as qutil
import zipline.messaging as qmsg

class DataSource(qmsg.Component):
    """
    Baseclass for data sources. Subclass and implement send_all - usually this 
    means looping through all records in a store, converting to a dict, and
    calling send(map).
    """
    def __init__(self, source_id):
        self.id                     = source_id
        self.host                   = host
        self.data_address           = None
        self.sync                   = None
        self.cur_event              = None
        self.context                = None
        self.data_socket            = None
        
    def get_id(self):
        return self.id
    
    def set_addresses(self, addresses):
        self.data_address = addresses['data_address']
        
    def open(self):    
        """create zmq context and socket"""
        qutil.LOGGER.info(
            "starting data source:{id} on {addr}"
                .format(id=self.id, addr=self.data_address))
        
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_socket.connect(self.data_address)
        #self.data_socket.setsockopt(zmq.LINGER,0)
        
        self.sync.open()
    
    def run(self):  
        """Fully execute this datasource."""  
        try:
            self.open()
            self.send_all()
        except:
            qutil.LOGGER.info("Exception running datasource.")
        finally:
            self.close()    
    
    def send_all(self):
        """Subclasses must implement this method."""
        raise NotImplementedError()
            
    def send(self, event):
        """
            event is expected to be a dict
            sets id and type properties in the dict
            sends to the data_socket.
        """
        event['id'] = self.id             
        event['type'] = self.get_type()
        self.data_socket.send(json.dumps(event), zmq.NOBLOCK)
        
    def get_type(self):
        raise NotImplemented
        
    def close(self):
        """
            Close the zmq context and sockets.
        """
        qutil.LOGGER.info("sending DONE message.")
        try:
            done_msg = {}
            done_msg['type'] = 'DONE'
            done_msg['s'] = self.id
            self.data_socket.send(json.dumps(done_msg), zmq.NOBLOCK)
            
            qutil.LOGGER.info("closing data socket")
            self.data_socket.close()
            qutil.LOGGER.info("closing sync")
            self.sync.close()
            qutil.LOGGER.info("closing context")
        except:
            qutil.LOGGER.exception("failed to send DONE message")
        finally:
            self.context.destroy()
        qutil.LOGGER.info("finished processing data source")
       
class RandomEquityTrades(DataSource):
    """Generates a random stream of trades for testing."""
    
    def __init__(self, sid, source_id, count):
        DataSource.__init__(self, source_id)
        self.count = count
        self.sid = sid
    
    def get_type(self):
        return 'equity_trade'    
    
    def send_all(self):
        trade_start = datetime.datetime.now()
        minute = datetime.timedelta(minutes=1)
        price = random.uniform(5.0, 50.0)
        
        for i in range(self.count):
            if not self.sync.confirm():
                break
            price = price + random.uniform(-0.05, 0.05)
            event = {'sid':self.sid, 
                     'dt':qutil.format_date(trade_start + (minute * i)),
                     'price':price, 
                     'volume':random.randrange(100,10000,100)}
            self.send(event)
          
       
        
        
