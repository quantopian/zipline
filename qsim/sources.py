"""
Provides data handlers that can push messages to a qsim.core.DataFeed
"""
import datetime
import zmq
import json
import random

import qsim.util as qutil

class DataSource(object):
    """
    Baseclass for data sources. Subclass and implement send_all - usually this 
    means looping through all records in a store, converting to a dict, and
    calling send(map).
    """
    def __init__(self, source_id):
        self.source_id              = source_id
        self.data_address           = None
        self.sync                   = None
        self.cur_event              = None
        self.context                = None
        self.data_socket            = None
    
    def open(self):    
        """create zmq context and socket"""
        qutil.LOGGER.info(
            "starting data source:{source_id} on {addr}"
                .format(source_id=self.source_id, addr=self.data_address))
        
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_socket.connect(self.data_address)
        self.data_socket.setsockopt(zmq.LINGER,0)
        
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
            sets source_id and type properties in the dict
            sends to the data_socket.
        """
        self.sync.confirm()
        event['s'] = self.source_id             
        event['type'] = 'event'
        self.data_socket.send(json.dumps(event), zmq.NOBLOCK)
    
    def close(self):
        """
            Close the zmq context and sockets.
        """
        qutil.LOGGER.info("sending DONE message.")
        try:
            done_msg = {}
            done_msg['type'] = 'DONE'
            done_msg['s'] = self.source_id
            self.data_socket.send(json.dumps(done_msg), zmq.NOBLOCK)
        except:
            qutil.LOGGER.exception("failed to send DONE message")
            pass #continue with the closing.   
        
        qutil.LOGGER.info("closing data socket")
        self.data_socket.close()
        qutil.LOGGER.info("closing sync")
        self.sync.close()
        qutil.LOGGER.info("closing context")
        try:
            self.context.term()
            qutil.LOGGER.info("done")
        except:
            qutil.LOGGER.exception("error closing context")
        qutil.LOGGER.info("finished processing data source")
       
class RandomEquityTrades(DataSource):
    """Generates a random stream of trades for testing."""
    
    def __init__(self, sid, source_id, count):
        DataSource.__init__(self, source_id)
        self.count = count
        self.sid = sid
        
    def send_all(self):
        trade_start = datetime.datetime.now()
        minute = datetime.timedelta(minutes=1)
        price = random.uniform(5.0, 50.0)
        
        for i in range(self.count):
            price = price + random.uniform(-0.05, 0.05)
            event = {'sid':self.sid, 
                     'dt':qutil.format_date(trade_start + (minute * i)),
                     'price':price, 
                     'volume':random.randrange(100,10000,100)}
            self.send(event)
          
       
        
        
