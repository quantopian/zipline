"""
Provides data handlers that can push messages to a qsim.simulator.DataFeed
"""
import datetime
import zmq
import json
import multiprocessing
import random

import qsim.util as qutil
import qsim.messaging as qmsg

class DataSource(object):
    """
    Baseclass for data sources. Subclass and implement send_all - usually this 
    means looping through all records in a store, converting to a dict, and
    calling send(map).
    """
    def __init__(self, feed, source_id):
        self.source_id              = source_id
        self.feed                   = feed
        self.cur_event              = None
        self.context                = None
        self.data_socket            = None

    def start(self):
        """Launch the datasource in a separate process."""
        proc = multiprocessing.Process(target=self.run)
        proc.start()
        
        
    def open(self):    
        """create zmq context and socket"""
        qutil.logger.info(
            "starting data source:{source_id} on {addr}"
                .format(source_id=self.source_id, addr=self.feed.data_address))
        
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_socket.connect(self.feed.data_address)
        
        #signal we are ready
        sync = qmsg.FeedSync(self.feed, str(self.source_id))
        sync.confirm()
    
    def run(self):  
        """Fully execute this datasource."""  
        self.open()
        self.send_all()
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
        event['s'] = self.source_id             
        event['type'] = 'event'
        self.data_socket.send(json.dumps(event))
    
    def close(self):
        """
            Close the zmq context and sockets.
        """
        done_msg = {}
        done_msg['type'] = 'DONE'
        done_msg['s'] = self.source_id
        self.data_socket.send(json.dumps(done_msg))   
        self.data_socket.close()
        self.context.term()
        qutil.logger.info("finished processing data source")
       
class RandomEquityTrades(DataSource):
    """Generates a random stream of trades for testing."""
    
    def __init__(self, sid, feed, source_id, count):
        DataSource.__init__(self, feed, source_id)
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
          
   
       
        
        
