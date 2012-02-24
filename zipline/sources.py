"""
Provides data handlers that can push messages to a zipline.core.DataFeed
"""
import datetime
import json
import random

import zipline.util as qutil
import zipline.messaging as qmsg        
       
class RandomEquityTrades(qmsg.DataSource):
    """Generates a random stream of trades for testing."""
    
    def __init__(self, sid, source_id, count):
        qmsg.DataSource.__init__(self, source_id)
        self.count          = count
        self.incr           = 0
        self.sid            = sid
        self.trade_start    = datetime.datetime.now()
        self.minute         = datetime.timedelta(minutes=1)
        self.price          = random.uniform(5.0, 50.0)
    
    def get_type(self):
        return 'equity_trade'    
    
    def do_work(self):
        if(self.incr == self.count):
            self.signal_done()
            return
        self.price = self.price + random.uniform(-0.05, 0.05)
        event = {'sid':self.sid, 
                 'dt':qutil.format_date(self.trade_start + (self.minute * self.incr)),
                 'price':self.price, 
                 'volume':random.randrange(100,10000,100)}
        self.send(event)
        self.incr += 1
               
        
        
