"""
Provides data handlers that can push messages to a zipline.core.DataFeed
"""
import datetime
import random
import pytz

import zipline.util as qutil
import zipline.messaging as zm
import zipline.protocol as zp

class TradeDataSource(zm.DataSource):
    
    def send(self, event):
        """ :param dict event: is a trade event with data as per :py:func: `zipline.protocol.TRADE_FRAME`
            :rtype: None
        """
        event.source_id = self.get_id
        message = zp.DATASOURCE_FRAME(event)
        self.data_socket.send(message)

class RandomEquityTrades(TradeDataSource):
    """Generates a random stream of trades for testing."""
    
    def __init__(self, sid, source_id, count):
        zm.DataSource.__init__(self, source_id)
        self.count          = count
        self.incr           = 0
        self.sid            = sid
        self.trade_start    = datetime.datetime.now().replace(tzinfo=pytz.utc)
        self.minute         = datetime.timedelta(minutes=1)
        self.price          = random.uniform(5.0, 50.0)
    
    
    def get_type(self):
        return 'equity_trade'
    
    
    def do_work(self):
        if(self.incr == self.count):
            self.signal_done()
            return
        
        self.price = self.price + random.uniform(-0.05, 0.05)        
        self._send(self.sid, self.price, random.randrange(100,10000,100), self.trade_start + (self.minute * self.incr))
        self.incr += 1        

    def _send(self, sid, price, volume, dt):
        event = zp.namedict({'source_id': self.get_id, "type" : "TRADE", "sid":sid, "price":price, "volume":volume, "dt":dt})
        self.send(event)


class SpecificEquityTrades(TradeDataSource):
    """Generates a random stream of trades for testing."""

    def __init__(self, source_id, event_list):
        """
        :event_list: should be a chronologically ordered list of dictionaries in the following form:
                event = {
                    'sid'    : an integer for security id,
                    'dt'     : datetime object,
                    'price'  : float for price,
                    'volume' : integer for volume
                }
        """
        zm.DataSource.__init__(self, source_id)
        self.event_list = event_list

    def get_type(self):
        return 'equity_trade'

    def do_work(self):
        if(len(self.event_list) == 0):
            self.signal_done()
            return
        
        event = self.event_list.pop(0)
        self.send(zp.namedict(event))
        

