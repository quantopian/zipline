"""
Provides data handlers that can push messages to a zipline.core.DataFeed
"""
import datetime
import random
import pytz

from zipline.components import DataSource
from zipline.utils import ndict, namedict

import zipline.protocol as zp

class TradeDataSource(DataSource):

    def send(self, event):
        """
        Sends the event iff it matches the internal SID filter.
        :param dict event: is a trade event with data as per
                           :py:func: `zipline.protocol.TRADE_FRAME`
        :rtype: None
        """

        event.source_id = self.get_id
        if event.sid in self.filter['SID']:
            message = zp.DATASOURCE_FRAME(event)
        else:
            blank = namedict({
                "type"      : zp.DATASOURCE_TYPE.TRADE,
                "source_id" : self.get_id
            })
            message = zp.DATASOURCE_FRAME(blank)

        self.data_socket.send(message)


class RandomEquityTrades(TradeDataSource):
    """
    Generates a random stream of trades for testing.
    """

    def __init__(self, sid, source_id, count):
        DataSource.__init__(self, source_id)
        self.count          = count
        self.incr           = 0
        self.sid            = sid
        self.trade_start    = datetime.datetime.now().replace(tzinfo=pytz.utc)
        self.day            = datetime.timedelta(days=1)
        self.price          = random.uniform(5.0, 50.0)


    def get_type(self):
        zp.COMPONENT_TYPE.SOURCE

    def do_work(self):
        if not self.incr < self.count:
            self.signal_done()
            return

        self.price = self.price + random.uniform(-0.05, 0.05)
        volume = random.randrange(100,10000,100)

        event = zp.namedict({
            "type"      : zp.DATASOURCE_TYPE.TRADE,
            "sid"       : self.sid,
            "price"     : self.price,
            "volume"    : volume,
            "dt"        : self.trade_start + (self.day * self.incr),
        })
        self.send(event)
        self.incr += 1


class SpecificEquityTrades(TradeDataSource):
    """
    Generates a random stream of trades for testing.
    """

    def __init__(self, source_id, event_list):
        """
        :param event_list: should be a chronologically ordered list of
                           dictionaries in the following form:

                event = {
                    'sid'    : an integer for security id,
                    'dt'     : datetime object,
                    'price'  : float for price,
                    'volume' : integer for volume
                }
        """
        DataSource.__init__(self, source_id)
        self.event_list = event_list
        self.count = 0

    def get_type(self):
        zp.COMPONENT_TYPE.SOURCE

    def do_work(self):
        if(len(self.event_list) == 0):
            self.signal_done()
            return

        event = self.event_list.pop(0)
        self.send(zp.namedict(event))
        self.count +=1
