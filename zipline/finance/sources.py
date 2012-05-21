"""
Provides data handlers that can push messages to a zipline.core.DataFeed

::
                   DataSource
                       |
                TradeDataSource
                  /          \
    RandomEquityTrades     SpecificEquityTrades

"""
import pytz
import random
import datetime
from mock import Mock

from zipline.components import DataSource
from zipline.utils import ndict

import zipline.protocol as zp

class TradeDataSource(DataSource):

    def init(self, source_id):
        self.source_id = source_id
        self.setup_source()

    @property
    def get_id(self):
        return 'TradeDataSource'

    def send(self, event):
        """
        Sends the event iff it matches the internal sid filter.
        :param dict event: is a trade event with data as per
                           :py:func: `zipline.protocol.TRADE_FRAME`
        :rtype: None
        """

        event.source_id = self.source_id

        if event.sid in self.filter['sid']:
            message = zp.DATASOURCE_FRAME(event)
        else:
            blank = ndict({
                "type"      : zp.DATASOURCE_TYPE.TRADE,
                "source_id" : self.source_id
            })
            message = zp.DATASOURCE_FRAME(blank)

        self.data_socket.send(message)


class RandomEquityTrades(TradeDataSource):
    """
    Generates a random stream of trades for testing.
    """

    def init(self, sid, source_id, count):
        self.source_id      = source_id
        self.count          = count
        self.incr           = 0
        self.sid            = sid
        self.trade_start    = datetime.datetime.now().replace(tzinfo=pytz.utc)
        self.day            = datetime.timedelta(days=1)
        self.price          = random.uniform(5.0, 50.0)

        self.setup_source()

    @property
    def get_id(self):
        return 'RandomEquityTrades'

    def do_work(self):
        if not self.incr < self.count:
            self.signal_done()
            return

        self.price = self.price + random.uniform(-0.05, 0.05)
        volume = random.randrange(100,10000,100)

        event = zp.ndict({
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

    def init(self, source_id, event_list):
        """
        :param event_list: should be a chronologically ordered list of
        dictionaries in the following form::

            event = {
                'sid'    : an integer for security id,
                'dt'     : datetime object,
                'price'  : float for price,
                'volume' : integer for volume
            }
        """
        self.source_id = source_id
        self.event_list = event_list
        self.count = 0

        # TODO temporary hack
        self.control_out = Mock()
        self.setup_source()

    @property
    def get_id(self):
        return "SpecificEquityTrades"

    def do_work(self):
        if(len(self.event_list) == 0):
            self.signal_done()
            return

        event = self.event_list.pop(0)
        self.send(zp.ndict(event))
        self.count +=1
