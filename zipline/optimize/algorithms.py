from logbook import Logger
from zipline import TradingAlgorithm

logger = Logger('Algo')

class BuySellAlgorithm(object):
    """Algorithm that buys and sells alternatingly. The amount for
    each order can be specified. In addition, an offset that will
    quadratically reduce the amount that will be bought can be
    specified.

    This algorithm is used to test the parameter optimization
    framework. If combined with the UpDown trade source, an offset of
    0 will produce maximum returns.

    """

    def __init__(self, sid, amount, offset):
        self.sid = sid
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.buy_or_sell = -1
        self.offset = offset
        self.orders = []
        self.prices = []

    def initialize(self):
        pass

    def set_order(self, order_callable):
        self.order = order_callable

    def set_portfolio(self, portfolio):
        self.portfolio = portfolio

    def handle_data(self, frame):
        print frame.sid
        order_size = self.buy_or_sell * (self.amount - (self.offset**2))
        self.order(self.sid, order_size)

        #sell next time around.
        self.buy_or_sell *= -1

        self.orders.append(order_size)

        self.frame_count += 1
        self.incr += 1

    def get_sid_filter(self):
        return [self.sid]


class BuySellAlgorithmNew(TradingAlgorithm):
    """Algorithm that buys and sells alternatingly. The amount for
    each order can be specified. In addition, an offset that will
    quadratically reduce the amount that will be bought can be
    specified.

    This algorithm is used to test the parameter optimization
    framework. If combined with the UpDown trade source, an offset of
    0 will produce maximum returns.

    """

    def __init__(self, sids, amount, offset):
        self.sids = sids
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.buy_or_sell = -1
        self.offset = offset
        self.orders = []
        self.prices = []

    def handle_data(self, data):
        order_size = self.buy_or_sell * (self.amount - (self.offset**2))
        self.order(self.sids[0], order_size)
        logger.debug("ordering" + str(order_size))

        #sell next time around.
        self.buy_or_sell *= -1

        self.orders.append(order_size)

        self.frame_count += 1
        self.incr += 1

