from zipline.gens.mavg import MovingAverage
from datetime import datetime, timedelta

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
        order_size = self.buy_or_sell * (self.amount - (self.offset**2))
        self.order(self.sid, order_size)

        #sell next time around.
        self.buy_or_sell *= -1

        self.orders.append(order_size)

        self.frame_count += 1
        self.incr += 1

    def get_sid_filter(self):
        return [self.sid]

# Algorithm base class, user algorithms inherit from this as they
# don't want to have to copy and know about set_order and
# set_portfolio
class Algorithm(object):
    def set_order(self, order_callable):
        self.order = order_callable

    def get_sid_filter(self):
        return [self.sid]

    def set_logger(self, logger):
        self.logger = logger

    def initialize(self):
        pass

    def add_transform(self, transform_class, tag, *args, **kwargs):
        if not hasattr(self, 'registered_transforms'):
            self.registered_transforms = {}

        self.registered_transforms[tag] = {'class': transform_class,
                                           'args': args,
                                           'kwargs': kwargs}


# Inherits from Algorithm base class
class DMA(Algorithm):
    """Dual Moving Average algorithm.
    """

    def __init__(self, sid, amount, short_window=20, long_window=40):
        self.sid = sid
        self.amount = amount
        self.done = False
        self.order = None
        self.frame_count = 0
        self.portfolio = None
        self.orders = []
        self.market_entered = False
        self.prices = []
        self.events = 0
        self.add_transform(MovingAverage, 'short_mavg', ['price'], market_aware=False, delta=timedelta(days=short_window))
        self.add_transform(MovingAverage, 'long_mavg', ['price'], market_aware=False, delta=timedelta(days=long_window))

    def handle_data(self, data):
        self.events += 1
        # access transforms via their user-defined tag
        if (data[self.sid].short_mavg > data[self.sid].long_mavg) and not self.market_entered:
            self.order(self.sid, 100)
            self.market_entered = True
        elif (data[self.sid].short_mavg < data[self.sid].long_mavg) and self.market_entered:
            self.order(self.sid, -100)
            self.market_entered = False
