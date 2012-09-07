from numbers import Number
from datetime import datetime, timedelta
from collections import defaultdict
from math import sqrt

from zipline import ndict
from zipline.gens.transform import EventWindow, TransformMeta

class MovingStandardDev(object):
    """
    Class that maintains a dicitonary from sids to
    MovingStandardDevWindows.  For each sid, we maintain a the
    standard deviation of all events falling within the specified
    window.
    """
    __metaclass__ = TransformMeta

    def __init__(self, market_aware, days = None, delta = None):

        self.market_aware = market_aware

        self.delta = delta
        self.days = days

        # Market-aware mode only works with full-day windows.
        if self.market_aware:
            assert self.days and not self.delta,\
                "Market-aware mode only works with full-day windows."

        # Non-market-aware mode requires a timedelta.
        else:
            assert self.delta and not self.days, \
                "Non-market-aware mode requires a timedelta."
        
        # No way to pass arguments to the defaultdict factory, so we
        # need to define a method to generate the correct EventWindows.
        self.sid_windows = defaultdict(self.create_window)
        
    def create_window(self):
        """
        Factory method for self.sid_windows.
        """
        return MovingStandardDevWindow(
            self.market_aware,
            self.days,
            self.delta
        )
    
    def update(self, event):
        """
        Update the event window for this event's sid.  Return an ndict
        from tracked fields to moving averages.
        """
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_stddev()
    
class MovingStandardDevWindow(EventWindow):
    """
    Iteratively calculates standard deviation for a particular sid
    over a given time window.  The expected functionality of this
    class is to be instantiated inside a MovingStandardDev.
    """
    
    def __init__(self, market_aware, days, delta):
        
        # Call the superclass constructor to set up base EventWindow
        # infrastructure.
        EventWindow.__init__(self, market_aware, days, delta)

        self.sum = 0.0
        self.sum_sqr = 0.0
                
    def handle_add(self, event):
        assert event.has_key('price')
        assert isinstance(event.price, Number)

        self.sum += event.price
        self.sum_sqr += event.price ** 2
                
    def handle_remove(self, event):
        assert event.has_key('price')
        assert isinstance(event.price, Number)
        
        self.sum -= event.price
        self.sum_sqr -= event.price ** 2
        
    def get_stddev(self):
        
        # Sample standard deviation is undefined for a single event or
        # no events.
        if len(self) <= 1:
            return None

        else:
            average = self.sum /len(self)
            s_squared = (self.sum_sqr - self.sum*average) / (len(self) - 1) 
            stddev = sqrt(s_squared)
        return stddev
