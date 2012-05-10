import pandas
from datetime import timedelta
from collections import defaultdict

from zipline.messaging import BaseTransform

class WindowTransform(BaseTransform):
    
    def init(self, daycount=3):
        self.daycount = daycount
        self.by_sid = defaultdict(DailyReturns)
        
    def transform(self, event):
        cur = self.by_sid(event.sid)
        cur.update(event)
        self.state['value'] = cur.vwap
        return self.state

class ReturnsFromPriorClose(object):
    """
    Calculates a security's returns since the previous close, using the
    current price.
    """
    
    def __init__(self):
        self.last_close = None
        self.last_event = None
        self.returns = 0.0
        
    def update(self, event):
        next_close = None
        if self.last_close:
            change = event.price - self.last_close.price
            self.returns = change / self.last_close.price
            
        if self.last_event:
            if self.last_event.dt.day != event.dt.day:
                # the current event is from the day after
                # the last event. Therefore the last event was
                # the last close
                self.last_close = self.last_event
        
        # the current event is now the last_event
        self.last_event = event