import pandas
from datetime import timedelta
from collections import defaultdict

from zipline.messaging import BaseTransform
from zipline.finance.movingaverage import EventWindow, EventHistory

class VWAPTransform(BaseTransform):
    
    def init(self, daycount=3):
        self.daycount = daycount
        self.by_sid = defaultdict(self.create_vwap)
        
    def transform(self, event):
        cur = self.by_sid(event.sid)
        cur.update(event)
        self.state['value'] = cur.vwap
        return self.state
    
    def create_vwap(self):
        return DailyVWAP(self.daycount)

class DailyVWAP:
    """A class that tracks the volume weighted average price
       based on tick updates."""
    def __init__(self, daycount):
        self.window = EventWindow(daycount)
        self.flux = 0.0
        self.volume = 0
        self.vwap = 0.0
        self.delta = timedelta(days=daycount)

    def update(self, event):

        # update the event window
        self.window.update(event)

        # add the current event's flux and volume to the tracker
        flux, volume = self.calculate_flux([event])
        self.flux += flux
        self.volume += volume

        # subract the expired events flux and volume from the tracker
        dropped = self.window.dropped_ticks
        dropped_flux, dropped_volume = self.calculate_flux(dropped)

        self.flux -= dropped_flux
        self.volume -= dropped_volume

        if(self.volume != 0):
            self.vwap = self.flux / self.volume
        else:
            self.vwap = None

    def calculate_flux(self, ticks):
        flux = 0.0
        volume = 0
        for tick in ticks:
            flux += tick['volume'] * tick['price']
            volume += tick['volume']
        return flux, volume


# ------------------------------
# Experimental
# ------------------------------
        
class DailyVWAP_df(object):
    
    def __init__(self, daycount=3):
        self.history = EventHistory(daycount)
        self.vwap = None
        
    def update(self, event):
        self.history.update(event)
        frame = self.history.frame
        
        window = len(frame)
        value = pandas.rolling_sum(
            frame['price'] * frame['volume'],
            window
        )
        volume = pandas.rolling_sum(
            frame['volume'],
            window
        )
        
        vwap = value / volume
        self.vwap = vwap[-1]
        
