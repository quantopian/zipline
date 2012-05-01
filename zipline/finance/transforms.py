from datetime import timedelta
from itertools import ifilter
from collections import defaultdict

from zipline.messaging import BaseTransform

class VWAPTransform(BaseTransform):
    
    def init(self, daycount=3):
        self.daycount = daycount
        
        
class DailyVWAP:
    """A class that tracks the volume weighted average price
       based on tick updates."""
    def __init__(self, daycount=3):
        self.ticks = []
        self.dropped_ticks = []
        self.flux = 0.0
        self.volume = 0
        self.lastTick = None
        self.vwap = 0.0
        self.delta = timedelta(days=daycount)
    
    def update(self, event):
        
        self.ticks.append(event)
        flux, volume = self.calculate_flux([event])
        self.flux += flux
        self.volume += volume
        
        self.last_date = event['dt']
        self.first_date = self.last_date - self.delta
        #use a list comprehension to filter the ticks to those within 
        #desired day range. The dt properties are full datetime objects
        #and provide overloads for arithmetic operations. 
        self.dropped_ticks = []
        for tick in self.ticks:
            if tick['dt'] < self.first_date:
                self.dropped_ticks.append(tick)
              
        slice_index = len(self.dropped_ticks)      
        self.ticks = self.ticks[slice_index:]

        dropped_flux, dropped_volume = self.calculate_flux(self.dropped_ticks)
        
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