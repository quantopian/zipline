"""
Transformations for common technical indicators.
TODO: add MACD transform
TODO: add trailing stop

"""
import datetime
import qsim.util as qutil

from qsim.transforms.core import BaseTransform

class MovingAverage(BaseTransform):
    """
        Calculate a unweighted moving average for props['sid'] security
        TODO: add sid filter.
    """
    
    def __init__(self, feed, props, result_address): 
        BaseTransform.__init__(self, feed, props, result_address)
        self.events = []
        
        self.window = datetime.timedelta(days           = self.config.get_integer('days'), 
                                        seconds         = self.config.get_integer('seconds'), 
                                        microseconds    = self.config.get_integer('microseconds'), 
                                        milliseconds    = self.config.get_integer('milliseconds'),
                                        minutes         = self.config.get_integer('minutes'),
                                        hours           = self.config.get_integer('hours'),
                                        weeks           = self.config.get_integer('weeks'))
    
        
  
        
    def transform(self, event):
        """Update the moving average with the latest data point."""
        
        self.events.append(event)
        
        #filter the event list to the window length.
        self.events = [x for x in self.events if (qutil.parse_date(x['dt']) - qutil.parse_date(event['dt'])) <= self.window]
        
        if(len(self.events) == 0):
            return 0.0
            
        total = 0.0
        for event in self.events:
            total += event['price']
        
        self.average = total/len(self.events)
        
        self.state['value'] = self.average
        
        return self.state
        