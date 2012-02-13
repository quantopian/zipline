"""
Transformations for common technical indicators.
TODO: add MACD transform
TODO: add trailing stop

"""
import datetime
from qsim.core import BaseTransform

class MovingAverage(BaseTransform):
    """
        Calculate a unweighted moving average for props['sid'] security
        TODO: add sid -> mvavg dict.
    """
    
    def __init__(self, name, days): 
        BaseTransform.__init__(self, name)
        self.events = []
        
        self.window = datetime.timedelta(days = days)
    
        
  
        
    def transform(self, event):
        """Update the moving average with the latest data point."""
        
        #self.events.append(event)
        
        #filter the event list to the window length.
        #self.events = [x for x in self.events if (qutil.parse_date(x['dt']) - qutil.parse_date(event['dt'])) <= self.window]
        
        #if(len(self.events) == 0):
        #    return 0.0
            
        #total = 0.0
        #for event in self.events:
        #    total += event['price']
        
        #self.average = total/len(self.events)
        
        #self.state['value'] = self.average
        self.state['value'] = 10
        return self.state
        