"""
Transformations for common technical indicators.
TODO: add MACD transform
TODO: add trailing stop
"""

import datetime
from zipline.messaging import BaseTransform
import zipline.util as qutil

class MovingAverage(BaseTransform):
    """
    Calculate a unweighted moving average for props['sid'] security
    TODO: add sid -> mvavg dict.
    """

    def __init__(self, name, days):
        BaseTransform.__init__(self, name)
        self.events         = []
        self.current_total  = 0
        self.window         = datetime.timedelta(days = days)

    def transform(self, event):
        """Update the moving average with the latest data point."""

        self.events.append(event)
        self.current_total += event['price']
        event_date = qutil.parse_date(event['dt'])

        index = 0
        for cur_event in self.events:
            cur_date = qutil.parse_date(cur_event['dt'])
            if(cur_date - event_date):
                self.events.pop(index)
                self.current_total -= cur_event['price']
                index += 1
            else:
                break

        if(len(self.events) == 0):
            return 0.0

        self.average = self.current_total/len(self.events)

        self.state['value'] = self.average
        return self.state

