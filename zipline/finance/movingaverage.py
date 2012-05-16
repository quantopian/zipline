from datetime import timedelta
from collections import defaultdict

from zipline.transforms.base import BaseTransform

class MovingAverageTransform(BaseTransform):

    def init(self, daycount=3):
        self.daycount = daycount
        self.by_sid = defaultdict(self._create)

    def transform(self, event):
        cur = self.by_sid[event.sid]
        cur.update(event)
        self.state['value'] = cur.average
        return self.state

    def _create(self):
        return MovingAverage(self.daycount)

class MovingAverage(object):

    def init(self, daycount):
        self.window = EventWindow(daycount)
        self.total = 0.0
        self.average = 0.0

    def update(self, event):
        self.window.update(event)

        self.total += event.price

        for dropped in self.window.dropped_ticks:
            self.total -= dropped.price

        if len(self.window.ticks) > 0:
            self.average = self.total / len(self.window.ticks)
        else:
            self.average = 0.0

class EventWindow(object):
    """
    Tracks a window of the event history. Use an instance to track the events
    inside your window to efficiently calculate rolling statistics.
    """
    def init(self, daycount):
        self.ticks = []
        self.dropped_ticks = []
        self.delta = timedelta(days=daycount)

    def update(self, event):
        # add new event
        self.ticks.append(event)
        # determine which events are expired
        last_date = event['dt']
        first_date = last_date - self.delta

        self.dropped_ticks = []
        for tick in self.ticks:
            if tick['dt'] <= first_date:
                self.dropped_ticks.append(tick)

        # remove the expired events
        slice_index = len(self.dropped_ticks)
        self.ticks = self.ticks[slice_index:]
