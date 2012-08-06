
from numbers import Number
from datetime import datetime, timedelta
from collections import defaultdict

from zipline import ndict
from zipline.gens.transform import EventWindow

class MovingAverage(object):
    """
    Class that maintains a dictionary from sids to EventWindows
    Upon receipt of each message we update the
    corresponding window and return the calculated average.
    """

    def __init__(self, delta, fields):
        self.delta = delta
        self.fields = fields

        # No way to pass arguments to the defaultdict factory, so we
        # need to define a method to generate the correct EventWindows.
        self.sid_windows = defaultdict(self.create_window)

    def create_window(self):
        """Factory method for self.sid_windows."""
        return MovingAverageEventWindow(self.delta, self.fields)

    def update(self, event):
        """
        Update the event window for this event's sid.  Return an ndict from
        tracked fields to averages.
        """
        assert isinstance(event, ndict),"Bad event in MovingAverage: %s" % event
        assert event.has_key('sid'), "No sid in MovingAverage: %s" % event
        assert event.has_key('dt'), "No dt in MovingAverage: %s" % event

        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_averages()

class MovingAverageEventWindow(EventWindow):
    """
    Calculates a moving average over all specified fields.
    """
    # Subclass initializer.  The superclass also requires a timedelta
    # argument, so instantiation should look like:
    # mavg = MovingAverageEventWindow(timedelta(minutes=1), ['price']) 
    def init(self, fields):
        self.fields = fields
        self.totals = defaultdict(float)

    def handle_add(self, event):
        # Sanity check on the event.
        self.assert_all_fields(event)
        # Increment our running totals with data from the event.
        for field in self.fields:
            self.totals[field] += event[field]

    def handle_remove(self, event):
        # Decrement our running totals with data from the event.
        for field in self.fields:
            self.totals[field] -= event[field]

    def average(self, field):
        """
        Calculate the average value of our ticks over a given field.
        """
        # Sanity check.
        assert field in self.fields

        # Averages are 0 by convention if we have no ticks.
        if len(self.ticks) == 0:
            return 0.0
        
        # Calculate and return the average.  len(self.ticks) is O(1).
        else:
            return self.totals[field] / len(self.ticks)

    def get_averages(self):
        """
        Return an ndict of all our tracked averages.
        """
        out = ndict()
        
        for field in self.fields:
            out[field] = self.average(field)
        return out

    def assert_all_fields(self, event):
        """
        We only track events with all the fields we care about.
        """
        for field in self.fields:
            assert event.has_key(field), \
                "Event missing [%s] in MovingAverageEventWindow" % field
            assert isinstance(event[field], Number), \
                "Got %s for %s in MovingAverageEventWindow" % (event[field], field)
