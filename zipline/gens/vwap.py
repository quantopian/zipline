from numbers import Number
from datetime import datetime, timedelta
from collections import defaultdict

from zipline import ndict
from zipline.gens.transform import EventWindow

class VWAP(object):
    """
    Class that maintains a dictionary from sids to VWAPEventWindows.
    """
    def __init__(self, delta):
        self.delta = delta

        # No way to pass arguments to the defaultdict factory, so we
        # need to define a method to generate the correct EventWindows.
        self.sid_windows = defaultdict(self.create_window)

    def create_window(self):
        """Factory method for self.sid_windows."""
        return VWAPEventWindow(self.delta)

    def update(self, event):
        """
        Update the event window for this event's sid. Returns the
        current vwap for the sid.
        """
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_vwap()


class VWAPEventWindow(EventWindow):
    """
    Iteratively maintains a vwap for a single sid over a given
    timedelta.
    """
    def __init__(self, delta):
        EventWindow.__init__(self, delta)
        self.flux = 0.0
        self.totalvolume = 0.0

    # Subclass customization for adding new events.
    def handle_add(self, event):
        # Sanity check on the event.
        self.assert_required_fields(event)
        self.flux += event.volume * event.price
        self.totalvolume += event.volume

    # Subclass customization for removing expired events.
    def handle_remove(self, event):
        self.flux -= event.volume * event.price
        self.totalvolume -= event.volume
    
    def get_vwap(self):
        """
        Return the calculated vwap for this sid.
        """
        # By convention, vwap is None if we have no events.
        if len(self.ticks) == 0:
            return None
        else:
            return (self.flux / self.totalvolume)

    # We need numerical price and volume to calculate a vwap.
    def assert_required_fields(self, event):
        assert isinstance(event.price, Number)
        assert isinstance(event.volume, Number)
