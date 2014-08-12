#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from six import with_metaclass

from zipline.transforms.utils import EventWindow, TransformMeta


class MovingVWAP(with_metaclass(TransformMeta)):
    """
    Class that maintains a dictionary from sids to VWAPEventWindows.
    """

    def __init__(self, market_aware=True, delta=None, window_length=None):

        self.market_aware = market_aware
        self.delta = delta
        self.window_length = window_length

        # Market-aware mode only works with full-day windows.
        if self.market_aware:
            assert self.window_length and not self.delta,\
                "Market-aware mode only works with full-day windows."

        # Non-market-aware mode requires a timedelta.
        else:
            assert self.delta and not self.window_length, \
                "Non-market-aware mode requires a timedelta."

        # No way to pass arguments to the defaultdict factory, so we
        # need to define a method to generate the correct EventWindows.
        self.sid_windows = defaultdict(self.create_window)

    def create_window(self):
        """Factory method for self.sid_windows."""
        return VWAPEventWindow(
            self.market_aware,
            window_length=self.window_length,
            delta=self.delta
        )

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
    def __init__(self, market_aware=True, window_length=None, delta=None):
        EventWindow.__init__(self, market_aware, window_length, delta)
        self._fields = ('price', 'volume')
        self.flux = 0.0
        self.totalvolume = 0.0

    @property
    def fields(self):
        return self._fields

    # Subclass customization for adding new events.
    def handle_add(self, event):
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
