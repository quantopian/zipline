#
# Copyright 2013 Quantopian, Inc.
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

from six import string_types, with_metaclass

from zipline.transforms.utils import EventWindow, TransformMeta


class MovingAverage(with_metaclass(TransformMeta)):
    """
    Class that maintains a dictionary from sids to
    MovingAverageEventWindows.  For each sid, we maintain moving
    averages over any number of distinct fields (For example, we can
    maintain a sid's average volume as well as its average price.)
    """

    def __init__(self, fields='price',
                 market_aware=True, window_length=None, delta=None):

        if isinstance(fields, string_types):
            fields = [fields]
        self.fields = fields

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
        """
        Factory method for self.sid_windows.
        """
        return MovingAverageEventWindow(
            self.fields,
            self.market_aware,
            self.window_length,
            self.delta
        )

    def update(self, event):
        """
        Update the event window for this event's sid.  Return a dict
        from tracked fields to moving averages.
        """
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_averages()


class Averages(object):
    """
    Container for averages.
    """

    def __getitem__(self, name):
        """
        Allow dictionary lookup.
        """
        return self.__dict__[name]


class MovingAverageEventWindow(EventWindow):
    """
    Iteratively calculates moving averages for a particular sid over a
    given time window.  We can maintain averages for arbitrarily many
    fields on a single sid.  (For example, we might track average
    price as well as average volume for a single sid.) The expected
    functionality of this class is to be instantiated inside a
    MovingAverage transform.
    """

    def __init__(self, fields, market_aware, days, delta):

        # Call the superclass constructor to set up base EventWindow
        # infrastructure.
        EventWindow.__init__(self, market_aware, days, delta)

        # We maintain a dictionary of totals for each of our tracked
        # fields.
        self._fields = fields
        self.totals = defaultdict(float)

    @property
    def fields(self):
        return self._fields

    # Subclass customization for adding new events.
    def handle_add(self, event):
        # Sanity check on the event.
        # Increment our running totals with data from the event.
        for field in self.fields:
            self.totals[field] += event[field]

    # Subclass customization for removing expired events.
    def handle_remove(self, event):
        # Decrement our running totals with data from the event.
        for field in self.fields:
            self.totals[field] -= event[field]

    def average(self, field):
        """
        Calculate the average value of our ticks over a single field.
        """
        # Sanity check.
        assert field in self.fields

        # Averages are None by convention if we have no ticks.
        if len(self.ticks) == 0:
            return 0.0

        # Calculate and return the average.  len(self.ticks) is O(1).
        else:
            return self.totals[field] / len(self.ticks)

    def get_averages(self):
        """
        Return a dict of all our tracked averages.
        """
        out = Averages()
        for field in self.fields:
            out.__dict__[field] = self.average(field)
        return out
