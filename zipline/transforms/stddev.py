#
# Copyright 2012 Quantopian, Inc.
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

from numbers import Number
from collections import defaultdict
from math import sqrt

from zipline import ndict
from zipline.transforms.utils import EventWindow, TransformMeta


class MovingStandardDev(object):
    """
    Class that maintains a dictionary from sids to
    MovingStandardDevWindows.  For each sid, we maintain standard 
    deviations over any number of distinct fields. (For example, we can
    maintain a sid's moving standard deviation of returns as well as its
    moving standard deviation of prices.
    """
    __metaclass__ = TransformMeta

    def __init__(self, fields,
                 market_aware=True, window_length=None, delta=None):

        self.fields = fields        
        self.market_aware = market_aware

        self.delta = delta
        self.window_length = window_length

        # Market-aware mode only works with full-day windows.
        if self.market_aware:
            # Window length must be 1 or greater
            assert self.window_length >= 1

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
        return MovingStandardDevWindow(
            self.fields,            
            self.market_aware,
            self.window_length,
            self.delta
        )

    def update(self, event):
        """
        Update the event window for this event's sid.  Return an ndict
        from tracked fields to moving standard deviations.
        """
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_stddevs()


class MovingStandardDevWindow(EventWindow):
    """
    Iteratively calculates moving standard deviations for a particular sid
    over a given time window. We can maintain standard deviations for 
    arbitrarily many fields on a single sid. (For example, we might track 
    moving standard deviation of returns as well as its moving standard 
    deviation of prices.) The expected functionality of this class is to be 
    instantiated inside a MovingStandardDev.
    """

    def __init__(self, fields, market_aware, window_length, delta):
        # Call the superclass constructor to set up base EventWindow
        # infrastructure.
        EventWindow.__init__(self, market_aware, window_length, delta)

        self.fields = fields
        self.sum = defaultdict(float)
        self.sum_sqr = defaultdict(float)

    def handle_add(self, event):
        # Sanity check on the event.
        self.assert_required_fields(event)

        # Increment our running totals with data from the event.
        for field in self.fields:
            self.sum[field] += event[field]
            self.sum_sqr[field] += event[field] ** 2

    def handle_remove(self, event):
        # Sanity check on the event.
        self.assert_required_fields(event)

        # Decrement our running totals with data from the event.
        for field in self.fields:
            self.sum[field] -= event[field]
            self.sum_sqr[field] -= event[field] ** 2

    def stdev(self, field):
        """
        Calculate the standard deviation of our ticks over a single field
        using a naive algorithm (see http://goo.gl/wPFtf).
        """
        # Sanity check.
        assert field in self.fields
        # Standard deviation is undefined for no event and 0 for one event
        if len(self.ticks) <= 1:
            return None

        # Calculate and return the standard deviation. 
        else:
            _mean = self.sum[field] / len(self.ticks)
            _var = (self.sum_sqr[field] - 
                self.sum[field] * _mean) / (len(self.ticks) - 1)
            return sqrt(_var)

    def get_stddevs(self):
        """
        Return an ndict of all our tracked standard deviations.
        """
        out = ndict()
        for field in self.fields:
            out[field] = self.stdev(field)
        return out

    def assert_required_fields(self, event):
        """
        We only allow events with all of our tracked fields.
        """
        for field in self.fields:
            assert field in event, \
                "Event missing [%s] in MovingStandardDevEventWindow" % field
            assert isinstance(event[field], Number), \
                "Got %s for %s in MovingStandardDevEventWindow" \
                % (event[field], field)
