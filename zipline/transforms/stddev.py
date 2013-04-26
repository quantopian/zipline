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
from math import sqrt

from zipline.errors import WrongDataForTransform
from zipline.transforms.utils import EventWindow, TransformMeta
import zipline.utils.math_utils as zp_math


class MovingStandardDev(object):
    """
    Class that maintains a dictionary from sids to
    MovingStandardDevWindows.  For each sid, we maintain a the
    standard deviation of all events falling within the specified
    window.
    """
    __metaclass__ = TransformMeta

    def __init__(self, market_aware=True, window_length=None, delta=None):

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
        return MovingStandardDevWindow(
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
        return window.get_stddev()

    def assert_required_fields(self, event):
        """
        We only allow events with a price field to be run through
        the returns transform.
        """
        if 'price' not in event:
                raise WrongDataForTransform(
                    transform="StdDevEventWindow",
                    fields='price')


class MovingStandardDevWindow(EventWindow):
    """
    Iteratively calculates standard deviation for a particular sid
    over a given time window.  The expected functionality of this
    class is to be instantiated inside a MovingStandardDev.
    """

    def __init__(self, market_aware=True, window_length=None, delta=None):
        # Call the superclass constructor to set up base EventWindow
        # infrastructure.
        EventWindow.__init__(self, market_aware, window_length, delta)

        self.sum = 0.0
        self.sum_sqr = 0.0

    def handle_add(self, event):
        self.sum += event.price
        self.sum_sqr += event.price ** 2

    def handle_remove(self, event):
        self.sum -= event.price
        self.sum_sqr -= event.price ** 2

    def get_stddev(self):
        # Sample standard deviation is undefined for a single event or
        # no events.
        if len(self) <= 1:
            return None

        else:
            average = self.sum / len(self)
            s_squared = (self.sum_sqr - self.sum * average) \
                / (len(self) - 1)

            if zp_math.tolerant_equals(0, s_squared):
                return 0.0
            stddev = sqrt(s_squared)
        return stddev
