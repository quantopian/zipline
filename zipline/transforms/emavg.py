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

from zipline.transforms.utils import EventWindow, TransformMeta


class ExponentialMovingAverage(object):

    __metaclass__ = TransformMeta

    def __init__(self, fields='price',
                 market_aware=True, window_length=None, delta=None):
        if isinstance(fields, basestring):
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
        return ExponentialMovingAverageEventWindow(
            self.fields,
            self.market_aware,
            self.window_length,
            self.delta
        )

    def update(self, event):
        """
        Update the event window for this event's sid.  Return an ndict
        from tracked fields to moving averages.
        """
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_emas()


class Averages(object):
    """
    Container for averages.
    """

    def __getitem__(self, name):
        """
        Allow dictionary lookup.
        """
        return self.__dict__[name]


class ExponentialMovingAverageEventWindow(EventWindow):
    """
    Iteratively calculates an exponentially weighted moving average
    for a particular sid over a given time window.  The time window
    must be days or minutes.  The expected functionality of this class
    is to be instantiated inside an ExponentialMovingAverage transform.
    """

    def __init__(self, fields, market_aware, days, delta):

        EventWindow.__init__(self, market_aware, days, delta)

        self.fields = fields
        self.emas = defaultdict(list)

        if market_aware is True:
            # If we are market aware, we only operate with full day windows
            periods = round((2.0 / (self.window_length + 1)), 4)
        else:
            # If we are not market aware, we need to figure out how many
            # periods we are using to calculate the EMA. This assumes day
            # long periods or minute long periods, anything else will give
            # broken results.
            if delta.days == 0:
                periods = delta.seconds / 60
            else:
                periods = delta.days

        if periods < 0.000001:
            # If we somehow end up with a near 0 period, use 50 as the default
            periods = 50

        self.multiplier = round((2.0 / (periods + 1)), 4)

    def handle_add(self, event):
        # Sanity check on the event.
        self.assert_required_fields(event)

        # Calculate the current EMA based on the event value
        # If the field is empty, use the event's value to get started
        for field in self.fields:
            if not self.emas[field]:
                self.emas[field] = event[field]
            else:
                prev_ema = self.emas[field]
                self.emas[field] = (event[field] - prev_ema) * \
                    self.multiplier + prev_ema

    # We only store one data point, the EMA, so we never remove anything
    def handle_remove(self, event):
        pass

    def get_emas(self):
        """
        Return an ndict of all our tracked averages.
        """
        out = Averages()
        for field in self.fields:
            out.__dict__[field] = round(self.emas[field], 2)
        return out

    def assert_required_fields(self, event):
        """
        We only allow events with all of our tracked fields.
        """
        for field in self.fields:
            assert isinstance(event[field], Number), \
                "Got %s for %s in ExponentialMovingAverageEventWindow" % \
                (event[field], field)
