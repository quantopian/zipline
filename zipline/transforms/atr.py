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

from six import with_metaclass

from zipline.transforms.utils import EventWindow, TransformMeta


class AverageTrueRange(with_metaclass(TransformMeta)):
    """
    Class that maintains a dictionary from sids to
    ATREventWindows.  For each sid, we maintain the ATR.
    """

    def __init__(self, window_length=14, market_aware=True, delta=None):
        # Consistent with Ta-Lib, use 14 day default window length
        self.window_length = window_length
        self.market_aware = market_aware
        self.delta = delta

        # Keep track of prior closing price
        self.prior_close = None

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
        return ATREventWindow(
            self.market_aware,
            self.window_length,
            self.delta
        )

    def update(self, event):
        """
        Update the event window for this event's sid.  Return a dict
        from tracked fields to ATRs.
        """
        # This will create a new EventWindow if this is the first
        # message for this sid.
        window = self.sid_windows[event.sid]
        window.update(event)
        return window.get_atr()


class ATRs(object):
    """
    Container for ATRs.
    """

    def __getitem__(self, name):
        """
        Allow dictionary lookup.
        """
        return self.__dict__[name]


class ATREventWindow(EventWindow):
    """
    Iteratively calculates ATRs for a particular sid over a
    given time window.  We can maintain ATRs for arbitrarily many
    fields on a single sid.  (For example, we might track ATR of
    price as well as ATR of volume for a single sid.) The expected
    functionality of this class is to be instantiated inside an
    ATR transform.
    """

    def __init__(self, market_aware, days, delta):

        # Call the superclass constructor to set up base EventWindow
        # infrastructure.
        EventWindow.__init__(self, market_aware, days, delta)

        self.market_aware = market_aware
        self.days = days
        self.delta = delta
        self.atr = 0.0
        self.prior_close = None

    # Subclass customization for adding new events.
    def handle_add(self, event):

        # Calculate True Range
        if self.prior_close is not None:
            # Calculate true range using current high & low and prior close
            tr = max(event.high, self.prior_close) - \
                min(event.low, self.prior_close)

            # Calculate Average True Range
            if len(self.ticks) <= self.days:
                # Just return the simple average during warm-up period
                self.atr = (self.atr * len(self.ticks - 1) + tr) \
                    / len(self.ticks)
            else:
                # atr = prior atr * (days-1)/days + tr/days
                self.atr *= (self.days - 1) / self.days
                self.atr += tr / self.days

        # Save closing price for next tick's calculation
        self.prior_close = event.close_price

    # Subclass customization for removing expired events.
    def handle_remove(self, event):
        # Not required.
        pass

    def get_atr(self):
        """
        Calculate the average true range of our ticks over a single field.
        """
        return self.atr
