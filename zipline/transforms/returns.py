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

from zipline.errors import WrongDataForTransform
from zipline.transforms.utils import TransformMeta
from collections import defaultdict, deque


class Returns(object):
    """
    Class that maintains a dictionary from sids to the sid's
    closing price N trading days ago.
    """
    __metaclass__ = TransformMeta

    def __init__(self, window_length):
        self.window_length = window_length
        self.mapping = defaultdict(self._create)

    def update(self, event):
        """
        Update and return the calculated returns for this event's sid.
        """
        tracker = self.mapping[event.sid]
        tracker.update(event)

        return tracker.returns

    def _create(self):
        return ReturnsFromPriorClose(
            self.window_length
        )


class ReturnsFromPriorClose(object):
    """
    Records the last N closing events for a given security as well as the
    last event for the security.  When we get an event for a new day, we
    treat the last event seen as the close for the previous day.
    """

    def __init__(self, window_length):
        self.closes = deque()
        self.last_event = None
        self.returns = 0.0
        self.window_length = window_length

    def update(self, event):
        self.assert_required_fields(event)
        if self.last_event:

            # Day has changed since the last event we saw.  Treat
            # the last event as the closing price for its day and
            # clear out the oldest close if it has expired.
            if self.last_event.dt.date() != event.dt.date():

                self.closes.append(self.last_event)

                # We keep an event for the end of each trading day, so
                # if the number of stored events is greater than the
                # number of days we want to track, the oldest close
                # is expired and should be discarded.
                while len(self.closes) > self.window_length:
                    # Pop the oldest event.
                    self.closes.popleft()

        # We only generate a return value once we've seen enough days
        # to give a sensible value.  Would be nice if we could query
        # db for closes prior to our initial event, but that would
        # require giving this transform database creds, which we want
        # to avoid.

        if len(self.closes) == self.window_length:
            last_close = self.closes[0].price
            change = event.price - last_close
            self.returns = change / last_close

        # the current event is now the last_event
        self.last_event = event

    def assert_required_fields(self, event):
        """
        We only allow events with a price field to be run through
        the returns transform.
        """
        if 'price' not in event:
            raise WrongDataForTransform(
                transform="ReturnsEventWindow",
                fields='price')
