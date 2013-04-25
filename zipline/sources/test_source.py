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

"""
A source to be used in testing.
"""

import pytz

from itertools import cycle, ifilter, izip
from datetime import datetime, timedelta
import numpy as np

from zipline.protocol import (
    Event,
    DATASOURCE_TYPE
)
from zipline.gens.utils import hash_args
from zipline.utils.tradingcalendar import trading_days


def create_trade(sid, price, amount, datetime, source_id="test_factory"):

    trade = Event()

    trade.source_id = source_id
    trade.type = DATASOURCE_TYPE.TRADE
    trade.sid = sid
    trade.dt = datetime
    trade.price = price
    trade.close = price
    trade.open = price
    trade.low = price * .95
    trade.high = price * 1.05
    trade.volume = amount
    trade.TRANSACTION = None

    return trade


def date_gen(start=datetime(2006, 6, 6, 12, tzinfo=pytz.utc),
             delta=timedelta(minutes=1),
             count=100,
             repeats=None):
    """
    Utility to generate a stream of dates.
    """
    one_day = timedelta(days=1)
    cur = start
    if delta == one_day:
        # if we are producing daily timestamps, we
        # use midnight
        cur = cur.replace(hour=0, minute=0, second=0,
                          microsecond=0)

    # yield count trade events, all on trading days, and
    # during trading hours.
    # NB: Being inside of trading hours is currently dependent upon the
    # count parameter being less than the number of trading minutes in a day
    for i in xrange(count):
        if repeats:
            for j in xrange(repeats):
                yield cur
        else:
            yield cur

        cur = cur + delta
        cur_midnight = cur.replace(hour=0, minute=0, second=0, microsecond=0)
        # skip over any non-trading days
        while cur_midnight not in trading_days:
            cur = cur + one_day
            cur_midnight = cur.replace(hour=0, minute=0, second=0,
                                       microsecond=0)
            cur = cur.replace(day=cur_midnight.day)


def mock_prices(count):
    """
    Utility to generate a stream of mock prices. By default
    cycles through values from 0.0 to 10.0, n times.
    """
    return (float(i % 10) + 1.0 for i in xrange(count))


def mock_volumes(count):
    """
    Utility to generate a set of volumes. By default cycles
    through values from 100 to 1000, incrementing by 50.
    """
    return ((i * 50) % 900 + 100 for i in xrange(count))


class SpecificEquityTrades(object):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    count  : integer representing number of trades
    sids   : list of values representing simulated internal sids
    start  : start date
    delta  : timedelta between internal events
    filter : filter to remove the sids
    """

    def __init__(self, *args, **kwargs):
        # We shouldn't get any positional arguments.
        assert len(args) == 0

        # Default to None for event_list and filter.
        self.event_list = kwargs.get('event_list')
        self.filter = kwargs.get('filter')

        if self.event_list is not None:
            # If event_list is provided, extract parameters from there
            # This isn't really clean and ultimately I think this
            # class should serve a single purpose (either take an
            # event_list or autocreate events).
            self.count = kwargs.get('count', len(self.event_list))
            self.sids = kwargs.get(
                'sids',
                np.unique([event.sid for event in self.event_list]).tolist())
            self.start = kwargs.get('start', self.event_list[0].dt)
            self.end = kwargs.get('start', self.event_list[-1].dt)
            self.delta = kwargs.get(
                'delta',
                self.event_list[1].dt - self.event_list[0].dt)
            self.concurrent = kwargs.get('concurrent', False)

        else:
            # Unpack config dictionary with default values.
            self.count = kwargs.get('count', 500)
            self.sids = kwargs.get('sids', [1, 2])
            self.start = kwargs.get(
                'start',
                datetime(2008, 6, 6, 15, tzinfo=pytz.utc))
            self.delta = kwargs.get(
                'delta',
                timedelta(minutes=1))
            self.concurrent = kwargs.get('concurrent', False)

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(*args, **kwargs)

        self.generator = self.create_fresh_generator()

    def __iter__(self):
        return self

    def next(self):
        return self.generator.next()

    def rewind(self):
        self.generator = self.create_fresh_generator()

    def get_hash(self):
        return self.__class__.__name__ + "-" + self.arg_string

    def update_source_id(self, gen):
        for event in gen:
            event.source_id = self.get_hash()
            yield event

    def create_fresh_generator(self):

        if self.event_list:
            event_gen = (event for event in self.event_list)
            unfiltered = self.update_source_id(event_gen)

        # Set up iterators for each expected field.
        else:
            if self.concurrent:
                # in this context the count is the number of
                # trades per sid, not the total.
                dates = date_gen(
                    count=self.count,
                    start=self.start,
                    delta=self.delta,
                    repeats=len(self.sids),
                )
            else:
                dates = date_gen(
                    count=self.count,
                    start=self.start,
                    delta=self.delta
                )

            prices = mock_prices(self.count)
            volumes = mock_volumes(self.count)

            sids = cycle(self.sids)

            # Combine the iterators into a single iterator of arguments
            arg_gen = izip(sids, prices, volumes, dates)

            # Convert argument packages into events.
            unfiltered = (create_trade(*args, source_id=self.get_hash())
                          for args in arg_gen)

        # If we specified a sid filter, filter out elements that don't
        # match the filter.
        if self.filter:
            filtered = ifilter(
                lambda event: event.sid in self.filter, unfiltered)

        # Otherwise just use all events.
        else:
            filtered = unfiltered

        # Return the filtered event stream.
        return filtered
