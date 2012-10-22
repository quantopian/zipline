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

"""
Tools to generate data sources.
"""

__all__ = ['DataFrameSource', 'SpecificEquityTrades']

import random
import pytz

from itertools import cycle, ifilter, izip
from datetime import datetime, timedelta
import pandas as pd
from copy import copy
import numpy as np

from zipline.protocol import DATASOURCE_TYPE
from zipline.utils import ndict
from zipline.gens.utils import hash_args, create_trade


def date_gen(start=datetime(2006, 6, 6, 12, tzinfo=pytz.utc),
             delta=timedelta(minutes=1),
             count=100,
             repeats=None):
    """
    Utility to generate a stream of dates.
    """
    if repeats:
        return (start + (i * delta)
                for i in xrange(count)
                for n in xrange(repeats))
    else:
        return (start + (i * delta) for i in xrange(count))


def mock_prices(count, rand=False):
    """
    Utility to generate a stream of mock prices. By default
    cycles through values from 0.0 to 10.0, n times.  Optional
    flag to give random values between 0.0 and 10.0
    """

    if rand:
        return (random.uniform(1.0, 10.0) for i in xrange(count))
    else:
        return (float(i % 10) + 1.0 for i in xrange(count))


def mock_volumes(count, rand=False):
    """
    Utility to generate a set of volumes. By default cycles
    through values from 100 to 1000, incrementing by 50.  Optional
    flag to give random values between 100 and 1000.
    """
    if rand:
        return (random.randrange(100, 1000) for i in xrange(count))
    else:
        return ((i * 50) % 900 + 100 for i in xrange(count))


def fuzzy_dates(count=500):
    """
    Add +-10 seconds to each event from a date_gen.  Note that this
    still guarantees sorting, since the default on date_gen is minute
    separation of events.
    """
    for date in date_gen(count=count):
        yield date + timedelta(seconds=random.randint(-10, 10))


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


class DataFrameSource(SpecificEquityTrades):
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

    def __init__(self, data, **kwargs):
        assert isinstance(data.index, pd.tseries.index.DatetimeIndex)

        self.data = data
        # Unpack config dictionary with default values.
        self.count = kwargs.get('count', len(data))
        self.sids = kwargs.get('sids', data.columns)
        self.start = kwargs.get('start', data.index[0])
        self.end = kwargs.get('end', data.index[-1])
        self.delta = kwargs.get('delta', data.index[1] - data.index[0])

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(data, **kwargs)

        self.generator = self.create_fresh_generator()

    def create_fresh_generator(self):
        def _generator(df=self.data):
            for dt, series in df.iterrows():
                if (dt < self.start) or (dt > self.end):
                    continue
                event = {'dt': dt,
                         'source_id': self.get_hash(),
                         'type': DATASOURCE_TYPE.TRADE
                }

                for sid, price in series.iterkv():
                    event = copy(event)
                    event['sid'] = sid
                    event['price'] = price
                    event['volume'] = 1000

                    yield ndict(event)

        # Return the filtered event stream.
        drop_sids = lambda x: x.sid in self.sids
        return ifilter(drop_sids, _generator())
