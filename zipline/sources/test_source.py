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

from six.moves import filter
from datetime import datetime, timedelta
import itertools

from six.moves import range

from zipline.protocol import (
    Event,
    DATASOURCE_TYPE
)
from zipline.gens.utils import hash_args


def create_trade(sid, price, amount, datetime, source_id="test_factory"):

    trade = Event()

    trade.source_id = source_id
    trade.type = DATASOURCE_TYPE.TRADE
    trade.sid = sid
    trade.dt = datetime
    trade.price = price
    trade.close_price = price
    trade.open_price = price
    trade.low = price * .95
    trade.high = price * 1.05
    trade.volume = amount

    return trade


def date_gen(start,
             end,
             env,
             delta=timedelta(minutes=1),
             repeats=None):
    """
    Utility to generate a stream of dates.
    """
    daily_delta = not (delta.total_seconds()
                       % timedelta(days=1).total_seconds())
    cur = start
    if daily_delta:
        # if we are producing daily timestamps, we
        # use midnight
        cur = cur.replace(hour=0, minute=0, second=0,
                          microsecond=0)

    def advance_current(cur):
        """
        Advances the current dt skipping non market days and minutes.
        """
        cur = cur + delta

        if not (env.is_trading_day
                if daily_delta
                else env.is_market_hours)(cur):
            if daily_delta:
                return env.next_trading_day(cur)
            else:
                return env.next_open_and_close(cur)[0]
        else:
            return cur

    # yield count trade events, all on trading days, and
    # during trading hours.
    while cur < end:
        if repeats:
            for j in range(repeats):
                yield cur
        else:
            yield cur

        cur = advance_current(cur)


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
    def __init__(self, env, *args, **kwargs):
        # We shouldn't get any positional arguments.
        assert len(args) == 0

        self.env = env

        # Default to None for event_list and filter.
        self.event_list = kwargs.get('event_list')
        self.filter = kwargs.get('filter')
        if self.event_list is not None:
            # If event_list is provided, extract parameters from there
            # This isn't really clean and ultimately I think this
            # class should serve a single purpose (either take an
            # event_list or autocreate events).
            self.count = kwargs.get('count', len(self.event_list))
            self.start = kwargs.get('start', self.event_list[0].dt)
            self.end = kwargs.get('end', self.event_list[-1].dt)
            self.delta = delta = kwargs.get('delta')
            if delta is None:
                self.delta = self.event_list[1].dt - self.event_list[0].dt
            self.concurrent = kwargs.get('concurrent', False)

            self.identifiers = kwargs.get(
                'sids',
                set(event.sid for event in self.event_list)
            )
            assets_by_identifier = {}
            for identifier in self.identifiers:
                assets_by_identifier[identifier] = env.asset_finder.\
                    lookup_generic(identifier, datetime.now())[0]
            self.sids = [asset.sid for asset in assets_by_identifier.values()]
            for event in self.event_list:
                event.sid = assets_by_identifier[event.sid].sid

        else:
            # Unpack config dictionary with default values.
            self.count = kwargs.get('count', 500)
            self.start = kwargs.get(
                'start',
                datetime(2008, 6, 6, 15, tzinfo=pytz.utc))
            self.end = kwargs.get(
                'end',
                datetime(2008, 6, 6, 15, tzinfo=pytz.utc))
            self.delta = kwargs.get(
                'delta',
                timedelta(minutes=1))
            self.concurrent = kwargs.get('concurrent', False)

            self.identifiers = kwargs.get('sids', [1, 2])
            assets_by_identifier = {}
            for identifier in self.identifiers:
                assets_by_identifier[identifier] = env.asset_finder.\
                    lookup_generic(identifier, datetime.now())[0]
            self.sids = [asset.sid for asset in assets_by_identifier.values()]

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(*args, **kwargs)

        self.generator = self.create_fresh_generator()

    def __iter__(self):
        return self

    def next(self):
        return self.generator.next()

    def __next__(self):
        return next(self.generator)

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
                date_generator = date_gen(
                    start=self.start,
                    end=self.end,
                    delta=self.delta,
                    repeats=len(self.sids),
                    env=self.env,
                )
            else:
                date_generator = date_gen(
                    start=self.start,
                    end=self.end,
                    delta=self.delta,
                    env=self.env,
                )

            source_id = self.get_hash()

            unfiltered = (
                create_trade(
                    sid=sid,
                    price=float(i % 10) + 1.0,
                    amount=(i * 50) % 900 + 100,
                    datetime=date,
                    source_id=source_id,
                ) for (i, date), sid in itertools.product(
                    enumerate(date_generator), self.sids
                )
            )

        # If we specified a sid filter, filter out elements that don't
        # match the filter.
        if self.filter:
            filtered = filter(
                lambda event: event.sid in self.filter, unfiltered)

        # Otherwise just use all events.
        else:
            filtered = unfiltered

        # Return the filtered event stream.
        return filtered
