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

from datetime import datetime, timedelta
import itertools

from six.moves import range

from zipline.protocol import (
    Event,
)
from zipline.gens.utils import hash_args


def create_trade(sid, price, amount, datetime):

    trade = Event()

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
             trading_calendar,
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

        currently_executing = \
            (daily_delta and (cur in trading_calendar.all_sessions)) or \
            (trading_calendar.is_open_on_minute(cur))

        if currently_executing:
            return cur
        else:
            if daily_delta:
                return trading_calendar.minute_to_session_label(cur)
            else:
                return trading_calendar.open_and_close_for_session(
                    trading_calendar.minute_to_session_label(cur)
                )[0]

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
    def __init__(self, env, trading_calendar, *args, **kwargs):
        # We shouldn't get any positional arguments.
        assert len(args) == 0

        self.env = env
        self.trading_calendar = trading_calendar

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

        self.generator = self.create_fresh_generator()

    def __iter__(self):
        return self

    def next(self):
        return self.generator.next()

    def __next__(self):
        return next(self.generator)

    def rewind(self):
        self.generator = self.create_fresh_generator()

    def create_fresh_generator(self):

        # Set up iterators for each expected field.
        date_generator = date_gen(
            start=self.start,
            end=self.end,
            delta=self.delta,
            trading_calendar=self.trading_calendar,
        )

        return (
            create_trade(
                sid=sid,
                price=float(i % 10) + 1.0,
                amount=(i * 50) % 900 + 100,
                datetime=date,
            ) for (i, date), sid in itertools.product(
                enumerate(date_generator), self.sids
            )
        )
