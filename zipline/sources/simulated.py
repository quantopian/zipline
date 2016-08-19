#
# Copyright 2014 Quantopian, Inc.
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

from copy import copy
import six

import numpy as np
from datetime import timedelta
import pandas as pd

from zipline.sources.data_source import DataSource
from zipline.utils import tradingcalendar as calendar_nyse
from zipline.gens.utils import hash_args


class RandomWalkSource(DataSource):
    """RandomWalkSource that emits events with prices that follow a
    random walk. Will generate valid datetimes that match market hours
    of the supplied calendar and can generate emit events with
    user-defined frequencies (e.g. minutely).

    """
    VALID_FREQS = frozenset(('daily', 'minute'))

    def __init__(self, start_prices=None, freq='minute', start=None,
                 end=None, drift=0.1, sd=0.1, calendar=calendar_nyse):
        """
        :Arguments:
            start_prices : dict
                 sid -> starting price.
                 Default: {0: 100, 1: 500}
            freq : str <default='minute'>
                 Emits events according to freq.
                 Can be 'daily' or 'minute'
            start : datetime <default=start of calendar>
                 Start dt to emit events.
            end : datetime <default=end of calendar>
                 End dt until to which emit events.
            drift: float <default=0.1>
                 Constant drift of the price series.
            sd: float <default=0.1>
                 Standard deviation of the price series.
            calendar : calendar object <default: NYSE>
                 Calendar to use.
                 See zipline.utils for different choices.

        :Example:
            # Assumes you have instantiated your Algorithm
            # as myalgo.
            myalgo = MyAlgo()
            source = RandomWalkSource()
            myalgo.run(source)

        """
        # Hash_value for downstream sorting.
        self.arg_string = hash_args(start_prices, freq, start, end,
                                    calendar.__name__)

        if freq not in self.VALID_FREQS:
            raise ValueError('%s not in %s' % (freq, self.VALID_FREQS))

        self.freq = freq
        if start_prices is None:
            self.start_prices = {0: 100,
                                 1: 500}
        else:
            self.start_prices = start_prices

        self.calendar = calendar
        if start is None:
            self.start = calendar.start
        else:
            self.start = start
        if end is None:
            self.end = calendar.end_base
        else:
            self.end = end

        self.drift = drift
        self.sd = sd

        self.sids = self.start_prices.keys()

        self.open_and_closes = \
            calendar.open_and_closes[self.start:self.end]

        self._raw_data = None

    @property
    def instance_hash(self):
        return self.arg_string

    @property
    def mapping(self):
        return {
            'dt': (lambda x: x, 'dt'),
            'sid': (lambda x: x, 'sid'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
            'open_price': (float, 'open_price'),
            'high': (float, 'high'),
            'low': (float, 'low'),
        }

    def _gen_next_step(self, x):
        x += np.random.randn() * self.sd + self.drift
        return max(x, 0.1)

    def _gen_events(self, cur_prices, current_dt):
        for sid, price in six.iteritems(cur_prices):
            cur_prices[sid] = self._gen_next_step(cur_prices[sid])

            event = {
                'dt': current_dt,
                'sid': sid,
                'price': cur_prices[sid],
                'volume': np.random.randint(1e5, 1e6),
                'open_price': cur_prices[sid],
                'high': cur_prices[sid] + .1,
                'low': cur_prices[sid] - .1,
            }

            yield event

    def raw_data_gen(self):
        cur_prices = copy(self.start_prices)
        for _, (open_dt, close_dt) in self.open_and_closes.iterrows():
            current_dt = copy(open_dt)
            if self.freq == 'minute':
                # Emit minutely trade signals from open to close
                while current_dt <= close_dt:
                    for event in self._gen_events(cur_prices, current_dt):
                        yield event
                    current_dt += timedelta(minutes=1)
            elif self.freq == 'daily':
                # Emit one signal per day at close
                for event in self._gen_events(
                        cur_prices, pd.tslib.normalize_date(close_dt)):
                    yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data
