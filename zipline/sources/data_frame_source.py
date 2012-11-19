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
from copy import copy
from itertools import ifilter

import pandas as pd

from zipline.gens.utils import hash_args

from zipline.protocol import DATASOURCE_TYPE
from zipline.utils import ndict

from zipline.sources.test_source import SpecificEquityTrades


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
                event = {
                    'dt': dt,
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
