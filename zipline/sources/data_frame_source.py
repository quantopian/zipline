
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
Tools to generate data sources.
"""
import pandas as pd

from zipline.gens.utils import hash_args

from zipline.sources.data_source import DataSource


class DataFrameSource(DataSource):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    sids   : list of values representing simulated internal sids
    start  : start date
    delta  : timedelta between internal events
    filter : filter to remove the sids
    """

    def __init__(self, data, **kwargs):
        assert isinstance(data.index, pd.tseries.index.DatetimeIndex)

        self.data = data
        # Unpack config dictionary with default values.
        self.sids = kwargs.get('sids', data.columns)
        self.start = kwargs.get('start', data.index[0])
        self.end = kwargs.get('end', data.index[-1])

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(data, **kwargs)

        self._raw_data = None

    @property
    def mapping(self):
        return {
            'dt': (lambda x: x, 'dt'),
            'sid': (lambda x: x, 'sid'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
        }

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        for dt, series in self.data.iterrows():
            for sid, price in series.iteritems():
                if sid in self.sids:
                    event = {
                        'dt': dt,
                        'sid': sid,
                        'price': price,
                        'volume': 1000,
                    }
                    yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data


class DataPanelSource(DataSource):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    sids   : list of values representing simulated internal sids
    start  : start date
    delta  : timedelta between internal events
    filter : filter to remove the sids
    """

    def __init__(self, data, **kwargs):
        assert isinstance(data.major_axis, pd.tseries.index.DatetimeIndex)

        self.data = data
        # Unpack config dictionary with default values.
        self.sids = kwargs.get('sids', data.items)
        self.start = kwargs.get('start', data.major_axis[0])
        self.end = kwargs.get('end', data.major_axis[-1])

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(data, **kwargs)

        self._raw_data = None

    _mapping = None

    @property
    def mapping(self):
        # cache the mapping
        if self._mapping is None:
            mapping = {
                'price': (float, 'price'),
                'volume': (int, 'volume'),
            }
            self._mapping = mapping

        return self._mapping

    def apply_mapping(self, raw_row):
        """
        Override this to hand craft conversion of row.
        """
        row = raw_row.copy()
        for target, (mapping_func, source_key) in self.mapping.items():
            row[target] = mapping_func(raw_row[source_key])

        row['source_id'] = self.get_hash()
        row['type'] = self.event_type
        return row

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        values = self.data.values
        major_axis = self.data.major_axis
        minor_axis = self.data.minor_axis
        items = self.data.items

        for i, dt in enumerate(major_axis):
            df = values[:, i, :]
            for k, sid in enumerate(items):
                if sid in self.sids:
                    event = {
                        'dt': dt,
                        'sid': sid
                    }
                    series = df[k]
                    event.update(dict(zip(minor_axis, series)))
                    yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data
