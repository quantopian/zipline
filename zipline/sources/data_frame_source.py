#
# Copyright 2015 Quantopian, Inc.
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
import numpy as np
import pandas as pd

from zipline.gens.utils import hash_args

from zipline.sources.data_source import DataSource


class DataFrameSource(DataSource):
    """
    Data source that yields from a pandas DataFrame.

    :Axis layout:
        * columns : sids
        * index : datetime

    :Note:
        Bars where the price is nan are filtered out.
    """

    def __init__(self, data, **kwargs):
        assert isinstance(data.index, pd.tseries.index.DatetimeIndex)
        # Only accept integer SIDs as the items of the DataFrame
        assert isinstance(data.columns, pd.Int64Index)
        # TODO is ffilling correct/necessary?
        # Forward fill prices
        self.data = data.fillna(method='ffill')
        # Unpack config dictionary with default values.
        self.start = kwargs.get('start', self.data.index[0])
        self.end = kwargs.get('end', self.data.index[-1])
        self.sids = self.data.columns

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(data, **kwargs)

        self._raw_data = None

        self.started_sids = set()

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
                # Skip SIDs that can not be forward filled
                if np.isnan(price) and \
                   sid not in self.started_sids:
                    continue
                self.started_sids.add(sid)

                event = {
                    'dt': dt,
                    'sid': sid,
                    'price': price,
                    # Just chose something large
                    # if no volume available.
                    'volume': 1e9,
                }
                yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data


class DataPanelSource(DataSource):
    """
    Data source that yields from a pandas Panel.

    :Axis layout:
        * items : sids
        * major_axis : datetime
        * minor_axis : price, volume, ...

    :Note:
        Bars where the price is nan are filtered out.
    """

    def __init__(self, data, **kwargs):
        assert isinstance(data.major_axis, pd.tseries.index.DatetimeIndex)
        # Only accept integer SIDs as the items of the Panel
        assert isinstance(data.items, pd.Int64Index)
        # TODO is ffilling correct/necessary?
        # forward fill with volumes of 0
        self.data = data.fillna(value={'volume': 0})
        # Unpack config dictionary with default values.
        self.start = kwargs.get('start', self.data.major_axis[0])
        self.end = kwargs.get('end', self.data.major_axis[-1])
        self.sids = self.data.items

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(data, **kwargs)

        self._raw_data = None

        self.started_sids = set()

    @property
    def mapping(self):
        mapping = {
            'dt': (lambda x: x, 'dt'),
            'sid': (lambda x: x, 'sid'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
        }

        # Add additional fields.
        for field_name in self.data.minor_axis:
            if field_name in ['price', 'volume', 'dt', 'sid']:
                continue
            mapping[field_name] = (lambda x: x, field_name)

        return mapping

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        for dt in self.data.major_axis:
            df = self.data.major_xs(dt)
            for sid, series in df.iteritems():
                # Skip SIDs that can not be forward filled
                if np.isnan(series['price']):
                    continue
                self.started_sids.add(sid)

                event = {
                    'dt': dt,
                    'sid': sid,
                }
                for field_name, value in series.iteritems():
                    event[field_name] = value

                yield event

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data
