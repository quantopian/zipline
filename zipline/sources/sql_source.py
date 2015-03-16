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


from zipline.sources.data_source import DataSource
import pandas as pd

from zipline.gens.utils import hash_args


class SqlSource(DataSource):

    def __init__(self, engine, table, **kwargs):

        self.engine = engine
        self.table = table

        self.data = pd.read_sql_table(table, engine, index_col='index')
        self.data = self.data.tz_localize('UTC')
        self.sids = kwargs.get('sids', self.data.columns)
        self.start = self.data.index[0]
        self.end = self.data.index[-1]

        # Hash_value for downstream sorting.
        self.arg_string = hash_args(engine, table, **kwargs)

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
