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
leverage work of briancappello and quantopian team
(especcially twiecki, eddie, and fawce)
michaelws
"""
import pandas as pd
from zipline.gens.utils import hash_args
from zipline.sources.data_source import DataSource
import datetime
import csv
import numpy as np
import dateutil.parser


def gen_ts(date, time):
    return pd.Timestamp(datetime.datetime.combine(date, time))


class DatasourceCSVohlc(DataSource):
    """ expects dictReader for a csv file
     with the following columns in the  header
    dt, sid, open, high, low, close, volume
    dt expected in ISO format and order does not matter"""
    def __init__(self, data, **kwargs):
        isinstance(data, csv.DictReader)
        self.data = data
        # Unpack config dictionary with default values.
        self.tz_in = kwargs.get('tz_in', "US/Eastern")
        self.start = pd.Timestamp(np.datetime64(kwargs.get('start')))
        self.start = self.start.tz_localize('utc')
        self.end = pd.Timestamp(np.datetime64(kwargs.get('end')))
        self.end = self.end.tz_localize('utc')
        start_time_str = kwargs.get("start_time", "9:30")
        end_time_str = kwargs.get("end_time", "16:00")
        self.sid_filter = kwargs.get('sid_filter', None)
        self.source_id = kwargs.get("source_id", None)
        self.sids = kwargs.get('sidsF', None)
        self.start_time = dateutil.parser.parse(start_time_str).time()
        self.end_time = dateutil.parser.parse(end_time_str).time()
        self._raw_data = None
        self.arg_string = hash_args(data, **kwargs)

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        previous_ts = None
        cols = np.array(["open", "high", "low"])
        for row in self.data:
            dt64 = pd.Timestamp(np.datetime64(row["dt"]))
            ts = pd.Timestamp(dt64).tz_localize(self.tz_in).tz_convert('utc')
            if ts < self.start or ts > self.end:
                continue
            if previous_ts is None or ts.date() != previous_ts.date():
                start_ts = datetime.date(ts.date(), self.start_time)
                end_ts = gen_ts(ts.date(), self.end_time)
            volumes = {}
            price_volumes = {}
            sid = row["sid"]
            if self.sid_filter and sid in self.sid_filter:
                continue
            elif self.sids is None or sid in self.sids:
                if sid not in volumes:
                    volumes[sid] = 0
                    price_volumes[sid] = 0
                if ts < start_ts or ts > end_ts:
                    continue
                event = {"sid": sid, "type": "TRADE", "symbol": sid}
                event["dt"] = ts
                event["price"] = float(row["close"])
                event["close"] = event["price"]
                event["volume"] = int(row["volume"])
                volumes[sid] += float(event["volume"])
                price_volumes[sid] += event["price"] * event["volume"]
                event["vwap"] = price_volumes[sid] / volumes[sid]
                for field in cols:
                    event[field] = float(row[field])
                yield event
            previous_ts = ts

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data

    @property
    def mapping(self):
        return {
            'sid': (lambda x: x, 'sid'),
            'dt': (lambda x: x, 'dt'),
            'open': (float, 'open'),
            'high': (float, 'high'),
            'low': (float, 'low'),
            'close': (float, 'close'),
            'price': (float, 'price'),
            'volume': (int, 'volume'),
            'vwap': (lambda x: x, 'vwap')
        }


class DataSourceCSVSignal(DataSource):
    """ expects dictReader for a csv file in form with header
        dt, sid, signal
        dt expected in ISO format"""
    def __init__(self, data, **kwargs):
        assert isinstance(data, csv.DictReader)
        self.data = data
        self.source_id = kwargs.get("source_id", None)
        # Unpack config dictionary with default values.
        self.start = kwargs.get('start')
        self.end = kwargs.get('end')
        self.sids = kwargs.get('sids', None)
        self.sid_filter = kwargs.get('sid_filter', None)
        self.arg_string = hash_args(data, **kwargs)
        self._raw_data = None

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        previous_ts = None
        for row in self.data:
            dt64 = pd.Timestamp(np.datetime64(row["dt"]))
            ts = pd.Timestamp(dt64).tz_localize(self.tz_in).tz_convert('utc')
            if ts < self.start or ts > self.end:
                continue
            if previous_ts is None or ts.date() != previous_ts.date():
                start_ts = gen_ts(ts.date(), self.start_time)
                end_ts = gen_ts(ts.date(), self.end_time)
            volumes = {}
            price_volumes = {}
            sid = row["sid"]
            if self.sid_filter and sid in self.sid_filter:
                continue
            elif self.sids is None or sid in self.sids:
                if sid not in volumes:
                    volumes[sid] = 0
                    price_volumes[sid] = 0
                if ts < start_ts or ts > end_ts:
                    continue
                    event = {"sid": sid, "type": "CUSTOM", "dt": ts,
                             "signal": row["signal"]}
                    yield event
            previous_ts = ts

    @property
    def raw_data(self):
        if not self._raw_data:
            self._raw_data = self.raw_data_gen()
        return self._raw_data

    @property
    def mapping(self):
        return {
            'sid': (lambda x: x, 'symbol'),
            'dt': (lambda x: x, 'dt'),
            'signal': (lambda x: x, 'signal'),
        }
