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
(especially twiecki, eddie, and fawce)
"""
import pandas as pd
from zipline.gens.utils import hash_args
from zipline.sources.data_source import DataSource
import datetime
import numpy as np
import dateutil.parser
import tables


def _iterate_ohlc(date_node, sid_filter, sids, start_ts, end_ts):
    last_stamp = None
    last_ts = None
    volumes = {}
    price_volumes = {}
    cols = np.array(["open", "high", "low", "close"])
    for row in date_node.iterrows():
        sid = row["sid"]
        if sid_filter and sid in sid_filter:
            continue
        elif sids is None or sid in sids:
            if sid not in volumes:
                volumes[sid] = 0
                price_volumes[sid] = 0
            if last_stamp and row["dt"] == last_stamp:
                ts = last_ts
            else:
                ts = pd.Timestamp(np.datetime64(row["dt"], "s"), tz='utc')
                last_ts = ts
                last_stamp = row["dt"]
            if (start_ts > ts) or (ts > end_ts):
                continue
            event = {"sid": sid, "type": "TRADE", "symbol": sid}
            event["dt"] = ts
            event["price"] = row["close"]
            event["volume"] = row["volume"]
            volumes[sid] += event["volume"]
            price_volumes[sid] += event["price"] * event["volume"]
            event["vwap"] = price_volumes[sid] / volumes[sid]
            last_ts = ts
            for field in cols:
                event[field] = row[field]
            yield event


def _iterate_signal(date_node, sids, sid_filter, start_ts, end_ts):
    last_stamp = None
    last_ts = None
    volumes = {}
    price_volumes = {}
    for row in date_node.iterrows():
        sid = row["sid"]
        if sid_filter and sid in sid_filter:
            continue
        elif sids is None or sid in sids:
            if sid not in volumes:
                volumes[sid] = 0
                price_volumes[sid] = 0
            if last_stamp and row["dt"] == last_stamp:
                ts = last_ts
            else:
                ts = pd.Timestamp(np.datetime64(row["dt"], "s"), tz='utc')
                last_ts = ts
                last_stamp = row["dt"]
            if (start_ts > ts) or (ts > end_ts):
                continue
            event = {"sid": sid, "type": "CUSTOM",
                     "signal": row["signal"]}
            yield event


class DataSourceTablesOHLC(DataSource):
    """
    Yields all events in event_list that match the given sid_filter.
    If no event_list is specified, generates an internal stream of events
    to filter.  Returns all events if filter is None.

    Configuration options:

    sids   : list of values representing simulated internal sids
    start  : start date
    tz_in : timezzone of table
    filter : filter to remove the sids
    start_time: what time trading should start
    end_time: what time trading should end
    """
    def __init__(self, data, **kwargs):
        assert isinstance(data, tables.file.File)
        self.data = data
        # Unpack config dictionary with default values.
        if 'symbols' in kwargs:
            self.sids = kwargs.get('symbols')
        else:
            self.sids = None
        self.tz_in = kwargs.get('tz_in', "US/Eastern")
        self.source_id = kwargs.get("source_id", None)
        self.sid_filter = kwargs.get("filter", None)
        self.start = pd.Timestamp(np.datetime64(kwargs.get('start')))
        self.start = self.start.tz_localize('utc')
        self.end = pd.Timestamp(np.datetime64(kwargs.get('end')))
        self.end = self.end.tz_localize('utc')
        start_time_str = kwargs.get("start_time", "9:30")
        end_time_str = kwargs.get("end_time", "16:00")
        self.start_time = dateutil.parser.parse(start_time_str).time()
        self.end_time = dateutil.parser.parse(end_time_str).time()
        self._raw_data = None
        self.arg_string = hash_args(data, **kwargs)
        self.root_node = "/" + kwargs.get('root', "TD") + "/"

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        for date_node in self.data.walkNodes(self.root_node):
            if isinstance(date_node, tables.group.Group):
                continue
            date = dateutil.parser.parse(date_node.name.split("_")[1])
            dt64 = np.datetime64(date)
            table_dt = pd.Timestamp(dt64).tz_localize("utc")
            if table_dt < self.start or table_dt > self.end:
                continue
            start_ts = pd.Timestamp(datetime.datetime.combine(table_dt.date(),
                                                              self.start_time),
                                    tz=self.tz_in)
            start_ts = start_ts.tz_convert("utc")
            end_ts = pd.Timestamp(datetime.datetime.combine(table_dt.date(),
                                                            self.end_time),
                                  tz=self.tz_in)
            end_ts = end_ts.tz_convert("utc")
            for item in _iterate_ohlc(date_node, self.sids, self.sid_filter,
                                      start_ts, end_ts):
                yield item

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
            'open': (lambda x: x, 'open'),
            'high': (lambda x: x, 'high'),
            'low': (lambda x: x, 'low'),
            'close': (lambda x: x, 'close'),
            'price': (lambda x: x, 'price'),
            'volume': (lambda x: x, 'volume'),
            'vwap': (lambda x: x, 'vwap')
        }


class DataSourceTablesSignal(DataSource):
    def __init__(self, data, **kwargs):
        assert isinstance(data, tables.file.File)
        self.h5file = data
        self.sids = kwargs.get('sids', None)
        self.start = kwargs.get('start')
        self.end = kwargs.get('end')
        self.source_id = kwargs.get("source_id", None)
        self.arg_string = hash_args(data, **kwargs)
        self._raw_data = None
        self.root_node = +"/" + kwargs.get('root', "signal") + "/"

    @property
    def instance_hash(self):
        return self.arg_string

    def raw_data_gen(self):
        for date_node in self.data.walkNodes(self.root_node):
            if isinstance(date_node, tables.group.Group):
                continue
            date = dateutil.parser.parse(date_node.name.split("_")[1])
            dt64 = np.datetime64(date)
            table_dt = pd.Timestamp(dt64).tz_localize("utc")
            if table_dt < self.start or table_dt > self.end:
                continue
            start_ts = pd.Timestamp(datetime.datetime.combine(table_dt.date(),
                                                              self.start_time),
                                    tz=self.tz_in)
            start_ts = start_ts.tz_convert("utc")
            end_ts = pd.Timestamp(datetime.datetime.combine(table_dt.date(),
                                                            self.end_time),
                                  tz=self.tz_in)
            end_ts = end_ts.tz_convert("utc")
            table = self.data.getNode(date_node)
            for row in _iterate_signal(table, self.sids, self.sid_filter,
                                       start_ts, end_ts):
                yield row

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
