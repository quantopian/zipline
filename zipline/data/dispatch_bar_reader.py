#
# Copyright 2016 Quantopian, Inc.
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
from abc import ABCMeta, abstractmethod

from numpy import (
    full,
    nan,
    int64,
    zeros
)
from six import iteritems, with_metaclass

from zipline.utils.memoize import lazyval


class AssetDispatchBarReader(with_metaclass(ABCMeta)):
    """

    Parameters
    ----------
    - trading_calendar : zipline.utils.trading_calendar.TradingCalendar
    - asset_finder : zipline.assets.AssetFinder
    - readers : dict
        A dict mapping Asset type to the corresponding
        [Minute|Session]BarReader
    - last_available_dt : pd.Timestamp or None, optional
        If not provided, infers it by using the min of the
        last_available_dt values of the underlying readers.
    """
    def __init__(
        self,
        trading_calendar,
        asset_finder,
        readers,
        last_available_dt=None,
    ):
        self._trading_calendar = trading_calendar
        self._asset_finder = asset_finder
        self._readers = readers
        self._last_available_dt = last_available_dt

        for t, r in iteritems(self._readers):
            assert trading_calendar == r.trading_calendar, \
                "All readers must share target trading_calendar. " \
                "Reader={0} for type={1} uses calendar={2} which does not " \
                "match the desired shared calendar={3} ".format(
                    r, t, r.trading_calendar, trading_calendar)

    @abstractmethod
    def _dt_window_size(self, start_dt, end_dt):
        pass

    @property
    def _asset_types(self):
        return self._readers.keys()

    def _make_raw_array_shape(self, start_dt, end_dt, num_sids):
        return self._dt_window_size(start_dt, end_dt), num_sids

    def _make_raw_array_out(self, field, shape):
        if field != 'volume' and field != 'sid':
            out = full(shape, nan)
        else:
            out = zeros(shape, dtype=int64)
        return out

    @property
    def trading_calendar(self):
        return self._trading_calendar

    @lazyval
    def last_available_dt(self):
        if self._last_available_dt is not None:
            return self._last_available_dt
        else:
            return max(r.last_available_dt for r in self._readers.values())

    @lazyval
    def first_trading_day(self):
        return min(r.first_trading_day for r in self._readers.values())

    def get_value(self, sid, dt, field):
        asset = self._asset_finder.retrieve_asset(sid)
        r = self._readers[type(asset)]
        return r.get_value(asset, dt, field)

    def get_last_traded_dt(self, asset, dt):
        r = self._readers[type(asset)]
        return r.get_last_traded_dt(asset, dt)

    def load_raw_arrays(self, fields, start_dt, end_dt, sids):
        asset_types = self._asset_types
        sid_groups = {t: [] for t in asset_types}
        out_pos = {t: [] for t in asset_types}

        assets = self._asset_finder.retrieve_all(sids)

        for i, asset in enumerate(assets):
            t = type(asset)
            sid_groups[t].append(asset)
            out_pos[t].append(i)

        batched_arrays = {
            t: self._readers[t].load_raw_arrays(fields,
                                                start_dt,
                                                end_dt,
                                                sid_groups[t])
            for t in asset_types if sid_groups[t]}

        results = []
        shape = self._make_raw_array_shape(start_dt, end_dt, len(sids))

        for i, field in enumerate(fields):
            out = self._make_raw_array_out(field, shape)
            for t, arrays in iteritems(batched_arrays):
                out[:, out_pos[t]] = arrays[i]
            results.append(out)

        return results


class AssetDispatchMinuteBarReader(AssetDispatchBarReader):

    def _dt_window_size(self, start_dt, end_dt):
        return len(self.trading_calendar.minutes_in_range(start_dt, end_dt))


class AssetDispatchSessionBarReader(AssetDispatchBarReader):

    def _dt_window_size(self, start_dt, end_dt):
        return len(self.trading_calendar.sessions_in_range(start_dt, end_dt))

    @lazyval
    def sessions(self):
        return self.trading_calendar.sessions_in_range(
            self.first_trading_day,
            self.last_available_dt)
