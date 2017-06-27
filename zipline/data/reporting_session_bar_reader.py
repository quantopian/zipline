#
# Copyright 2017 Quantopian, Inc.
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
from zipline.data.bar_reader import BarReader
from zipline.utils.memoize import lazyval


class ReportingSessionBarReader(BarReader):
    """
    Stitches together a two session bar readers, where one is used to provide
    the standard fields (e.g. OHLCV, etc.), and the other is used to provide
    fields as used for reporting (e.g. reporting_close).

    This is useful for cases like our standard pricing data for an asset type
    runs on a different calendar than desired for price reporting.
    """
    def __init__(self, reader, reporting_reader):
        self._reader = reader
        self._reporting_reader = reporting_reader

    @property
    def trading_calendar(self):
        return self._reader.trading_calendar

    @lazyval
    def last_available_dt(self):
        return min(
            self._reader.last_available_dt,
            self._reporting_reader.last_available_dt,
        )

    @lazyval
    def first_trading_day(self):
        return max(
            self._reader.first_trading_day,
            self._reporting_reader.first_trading_day,
        )

    def get_last_traded_dt(self, asset, dt):
        return min(
            self._reader.get_last_traded_dt(asset, dt),
            self._reporting_reader.get_last_traded_dt(asset, dt),
        )

    def get_value(self, sid, dt, field):
        if field == 'reporting_close':
            return self._reporting_reader.get_value(sid, dt, 'close')
        else:
            return self._reader(sid, dt, field)

    def load_raw_arrays(self, fields, start_dt, end_dt, sids):
        standard_fields = [f for f in fields if f != 'reporting_close']
        arrays = self._reader.load_raw_arrays(
            standard_fields,
            start_dt,
            end_dt,
            sids,
        )

        try:
            reporting_close_index = fields.index('reporting_close')
        except IndexError:
            return arrays

        reporting_close_array = self._reporting_reader.load_raw_arrays(
            'close',
            start_dt,
            end_dt,
            sids,
        )
        arrays.insert(reporting_close_index, reporting_close_array)

        return arrays
