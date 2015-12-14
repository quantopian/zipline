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
from numpy import (
    float64,
    uint32
)
from bcolz import ctable

from zipline.data.us_equity_pricing import (
    BcolzDailyBarWriter,
    OHLC,
    UINT32_MAX
)


class DailyBarWriterFromDataFrames(BcolzDailyBarWriter):
    _csv_dtypes = {
        'open': float64,
        'high': float64,
        'low': float64,
        'close': float64,
        'volume': float64,
    }

    def __init__(self, asset_map):
        self._asset_map = asset_map

    def gen_tables(self, assets):
        for asset in assets:
            yield asset, ctable.fromdataframe(assets[asset])

    def to_uint32(self, array, colname):
        arrmax = array.max()
        if colname in OHLC:
            self.check_uint_safe(arrmax * 1000, colname)
            return (array * 1000).astype(uint32)
        elif colname == 'volume':
            self.check_uint_safe(arrmax, colname)
            return array.astype(uint32)
        elif colname == 'day':
            nanos_per_second = (1000 * 1000 * 1000)
            self.check_uint_safe(arrmax.view(int) / nanos_per_second, colname)
            return (array.view(int) / nanos_per_second).astype(uint32)

    @staticmethod
    def check_uint_safe(value, colname):
        if value >= UINT32_MAX:
            raise ValueError(
                "Value %s from column '%s' is too large" % (value, colname)
            )
