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
from interface import implements
from numpy import iinfo, uint32

from zipline.lib.adjusted_array import AdjustedArray

from .base import PipelineLoader
from .utils import shift_dates

UINT32_MAX = iinfo(uint32).max


class EquityPricingLoader(implements(PipelineLoader)):
    """A PipelineLoader for loading daily OHLCV data.

    Parameters
    ----------
    raw_price_reader : zipline.data.session_bars.SessionBarReader
        Reader providing raw prices.
    adjustments_reader : zipline.data.adjustments.SQLiteAdjustmentReader
        Reader providing price/volume adjustments.
    """
    def __init__(self, raw_price_reader, adjustments_reader):
        self.raw_price_reader = raw_price_reader
        self.adjustments_reader = adjustments_reader

    def load_adjusted_array(self, domain, columns, dates, sids, mask):
        # load_adjusted_array is called with dates on which the user's algo
        # will be shown data, which means we need to return the data that would
        # be known at the start of each date.  We assume that the latest data
        # known on day N is the data from day (N - 1), so we shift all query
        # dates back by a day.
        sessions = domain.all_sessions()
        start_date, end_date = shift_dates(
            sessions, dates[0], dates[-1], shift=1,
        )
        colnames = [c.name for c in columns]
        raw_arrays = self.raw_price_reader.load_raw_arrays(
            colnames,
            start_date,
            end_date,
            sids,
        )
        adjustments = self.adjustments_reader.load_pricing_adjustments(
            colnames,
            dates,
            sids,
        )

        out = {}
        for c, c_raw, c_adjs in zip(columns, raw_arrays, adjustments):
            out[c] = AdjustedArray(
                c_raw.astype(c.dtype),
                c_adjs,
                c.missing_value,
            )
        return out


# Backwards compat alias.
USEquityPricingLoader = EquityPricingLoader
