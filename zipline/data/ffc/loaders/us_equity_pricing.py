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

from zipline.data.baseloader import DataLoader
from zipline.data.ffc.loaders._us_equity_pricing import (
    load_raw_arrays_from_bcolz,
    load_adjustments_from_sqlite
)

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)


class USEquityPricingLoader(DataLoader):

    def __init__(self, raw_price_loader, adjustments_loader):
        self.raw_price_loader = raw_price_loader
        self.adjustments_loader = adjustments_loader

    def load_adjusted_array(self, columns, assets, dates):
        raw_arrays = self.raw_price_loader.load_raw_arrays(
            columns, assets, dates)
        adjustments = self.adjustments_loader.load_adjustments(
            columns, assets, dates)
        return [
            adjusted_array(raw_array,
                           NOMASK,
                           col_adjustments)
            for raw_array, col_adjustments in zip(raw_arrays, adjustments)
        ]


class BcolzRawPriceLoader(object):
    """
    Returns the raw pricing information using a bcolz table.

    The bcolz table that backs this loader contains the following columns
    and attributes.

    Columns:

    - open, an integer column which is the price * 1000

    - high, an integer column which is the price * 1000

    - low, an integer column which is the price * 1000

    - close, an integer column which is the price * 1000

    - volume, an integer column of the volume price information unchanged.

    - sid, an integer column reperesting the asset id (unused by loader)

    - day, an integer column of the midnight of the day the row in seconds
    since epoch. (unused by loader)

    The data in each column is grouped by asset and then sorted by day within
    each asset block.

    The table is built to represent a long time range of data, e.g. ten years
    of equity data, so the lengths of each asset block is not equal to each
    other. The blocks are clipped to the known start and end date of each
    asset, to cut down on the number of empty values that would need to be
    included to make a regular/cubic dataset.

    When read across the open, high, low, close, and volume with the same
    index should represent the same asset and day.

    Attributes:

    - start_pos, a dictionary keyed by asset id of the index of the first row
    of each asset, used as the basis for finding the start index with which
    to read the data from the table.

    - start_day_offset, the number of days since the start date of the entire
    dataset that the individual equity starts, used with the start_pos to
    determine the index from which to start reading data.

    - end_day_offset, the number of days since the end date of the entire that
    the individual equity ends at, used to prevent the indexing logic from
    including values from the next block of data.
    """
    def __init__(self, table, trading_days):
        self.table = table
        self.trading_days = trading_days
        self.start_pos = {int(k): v for k, v in
                          table.attrs['start_pos'].iteritems()}
        self.start_day_offset = {int(k): v for k, v in
                                 table.attrs['start_day_offset'].iteritems()}
        self.end_day_offset = {int(k): v for k, v in
                               table.attrs['end_day_offset'].iteritems()}

    def load_raw_arrays(self, columns, assets, dates):
        return load_raw_arrays_from_bcolz(self.table,
                                          self.start_pos,
                                          self.start_day_offset,
                                          self.end_day_offset,
                                          self.trading_days,
                                          columns,
                                          assets,
                                          dates)


class SQLiteAdjustmentLoader(object):
    """
    Loads adjustments based on corporate actions from a SQLite database.

    The database has tables for mergers, dividends, and splits.

    Each table has the columns:
    - sid, the asset identifier

    - effective_date, the midnight of date, in seconds, on which the adjustment
    starts. Adjustments are applied on the effective_date and all dates before
    it.

    - ratio, the price and/or volume multiplier.

    Corporate Actions Types:

    mergers, modify the price (ohlc)

    splits, modify the price (ohlc), and the volume. Volume is modify the
    inverse of the price adjustment.

    dividends, modify the price (ohlc). The dividend ratio is calculated as:
    1.0 - (dividend / "close of the market day before the ex_date"
    """

    def __init__(self, conn):
        self.conn = conn

    def load_adjustments(self, columns, assets, dates):
        return load_adjustments_from_sqlite(self.conn, columns, assets, dates)
