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

from bcolz import ctable
import pandas as pd

from sid import string_types


class NoDataOnDate(Exception):
    """
    Raised when a spot price can be found for the sid and date.
    """
    pass


class BcolzDailyBarSpotReader(object):
    """
    Reads inividual price moments from the daily bcolz file format shared
    with pipeline.loaders.equity_pricing_loader.BcolzOHLCVReader

    This is distinct from the BcolzOHLCVReader since the queries are on moments
    instead of windows.
    """

    def __init__(self, table):
        if isinstance(table, string_types):
            table = ctable(rootdir=table, mode='r')
        self.daily_bar_table = table
        calendar = self.daily_bar_table.attrs['calendar']
        self.trading_days = pd.DatetimeIndex(calendar, tz='UTC')

        # For indexing into daily bars.
        # We may be able to reuse code from DataPortal when that is ready.
        self.first_rows = {
            int(k): v for k, v
            in self.daily_bar_table.attrs['first_row'].iteritems()}
        self.last_rows = {
            int(k): v for k, v
            in self.daily_bar_table.attrs['last_row'].iteritems()}
        self.calendar_offset = {
            int(k): v for k, v
            in self.daily_bar_table.attrs['calendar_offset'].iteritems()}

        self._cols = {}

    def daily_bar_col(self, colname):
        """
        Get the colname from daily_bar_table and read all of it into memory,
        caching the result.

        Parameters
        ----------
        colname : string
            A name of a OHLCV carray in the daily_bar_table

        Returns
        -------
        array (uint32)
            Full read array of the carray in the daily_bar_table with the
            given colname.
        """
        try:
            col = self._cols[colname]
        except KeyError:
            col = self._cols[colname] = self.daily_bar_table[colname][:]
        return col

    def unadjusted_spot_price(self, sid, day, colname):
        """
        Parameters
        ----------
        sid : int
            The asset identifier.
        day : datetime64
            Midnight of the day for which data is requested.
        colname : string
            The price field. e.g. ('open', 'high', 'low', 'close', 'volume')

        Returns
        -------
        float
            The spot price for colnmae of the given sid on the given day.
            Raises a NoDataOnDate exception if there is no data available
            for the given day and sid.
        """
        day_loc = self.trading_days.get_loc(day)
        offset = day_loc - self.calendar_offset[sid]
        if offset < 0:
            raise NoDataOnDate(
                "No data on or before day={0} for sid={1}".format(
                    day, sid))
        ix = self.first_rows[sid] + offset
        if ix > self.last_rows[sid]:
            raise NoDataOnDate(
                "No data on or after day={0} for sid={1}".format(
                    day, sid))
        return self.daily_bar_col(colname)[ix] * 0.001
