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

import bcolz
from logbook import Logger

import numpy as np
import pandas as pd
from pandas.tslib import normalize_date

from zipline.assets import Future, Equity
from zipline.data.us_equity_pricing import NoDataOnDate

from zipline.utils import tradingcalendar
from zipline.errors import (
    NoTradeDataAvailableTooEarly,
    NoTradeDataAvailableTooLate
)

log = Logger('DataPortal')

BASE_FIELDS = {
    'open': 'open',
    'open_price': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'close_price': 'close',
    'volume': 'volume',
    'price': 'close'
}


class DataPortal(object):
    def __init__(self,
                 env,
                 sim_params=None,
                 equity_daily_reader=None,
                 equity_minute_reader=None,
                 minutes_futures_path=None,
                 adjustment_reader=None,
                 futures_sid_path_func=None):
        self.env = env

        # Internal pointers to the current dt (can be minute) and current day.
        # In daily mode, they point to the same thing. In minute mode, it's
        # useful to have separate pointers to the current day and to the
        # current minute.  These pointers are updated by the
        # AlgorithmSimulator's transform loop.
        self.current_dt = None
        self.current_day = None

        self.views = {}

        self._minutes_futures_path = minutes_futures_path

        self._asset_finder = env.asset_finder

        self._carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {},
            'sid': {},
            'dt': {},
        }

        self._adjustment_reader = adjustment_reader

        # caches of sid -> adjustment list
        self._splits_dict = {}
        self._mergers_dict = {}
        self._dividends_dict = {}

        # Cache of sid -> the first trading day of an asset, even if that day
        # is before 1/2/2002.
        self._asset_start_dates = {}
        self._asset_end_dates = {}

        # Handle extra sources, like Fetcher.
        self._augmented_sources_map = {}
        self._extra_source_df = None

        if self._sim_params is not None:
            self._data_frequency = self._sim_params.data_frequency
        else:
            self._data_frequency = "minute"

        self._futures_sid_path_func = futures_sid_path_func

        self.MINUTE_PRICE_ADJUSTMENT_FACTOR = 0.001

        self._equity_daily_reader = equity_daily_reader
        self._equity_minute_reader = equity_minute_reader

    def _open_minute_file(self, field, asset):
        sid_str = str(int(asset))

        try:
            carray = self._carrays[field][sid_str]
        except KeyError:
            carray = self._carrays[field][sid_str] = \
                self._get_ctable(asset)[field]

        return carray

    def _get_ctable(self, asset):
        sid = int(asset)

        if isinstance(asset, Future):
            if self._futures_sid_path_func is not None:
                path = self._futures_sid_path_func(
                    self._minutes_futures_path, sid
                )
            else:
                path = "{0}/{1}.bcolz".format(self._minutes_futures_path, sid)
        elif isinstance(asset, Equity):
            if self._equity_minute_reader.sid_path_func is not None:
                path = self._equity_minute_reader.sid_path_func(
                    self._equity_minute_reader.rootdir, sid
                )
            else:
                path = "{0}/{1}.bcolz".format(
                    self._equity_minute_reader.rootdir, sid)

        return bcolz.open(path, mode='r')

    def get_spot_value(self, asset, field, dt=None):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt, or this data
        portal's current_dt.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.

        field: string
            The desired field of the asset.  Valid values are "open",
            "open_price", "high", "low", "close", "close_price", "volume", and
            "price".

        dt: pd.Timestamp
            (Optional) The timestamp for the desired value.

        Returns
        -------
        The value of the desired field at the desired time.
        """
        if field not in BASE_FIELDS:
            raise KeyError("Invalid column: " + str(field))

        column_to_use = BASE_FIELDS[field]

        if isinstance(asset, int):
            asset = self._asset_finder.retrieve_asset(asset)

        self._check_is_currently_alive(asset, dt)

        if self._data_frequency == "daily":
            day_to_use = dt or self.current_day
            day_to_use = normalize_date(day_to_use)
            return self._get_daily_data(asset, column_to_use, day_to_use)
        else:
            dt_to_use = dt or self.current_dt

            if isinstance(asset, Future):
                return self._get_minute_spot_value_future(
                    asset, column_to_use, dt_to_use)
            else:
                return self._get_minute_spot_value(
                    asset, column_to_use, dt_to_use)

    def _get_minute_spot_value_future(self, asset, column, dt):
        # Futures bcolz files have 1440 bars per day (24 hours), 7 days a week.
        # The file attributes contain the "start_dt" and "last_dt" fields,
        # which represent the time period for this bcolz file.

        # The start_dt is midnight of the first day that this future started
        # trading.

        # figure out the # of minutes between dt and this asset's start_dt
        start_date = self._get_asset_start_date(asset)
        minute_offset = int((dt - start_date).total_seconds() / 60)

        if minute_offset < 0:
            # asking for a date that is before the asset's start date, no dice
            return 0.0

        # then just index into the bcolz carray at that offset
        carray = self._open_minute_file(column, asset)
        result = carray[minute_offset]

        # if there's missing data, go backwards until we run out of file
        while result == 0 and minute_offset > 0:
            minute_offset -= 1
            result = carray[minute_offset]

        if column != 'volume':
            return result * self.MINUTE_PRICE_ADJUSTMENT_FACTOR
        else:
            return result

    def _get_minute_spot_value(self, asset, column, dt):
        # if dt is before the first market minute, minute_index
        # will be 0.  if it's after the last market minute, it'll
        # be len(minutes_for_day)
        given_day = pd.Timestamp(dt.date(), tz='utc')
        day_index = self._equity_minute_reader.trading_days.searchsorted(
            given_day)

        # if dt is before the first market minute, minute_index
        # will be 0.  if it's after the last market minute, it'll
        # be len(minutes_for_day)
        minute_index = self.env.market_minutes_for_day(given_day).\
            searchsorted(dt)

        minute_offset_to_use = (day_index * 390) + minute_index

        carray = self._equity_minute_reader._open_minute_file(column, asset)
        result = carray[minute_offset_to_use]

        if result == 0:
            # if the given minute doesn't have data, we need to seek
            # backwards until we find data. This makes the data
            # forward-filled.

            # get this asset's start date, so that we don't look before it.
            start_date = self._get_asset_start_date(asset)
            start_date_idx = self._equity_minute_reader.trading_days.\
                searchsorted(start_date)
            start_day_offset = start_date_idx * 390

            original_start = minute_offset_to_use

            while result == 0 and minute_offset_to_use > start_day_offset:
                minute_offset_to_use -= 1
                result = carray[minute_offset_to_use]

            # once we've found data, we need to check whether it needs
            # to be adjusted.
            if result != 0:
                minutes = self.env.market_minute_window(
                    start=(dt or self.current_dt),
                    count=(original_start - minute_offset_to_use + 1),
                    step=-1
                ).order()

                # only need to check for adjustments if we've gone back
                # far enough to cross the day boundary.
                if minutes[0].date() != minutes[-1].date():
                    # create a np array of size minutes, fill it all with
                    # the same value.  and adjust the array.
                    arr = np.array([result] * len(minutes),
                                   dtype=np.float64)
                    self._apply_all_adjustments(
                        data=arr,
                        asset=asset,
                        dts=minutes,
                        field=column
                    )

                    # The first value of the adjusted array is the value
                    # we want.
                    result = arr[0]

        if column != 'volume':
            return result * self.MINUTE_PRICE_ADJUSTMENT_FACTOR
        else:
            return result

    def _get_daily_data(self, asset, column, dt):
        while True:
            try:
                value = self._equity_daily_reader.spot_price(
                    asset, dt, column)
                if value != -1:
                    return value
                else:
                    dt -= tradingcalendar.trading_day
            except NoDataOnDate:
                return 0

    def _apply_all_adjustments(self, data, asset, dts, field,
                               price_adj_factor=1.0):
        """
        Internal method that applies all the necessary adjustments on the
        given data array.

        The adjustments are:
        - splits
        - if field != "volume":
            - mergers
            - dividends
            - * 0.001
            - any zero fields replaced with NaN
        - all values rounded to 3 digits after the decimal point.

        Parameters
        ----------
        data : np.array
            The data to be adjusted.

        asset: Asset
            The asset whose data is being adjusted.

        dts: pd.DateTimeIndex
            The list of minutes or days representing the desired window.

        field: string
            The field whose values are in the data array.

        price_adj_factor: float
            Factor with which to adjust OHLC values.
        Returns
        -------
        None.  The data array is modified in place.
        """
        self._apply_adjustments_to_window(
            self._get_adjustment_list(
                asset, self._splits_dict, "SPLITS"
            ),
            data,
            dts,
            field != 'volume'
        )

        if field != 'volume':
            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    asset, self._mergers_dict, "MERGERS"
                ),
                data,
                dts,
                True
            )

            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    asset, self._dividends_dict, "DIVIDENDS"
                ),
                data,
                dts,
                True
            )

            data *= price_adj_factor

            # if anything is zero, it's a missing bar, so replace it with NaN.
            # we only want to do this for non-volume fields, because a missing
            # volume should be 0.
            data[data == 0] = np.NaN

        np.around(data, 3, out=data)

    @staticmethod
    def _apply_adjustments_to_window(adjustments_list, window_data,
                                     dts_in_window, multiply):
        if len(adjustments_list) == 0:
            return

        # advance idx to the correct spot in the adjustments list, based on
        # when the window starts
        idx = 0

        while idx < len(adjustments_list) and dts_in_window[0] >\
                adjustments_list[idx][0]:
            idx += 1

        # if we've advanced through all the adjustments, then there's nothing
        # to do.
        if idx == len(adjustments_list):
            return

        while idx < len(adjustments_list):
            adjustment_to_apply = adjustments_list[idx]

            if adjustment_to_apply[0] > dts_in_window[-1]:
                break

            range_end = dts_in_window.searchsorted(adjustment_to_apply[0])
            if multiply:
                window_data[0:range_end] *= adjustment_to_apply[1]
            else:
                window_data[0:range_end] /= adjustment_to_apply[1]

            idx += 1

    def _get_adjustment_list(self, asset, adjustments_dict, table_name):
        """
        Internal method that returns a list of adjustments for the given sid.

        Parameters
        ----------
        asset : Asset
            The asset for which to return adjustments.

        adjustments_dict: dict
            A dictionary of sid -> list that is used as a cache.

        table_name: string
            The table that contains this data in the adjustments db.

        Returns
        -------
        adjustments: list
            A list of [multiplier, pd.Timestamp], earliest first

        """
        if self._adjustment_reader is None:
            return []

        sid = int(asset)

        try:
            adjustments = adjustments_dict[sid]
        except KeyError:
            adjustments = adjustments_dict[sid] = self._adjustment_reader.\
                get_adjustments_for_sid(table_name, sid)

        return adjustments

    def _check_is_currently_alive(self, asset, dt):
        if dt is None:
            dt = self.current_day

        sid = int(asset)

        if sid not in self._asset_start_dates:
            self._get_asset_start_date(asset)

        start_date = self._asset_start_dates[sid]
        if self._asset_start_dates[sid] > dt:
            raise NoTradeDataAvailableTooEarly(
                sid=sid,
                dt=dt,
                start_dt=start_date
            )

        end_date = self._asset_end_dates[sid]
        if self._asset_end_dates[sid] < dt:
            raise NoTradeDataAvailableTooLate(
                sid=sid,
                dt=dt,
                end_dt=end_date
            )

    def _get_asset_start_date(self, asset):
        self._ensure_asset_dates(asset)
        return self._asset_start_dates[asset]

    def _get_asset_end_date(self, asset):
        self._ensure_asset_dates(asset)
        return self._asset_end_dates[asset]

    def _ensure_asset_dates(self, asset):
        sid = int(asset)

        if sid not in self._asset_start_dates:
            self._asset_start_dates[sid] = asset.start_date
            self._asset_end_dates[sid] = asset.end_date
