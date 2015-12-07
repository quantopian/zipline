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

from datetime import datetime
import bcolz
from logbook import Logger

import numpy as np
import pandas as pd
from pandas.tslib import normalize_date
from six import iteritems

from zipline.assets import Asset, Future, Equity
from zipline.data.us_equity_pricing import (
    BcolzDailyBarReader,
    NoDataOnDate
)
from zipline.pipeline.data.equity_pricing import USEquityPricing

from zipline.utils import tradingcalendar
from zipline.errors import (
    NoTradeDataAvailableTooEarly,
    NoTradeDataAvailableTooLate
)

log = Logger('DataPortal')

HISTORY_FREQUENCIES = ["1d", "1m"]

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


US_EQUITY_COLUMNS = {
    'open': USEquityPricing.open,
    'open_price': USEquityPricing.open,
    'high': USEquityPricing.high,
    'low': USEquityPricing.low,
    'close': USEquityPricing.close,
    'close_price': USEquityPricing.close,
    'volume': USEquityPricing.volume,
    'price': USEquityPricing.close,
}


class DataPortal(object):
    def __init__(self,
                 env,
                 sim_params=None,
                 equity_minute_reader=None,
                 minutes_futures_path=None,
                 daily_equities_path=None,
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

        # This is a bit ugly, but is here for performance reasons.  In minute
        # simulations, we need to very quickly go from dt -> (# of minutes
        # since Jan 1 2002 9:30 Eastern).
        #
        # The clock that heartbeats the simulation has all the necessary
        # information to do this calculation very quickly.  This value is
        # calculated there, and then set here
        self.cur_data_offset = 0

        self.views = {}

        self._daily_equities_path = daily_equities_path
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

        self._sim_params = sim_params
        if self._sim_params is not None:
            self._data_frequency = self._sim_params.data_frequency
        else:
            self._data_frequency = "minute"

        self._futures_sid_path_func = futures_sid_path_func

        self.MINUTE_PRICE_ADJUSTMENT_FACTOR = 0.001

        if daily_equities_path is not None:
            self._daily_bar_reader = BcolzDailyBarReader(daily_equities_path)
        else:
            self._daily_bar_reader = None

        self._equity_minute_reader = equity_minute_reader

        # The following values are used by _minute_offset to calculate the
        # index into the minute bcolz date.

        # A lookup of table every minute to the corresponding day, to avoid
        # calling `.date()` on every lookup.
        self._minutes_to_day = {}
        # A map of days (keyed by midnight) to a DatetimeIndex of market
        # minutes for that day.
        self._minutes_by_day = {}
        # A dict of day to the offset into the minute bcolz on which that
        # days data starts.
        self._day_offsets = None

    def handle_extra_source(self, source_df):
        """
        Extra sources always have a sid column.

        We expand the given data (by forward filling) to the full range of
        the simulation dates, so that lookup is fast during simulation.
        """
        if source_df is None:
            return

        self._extra_source_df = source_df

        # source_df's sid column can either consist of assets we know about
        # (such as sid(24)) or of assets we don't know about (such as
        # palladium).
        #
        # In both cases, we break up the dataframe into individual dfs
        # that only contain a single asset's information.  ie, if source_df
        # has data for PALLADIUM and GOLD, we split source_df into two
        # dataframes, one for each. (same applies if source_df has data for
        # AAPL and IBM).
        #
        # We then take each child df and reindex it to the simulation's date
        # range by forward-filling missing values. this makes reads simpler.
        #
        # Finally, we store the data. For each column, we store a mapping in
        # self.augmented_sources_map from the column to a dictionary of
        # asset -> df.  In other words,
        # self.augmented_sources_map['days_to_cover']['AAPL'] gives us the df
        # holding that data.

        if self._sim_params.emission_rate == "daily":
            source_date_index = self.env.days_in_range(
                start=self._sim_params.period_start,
                end=self._sim_params.period_end
            )
        else:
            source_date_index = self.env.minutes_for_days_in_range(
                start=self._sim_params.period_start,
                end=self._sim_params.period_end
            )

        # Break the source_df up into one dataframe per sid.  This lets
        # us (more easily) calculate accurate start/end dates for each sid,
        # de-dup data, and expand the data to fit the backtest start/end date.
        grouped_by_sid = source_df.groupby(["sid"])
        group_names = grouped_by_sid.groups.keys()
        group_dict = {}
        for group_name in group_names:
            group_dict[group_name] = grouped_by_sid.get_group(group_name)

        for identifier, df in iteritems(group_dict):
            # Before reindexing, save the earliest and latest dates
            earliest_date = df.index[0]
            latest_date = df.index[-1]

            # Since we know this df only contains a single sid, we can safely
            # de-dupe by the index (dt)
            df = df.groupby(level=0).last()

            # Reindex the dataframe based on the backtest start/end date.
            # This makes reads easier during the backtest.
            df = df.reindex(index=source_date_index, method='ffill')

            if not isinstance(identifier, Asset):
                # for fake assets we need to store a start/end date
                self._asset_start_dates[identifier] = earliest_date
                self._asset_end_dates[identifier] = latest_date

            for col_name in df.columns.difference(['sid']):
                if col_name not in self._augmented_sources_map:
                    self._augmented_sources_map[col_name] = {}

                self._augmented_sources_map[col_name][identifier] = df

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

        else:
            # TODO: Figure out if assets should be allowed if neither, and
            # why this code path is being hit.
            if self._equity_minute_reader.sid_path_func is not None:
                path = self._equity_minute_reader.sid_path_func(
                    self._equity_minute_reader.rootdir, sid
                )
            else:
                path = "{0}/{1}.bcolz".format(
                    self._equity_minute_reader.rootdir, sid)

        return bcolz.open(path, mode='r')

    def get_previous_value(self, asset, field, dt):
        """
        Given an asset and a column and a dt, returns the previous value for
        the same asset/column pair.  If this data portal is in minute mode,
        it's the previous minute value, otherwise it's the previous day's
        value.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.

        field: string
            The desired field of the asset.  Valid values are "open",
            "open_price", "high", "low", "close", "close_price", "volume", and
            "price".

        dt: pd.Timestamp
            The timestamp from which to go back in time one slot.

        Returns
        -------
        The value of the desired field at the desired time.
        """
        if self._data_frequency == 'daily':
            prev_dt = self.env.previous_trading_day(dt)
        elif self._data_frequency == 'minute':
            prev_dt = self.env.previous_market_minute(dt)

        return self.get_spot_value(asset, field, prev_dt)

    def _check_extra_sources(self, asset, column, day):
        # If we have an extra source with a column called "price", only look
        # at it if it's on something like palladium and not AAPL (since our
        # own price data always wins when dealing with assets).
        look_in_augmented_sources = column in self._augmented_sources_map and \
            not (column in BASE_FIELDS and isinstance(asset, Asset))

        if look_in_augmented_sources:
            # we're being asked for a field in an extra source
            try:
                return self._augmented_sources_map[column][asset].\
                    loc[day, column]
            except:
                log.error(
                    "Could not find value for asset={0}, current_day={1},"
                    "column={2}".format(
                        str(asset),
                        str(self.current_day),
                        str(column)))

                raise KeyError

    def get_spot_value(self, asset, field, dt=None):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt, or this data
        portal's current_dt.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.gith

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
        extra_source_val = self._check_extra_sources(
            asset,
            field,
            (dt or self.current_dt)
        )

        if extra_source_val is not None:
            return extra_source_val

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

    def setup_offset_cache(self, minutes_by_day, minutes_to_day):
        # TODO: This case should not be hit, but is when tests are setup
        # with data_frequency of daily, but run with minutely.
        if self._equity_minute_reader is None:
            return

        self._minutes_to_day = minutes_to_day
        self._minutes_by_day = minutes_by_day
        if self._sim_params is not None:
            start = self._sim_params.trading_days[0]
            first_trading_day_idx = self._equity_minute_reader.trading_days.\
                searchsorted(start)
            self._day_offsets = {
                day: (i + first_trading_day_idx) * 390
                for i, day in enumerate(
                    self._sim_params.trading_days)}

    def _minute_offset(self, dt):
        if self._day_offsets is not None:
            try:
                day = self._minutes_to_day[dt]
                minutes = self._minutes_by_day[day]
                return self._day_offsets[day] + minutes.get_loc(dt)
            except KeyError:
                return None

    def _get_minute_spot_value(self, asset, column, dt):
        # if dt is before the first market minute, minute_index
        # will be 0.  if it's after the last market minute, it'll
        # be len(minutes_for_day)
        minute_offset_to_use = self._minute_offset(dt)

        if minute_offset_to_use is None:
            given_day = pd.Timestamp(dt.date(), tz='utc')
            day_index = self._equity_minute_reader.trading_days.searchsorted(
                given_day)

            # if dt is before the first market minute, minute_index
            # will be 0.  if it's after the last market minute, it'll
            # be len(minutes_for_day)
            minute_index = self.env.market_minutes_for_day(given_day).\
                searchsorted(dt)

            minute_offset_to_use = (day_index * 390) + minute_index

        carray = self._open_minute_file(column, asset)
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
                value = self._daily_bar_reader.spot_price(asset, dt, column)
                if value != -1:
                    return value
                else:
                    dt -= tradingcalendar.trading_day
            except NoDataOnDate:
                return 0

    def _get_history_daily_window(self, assets, end_dt, bar_count,
                                  field_to_use):
        """
        Internal method that returns a dataframe containing history bars
        of daily frequency for the given sids.
        """
        day_idx = tradingcalendar.trading_days.searchsorted(end_dt.date())
        days_for_window = tradingcalendar.trading_days[
            (day_idx - bar_count + 1):(day_idx + 1)]

        if len(assets) == 0:
            return pd.DataFrame(None,
                                index=days_for_window,
                                columns=None)

        data = []

        for asset in assets:
            if isinstance(asset, Future):
                data.append(self._get_history_daily_window_future(
                    asset, days_for_window, end_dt, field_to_use
                ))
            else:
                data.append(self._get_history_daily_window_equity(
                    asset, days_for_window, end_dt, field_to_use
                ))

        return pd.DataFrame(
            np.array(data).T,
            index=days_for_window,
            columns=assets
        )

    def _get_history_daily_window_future(self, asset, days_for_window,
                                         end_dt, column):
        # Since we don't have daily bcolz files for futures (yet), use minute
        # bars to calculate the daily values.
        data = []
        data_groups = []

        # get all the minutes for the days NOT including today
        for day in days_for_window[:-1]:
            minutes = self.env.market_minutes_for_day(day)

            values_for_day = np.zeros(len(minutes), dtype=np.float64)

            for idx, minute in enumerate(minutes):
                minute_val = self._get_minute_spot_value_future(
                    asset, column, minute
                )

                values_for_day[idx] = minute_val

            data_groups.append(values_for_day)

        # get the minutes for today
        last_day_minutes = pd.date_range(
            start=self.env.get_open_and_close(end_dt)[0],
            end=end_dt,
            freq="T"
        )

        values_for_last_day = np.zeros(len(last_day_minutes), dtype=np.float64)

        for idx, minute in enumerate(last_day_minutes):
            minute_val = self._get_minute_spot_value_future(
                asset, column, minute
            )

            values_for_last_day[idx] = minute_val

        data_groups.append(values_for_last_day)

        for group in data_groups:
            if len(group) == 0:
                continue

            if column == 'volume':
                data.append(np.sum(group))
            elif column == 'open':
                data.append(group[0])
            elif column == 'close':
                data.append(group[-1])
            elif column == 'high':
                data.append(np.amax(group))
            elif column == 'low':
                data.append(np.amin(group))

        return data

    def _get_history_daily_window_equity(self, asset, days_for_window,
                                         end_dt, field_to_use):
        sid = int(asset)
        ends_at_midnight = end_dt.hour == 0 and end_dt.minute == 0

        # get the start and end dates for this sid
        end_date = self._get_asset_end_date(asset)

        if ends_at_midnight or (days_for_window[-1] > end_date):
            # two cases where we use daily data for the whole range:
            # 1) the history window ends at midnight utc.
            # 2) the last desired day of the window is after the
            # last trading day, use daily data for the whole range.
            return self._get_daily_window_for_sid(
                asset,
                field_to_use,
                days_for_window,
                extra_slot=False
            )
        else:
            # for the last day of the desired window, use minute
            # data and aggregate it.
            all_minutes_for_day = self.env.market_minutes_for_day(
                pd.Timestamp(end_dt.date()))

            last_minute_idx = all_minutes_for_day.searchsorted(end_dt)

            # these are the minutes for the partial day
            minutes_for_partial_day =\
                all_minutes_for_day[0:(last_minute_idx + 1)]

            daily_data = self._get_daily_window_for_sid(
                sid,
                field_to_use,
                days_for_window[0:-1]
            )

            minute_data = self._get_minute_window_for_equity(
                sid,
                field_to_use,
                minutes_for_partial_day
            )

            if field_to_use == 'volume':
                minute_value = np.sum(minute_data)
            elif field_to_use == 'open':
                minute_value = minute_data[0]
            elif field_to_use == 'close':
                minute_value = minute_data[-1]
            elif field_to_use == 'high':
                minute_value = np.amax(minute_data)
            elif field_to_use == 'low':
                minute_value = np.amin(minute_data)

            # append the partial day.
            daily_data[-1] = minute_value

            return daily_data

    def _get_history_minute_window(self, assets, end_dt, bar_count,
                                   field_to_use):
        """
        Internal method that returns a dataframe containing history bars
        of minute frequency for the given sids.
        """
        # get all the minutes for this window
        minutes_for_window = self.env.market_minute_window(
            end_dt, bar_count, step=-1)[::-1]

        first_trading_day = self._equity_minute_reader.first_trading_day

        # but then cut it down to only the minutes after
        # the first trading day.
        modified_minutes_for_window = minutes_for_window[
            minutes_for_window.slice_indexer(first_trading_day)]

        modified_minutes_length = len(modified_minutes_for_window)

        if modified_minutes_length == 0:
            raise ValueError("Cannot calculate history window that ends"
                             "before 2002-01-02 14:31 UTC!")

        data = []
        bars_to_prepend = 0
        nans_to_prepend = None

        if modified_minutes_length < bar_count:
            first_trading_date = first_trading_day.date()
            if modified_minutes_for_window[0].date() == first_trading_date:
                # the beginning of the window goes before our global trading
                # start date
                bars_to_prepend = bar_count - modified_minutes_length
                nans_to_prepend = np.repeat(np.nan, bars_to_prepend)

        if len(assets) == 0:
            return pd.DataFrame(
                None,
                index=modified_minutes_for_window,
                columns=None
            )

        for asset in assets:
            asset_minute_data = self._get_minute_window_for_asset(
                asset,
                field_to_use,
                modified_minutes_for_window
            )

            if bars_to_prepend != 0:
                asset_minute_data = np.insert(asset_minute_data, 0,
                                              nans_to_prepend)

            data.append(asset_minute_data)

        return pd.DataFrame(
            np.array(data).T,
            index=minutes_for_window,
            columns=map(int, assets)
        )

    def get_history_window(self, assets, end_dt, bar_count, frequency, field,
                           ffill=True):
        """
        Public API method that returns a dataframe containing the requested
        history window.  Data is fully adjusted.

        Parameters
        ---------
        assets : list of zipline.data.Asset objects
            The assets whose data is desired.

        bar_count: int
            The number of bars desired.

        frequency: string
            "1d" or "1m"

        field: string
            The desired field of the asset.

        ffill: boolean
            Forward-fill missing values. Only has effect if field
            is 'price'.

        Returns
        -------
        A dataframe containing the requested data.
        """
        try:
            field_to_use = BASE_FIELDS[field]
        except KeyError:
            raise ValueError("Invalid history field: " + str(field))

        # sanity check in case sids were passed in
        assets = [(self.env.asset_finder.retrieve_asset(asset) if
                   isinstance(asset, int) else asset) for asset in assets]

        if frequency == "1d":
            df = self._get_history_daily_window(assets, end_dt, bar_count,
                                                field_to_use)
        elif frequency == "1m":
            df = self._get_history_minute_window(assets, end_dt, bar_count,
                                                 field_to_use)
        else:
            raise ValueError("Invalid frequency: {0}".format(frequency))

        # forward-fill if needed
        if field == "price" and ffill:
            df.fillna(method='ffill', inplace=True)

        return df

    def _get_minute_window_for_asset(self, asset, field, minutes_for_window):
        """
        Internal method that gets a window of adjusted minute data for an asset
        and specified date range.  Used to support the history API method for
        minute bars.

        Missing bars are filled with NaN.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.

        field: string
            The specific field to return.  "open", "high", "close_price", etc.

        minutes_for_window: pd.DateTimeIndex
            The list of minutes representing the desired window.  Each minute
            is a pd.Timestamp.

        Returns
        -------
        A numpy array with requested values.
        """
        if isinstance(asset, int):
            asset = self.env.asset_finder.retrieve_asset(asset)

        if isinstance(asset, Future):
            return self._get_minute_window_for_future(asset, field,
                                                      minutes_for_window)
        else:
            return self._get_minute_window_for_equity(asset, field,
                                                      minutes_for_window)

    def _get_minute_window_for_future(self, asset, field, minutes_for_window):
        # THIS IS TEMPORARY.  For now, we are only exposing futures within
        # equity trading hours (9:30 am to 4pm, Eastern).  The easiest way to
        # do this is to simply do a spot lookup for each desired minute.
        return_data = np.zeros(len(minutes_for_window), dtype=np.float64)
        for idx, minute in enumerate(minutes_for_window):
            return_data[idx] = \
                self._get_minute_spot_value_future(asset, field, minute)

        # Note: an improvement could be to find the consecutive runs within
        # minutes_for_window, and use them to read the underlying ctable
        # more efficiently.

        # Once futures are on 24-hour clock, then we can just grab all the
        # requested minutes in one shot from the ctable.

        # no adjustments for futures, yay.
        return return_data

    def _get_minute_window_for_equity(self, asset, field, minutes_for_window):
        # each sid's minutes are stored in a bcolz file
        # the bcolz file has 390 bars per day, starting at 1/2/2002, regardless
        # of when the asset started trading and regardless of half days.
        # for a half day, the second half is filled with zeroes.

        # find the position of start_dt in the entire timeline, go back
        # bar_count bars, and that's the unadjusted data
        raw_data = self._open_minute_file(field, asset)

        start_idx = max(self._find_position_of_minute(minutes_for_window[0]),
                        0)
        end_idx = self._find_position_of_minute(minutes_for_window[-1]) + 1

        if end_idx == 0:
            # No data to return for minute window.
            return np.full(len(minutes_for_window), np.nan)

        return_data = np.zeros(len(minutes_for_window), dtype=np.float64)

        data_to_copy = raw_data[start_idx:end_idx]

        num_minutes = len(minutes_for_window)

        # data_to_copy contains all the zeros (from 1pm to 4pm of an early
        # close).  num_minutes is the number of actual trading minutes.  if
        # these two have different lengths, that means that we need to trim
        # away data due to early closes.
        if len(data_to_copy) != num_minutes:
            # get a copy of the minutes in Eastern time, since we depend on
            # an early close being at 1pm Eastern.
            eastern_minutes = minutes_for_window.tz_convert("US/Eastern")

            # accumulate a list of indices of the last minute of an early
            # close day.  For example, if data_to_copy starts at 12:55 pm, and
            # there are five minutes of real data before 180 zeroes, we would
            # put 5 into last_minute_idx_of_early_close_day, because the fifth
            # minute is the last "real" minute of the day.
            last_minute_idx_of_early_close_day = []
            for minute_idx, minute_dt in enumerate(eastern_minutes):
                if minute_idx == (num_minutes - 1):
                    break

                if minute_dt.hour == 13 and minute_dt.minute == 0:
                    next_minute = eastern_minutes[minute_idx + 1]
                    if next_minute.hour != 13:
                        # minute_dt is the last minute of an early close day
                        last_minute_idx_of_early_close_day.append(minute_idx)

            # spin through the list of early close markers, and use them to
            # chop off 180 minutes at a time from data_to_copy.
            for idx, early_close_minute_idx in \
                    enumerate(last_minute_idx_of_early_close_day):
                early_close_minute_idx -= (180 * idx)
                data_to_copy = np.delete(
                    data_to_copy,
                    range(
                        early_close_minute_idx + 1,
                        early_close_minute_idx + 181
                    )
                )

        return_data[0:len(data_to_copy)] = data_to_copy

        self._apply_all_adjustments(
            return_data,
            asset,
            minutes_for_window,
            field,
            self.MINUTE_PRICE_ADJUSTMENT_FACTOR
        )

        return return_data

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

    def _find_position_of_minute(self, minute_dt):
        """
        Internal method that returns the position of the given minute in the
        list of every trading minute since market open on 1/2/2002.

        IMPORTANT: This method assumes every day is 390 minutes long, even
        early closes.  Our minute bcolz files are generated like this to
        support fast lookup.

        ex. this method would return 2 for 1/2/2002 9:32 AM Eastern.

        Parameters
        ----------
        minute_dt: pd.Timestamp
            The minute whose position should be calculated.

        Returns
        -------
        The position of the given minute in the list of all trading minutes
        since market open on 1/2/2002.
        """
        day = minute_dt.date()
        day_idx = self._equity_minute_reader.trading_days.searchsorted(day)
        if day_idx < 0:
            return -1

        day_open = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=9,
                minute=31),
            tz='US/Eastern').tz_convert('UTC')

        minutes_offset = int((minute_dt - day_open).total_seconds()) / 60

        return int((390 * day_idx) + minutes_offset)

    def _get_daily_window_for_sid(self, asset, field, days_in_window,
                                  extra_slot=True):
        """
        Internal method that gets a window of adjusted daily data for a sid
        and specified date range.  Used to support the history API method for
        daily bars.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.

        start_dt: pandas.Timestamp
            The start of the desired window of data.

        bar_count: int
            The number of days of data to return.

        field: string
            The specific field to return.  "open", "high", "close_price", etc.

        extra_slot: boolean
            Whether to allocate an extra slot in the returned numpy array.
            This extra slot will hold the data for the last partial day.  It's
            much better to create it here than to create a copy of the array
            later just to add a slot.

        Returns
        -------
        A numpy array with requested values.  Any missing slots filled with
        nan.

        """
        bar_count = len(days_in_window)
        # create an np.array of size bar_count
        if extra_slot:
            return_array = np.zeros((bar_count + 1,))
        else:
            return_array = np.zeros((bar_count,))

        return_array[:] = np.NAN

        start_date = self._get_asset_start_date(asset)
        end_date = self._get_asset_end_date(asset)
        day_slice = days_in_window.slice_indexer(start_date, end_date)
        active_days = days_in_window[day_slice]

        if active_days.shape[0]:
            data = self._daily_bar_reader.history_window(field,
                                                         active_days[0],
                                                         active_days[-1],
                                                         asset)
            return_array[day_slice] = data
            self._apply_all_adjustments(
                return_array,
                asset,
                active_days,
                field,
            )

        return return_array

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

    def get_equity_price_view(self, asset):
        """
        Returns a DataPortalSidView for the given asset.  Used to support the
        data[sid(N)] public API.  Not needed if DataPortal is used standalone.

        Parameters
        ----------
        asset : Asset
            Asset that is being queried.

        Returns
        -------
        DataPortalSidView: Accessor into the given asset's data.
        """
        try:
            view = self.views[asset]
        except KeyError:
            view = self.views[asset] = DataPortalSidView(asset, self)

        return view

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

    def get_splits(self, sids, dt):
        """
        Returns any splits for the given sids and the given dt.

        Parameters
        ----------
        sids : list
            Sids for which we want splits.

        dt: pd.Timestamp
            The date for which we are checking for splits.  Note: this is
            expected to be midnight UTC.

        Returns
        -------
        list: List of splits, where each split is a (sid, ratio) tuple.
        """
        if self._adjustment_reader is None or len(sids) == 0:
            return {}

        # convert dt to # of seconds since epoch, because that's what we use
        # in the adjustments db
        seconds = int(dt.value / 1e9)

        splits = self._adjustment_reader.conn.execute(
            "SELECT sid, ratio FROM SPLITS WHERE effective_date = ?",
            (seconds,)).fetchall()

        sids_set = set(sids)
        splits = [split for split in splits if split[0] in sids_set]

        return splits

    def get_stock_dividends(self, sid, trading_days):
        """
        Returns all the stock dividends for a specific sid that occur
        in the given trading range.

        Parameters
        ----------
        sid: int
            The asset whose stock dividends should be returned.

        trading_days: pd.DatetimeIndex
            The trading range.

        Returns
        -------
        list: A list of objects with all relevant attributes populated.
        All timestamp fields are converted to pd.Timestamps.
        """

        if self._adjustment_reader is None:
            return []

        if len(trading_days) == 0:
            return []

        start_dt = trading_days[0].value / 1e9
        end_dt = trading_days[-1].value / 1e9

        dividends = self._adjustment_reader.conn.execute(
            "SELECT * FROM stock_dividend_payouts WHERE sid = ? AND "
            "ex_date > ? AND pay_date < ?", (int(sid), start_dt, end_dt,)).\
            fetchall()

        dividend_info = []
        for dividend_tuple in dividends:
            dividend_info.append({
                "declared_date": dividend_tuple[1],
                "ex_date": pd.Timestamp(dividend_tuple[2], unit="s"),
                "pay_date": pd.Timestamp(dividend_tuple[3], unit="s"),
                "payment_sid": dividend_tuple[4],
                "ratio": dividend_tuple[5],
                "record_date": pd.Timestamp(dividend_tuple[6], unit="s"),
                "sid": dividend_tuple[7]
            })

        return dividend_info

    def contains(self, asset, field):
        return field in BASE_FIELDS or \
            (field in self._augmented_sources_map and
             asset in self._augmented_sources_map[field])

    def get_fetcher_assets(self):
        """
        Returns a list of assets for the current date, as defined by the
        fetcher data.

        Notes
        -----
        Data is forward-filled.  If there is no fetcher data defined for day
        N, we use day N-1's data (if available, otherwise we keep going back).

        Returns
        -------
        list: a list of Asset objects.
        """
        # return a list of assets for the current date, as defined by the
        # fetcher source
        if self._extra_source_df is None:
            return []

        if self.current_day in self._extra_source_df.index:
            date_to_use = self.current_day
        else:
            # current day isn't in the fetcher df, go back the last
            # available day
            idx = self._extra_source_df.index.searchsorted(self.current_day)
            if idx == 0:
                return []

            date_to_use = self._extra_source_df.index[idx - 1]

        asset_list = self._extra_source_df.loc[date_to_use]["sid"]

        # make sure they're actually assets
        asset_list = [asset for asset in asset_list
                      if isinstance(asset, Asset)]

        return asset_list


class DataPortalSidView(object):
    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_spot_value(self.asset, column)

    def __contains__(self, column):
        return self.portal.contains(self.asset, column)

    def __getitem__(self, column):
        return self.__getattr__(column)
