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
from operator import mul

import bcolz
from logbook import Logger

import numpy as np
import pandas as pd
from pandas.tslib import normalize_date
from six import iteritems
from six.moves import reduce

from zipline.assets import Asset, Future, Equity
from zipline.data.us_equity_pricing import NoDataOnDate
from zipline.data.us_equity_loader import USEquityHistoryLoader

from zipline.utils import tradingcalendar
from zipline.utils.math_utils import (
    nansum,
    nanmean,
    nanstd
)
from zipline.utils.memoize import remember_last
from zipline.errors import (
    NoTradeDataAvailableTooEarly,
    NoTradeDataAvailableTooLate
)

log = Logger('DataPortal')

BASE_FIELDS = frozenset([
    "open", "high", "low", "close", "volume", "price", "last_traded"
])

OHLCV_FIELDS = frozenset([
    "open", "high", "low", "close", "volume"
])

OHLCVP_FIELDS = frozenset([
    "open", "high", "low", "close", "volume", "price"
])

HISTORY_FREQUENCIES = set(["1m", "1d"])


class DataPortal(object):
    def __init__(self,
                 env,
                 equity_daily_reader=None,
                 equity_minute_reader=None,
                 future_daily_reader=None,
                 future_minute_reader=None,
                 adjustment_reader=None):
        self.env = env

        self.views = {}

        self._asset_finder = env.asset_finder

        self._carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {},
            'sid': {},
        }

        self._adjustment_reader = adjustment_reader

        # caches of sid -> adjustment list
        self._splits_dict = {}
        self._mergers_dict = {}
        self._dividends_dict = {}

        # Cache of sid -> the first trading day of an asset.
        self._asset_start_dates = {}
        self._asset_end_dates = {}

        # Handle extra sources, like Fetcher.
        self._augmented_sources_map = {}
        self._extra_source_df = None

        self._equity_daily_reader = equity_daily_reader
        if self._equity_daily_reader is not None:
            self._equity_history_loader = USEquityHistoryLoader(
                self.env,
                self._equity_daily_reader,
                self._adjustment_reader
            )
        self._equity_minute_reader = equity_minute_reader
        self._future_daily_reader = future_daily_reader
        self._future_minute_reader = future_minute_reader

        self._first_trading_day = None

        if self._equity_minute_reader is not None:
            self.MINUTE_PRICE_ADJUSTMENT_FACTOR = \
                self._equity_minute_reader._ohlc_inverse

        # get the first trading day from our readers.
        if self._equity_daily_reader is not None:
            self._first_trading_day = \
                self._equity_daily_reader.first_trading_day
        elif self._equity_minute_reader is not None:
            self._first_trading_day = \
                self._equity_minute_reader.first_trading_day

        # The `equity_daily_reader_array` lookups provide lru cache of 1 for
        # daily history reads from the daily_reader.
        # `last_remembered` or lru_cache can not be used, because the inputs
        # to the function are not hashable types, and the order of the assets
        # iterable needs to be preserved.
        # The function that implements the cache will insert a value for the
        # given field of (frozenset(assets), start_dt, end_dt).
        #
        # This is  optimized for algorithms that call history once per field
        # in handle_data or a scheduled function.
        self._equity_daily_reader_array_keys = {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': None,
            'price': None
        }
        self._equity_daily_reader_array_data = {}

        self._in_bts = False

    def handle_extra_source(self, source_df, sim_params):
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

        if sim_params.emission_rate == "daily":
            source_date_index = self.env.days_in_range(
                start=sim_params.period_start,
                end=sim_params.period_end
            )
        else:
            source_date_index = self.env.minutes_for_days_in_range(
                start=sim_params.period_start,
                end=sim_params.period_end
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
            if self._future_minute_reader.sid_path_func is not None:
                path = self._future_minute_reader.sid_path_func(
                    self._future_minute_reader.rootdir, sid
                )
            else:
                path = "{0}/{1}.bcolz".format(
                    self._future_minute_reader.rootdir, sid)
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

    def get_last_traded_dt(self, asset, dt, data_frequency):
        """
        Given an asset and dt, returns the last traded dt from the viewpoint
        of the given dt.

        If there is a trade on the dt, the answer is dt provided.
        """
        if data_frequency == 'minute':
            return self._equity_minute_reader.get_last_traded_dt(asset, dt)
        elif data_frequency == 'daily':
            return self._equity_daily_reader.get_last_traded_dt(asset, dt)

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
                    "Could not find value for asset={0}, day={1},"
                    "column={2}".format(
                        str(asset),
                        str(day),
                        str(column)))

                raise KeyError

    def get_spot_value(self, asset, field, dt, data_frequency):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.

        field: string
            The desired field of the asset.  Valid values are "open", "high",
            "low", "close", "volume", "price", and "last_traded".

        dt: pd.Timestamp
            The timestamp for the desired value.

        data_frequency: string
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        The value of the desired field at the desired time.
        """
        extra_source_val = self._check_extra_sources(asset, field, dt)

        if extra_source_val is not None:
            return extra_source_val

        if field not in BASE_FIELDS:
            raise KeyError("Invalid column: " + str(field))

        if isinstance(asset, int):
            asset = self.env.asset_finder.retrieve_asset(asset)

        if dt < asset.start_date or \
                (data_frequency == "daily" and dt > asset.end_date) or \
                (data_frequency == "minute" and
                 normalize_date(dt) > asset.end_date):
            if field == "volume":
                return 0
            elif field != "last_traded":
                return np.NaN

        if data_frequency == "daily":
            day_to_use = dt
            day_to_use = normalize_date(day_to_use)
            return self._get_daily_data(asset, field, day_to_use)
        else:
            if isinstance(asset, Future):
                return self._get_minute_spot_value_future(
                    asset, field, dt)
            else:
                if field == "last_traded":
                    return self._equity_minute_reader.get_last_traded_dt(
                        asset, dt
                    )
                elif field == "price":
                    return self._get_minute_spot_value(asset, "close", dt,
                                                       True)
                else:
                    return self._get_minute_spot_value(asset, field, dt)

    def _get_adjusted_value(self, asset, field, dt,
                            perspective_dt,
                            data_frequency,
                            spot_value=None):
        """
        Private method that returns a scalar value representing the value
        of the desired asset's field at the given dt with adjustments applied.

        Parameters
        ---------
        asset : Asset
            The asset whose data is desired.

        field: string
            The desired field of the asset.  Valid values are "open",
            "open_price", "high", "low", "close", "close_price", "volume", and
            "price".

        dt: pd.Timestamp
            The timestamp for the desired value.

        perspective_dt : pd.Timestamp
            The timestamp from which the data is being viewed back from.

        data_frequency: string
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        The value of the desired field at the desired time.
        """
        if isinstance(asset, int):
            asset = self._asset_finder.retrieve_asset(asset)

        if spot_value is None:
            spot_value = self.get_spot_value(asset, field, dt, data_frequency)

        if isinstance(asset, Equity):
            adjs = []
            split_adjustments = self._get_adjustment_list(
                asset, self._splits_dict, "SPLITS"
            )
            for adj_dt, adj in split_adjustments:
                if dt <= adj_dt <= perspective_dt:
                    if field != 'volume':
                        adjs.append(adj)
                    else:
                        adjs.append(1.0 / adj)
                if adj_dt >= perspective_dt:
                    break

            if field != 'volume':
                merger_adjustments = self._get_adjustment_list(
                    asset, self._mergers_dict, "MERGERS"
                )
                for adj_dt, adj in merger_adjustments:
                    if dt <= adj_dt <= perspective_dt:
                        adjs.append(adj)
                    if adj_dt >= perspective_dt:
                        break
                div_adjustments = self._get_adjustment_list(
                    asset, self._dividends_dict, "DIVIDENDS",
                )
                for adj_dt, adj in div_adjustments:
                    if dt <= adj_dt <= perspective_dt:
                        adjs.append(adj)
                    if adj_dt >= perspective_dt:
                        break

            ratio = reduce(mul, adjs, 1.0)

            spot_value *= ratio

        return spot_value

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
            # FIXME switch to a futures reader
            return result * 0.001
        else:
            return result

    def _get_minute_spot_value(self, asset, column, dt, ffill=False):
        result = self._equity_minute_reader.get_value(
            asset.sid, dt, column
        )

        if column == "volume":
            if result == 0:
                return 0
        elif not ffill or not np.isnan(result):
            # if we're not forward filling, or we found a result, return it
            return result

        # we are looking for price, and didn't find one. have to go hunting.
        last_traded_dt = \
            self._equity_minute_reader.get_last_traded_dt(asset, dt)

        if last_traded_dt is pd.NaT:
            # no last traded dt, bail
            return np.nan

        # get the value as of the last traded dt
        result = self._equity_minute_reader.get_value(
            asset.sid,
            last_traded_dt,
            column
        )

        if np.isnan(result):
            return np.nan

        if dt == last_traded_dt or dt.date() == last_traded_dt.date():
            return result

        # the value we found came from a different day, so we have to adjust
        # the data if there are any adjustments on that day barrier
        return self._get_adjusted_value(
            asset, column, last_traded_dt,
            dt, "minute", spot_value=result
        )

    def _get_daily_data(self, asset, column, dt):
        if column == "last_traded":
            last_traded_dt = \
                self._equity_daily_reader.get_last_traded_dt(asset, dt)

            if pd.isnull(last_traded_dt):
                return pd.NaT
            else:
                return last_traded_dt
        elif column in OHLCV_FIELDS:
            # don't forward fill
            try:
                val = self._equity_daily_reader.spot_price(asset, dt, column)
                if val == -1:
                    if column == "volume":
                        return 0
                    else:
                        return np.nan
                else:
                    return val
            except NoDataOnDate:
                return np.nan
        elif column == "price":
            found_dt = dt
            while True:
                try:
                    value = self._equity_daily_reader.spot_price(
                        asset, found_dt, "close"
                    )
                    if value != -1:
                        if dt == found_dt:
                            return value
                        else:
                            # adjust if needed
                            return self._get_adjusted_value(
                                asset, column, found_dt, dt, "minute",
                                spot_value=value
                            )
                    else:
                        found_dt -= tradingcalendar.trading_day
                except NoDataOnDate:
                    return np.nan

    @remember_last
    def _get_days_for_window(self, end_date, bar_count):
        day_idx = tradingcalendar.trading_days.searchsorted(end_date)
        return tradingcalendar.trading_days[
            (day_idx - bar_count + 1):(day_idx + 1)]

    def _get_history_daily_window(self, assets, end_dt, bar_count,
                                  field_to_use):
        """
        Internal method that returns a dataframe containing history bars
        of daily frequency for the given sids.
        """
        days_for_window = self._get_days_for_window(end_dt.date(), bar_count)

        if len(assets) == 0:
            return pd.DataFrame(None,
                                index=days_for_window,
                                columns=None)

        future_data = []
        eq_assets = []

        for asset in assets:
            if isinstance(asset, Future):
                future_data.append(self._get_history_daily_window_future(
                    asset, days_for_window, end_dt, field_to_use
                ))
            else:
                eq_assets.append(asset)
        eq_data = self._get_history_daily_window_equities(
            eq_assets, days_for_window, end_dt, field_to_use
        )
        if future_data:
            # TODO: This case appears to be uncovered by testing.
            data = np.concatenate(eq_data, np.array(future_data).T)
        else:
            data = eq_data
        return pd.DataFrame(
            data,
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

    @remember_last
    def _get_market_minutes_for_day(self, end_date):
        return self.env.market_minutes_for_day(pd.Timestamp(end_date))

    def _get_history_daily_window_equities(
            self, assets, days_for_window, end_dt, field_to_use):
        ends_at_midnight = end_dt.hour == 0 and end_dt.minute == 0

        if ends_at_midnight:
            # two cases where we use daily data for the whole range:
            # 1) the history window ends at midnight utc.
            # 2) the last desired day of the window is after the
            # last trading day, use daily data for the whole range.
            return self._get_daily_window_for_sids(
                assets,
                field_to_use,
                days_for_window,
                extra_slot=False
            )
        else:
            all_minutes_for_day = self._get_market_minutes_for_day(
                end_dt.date())
            # for the last day of the desired window, use minute
            # data and aggregate it.
            last_minute_idx = all_minutes_for_day.searchsorted(end_dt)

            # these are the minutes for the partial day
            minutes_for_partial_day =\
                all_minutes_for_day[0:(last_minute_idx + 1)]

            daily_data = self._get_daily_window_for_sids(
                assets,
                field_to_use,
                days_for_window[0:-1]
            )

            minute_data = self._get_minute_window_for_equities(
                assets,
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

        if minutes_for_window[0] < first_trading_day:

            # but then cut it down to only the minutes after
            # the first trading day.
            modified_minutes_for_window = minutes_for_window[
                minutes_for_window.slice_indexer(first_trading_day)]

            modified_minutes_length = len(modified_minutes_for_window)

            if modified_minutes_length == 0:
                raise ValueError("Cannot calculate history window that ends "
                                 "before the first trading day!")

            bars_to_prepend = 0
            nans_to_prepend = None

            if modified_minutes_length < bar_count:
                first_trading_date = first_trading_day.date()
                if modified_minutes_for_window[0].date() == first_trading_date:
                    # the beginning of the window goes before our global
                    # trading start date
                    bars_to_prepend = bar_count - modified_minutes_length
                    nans_to_prepend = np.repeat(np.nan, bars_to_prepend)

            if len(assets) == 0:
                return pd.DataFrame(
                    None,
                    index=modified_minutes_for_window,
                    columns=None
                )
            query_minutes = modified_minutes_for_window
        else:
            query_minutes = minutes_for_window
            bars_to_prepend = 0

        asset_minute_data = self._get_minute_window_for_assets(
            assets,
            field_to_use,
            query_minutes,
        )

        if bars_to_prepend != 0:
            if field_to_use == "volume":
                filler = np.zeros((len(nans_to_prepend), len(assets)))
            else:
                filler = np.full((len(nans_to_prepend), len(assets)), np.nan)

            asset_minute_data = np.concatenate([filler, asset_minute_data])

        return pd.DataFrame(
            asset_minute_data,
            index=minutes_for_window,
            columns=assets
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
        if field not in OHLCVP_FIELDS:
            raise ValueError("Invalid field: {0}".format(field))

        if frequency == "1d":
            if field == "price":
                df = self._get_history_daily_window(assets, end_dt, bar_count,
                                                    "close")
            else:
                df = self._get_history_daily_window(assets, end_dt, bar_count,
                                                    field)
        elif frequency == "1m":
            if field == "price":
                df = self._get_history_minute_window(assets, end_dt, bar_count,
                                                     "close")
            else:
                df = self._get_history_minute_window(assets, end_dt, bar_count,
                                                     field)
        else:
            raise ValueError("Invalid frequency: {0}".format(frequency))

        # forward-fill price
        if field == "price":
            if frequency == "1m":
                data_frequency = 'minute'
            elif frequency == "1d":
                data_frequency = 'daily'
            else:
                raise Exception(
                    "Only 1d and 1m are supported for forward-filling.")

            dt_to_fill = df.index[0]

            perspective_dt = df.index[-1]
            assets_with_leading_nan = np.where(pd.isnull(df.iloc[0]))[0]
            for missing_loc in assets_with_leading_nan:
                asset = assets[missing_loc]
                previous_dt = self.get_last_traded_dt(
                    asset, dt_to_fill, data_frequency)
                if pd.isnull(previous_dt):
                    continue
                previous_value = self._get_adjusted_value(
                    asset,
                    field,
                    previous_dt,
                    perspective_dt,
                    data_frequency,
                )
                df.iloc[0, missing_loc] = previous_value

            df.fillna(method='ffill', inplace=True)

            for asset in df.columns:
                if df.index[-1] >= asset.end_date:
                    # if the window extends past the asset's end date, set
                    # all post-end-date values to NaN in that asset's series
                    series = df[asset]
                    series[series.index >= asset.end_date] = np.NaN

        return df

    def _get_minute_window_for_assets(self, assets, field, minutes_for_window):
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
        if isinstance(assets, Future):
            return self._get_minute_window_for_future([assets], field,
                                                      minutes_for_window)
        else:
            # TODO: Make caller accept assets.
            window = self._get_minute_window_for_equities(assets, field,
                                                          minutes_for_window)
            return window

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

    def _get_minute_window_for_equities(
            self, assets, field, minutes_for_window):
        # each sid's minutes are stored in a bcolz file
        # the bcolz file has 390 bars per day, regardless
        # of when the asset started trading and regardless of half days.
        # for a half day, the second half is filled with zeroes.
        # all the minutely bcolz files start on the same day.

        try:
            start_idx = self._equity_minute_reader._find_position_of_minute(
                minutes_for_window[0])
        except KeyError:
            start_idx = 0

        try:
            end_idx = self._equity_minute_reader._find_position_of_minute(
                minutes_for_window[-1]) + 1
        except KeyError:
            end_idx = 0

        if end_idx == 0:
            # No data to return for minute window.
            return np.full((len(minutes_for_window), len(assets), np.nan))

        num_minutes = len(minutes_for_window)
        start_date = normalize_date(minutes_for_window[0])
        end_date = normalize_date(minutes_for_window[-1])

        return_data = np.zeros((len(minutes_for_window), len(assets)),
                               dtype=np.float64)

        for i, asset in enumerate(assets):
            # find the position of start_dt in the entire timeline, go back
            # bar_count bars, and that's the unadjusted data
            raw_data = self._equity_minute_reader._open_minute_file(
                field, asset)

            data_to_copy = raw_data[start_idx:end_idx]

            # data_to_copy contains all the zeros (from 1pm to 4pm of an early
            # close).  num_minutes is the number of actual trading minutes.  if
            # these two have different lengths, that means that we need to trim
            # away data due to early closes.
            if len(data_to_copy) != num_minutes:
                # get a copy of the minutes in Eastern time, since we depend on
                # an early close being at 1pm Eastern.
                eastern_minutes = minutes_for_window.tz_convert("US/Eastern")

                # accumulate a list of indices of the last minute of an early
                # close day.  For example, if data_to_copy starts at 12:55 pm,
                # and there are five minutes of real data before 180 zeroes,
                # we would put 5 into last_minute_idx_of_early_close_day,
                # because the fifth minute is the last "real" minute of
                # the day.
                last_minute_idx_of_early_close_day = []
                for minute_idx, minute_dt in enumerate(eastern_minutes):
                    if minute_idx == (num_minutes - 1):
                        break

                    if minute_dt.hour == 13 and minute_dt.minute == 0:
                        next_minute = eastern_minutes[minute_idx + 1]
                        if next_minute.hour != 13:
                            # minute_dt is the last minute of an early close
                            # day
                            last_minute_idx_of_early_close_day.append(
                                minute_idx)

                # spin through the list of early close markers, and use them to
                # chop off 180 minutes at a time from data_to_copy.
                for idx, early_close_minute_idx in enumerate(
                        last_minute_idx_of_early_close_day):
                    early_close_minute_idx -= (180 * idx)
                    data_to_copy = np.delete(
                        data_to_copy,
                        range(
                            early_close_minute_idx + 1,
                            early_close_minute_idx + 181
                        )
                    )

            return_data[0:len(data_to_copy), i] = data_to_copy
            if start_date != end_date:
                self._apply_all_adjustments(
                    return_data[:, i],
                    asset,
                    minutes_for_window,
                    field,
                    self.MINUTE_PRICE_ADJUSTMENT_FACTOR
                )

        if start_date == end_date:
            if field != 'volume':
                # TODO: Use reader for this value.
                return_data[return_data == 0] = np.nan
                return_data *= self.MINUTE_PRICE_ADJUSTMENT_FACTOR

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

    def _equity_daily_reader_arrays(self, field, dts, assets):
        # Temporary cache shim before loader is pulled in.
        # Custom memoization, because of unhashable types.
        assets_key = frozenset(assets)
        key = (field, dts[0], dts[-1], assets_key)
        if self._equity_daily_reader_array_keys[field] == key:
            return self._equity_daily_reader_array_data[field]
        else:
            data = self._equity_history_loader.history(assets,
                                                       dts,
                                                       field)
            self._equity_daily_reader_array_keys[field] = key
            self._equity_daily_reader_array_data[field] = data
            return data

    def _get_daily_window_for_sids(
            self, assets, field, days_in_window, extra_slot=True):
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
            return_array = np.zeros((bar_count + 1, len(assets)))
        else:
            return_array = np.zeros((bar_count, len(assets)))

        if field != "volume":
            # volumes default to 0, so we don't need to put NaNs in the array
            return_array[:] = np.NAN

        data = self._equity_daily_reader_arrays(field,
                                                days_in_window,
                                                assets)

        if extra_slot:
            return_array[:len(return_array) - 1, :] = data
        else:
            return_array[:len(data)] = data
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

    def _check_is_currently_alive(self, asset, dt):
        sid = int(asset)

        if sid not in self._asset_start_dates:
            self._get_asset_start_date(asset)

        start_date = self._asset_start_dates[sid]
        if self._asset_start_dates[sid] > dt:
            raise NoTradeDataAvailableTooEarly(
                sid=sid,
                dt=normalize_date(dt),
                start_dt=start_date
            )

        end_date = self._asset_end_dates[sid]
        if self._asset_end_dates[sid] < dt:
            raise NoTradeDataAvailableTooLate(
                sid=sid,
                dt=normalize_date(dt),
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
            if self._first_trading_day is not None:
                self._asset_start_dates[sid] = \
                    max(asset.start_date, self._first_trading_day)
            else:
                self._asset_start_dates[sid] = asset.start_date

            self._asset_end_dates[sid] = asset.end_date

    def get_splits(self, sids, dt):
        """
        Returns any splits for the given sids and the given dt.

        Parameters
        ----------
        sids : container
            Sids for which we want splits.

        dt: pd.Timestamp
            The date for which we are checking for splits.  Note: this is
            expected to be midnight UTC.

        Returns
        -------
        list: List of splits, where each split is a (sid, ratio) tuple.
        """
        if self._adjustment_reader is None or not sids:
            return {}

        # convert dt to # of seconds since epoch, because that's what we use
        # in the adjustments db
        seconds = int(dt.value / 1e9)

        splits = self._adjustment_reader.conn.execute(
            "SELECT sid, ratio FROM SPLITS WHERE effective_date = ?",
            (seconds,)).fetchall()

        splits = [split for split in splits if split[0] in sids]

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

    def get_fetcher_assets(self, day):
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

        if day in self._extra_source_df.index:
            date_to_use = day
        else:
            # current day isn't in the fetcher df, go back the last
            # available day
            idx = self._extra_source_df.index.searchsorted(day)
            if idx == 0:
                return []

            date_to_use = self._extra_source_df.index[idx - 1]

        asset_list = self._extra_source_df.loc[date_to_use]["sid"]

        # make sure they're actually assets
        asset_list = [asset for asset in asset_list
                      if isinstance(asset, Asset)]

        return asset_list

    def get_simple_transform(self, asset, transform_name, dt, data_frequency,
                             bars=None):
        if transform_name == "returns":
            # returns is always calculated over the last 2 days, regardless
            # of the simulation's data frequency.
            hst = self.get_history_window(
                [asset], dt, 2, "1d", "price", ffill=True
            )[asset]

            return (hst.iloc[-1] - hst.iloc[0]) / hst.iloc[0]

        if bars is None:
            raise ValueError("bars cannot be None!")

        if data_frequency == "minute":
            freq_str = "1m"
        else:
            freq_str = "1d"

        price_arr = self.get_history_window(
            [asset], dt, bars, freq_str, "price", ffill=True
        )[asset]

        if transform_name == "mavg":
            return nanmean(price_arr)
        elif transform_name == "stddev":
            return nanstd(price_arr, ddof=1)
        elif transform_name == "vwap":
            volume_arr = self.get_history_window(
                [asset], dt, bars, freq_str, "volume", ffill=True
            )[asset]

            vol_sum = nansum(volume_arr)

            try:
                ret = nansum(price_arr * volume_arr) / vol_sum
            except ZeroDivisionError:
                ret = np.nan

            return ret
