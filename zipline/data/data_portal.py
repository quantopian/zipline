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
from zipline.data.us_equity_loader import (
    USEquityDailyHistoryLoader,
    USEquityMinuteHistoryLoader,
)

from zipline.utils import tradingcalendar
from zipline.utils.math_utils import (
    nansum,
    nanmean,
    nanstd
)
from zipline.utils.memoize import remember_last, weak_lru_cache
from zipline.errors import (
    NoTradeDataAvailableTooEarly,
    NoTradeDataAvailableTooLate,
    HistoryWindowStartsBeforeData,
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


class DailyHistoryAggregator(object):
    """
    Converts minute pricing data into a daily summary, to be used for the
    last slot in a call to history with a frequency of `1d`.

    This summary is the same as a daily bar rollup of minute data, with the
    distinction that the summary is truncated to the `dt` requested.
    i.e. the aggregation slides forward during a the course of simulation day.

    Provides aggregation for `open`, `high`, `low`, `close`, and `volume`.
    The aggregation rules for each price type is documented in their respective

    """

    def __init__(self, market_opens, minute_reader):
        self._market_opens = market_opens
        self._minute_reader = minute_reader

        # The caches are structured as (date, market_open, entries), where
        # entries is a dict of asset -> (last_visited_dt, value)
        #
        # Whenever an aggregation method determines the current value,
        # the entry for the respective asset should be overwritten with a new
        # entry for the current dt.value (int) and aggregation value.
        #
        # When the requested dt's date is different from date the cache is
        # flushed, so that the cache entries do not grow unbounded.
        #
        # Example cache:
        # cache = (date(2016, 3, 17),
        #          pd.Timestamp('2016-03-17 13:31', tz='UTC'),
        #          {
        #              1: (1458221460000000000, np.nan),
        #              2: (1458221460000000000, 42.0),
        #         })
        self._caches = {
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': None
        }

        # The int value is used for deltas to avoid extra computation from
        # creating new Timestamps.
        self._one_min = pd.Timedelta('1 min').value

    def _prelude(self, dt, field):
        date = dt.date()
        dt_value = dt.value
        cache = self._caches[field]
        if cache is None or cache[0] != date:
            market_open = self._market_opens.loc[date]
            cache = self._caches[field] = (dt.date(), market_open, {})

        _, market_open, entries = cache
        if dt != market_open:
            prev_dt = dt_value - self._one_min
        else:
            prev_dt = None
        return market_open, prev_dt, dt_value, entries

    def opens(self, assets, dt):
        """
        The open field's aggregation returns the first value that occurs
        for the day, if there has been no data on or before the `dt` the open
        is `nan`.

        Once the first non-nan open is seen, that value remains constant per
        asset for the remainder of the day.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'open')

        opens = []
        normalized_date = normalize_date(dt)

        for asset in assets:
            if not asset._is_alive(normalized_date, True):
                opens.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'open')
                entries[asset] = (dt_value, val)
                opens.append(val)
                continue
            else:
                try:
                    last_visited_dt, first_open = entries[asset]
                    if last_visited_dt == dt_value:
                        opens.append(first_open)
                        continue
                    elif not pd.isnull(first_open):
                        opens.append(first_open)
                        entries[asset] = (dt_value, first_open)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['open'],
                            after_last,
                            dt,
                            [asset],
                        )[0]
                        nonnan = window[~pd.isnull(window)]
                        if len(nonnan):
                            val = nonnan[0]
                        else:
                            val = np.nan
                        entries[asset] = (dt_value, val)
                        opens.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['open'],
                        market_open,
                        dt,
                        [asset],
                    )[0]
                    nonnan = window[~pd.isnull(window)]
                    if len(nonnan):
                        val = nonnan[0]
                    else:
                        val = np.nan
                    entries[asset] = (dt_value, val)
                    opens.append(val)
                    continue
        return np.array(opens)

    def highs(self, assets, dt):
        """
        The high field's aggregation returns the largest high seen between
        the market open and the current dt.
        If there has been no data on or before the `dt` the high is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'high')

        highs = []
        normalized_date = normalize_date(dt)

        for asset in assets:
            if not asset._is_alive(normalized_date, True):
                highs.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'high')
                entries[asset] = (dt_value, val)
                highs.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_max = entries[asset]
                    if last_visited_dt == dt_value:
                        highs.append(last_max)
                        continue
                    elif last_visited_dt == prev_dt:
                        curr_val = self._minute_reader.get_value(
                            asset, dt, 'high')
                        if pd.isnull(curr_val):
                            val = last_max
                        elif pd.isnull(last_max):
                            val = curr_val
                        else:
                            val = max(last_max, curr_val)
                        entries[asset] = (dt_value, val)
                        highs.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['high'],
                            after_last,
                            dt,
                            [asset],
                        )[0].T
                        val = max(last_max, np.nanmax(window))
                        entries[asset] = (dt_value, val)
                        highs.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['high'],
                        market_open,
                        dt,
                        [asset],
                    )[0].T
                    val = np.nanmax(window)
                    entries[asset] = (dt_value, val)
                    highs.append(val)
                    continue
        return np.array(highs)

    def lows(self, assets, dt):
        """
        The low field's aggregation returns the smallest low seen between
        the market open and the current dt.
        If there has been no data on or before the `dt` the low is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'low')

        lows = []
        normalized_date = normalize_date(dt)

        for asset in assets:
            if not asset._is_alive(normalized_date, True):
                lows.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'low')
                entries[asset] = (dt_value, val)
                lows.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_min = entries[asset]
                    if last_visited_dt == dt_value:
                        lows.append(last_min)
                        continue
                    elif last_visited_dt == prev_dt:
                        curr_val = self._minute_reader.get_value(
                            asset, dt, 'low')
                        val = np.nanmin([last_min, curr_val])
                        entries[asset] = (dt_value, val)
                        lows.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['low'],
                            after_last,
                            dt,
                            [asset],
                        )[0].T
                        window_min = np.nanmin(window)
                        if pd.isnull(window_min):
                            val = last_min
                        else:
                            val = min(last_min, window_min)
                        entries[asset] = (dt_value, val)
                        lows.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['low'],
                        market_open,
                        dt,
                        [asset],
                    )[0].T
                    val = np.nanmin(window)
                    entries[asset] = (dt_value, val)
                    lows.append(val)
                    continue
        return np.array(lows)

    def closes(self, assets, dt):
        """
        The close field's aggregation returns the latest close at the given
        dt.
        If the close for the given dt is `nan`, the most recent non-nan
        `close` is used.
        If there has been no data on or before the `dt` the close is `nan`.

        Returns
        -------
        np.array with dtype=float64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'close')

        closes = []
        normalized_dt = normalize_date(dt)

        for asset in assets:
            if not asset._is_alive(normalized_dt, True):
                closes.append(np.NaN)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'close')
                entries[asset] = (dt_value, val)
                closes.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_close = entries[asset]
                    if last_visited_dt == dt_value:
                        closes.append(last_close)
                        continue
                    elif last_visited_dt == prev_dt:
                        val = self._minute_reader.get_value(
                            asset, dt, 'close')
                        if pd.isnull(val):
                            val = last_close
                        entries[asset] = (dt_value, val)
                        closes.append(val)
                        continue
                    else:
                        val = self._minute_reader.get_value(
                            asset, dt, 'close')
                        if pd.isnull(val):
                            val = self.closes(
                                [asset],
                                pd.Timestamp(prev_dt, tz='UTC'))[0]
                        entries[asset] = (dt_value, val)
                        closes.append(val)
                        continue
                except KeyError:
                    val = self._minute_reader.get_value(
                        asset, dt, 'close')
                    if pd.isnull(val):
                        val = self.closes([asset],
                                          pd.Timestamp(prev_dt, tz='UTC'))[0]
                    entries[asset] = (dt_value, val)
                    closes.append(val)
                    continue
        return np.array(closes)

    def volumes(self, assets, dt):
        """
        The volume field's aggregation returns the sum of all volumes
        between the market open and the `dt`
        If there has been no data on or before the `dt` the volume is 0.

        Returns
        -------
        np.array with dtype=int64, in order of assets parameter.
        """
        market_open, prev_dt, dt_value, entries = self._prelude(dt, 'volume')

        volumes = []
        normalized_date = normalize_date(dt)

        for asset in assets:
            if not asset._is_alive(normalized_date, True):
                volumes.append(0)
                continue

            if prev_dt is None:
                val = self._minute_reader.get_value(asset, dt, 'volume')
                entries[asset] = (dt_value, val)
                volumes.append(val)
                continue
            else:
                try:
                    last_visited_dt, last_total = entries[asset]
                    if last_visited_dt == dt_value:
                        volumes.append(last_total)
                        continue
                    elif last_visited_dt == prev_dt:
                        val = self._minute_reader.get_value(
                            asset, dt, 'volume')
                        val += last_total
                        entries[asset] = (dt_value, val)
                        volumes.append(val)
                        continue
                    else:
                        after_last = pd.Timestamp(
                            last_visited_dt + self._one_min, tz='UTC')
                        window = self._minute_reader.load_raw_arrays(
                            ['volume'],
                            after_last,
                            dt,
                            [asset],
                        )[0]
                        val = np.nansum(window) + last_total
                        entries[asset] = (dt_value, val)
                        volumes.append(val)
                        continue
                except KeyError:
                    window = self._minute_reader.load_raw_arrays(
                        ['volume'],
                        market_open,
                        dt,
                        [asset],
                    )[0]
                    val = np.nansum(window)
                    entries[asset] = (dt_value, val)
                    volumes.append(val)
                    continue
        return np.array(volumes)


class DataPortal(object):
    """Interface to all of the data that a zipline simulation needs.

    This is used by the simulation runner to answer questions about the data,
    like getting the prices of assets on a given day or to service history
    calls.

    Parameters
    ----------
    env : TradingEnvironment
        The trading environment for the simulation. This includes the trading
        calendar and benchmark data.
    equity_daily_reader : BcolzDailyBarReader, optional
        The daily bar ready for equities. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    equity_minute_reader : BcolzMinuteBarReader, optional
        The minute bar reader for equities. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    future_daily_reader : BcolzDailyBarReader, optional
        The daily bar ready for futures. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    future_minute_reader : BcolzMinuteBarReader, optional
        The minute bar reader for futures. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    adjustment_reader : SQLiteAdjustmentWriter, optional
        The adjustment reader. This is used to apply splits, dividends, and
        other adjustment data to the raw data from the readers.
    """
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
            self._equity_history_loader = USEquityDailyHistoryLoader(
                self.env,
                self._equity_daily_reader,
                self._adjustment_reader
            )
        self._equity_minute_reader = equity_minute_reader
        self._future_daily_reader = future_daily_reader
        self._future_minute_reader = future_minute_reader

        self._first_trading_day = None

        if self._equity_minute_reader is not None:
            self._equity_daily_aggregator = DailyHistoryAggregator(
                self.env.open_and_closes.market_open,
                self._equity_minute_reader)
            self._equity_minute_history_loader = USEquityMinuteHistoryLoader(
                self.env,
                self._equity_minute_reader,
                self._adjustment_reader
            )
            self.MINUTE_PRICE_ADJUSTMENT_FACTOR = \
                self._equity_minute_reader._ohlc_inverse

        # get the first trading day from our readers.
        if self._equity_daily_reader is not None:
            self._first_trading_day = \
                self._equity_daily_reader.first_trading_day
        elif self._equity_minute_reader is not None:
            self._first_trading_day = \
                self._equity_minute_reader.first_trading_day

    def _reindex_extra_source(self, df, source_date_index):
        return df.reindex(index=source_date_index, method='ffill')

    def handle_extra_source(self, source_df, sim_params):
        """
        Extra sources always have a sid column.

        We expand the given data (by forward filling) to the full range of
        the simulation dates, so that lookup is fast during simulation.
        """
        if source_df is None:
            return

        # Normalize all the dates in the df
        source_df.index = source_df.index.normalize()

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
        source_date_index = self.env.days_in_range(
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

        # This will be the dataframe which we query to get fetcher assets at
        # any given time. Get's overwritten every time there's a new fetcher
        # call
        extra_source_df = pd.DataFrame()

        for identifier, df in iteritems(group_dict):
            # Before reindexing, save the earliest and latest dates
            earliest_date = df.index[0]
            latest_date = df.index[-1]

            # Since we know this df only contains a single sid, we can safely
            # de-dupe by the index (dt). If minute granularity, will take the
            # last data point on any given day
            df = df.groupby(level=0).last()

            # Reindex the dataframe based on the backtest start/end date.
            # This makes reads easier during the backtest.
            df = self._reindex_extra_source(df, source_date_index)

            if not isinstance(identifier, Asset):
                # for fake assets we need to store a start/end date
                self._asset_start_dates[identifier] = earliest_date
                self._asset_end_dates[identifier] = latest_date

            for col_name in df.columns.difference(['sid']):
                if col_name not in self._augmented_sources_map:
                    self._augmented_sources_map[col_name] = {}

                self._augmented_sources_map[col_name][identifier] = df

            # Append to extra_source_df the reindexed dataframe for the single
            # sid
            extra_source_df = extra_source_df.append(df)

        self._extra_source_df = extra_source_df

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

    @staticmethod
    def _is_extra_source(asset, field, map):
        """
        Internal method that determines if this asset/field combination
        represents a fetcher value or a regular OHLCVP lookup.
        """
        # If we have an extra source with a column called "price", only look
        # at it if it's on something like palladium and not AAPL (since our
        # own price data always wins when dealing with assets).

        return not (field in BASE_FIELDS and isinstance(asset, Asset))

    def _get_fetcher_value(self, asset, field, dt):
        day = normalize_date(dt)

        try:
            return \
                self._augmented_sources_map[field][asset].loc[day, field]
        except KeyError:
            return np.NaN

    def get_spot_value(self, asset, field, dt, data_frequency):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.
        field : {'open', 'high', 'low', 'close', 'volume',
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        value : float, int, or pd.Timestamp
            The spot value of ``field`` for ``asset`` The return type is based
            on the ``field`` requested. If the field is one of 'open', 'high',
            'low', 'close', or 'price', the value will be a float. If the
            ``field`` is 'volume' the value will be a int. If the ``field`` is
            'last_traded' the value will be a Timestamp.
        """
        if self._is_extra_source(asset, field, self._augmented_sources_map):
            return self._get_fetcher_value(asset, field, dt)

        if field not in BASE_FIELDS:
            raise KeyError("Invalid column: " + str(field))

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

    def get_adjustments(self, assets, field, dt, perspective_dt):
        """
        Returns a list of adjustments between the dt and perspective_dt for the
        given field and list of assets

        Parameters
        ----------
        assets : list of type Asset, or Asset
            The asset, or assets whose adjustments are desired.
        field : {'open', 'high', 'low', 'close', 'volume', \
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        perspective_dt : pd.Timestamp
            The timestamp from which the data is being viewed back from.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        adjustments : list[Adjustment]
            The adjustments to that field.
        """
        if isinstance(assets, Asset):
            assets = [assets]

        adjustment_ratios_per_asset = []
        split_adj_factor = lambda x: x if field != 'volume' else 1.0 / x

        for asset in assets:
            adjustments_for_asset = []
            split_adjustments = self._get_adjustment_list(
                asset, self._splits_dict, "SPLITS"
            )
            for adj_dt, adj in split_adjustments:
                if dt <= adj_dt <= perspective_dt:
                    adjustments_for_asset.append(split_adj_factor(adj))
                elif adj_dt > perspective_dt:
                    break

            if field != 'volume':
                merger_adjustments = self._get_adjustment_list(
                    asset, self._mergers_dict, "MERGERS"
                )
                for adj_dt, adj in merger_adjustments:
                    if dt <= adj_dt <= perspective_dt:
                        adjustments_for_asset.append(adj)
                    elif adj_dt > perspective_dt:
                        break

                dividend_adjustments = self._get_adjustment_list(
                    asset, self._dividends_dict, "DIVIDENDS",
                )
                for adj_dt, adj in dividend_adjustments:
                    if dt <= adj_dt <= perspective_dt:
                        adjustments_for_asset.append(adj)
                    elif adj_dt > perspective_dt:
                        break

            ratio = reduce(mul, adjustments_for_asset, 1.0)
            adjustment_ratios_per_asset.append(ratio)

        return adjustment_ratios_per_asset

    def get_adjusted_value(self, asset, field, dt,
                           perspective_dt,
                           data_frequency,
                           spot_value=None):
        """
        Returns a scalar value representing the value
        of the desired asset's field at the given dt with adjustments applied.

        Parameters
        ----------
        asset : Asset
            The asset whose data is desired.
        field : {'open', 'high', 'low', 'close', 'volume', \
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        perspective_dt : pd.Timestamp
            The timestamp from which the data is being viewed back from.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        value : float, int, or pd.Timestamp
            The value of the given ``field`` for ``asset`` at ``dt`` with any
            adjustments known by ``perspective_dt`` applied. The return type is
            based on the ``field`` requested. If the field is one of 'open',
            'high', 'low', 'close', or 'price', the value will be a float. If
            the ``field`` is 'volume' the value will be a int. If the ``field``
            is 'last_traded' the value will be a Timestamp.
        """
        if spot_value is None:
            # if this a fetcher field, we want to use perspective_dt (not dt)
            # because we want the new value as of midnight (fetcher only works
            # on a daily basis, all timestamps are on midnight)
            if self._is_extra_source(asset, field,
                                     self._augmented_sources_map):
                spot_value = self.get_spot_value(asset, field, perspective_dt,
                                                 data_frequency)
            else:
                spot_value = self.get_spot_value(asset, field, dt,
                                                 data_frequency)

        if isinstance(asset, Equity):
            ratio = self.get_adjustments(asset, field, dt, perspective_dt)[0]
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
        return self.get_adjusted_value(
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
                            return self.get_adjusted_value(
                                asset, column, found_dt, dt, "minute",
                                spot_value=value
                            )
                    else:
                        found_dt -= tradingcalendar.trading_day
                except NoDataOnDate:
                    return np.nan

    @remember_last
    def _get_days_for_window(self, end_date, bar_count):
        tds = self.env.trading_days
        end_loc = self.env.trading_days.get_loc(end_date)
        start_loc = end_loc - bar_count + 1
        if start_loc < 0:
            raise HistoryWindowStartsBeforeData(
                first_trading_day=self.env.first_trading_day.date(),
                bar_count=bar_count,
                suggested_start_day=tds[bar_count].date(),
            )
        return tds[start_loc:end_loc + 1]

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
            # minute mode, requesting '1d'
            daily_data = self._get_daily_window_for_sids(
                assets,
                field_to_use,
                days_for_window[0:-1]
            )

            if field_to_use == 'open':
                minute_value = self._equity_daily_aggregator.opens(
                    assets, end_dt)
            elif field_to_use == 'high':
                minute_value = self._equity_daily_aggregator.highs(
                    assets, end_dt)
            elif field_to_use == 'low':
                minute_value = self._equity_daily_aggregator.lows(
                    assets, end_dt)
            elif field_to_use == 'close':
                minute_value = self._equity_daily_aggregator.closes(
                    assets, end_dt)
            elif field_to_use == 'volume':
                minute_value = self._equity_daily_aggregator.volumes(
                    assets, end_dt)

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
        mm = self.env.market_minutes
        end_loc = mm.get_loc(end_dt)
        start_loc = end_loc - bar_count + 1
        if start_loc < 0:
            suggested_start_day = (mm[bar_count] + self.env.trading_day).date()
            raise HistoryWindowStartsBeforeData(
                first_trading_day=self.env.first_trading_day.date(),
                bar_count=bar_count,
                suggested_start_day=suggested_start_day,
            )
        minutes_for_window = mm[start_loc:end_loc + 1]

        asset_minute_data = self._get_minute_window_for_assets(
            assets,
            field_to_use,
            minutes_for_window,
        )

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
        ----------
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
                previous_value = self.get_adjusted_value(
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
                    series[series.index.normalize() > asset.end_date] = np.NaN

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
        return self._equity_minute_history_loader.history(assets,
                                                          minutes_for_window,
                                                          field)

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

            if price_adj_factor is not None:
                data *= price_adj_factor
                np.around(data, 3, out=data)

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

        if bar_count != 0:
            data = self._equity_history_loader.history(assets,
                                                       days_in_window,
                                                       field)
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
        dt : pd.Timestamp
            The date for which we are checking for splits. Note: this is
            expected to be midnight UTC.

        Returns
        -------
        splits : list[(int, float)]
            List of splits, where each split is a (sid, ratio) tuple.
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

    def get_fetcher_assets(self, dt):
        """
        Returns a list of assets for the current date, as defined by the
        fetcher data.

        Returns
        -------
        list: a list of Asset objects.
        """
        # return a list of assets for the current date, as defined by the
        # fetcher source
        if self._extra_source_df is None:
            return []

        day = normalize_date(dt)

        if day in self._extra_source_df.index:
            assets = self._extra_source_df.loc[day]['sid']
        else:
            return []

        if isinstance(assets, pd.Series):
            return [x for x in assets if isinstance(x, Asset)]
        else:
            return [assets] if isinstance(assets, Asset) else []

    @weak_lru_cache(20)
    def _get_minute_count_for_transform(self, ending_minute, days_count):
        # cache size picked somewhat loosely.  this code exists purely to
        # handle deprecated API.

        # bars is the number of days desired.  we have to translate that
        # into the number of minutes we want.
        # we get all the minutes for the last (bars - 1) days, then add
        # all the minutes so far today.  the +2 is to account for ignoring
        # today, and the previous day, in doing the math.
        previous_day = self.env.previous_trading_day(ending_minute)
        days = self.env.days_in_range(
            self.env.add_trading_days(-days_count + 2, previous_day),
            previous_day,
        )

        minutes_count = \
            sum(210 if day in self.env.early_closes else 390 for day in days)

        # add the minutes for today
        today_open = self.env.get_open_and_close(ending_minute)[0]
        minutes_count += \
            ((ending_minute - today_open).total_seconds() // 60) + 1

        return minutes_count

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
            calculated_bar_count = self._get_minute_count_for_transform(
                dt, bars
            )
        else:
            freq_str = "1d"
            calculated_bar_count = bars

        price_arr = self.get_history_window(
            [asset], dt, calculated_bar_count, freq_str, "price", ffill=True
        )[asset]

        if transform_name == "mavg":
            return nanmean(price_arr)
        elif transform_name == "stddev":
            return nanstd(price_arr, ddof=1)
        elif transform_name == "vwap":
            volume_arr = self.get_history_window(
                [asset], dt, calculated_bar_count, freq_str, "volume",
                ffill=True
            )[asset]

            vol_sum = nansum(volume_arr)

            try:
                ret = nansum(price_arr * volume_arr) / vol_sum
            except ZeroDivisionError:
                ret = np.nan

            return ret
