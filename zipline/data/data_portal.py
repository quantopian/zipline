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
from sqlite3 import OperationalError
import bcolz
from logbook import Logger

import numpy as np
import pandas as pd

from zipline.assets import Asset

from zipline.utils import tradingcalendar
from zipline.errors import (
    NoTradeDataAvailableTooEarly,
    NoTradeDataAvailableTooLate
)

# FIXME anything to do with 2002-01-02 probably belongs in qexec, right/
FIRST_TRADING_DAY = pd.Timestamp("2002-01-02 00:00:00", tz='UTC')
FIRST_TRADING_MINUTE = pd.Timestamp("2002-01-02 14:31:00", tz='UTC')

# FIXME should this be passed in (is this qexec specific?)?
INDEX_OF_FIRST_TRADING_DAY = 3028

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


class DataPortal(object):
    def __init__(self,
                 env,
                 sim_params=None,
                 minutes_equities_path=None,
                 daily_equities_path=None,
                 adjustment_reader=None,
                 asset_finder=None,
                 sid_path_func=None):
        self.env = env
        self.current_dt = None
        self.current_day = None
        self.cur_data_offset = 0

        self.views = {}

        if minutes_equities_path is None and daily_equities_path is None:
            raise ValueError("Must provide at least one of minute or "
                             "daily data path!")

        self.minutes_equities_path = minutes_equities_path
        self.daily_equities_path = daily_equities_path
        self.asset_finder = asset_finder

        self.carrays = {
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
        self.splits_dict = {}
        self.mergers_dict = {}
        self.dividends_dict = {}

        # Pointer to the daily bcolz file.
        self.daily_equities_data = None

        # Cache of int -> the first trading day of an asset, even if that day
        # is before 1/2/2002.
        self.asset_start_dates = {}
        self.asset_end_dates = {}

        self.augmented_sources_map = {}
        self.fetcher_df = None

        self.sim_params = sim_params
        if self.sim_params is not None:
            self.data_frequency = self.sim_params.data_frequency

        self.sid_path_func = sid_path_func

        self.DAILY_PRICE_ADJUSTMENT_FACTOR = 0.001
        self.MINUTE_PRICE_ADJUSTMENT_FACTOR = 0.001

    def handle_extra_source(self, source_df):
        """
        Extra sources always have a sid column.

        We expand the given data (by forward filling) to the full range of
        the simulation dates, so that lookup is fast during simulation.
        """
        if source_df is None:
            return

        self.fetcher_df = source_df

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
        # Finally, we store the data. For each fetcher column, we store a
        # mapping in self.augmented_sources_map from it to a dictionary of
        # asset -> df.  In other words,
        # self.augmented_sources_map['days_to_cover']['AAPL'] gives us the df
        # holding that data.

        if self.sim_params.emission_rate == "daily":
            fetcher_date_index = self.env.days_in_range(
                start=self.sim_params.period_start,
                end=self.sim_params.period_end
            )
        else:
            fetcher_date_index = self.env.minutes_for_days_in_range(
                start=self.sim_params.period_start,
                end=self.sim_params.period_end
            )

        # break the source_df up into one dataframe per sid.  this lets
        # us (more easily) calculate accurate start/end dates for each sid,
        # de-dup data, and expand the data to fit the backtest start/end date.
        grouped_by_sid = source_df.groupby(["sid"])
        group_names = grouped_by_sid.groups.keys()
        group_dict = {}
        for group_name in group_names:
            group_dict[group_name] = grouped_by_sid.get_group(group_name)

        for identifier, df in group_dict.iteritems():
            # before reindexing, save the earliest and latest dates
            earliest_date = df.index[0]
            latest_date = df.index[-1]

            # since we know this df only contains a single sid, we can safely
            # de-dupe by the index (dt)
            df = df.groupby(level=0).last()

            # reindex the dataframe based on the backtest start/end date.
            # this makes reads easier during the backtest.
            df = df.reindex(index=fetcher_date_index, method='ffill')

            if not isinstance(identifier, Asset):
                # for fake assets we need to store a start/end date
                self.asset_start_dates[identifier] = earliest_date
                self.asset_end_dates[identifier] = latest_date

            for col_name in df.columns.difference(['sid']):
                if col_name not in self.augmented_sources_map:
                    self.augmented_sources_map[col_name] = {}

                self.augmented_sources_map[col_name][identifier] = df

    def _open_daily_file(self):
        if self.daily_equities_data is None:
            self.daily_equities_data = bcolz.open(self.daily_equities_path)
            self.daily_equities_attrs = self.daily_equities_data.attrs

        return self.daily_equities_data, self.daily_equities_attrs

    def _open_minute_file(self, field, sid):
        if self.sid_path_func is None:
            path = "{0}/{1}.bcolz".format(self.minutes_equities_path, sid)
        else:
            path = self.sid_path_func(self.minutes_equities_path, sid)

        try:
            carray = self.carrays[field][path]
        except KeyError:
            carray = self.carrays[field][path] = bcolz.carray(
                rootdir=path + "/" + field, mode='r')

        return carray

    def get_previous_price(self, asset, column, dt):
        if self.data_frequency == 'daily':
            prev_dt = self.env.previous_trading_day(dt)
        elif self.data_frequency == 'minute':
            prev_dt = self.env.previous_market_minute(dt)
        return self.get_spot_value(asset, column, prev_dt)

    def _check_fetcher(self, asset, column, day):
        # if there is a fetcher column called "price", only look at it if
        # it's on something like palladium and not AAPL (since our own price
        # data always wins when dealing with assets)
        look_in_augmented_sources = column in self.augmented_sources_map and \
            not (column in BASE_FIELDS and isinstance(asset, Asset))

        if look_in_augmented_sources:
            # we're being asked for a fetcher field
            try:
                return self.augmented_sources_map[column][asset].\
                    loc[day, column]
            except:
                log.error(
                    "Could not find value for asset={0}, current_day={1},"
                    "column={2}".format(
                        str(asset),
                        str(self.current_day),
                        str(column)))

                raise KeyError

    def get_spot_value(self, asset, column, dt=None):
        fetcher_val = self._check_fetcher(asset, column,
                                          (dt or self.current_dt))

        if fetcher_val:
            return fetcher_val

        if column not in BASE_FIELDS:
            raise KeyError("Invalid column: " + str(column))

        asset_int = int(asset)
        column_to_use = BASE_FIELDS[column]

        self._check_is_currently_alive(asset_int, dt)

        if self.data_frequency == "daily":
            day_to_use = dt or self.current_day
            return self._get_daily_data(asset_int, column_to_use, day_to_use)
        else:
            # keeping minute data logic in-lined to avoid the cost of calling
            # another method.
            carray = self._open_minute_file(column_to_use, asset_int)

            if dt is None:
                # hope cur_data_offset is set correctly, since we weren't
                # given a dt to use.
                minute_offset_to_use = self.cur_data_offset
            else:
                if dt == self.current_dt:
                    minute_offset_to_use = self.cur_data_offset
                else:
                    # dt was passed in, so calculate the offset.
                    # = (390 * number of trading days since 1/2/2002) +
                    #   (index of minute in day)
                    given_day = pd.Timestamp(dt.date(), tz='utc')
                    day_index = tradingcalendar.trading_days.searchsorted(
                        given_day) - INDEX_OF_FIRST_TRADING_DAY

                    # if dt is before the first market minute, minute_index
                    # will be 0.  if it's after the last market minute, it'll
                    # be len(minutes_for_day)
                    minute_index = self.env.market_minutes_for_day(given_day).\
                        searchsorted(dt)

                    minute_offset_to_use = (day_index * 390) + minute_index

            result = carray[minute_offset_to_use]
            if result == 0:
                # if the given minute doesn't have data, we need to seek
                # backwards until we find data. This makes the data
                # forward-filled.

                # get this asset's start date, so that we don't look before it.
                start_date = self._get_asset_start_date(asset_int)
                start_date_idx = tradingcalendar.trading_days.searchsorted(
                    start_date) - INDEX_OF_FIRST_TRADING_DAY
                start_day_offset = start_date_idx * 390

                original_start = minute_offset_to_use

                while result == 0 and minute_offset_to_use > start_day_offset:
                    minute_offset_to_use -= 1
                    result = carray[minute_offset_to_use]

                # once we've found data, we need to check whether it needs
                # to be adjusted.
                if result != 0:
                    minutes = self.env.market_minute_window(
                        start=dt,
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
                            sid=asset_int,
                            dts=minutes,
                            field=column
                        )

                        # The first value of the adjusted array is the value
                        # we want.
                        result = arr[0]

            if column_to_use != 'volume':
                return result * self.MINUTE_PRICE_ADJUSTMENT_FACTOR
            else:
                return result

    def _get_daily_data(self, asset_int, column, dt):
        dt = pd.Timestamp(dt.date(), tz='utc')
        daily_data, daily_attrs = self._open_daily_file()

        # find the start index in the daily file for this asset
        asset_file_index = daily_attrs['first_row'][str(asset_int)]

        # find when the asset started trading
        asset_data_start_date = max(self._get_asset_start_date(asset_int),
                                    FIRST_TRADING_DAY)

        tradingdays = tradingcalendar.trading_days

        # figure out how many days it's been between now and when this
        # asset starting trading
        # FIXME can cache tradingdays.searchsorted(asset_data_start_date)
        window_offset = tradingdays.searchsorted(dt) - \
            tradingdays.searchsorted(asset_data_start_date)

        # and use that offset to find our lookup index
        lookup_idx = asset_file_index + window_offset

        # sanity check
        assert lookup_idx >= asset_file_index
        assert lookup_idx <= daily_attrs['last_row'][str(asset_int)] + 1

        ctable = daily_data[column]
        raw_value = ctable[lookup_idx]

        while raw_value == 0 and lookup_idx > asset_file_index:
            lookup_idx -= 1
            raw_value = ctable[lookup_idx]

        if column != 'volume':
            return raw_value * self.DAILY_PRICE_ADJUSTMENT_FACTOR
        else:
            return raw_value

    def _get_history_daily_window(self, sids, end_dt, bar_count, field_to_use):
        """
        Internal method that returns a dataframe containing history bars
        of daily frequency for the given sids.
        """
        data = []

        day = end_dt.date()
        day_idx = tradingcalendar.trading_days.searchsorted(day)
        days_for_window = tradingcalendar.trading_days[
            (day_idx - bar_count + 1):(day_idx + 1)]

        ends_at_midnight = end_dt.hour == 0 and end_dt.minute == 0

        if len(sids) == 0:
            return pd.DataFrame(None,
                                index=days_for_window,
                                columns=None)

        for sid in sids:
            sid = int(sid)

            # get the start and end dates for this sid
            if sid not in self.asset_start_dates:
                asset = self.asset_finder.retrieve_asset(sid)
                self.asset_start_dates[sid] = asset.start_date
                self.asset_end_dates[sid] = asset.end_date

            if ends_at_midnight or \
                    (days_for_window[-1] > self.asset_end_dates[sid]):
                # two cases where we use daily data for the whole range:
                # 1) the history window ends at midnight utc.
                # 2) the last desired day of the window is after the
                # last trading day, use daily data for the whole range.
                data.append(self._get_daily_window_for_sid(
                    sid,
                    field_to_use,
                    days_for_window,
                    extra_slot=False
                ))
            else:
                # for the last day of the desired window, use minute
                # data and aggregate it.
                all_minutes_for_day = self.env.market_minutes_for_day(
                    pd.Timestamp(day))

                last_minute_idx = all_minutes_for_day.searchsorted(end_dt)

                # these are the minutes for the partial day
                minutes_for_partial_day =\
                    all_minutes_for_day[0:(last_minute_idx + 1)]

                daily_data = self._get_daily_window_for_sid(
                    sid,
                    field_to_use,
                    days_for_window[0:-1]
                )

                minute_data = self._get_minute_window_for_sid(
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

                data.append(daily_data)

        return pd.DataFrame(np.array(data).T,
                            index=days_for_window,
                            columns=sids)

    def _get_history_minute_window(self, sids, end_dt, bar_count,
                                   field_to_use):
        """
        Internal method that returns a dataframe containing history bars
        of minute frequency for the given sids.
        """
        # get all the minutes for this window
        minutes_for_window = self.env.market_minute_window(
            end_dt, bar_count, step=-1)[::-1]

        # but then cut it down to only the minutes after
        # FIRST_TRADING_MINUTE
        modified_minutes_for_window = minutes_for_window[
            minutes_for_window.slice_indexer(FIRST_TRADING_MINUTE)]

        modified_minutes_length = len(modified_minutes_for_window)

        if modified_minutes_length == 0:
            raise ValueError("Cannot calculate history window that ends"
                             "before 2002-01-02 14:31 UTC!")

        data = []
        bars_to_prepend = 0
        nans_to_prepend = None

        if modified_minutes_length < bar_count and \
           (modified_minutes_for_window[0] == FIRST_TRADING_MINUTE):
            # the beginning of the window goes before our global trading
            # start date
            bars_to_prepend = bar_count - modified_minutes_length
            nans_to_prepend = np.repeat(np.nan, bars_to_prepend)

        if len(sids) == 0:
            return pd.DataFrame(
                None,
                index=modified_minutes_for_window,
                columns=None
            )

        for sid in sids:
            sid_minute_data = self._get_minute_window_for_sid(
                int(sid),
                field_to_use,
                modified_minutes_for_window
            )

            if bars_to_prepend != 0:
                sid_minute_data = np.insert(sid_minute_data, 0,
                                            nans_to_prepend)

            data.append(sid_minute_data)

        return pd.DataFrame(np.array(data).T,
                            index=minutes_for_window,
                            columns=sids)

    def get_history_window(self, sids, end_dt, bar_count, frequency, field,
                           ffill=True):
        """
        Public API method that returns a dataframe containing the requested
        history window.  Data is fully adjusted.

        Parameters
        ---------
        sids : list
            The sids whose data is desired.

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

        if frequency == "1d":
            df = self._get_history_daily_window(sids, end_dt, bar_count,
                                                field_to_use)
        elif frequency == "1m":
            df = self._get_history_minute_window(sids, end_dt, bar_count,
                                                 field_to_use)
        else:
            raise ValueError("Invalid frequency: {0}".format(frequency))

        # forward-fill if needed
        if field == "price" and ffill:
            df.fillna(method='ffill', inplace=True)

        return df

    def _get_minute_window_for_sid(self, sid, field, minutes_for_window):
        """
        Internal method that gets a window of adjusted minute data for a sid
        and specified date range.  Used to support the history API method for
        minute bars.

        Missing bars are filled with NaN.

        Parameters
        ----------
        sid : int
            The sid whose data is desired.

        field: string
            The specific field to return.  "open", "high", "close_price", etc.

        minutes_for_window: pd.DateTimeIndex
            The list of minutes representing the desired window.  Each minute
            is a pd.Timestamp.

        Returns
        -------
        A numpy array with requested values.
        """
        # each sid's minutes are stored in a bcolz file
        # the bcolz file has 390 bars per day, starting at 1/2/2002, regardless
        # of when the asset started trading and regardless of half days.
        # for a half day, the second half is filled with zeroes.

        # find the position of start_dt in the entire timeline, go back
        # bar_count bars, and that's the unadjusted data
        raw_data = self._open_minute_file(field, sid)

        start_idx = max(self._find_position_of_minute(minutes_for_window[0]),
                        0)
        end_idx = self._find_position_of_minute(minutes_for_window[-1]) + 1

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
            sid,
            minutes_for_window,
            field,
            self.MINUTE_PRICE_ADJUSTMENT_FACTOR
        )

        return return_data

    def _apply_all_adjustments(self, data, sid, dts, field,
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

        sid: integer
            The sid whose data is being adjusted.

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
                sid, self.splits_dict, "SPLITS"
            ),
            data,
            dts,
            field != 'volume'
        )

        if field != 'volume':
            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    sid, self.mergers_dict, "MERGERS"
                ),
                data,
                dts,
                True
            )

            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    sid, self.dividends_dict, "DIVIDENDS"
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
    def _find_position_of_minute(minute_dt):
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
        day_idx = tradingcalendar.trading_days.searchsorted(day) -\
            INDEX_OF_FIRST_TRADING_DAY

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

    def _get_daily_window_for_sid(self, sid, field, days_in_window,
                                  extra_slot=True):
        """
        Internal method that gets a window of adjusted daily data for a sid
        and specified date range.  Used to support the history API method for
        daily bars.

        Parameters
        ----------
        sid : int
            The sid whose data is desired.

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
        daily_data, daily_attrs = self._open_daily_file()

        # the daily file stores each sid's daily OHLCV in a contiguous block.
        # the first row per sid is either 1/2/2002, or the sid's start_date if
        # it started after 1/2/2002.  once a sid stops trading, there are no
        # rows for it.

        bar_count = len(days_in_window)

        # create an np.array of size bar_count
        if extra_slot:
            return_array = np.zeros((bar_count + 1,))
        else:
            return_array = np.zeros((bar_count,))

        return_array[:] = np.NAN

        # find the start index in the daily file for this asset
        asset_file_index = daily_attrs['first_row'][str(sid)]

        trading_days = tradingcalendar.trading_days

        # Calculate the starting day to use (either the asset's first trading
        # day, or 1/1/2002 (which is the 3028th day in the trading calendar).
        first_trading_day_to_use = max(trading_days.searchsorted(
            self.asset_start_dates[sid]), INDEX_OF_FIRST_TRADING_DAY)

        # find the # of trading days between max(asset's first trade date,
        # 2002-01-02) and start_dt
        window_offset = (trading_days.searchsorted(days_in_window[0]) -
                         first_trading_day_to_use)

        start_index = max(asset_file_index, asset_file_index + window_offset)

        if window_offset < 0 and (abs(window_offset) > bar_count):
            # consumer is requesting a history window that starts AND ends
            # before this equity started trading, so gtfo
            return return_array

        # find the end index in the daily file. make sure it doesn't extend
        # past the end of this asset's data in the daily file.
        if window_offset < 0:
            # if the window_offset is negative, we need to decrease the
            # end_index accordingly.
            end_index = min(start_index + window_offset + bar_count,
                            daily_attrs['last_row'][str(sid)] + 1)

            # get data from bcolz file
            data = daily_data[field][start_index:end_index]

            # have to leave a bunch of empty slots at the beginning of
            # return_array, since they represent days before this asset
            # started trading.
            return_array[abs(window_offset):bar_count] = data
        else:
            end_index = min(start_index + bar_count,
                            daily_attrs['last_row'][str(sid)])
            data = daily_data[field][start_index:(end_index + 1)]

            if len(data) > len(return_array):
                return_array[:] = data[0:len(return_array)]
            else:
                return_array[0:len(data)] = data

        self._apply_all_adjustments(
            return_array,
            sid,
            days_in_window,
            field,
            self.DAILY_PRICE_ADJUSTMENT_FACTOR
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

    def _get_adjustment_list(self, sid, adjustments_dict, table_name):
        """
        Internal method that returns a list of adjustments for the given sid.

        Parameters
        ----------
        sid : int
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

        if sid not in adjustments_dict:
            adjustments_for_sid = self._adjustment_reader.\
                get_adjustments_for_sid(table_name, sid)
            adjustments_dict[sid] = adjustments_for_sid

        return adjustments_dict[sid]

    def get_equity_price_view(self, asset):
        try:
            view = self.views[asset]
        except KeyError:
            view = self.views[asset] = DataPortalSidView(asset, self)

        return view

    def _check_is_currently_alive(self, name, dt):
        if dt is None:
            dt = self.current_day

        if name not in self.asset_start_dates:
            self._get_asset_start_date(name)

        start_date = self.asset_start_dates[name]
        if self.asset_start_dates[name] > dt:
            raise NoTradeDataAvailableTooEarly(
                sid=name,
                dt=dt,
                start_dt=start_date
            )

        end_date = self.asset_end_dates[name]
        if self.asset_end_dates[name] < dt:
            raise NoTradeDataAvailableTooLate(
                sid=name,
                dt=dt,
                end_dt=end_date
            )

    def _get_asset_start_date(self, sid):
        if sid not in self.asset_start_dates:
            asset = self.asset_finder.retrieve_asset(sid)
            self.asset_start_dates[sid] = asset.start_date
            self.asset_end_dates[sid] = asset.end_date

        return self.asset_start_dates[sid]

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
            (field in self.augmented_sources_map and
             asset in self.augmented_sources_map[field])

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
        if self.fetcher_df is None:
            return []

        if self.current_day in self.fetcher_df.index:
            date_to_use = self.current_day
        else:
            # current day isn't in the fetcher df, go back the last
            # available day
            idx = self.fetcher_df.index.searchsorted(self.current_day)
            if idx == 0:
                return []

            date_to_use = self.fetcher_df.index[idx - 1]

        asset_list = self.fetcher_df.loc[date_to_use]["sid"]

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
