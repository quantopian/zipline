from datetime import datetime
import bcolz
import sqlite3

import numpy as np
import pandas as pd

from zipline.utils import tradingcalendar
from zipline.finance.trading import TradingEnvironment

from zipline.utils.algo_instance import get_algo_instance
from zipline.utils.math_utils import nanstd, nanmean, nansum


class DataPortal(object):
    def __init__(self,
                 algo,  # FIXME hack
                 findata_dir=None,
                 daily_equities_path=None,
                 adjustments_path=None,
                 asset_finder=None):
        self.current_dt = None
        self.cur_data_offset = 0

        self.views = {}
        self.algo = algo

        if findata_dir is None:
            raise ValueError("Must provide findata dir!")

        if daily_equities_path is None:
            raise ValueError("Must provide daily equities path!")

        if adjustments_path is None:
            raise ValueError("Must provide adjustments path!")

        self.findata_dir = findata_dir
        self.daily_equities_path = daily_equities_path
        self.adjustments_path = adjustments_path

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

        # hack
        if self.algo is not None:
            self.benchmark_iter = iter(self.algo.benchmark_iter)

        self.column_lookup = {
            'open': 'open',
            'high': 'high',
            'lows': 'low',
            'close': 'close',
            'volume': 'volume',
            'open_price': 'open',
            'close_price': 'close',
            'price': 'close'
        }

        self.adjustments_conn = sqlite3.connect(self.adjustments_path)

        # We handle splits and mergers differently for point-in-time and
        # history windows.
        # For point-in-time (ie, data[sid(24)].price, the backtest clock
        # only ever goes forward in time, so we can keep around the current
        # adjustment ratio for a sid knowing that we'll never need an old
        # one again.  All that information is stored in these dictionaries.
        self.splits_dict = {}
        self.split_multipliers = {}
        self.mergers_dict = {}
        self.mergers_multipliers = {}

        # For history windows, just because we've been asked for an asset's
        # window started at day N doesn't mean that we won't be asked for
        # another window for this sid starting at day N-10.  Therefore,
        # we have to store all the adjustments per asset in these dictionaries.
        self.splits_dict_immutable = {}
        self.mergers_dict_immutable = {}

        # Pointer to the daily bcolz file.
        self.daily_equities_data = None

        self.asset_finder = None

        # Cache of int -> the first trading day of an asset, even if that day
        # is before 1/2/2002.
        self.asset_start_dates = {}

    def _open_daily_file(self):
        if self.daily_equities_data is None:
            self.daily_equities_data = bcolz.open(self.daily_equities_path)
            self.daily_equities_attrs = self.daily_equities_data.attrs

        return self.daily_equities_data, self.daily_equities_attrs

    def _open_minute_file(self, field, sid):
        path = "{0}/{1}.bcolz".format(self.findata_dir, sid)

        try:
            carray = self.carrays[field][path]
        except KeyError:
            carray = self.carrays[field][path] = bcolz.carray(
                rootdir=path + "/" + field, mode='r')

        return carray

    def get_current_price_data(self, asset, column):
        asset_int = int(asset)

        if column not in self.column_lookup:
            raise KeyError("Invalid column: " + str(column))

        column_to_use = self.column_lookup[column]
        carray = self._open_minute_file(column_to_use, asset_int)

        adjusted_dt = int(self.current_dt / 1e9)

        split_ratio = self._get_adjustment_ratio(
            asset_int, adjusted_dt, self.splits_dict,
            self.split_multipliers, "SPLITS")

        mergers_ratio = self._get_adjustment_ratio(
            asset_int, adjusted_dt, self.mergers_dict,
            self.mergers_multipliers, "MERGERS")

        if column_to_use == 'volume':
            return carray[self.cur_data_offset] / split_ratio
        else:
            return carray[self.cur_data_offset] * 0.001 * split_ratio * \
                mergers_ratio

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
            "daily" or "minute"

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
            field_to_use = self.column_lookup[field]
        except KeyError:
            raise KeyError("Invalid history field: " + str(field))

        if frequency == "daily":
            data = []

            day = end_dt.date()

            # for daily history, we need to get bar_count - 1 complete days,
            # and then a partial day constructed from the minute data of this
            # day.
            day_idx = tradingcalendar.trading_days.searchsorted(day)
            days_for_window = tradingcalendar.trading_days[
                (day_idx - bar_count + 1):(day_idx + 1)]

            all_minutes_for_day = TradingEnvironment.instance().\
                market_minutes_for_day(pd.Timestamp(day))

            last_minute_idx = all_minutes_for_day.searchsorted(end_dt)

            # these are the minutes for the partial day
            minutes_for_partial_day =\
                all_minutes_for_day[0:(last_minute_idx + 1)]

            for sid in sids:
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

                daily_data[-1] = minute_value

                data.append(daily_data)

            df = pd.DataFrame(np.array(data).T,
                              index=days_for_window,
                              columns=sids)

        else:
            minutes_for_window = TradingEnvironment.instance().\
                market_minute_window(end_dt, bar_count, step=-1)[::-1]

            data = []
            for sid in sids:
                data.append(self._get_minute_window_for_sid(
                    sid,
                    field_to_use,
                    minutes_for_window
                ))

            df = pd.DataFrame(np.array(data).T,
                              index=minutes_for_window,
                              columns=sids)

        # forward-fill if needed
        if field == "price" and ffill:
            df.fillna(method='backfill', inplace=True)

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

        start_idx = self._find_position_of_minute(minutes_for_window[0])
        end_idx = self._find_position_of_minute(minutes_for_window[-1]) + 1

        return_data = np.array(raw_data[start_idx:end_idx], dtype=np.float64)

        num_minutes = len(minutes_for_window)

        if len(return_data) != num_minutes:
            # there must be some early closes in this date range, have
            # to deal with them

            eastern_minutes = minutes_for_window.tz_convert("US/Eastern")

            # early close is 1pm.
            # walk over the minutes. each time we encounter a minute that is
            # 1pm not followed by 1:01pm, we know that's an early close.
            last_minute_idx_of_early_close_day = []
            for minute_idx, minute_dt in enumerate(eastern_minutes):
                if minute_idx == (num_minutes - 1):
                    break

                if minute_dt.hour == 13 and minute_dt.minute == 0:
                    next_minute = eastern_minutes[minute_idx + 1]
                    if next_minute.hour != 13:
                        # minute_dt is the last minute of an early close
                        last_minute_idx_of_early_close_day.append(minute_idx)

            # for each minute in last_minute_idx_of_early_close_day, find
            # its position in unadjusted_data and remove the subsequent 180
            # bars (since an early close removes 1-4pm from that trading day)
            for idx, early_close_minute_idx in \
                    enumerate(last_minute_idx_of_early_close_day):
                early_close_minute_idx -= (180 * idx)
                return_data = np.delete(
                    return_data,
                    range(early_close_minute_idx,
                          early_close_minute_idx + 180))

        # adjust the data
        self._apply_adjustments_to_window(
            self._get_adjustment_list(
                sid, self.splits_dict_immutable, "SPLITS"),
            return_data,
            minutes_for_window,
            field != 'volume'
        )

        if field != 'volume':
            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    sid, self.mergers_dict_immutable, "MERGERS"),
                return_data,
                minutes_for_window,
                True
            )

            return_data *= 0.001

            # if anything is zero, it's a missing bar, so replace it with NaN.
            # we only want to do this for non-volume fields, because a missing
            # volume should be 0.
            return_data[return_data == 0] = np.NaN

        return return_data

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
        day_idx = tradingcalendar.trading_days.searchsorted(day) - 3028

        day_open = pd.Timestamp(
            datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=9,
                minute=31),
            tz='US/Eastern').tz_convert('UTC')

        minutes_offset = int((minute_dt - day_open).total_seconds()) / 60

        return (390 * day_idx) + minutes_offset

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

        if self.asset_finder is None:
            self.asset_finder = create_asset_finder()

        # get the start date
        if sid not in self.asset_start_dates:
            asset = self.asset_finder.retrieve_asset(sid)
            self.asset_start_dates[sid] = asset.start_date

        # the daily file stores each sid's daily OHLCV in a contiguous block.
        # the first row per sid is either 1/2/2002, or the sid's start_date if
        # it started after 1/2/2002.  once a sid stops trading, there are no
        # rows for it.

        bar_count = len(days_in_window)

        # create an np.array of size bar_count
        if extra_slot:
            return_array = np.empty((bar_count + 1,))
        else:
            return_array = np.empty((bar_count,))

        return_array[:] = np.NAN

        # find the start index in the daily file for this asset
        asset_file_index = daily_attrs['first_row'][str(sid)]

        trading_days = tradingcalendar.trading_days

        # Calculate the starting day to use (either the asset's first trading
        # day, or 1/1/2002 (which is the 3028th day in the trading calendar).
        first_trading_day_to_use = max(trading_days.searchsorted(
            self.asset_start_dates[sid]), 3028)

        # find the # of trading days between max(asset's first trade date,
        # 2002-01-02) and start_dt
        window_offset = (trading_days.searchsorted(days_in_window[0]) -
                         first_trading_day_to_use)

        start_index = max(asset_file_index, asset_file_index + window_offset)

        # find the end index in the daily file. make sure it doesn't extend
        # past the end of this asset's data in the daily file.
        if window_offset < 0:
            # if the window_offset is negative, we need to decrease the
            # end_index accordingly.
            end_index = min(start_index + window_offset + bar_count,
                         daily_attrs['last_row'][str(sid)])
        else:
            end_index = min(start_index + bar_count,
                         daily_attrs['last_row'][str(sid)])

        if window_offset < 0 and (abs(window_offset) > bar_count):
            # consumer is requesting a history window that starts AND ends
            # before this equity started trading, so gtfo
            return return_array

        # fetch the data from the daily bcolz file
        data = daily_data[field][start_index:end_index]

        # put data into the right slot into return_data
        if window_offset < 0:
            return_array[abs(window_offset):bar_count] = data
        else:
            return_array[0:bar_count] = data

        self._apply_adjustments_to_window(
            self._get_adjustment_list(
                sid, self.splits_dict_immutable, "SPLITS"),
            return_array,
            days_in_window,
            field != 'volume'
        )

        if field != 'volume':
            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    sid, self.mergers_dict_immutable, "MERGERS"),
                return_array,
                days_in_window,
                True
            )

            return_array *= 0.001

            # if anything is zero, it's a missing bar, so replace it with NaN.
            # we only want to do this for non-volume fields, because a missing
            # volume should be 0.
            return_array[return_array == 0] = np.NaN

        return return_array

    @staticmethod
    def _apply_adjustments_to_window(adjustments_list, window_data,
                                     dts_in_window, multiply):

        if len(adjustments_list) == 0:
            return

        idx = 0

        # clear out all adjustments that happened before the first minute
        adjustments_length = len(adjustments_list)
        first_dt = dts_in_window[0]

        # advance idx to the correct spot in the adjustments list, based on
        # when the window starts
        while idx < adjustments_length - 1 and first_dt > \
                adjustments_list[idx][0]:
            idx += 1

        if idx == adjustments_length - 1:
            # if there are no applicable adjustments, get out.
            return

        first_applicable_adjustment = adjustments_list[idx]

        # optimization: if the last minute is before the first
        # adjustment, then the entire window is before the first adj,
        # so just apply the first adj to the entire window and gtfo.
        if dts_in_window[-1] < first_applicable_adjustment[0]:
            if multiply:
                window_data *= first_applicable_adjustment[1]
            else:
                window_data /= first_applicable_adjustment[1]
            return

        # walk over minutes in window.  for each minute, if it's less than
        # the next adjustment, adjust.  if the adjustment has passed, get rid
        # of it.
        range_start = 0
        for idx, minute_dt in enumerate(dts_in_window):
            adjustment_to_apply = adjustments_list[idx]
            if minute_dt > adjustment_to_apply[0]:
                idx += 1
                range_end = idx - 1

                if multiply:
                    window_data[range_start:range_end + 1] *=\
                        adjustment_to_apply[1]
                else:
                    window_data[range_start:range_end + 1] /=\
                        adjustment_to_apply[1]

                if idx == (adjustments_length - 1):
                    break

                range_start = idx

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
            A list of [cumulative_multiplier, pd.Timestamp], earliest first

        """
        if sid not in adjustments_dict:
            adjustments_for_sid = self.adjustments_conn.execute(
                "SELECT effective_date, ratio FROM %s WHERE sid = ?" %
                table_name, [sid]).fetchall()

            # now build up the adjustment factors
            # for example, if the data is:
            # 6/17/08, 0.5
            # 3/1/10, 0.5
            # 6/8/13, 0.5
            # then we want to convert that to:
            # [6/17/08, 0.125], [3/1/10: 0.25], [6/18/13: 0.5]

            # and using that list and a given dt, we can find the first date
            # that is greater than the current time, and get the multiplier for
            # that range.

            calculated_list = []
            multiplier = 1
            for adj_info in reversed(adjustments_for_sid):
                multiplier *= adj_info[1]
                calculated_list.insert(0, [pd.Timestamp(adj_info[0], unit='s', tz='UTC'),
                                           multiplier])

            adjustments_dict[sid] = calculated_list

        return adjustments_dict[sid]

    def _get_adjustment_ratio(self, sid, dt, adjustments_dict, multiplier_dict,
                              table_name):
        """
        Internal method that returns the value of the desired adjustment as of
        the given dt.

        IMPORTANT: This method assumes that the clock always goes forward.
        Once this method has been called with a dt, it has to be called with
        a greater-than-or-equal dt subsequently.

        Parameters
        ----------
        sid : int
            The asset for which to return adjustments.

        dt: int
            Epoch in seconds.

        adjustments_dict: dict
            Stores the not-used-yet adjustments for this sid.  As an adjustment
            is used up, it's removed from this dictionary.

        multiplier_dict: dict
            Stores the current multiplier for this adjustment, per sid.

        table_name: string
            The name of the table where this adjustment lives, in the sqlite
            db.


        Returns
        -------
        ratio: float
            The desired adjustment ratio.

        """

        # For each adjustment type (split, mergers) we keep two dictionaries
        # around:
        # - ADJUSTMENTTYPE_dict: dictionary of sid to a list of future
        #   adjustments
        # - ADJUSTMENTTYPE_multipliers: dictionary of sid to the current
        #   multiplier
        #
        # Each time we're asked to get a ratio:
        # - if this is the first time we've been asked for this adjustment/sid
        #   pair, we query the data from ADJUSTMENTS_PATH and store it in
        #   ADJUSTMENTTYPE_dict. We get the initial ratio by multiplying all
        #   the ratios together (since we always present pricing data with an
        #   as-of date of today). We then fast-forward to the desired date by
        #   dividing the initial ratio by any encountered adjustments.  The
        #   ratio is stored in ADJUSTMENTTYPE_multipliers.
        # - now that we have the current ratio as well as the current date;
        #   - if there are no adjustments left, just return 1.
        #   - else if the next adjustment's date is in the future, return the
        #     current ratio.
        #   - else apply the next adjustment for this sid, and remove it from
        #     ADJUSTMENTTYPE_dict[sid].  Save the new current ratio in
        #     ADJUSTMENTTYPE_multipliers, and return that.
        #
        # NOTE: This approach is optimized for single asset lookups on an ever-
        # moving-forward clock.  Once a split has been seen, it's consumed and
        # the current split multiplier for that sid is updated.  After that
        # happens, we can't look up an older ratio because it won't be there
        # anymore.
        if sid not in adjustments_dict:
            adjustments_for_sid = self.adjustments_conn.execute(
                "SELECT effective_date, ratio FROM %s WHERE sid = ?" %
                table_name, [sid]).fetchall()

            if (len(adjustments_for_sid) == 0) or \
               (adjustments_for_sid[-1][0] < dt):
                multiplier_dict[sid] = 1
                adjustments_dict[sid] = []
                return 1

            multiplier_dict[sid] = reduce(lambda x, y: x[1] * y[1],
                                          adjustments_for_sid)

            while (len(adjustments_dict) > 0) and \
                  (adjustments_for_sid[0][0] < dt):
                multiplier_dict[sid] /= adjustments_for_sid[0][1]
                adjustments_for_sid.pop(0)

            adjustments_dict[sid] = adjustments_for_sid

        adjustment_info = adjustments_dict[sid]

        # check that we haven't gone past an adjustment
        if len(adjustment_info) == 0:
            return 1
        elif adjustment_info[0][0] > dt:
            return multiplier_dict[sid]
        else:
            # new split encountered, adjust our current multiplier and remove
            # it from the list
            multiplier_dict[sid] /= adjustment_info[0][0]
            adjustment_info.pop(0)

            return multiplier_dict[sid]

    def get_equity_price_view(self, asset):
        try:
            view = self.views[asset]
        except KeyError:
            view = self.views[asset] = DataPortalSidView(asset, self)

        return view

    def get_benchmark_returns_for_day(self, day):
        # For now use benchmark iterator, and assume this is only called
        # once a day.
        return next(self.benchmark_iter).returns

    def get_simple_transform(self, sid, transform_name, bars=None):
        now = pd.Timestamp(get_algo_instance().datetime, tz='UTC')
        sid_int = int(sid)

        if transform_name == "returns":
            # returns is always calculated over the last 2 days, even though
            # we only support minutely backtests now.
            hst = self.get_history_window(
                [sid_int],
                now,
                2,
                "1d",
                "price",
                ffill=True
            )[sid_int]

            return (hst.iloc[-1] - hst.iloc[0]) / hst.iloc[0]

        if bars is None:
            raise ValueError("bars cannot be None!")

        price_arr = self.get_history_window(
            [sid_int],
            now,
            bars,
            "1m",
            "price",
            ffill=True
        )[sid_int]

        if transform_name == "mavg":
            return nanmean(price_arr)
        elif transform_name == "stddev":
            return nanstd(price_arr, ddof=1)
        elif transform_name == "vwap":
            volume_arr = self.get_history_window(
                [sid_int],
                now,
                bars,
                "1m",
                "volume",
                ffill=True
            )[sid_int]

            vol_sum = nansum(volume_arr)
            try:
                ret = nansum(price_arr * volume_arr) / vol_sum
            except ZeroDivisionError:
                ret = np.nan

            return ret


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)

    def mavg(self, minutes):
        return self.portal.get_simple_transform(self.asset, "mavg", bars=minutes)

    def stddev(self, minutes):
        return self.portal.get_simple_transform(self.asset, "stddev", bars=minutes)

    def vwap(self, minutes):
        return self.portal.get_simple_transform(self.asset, "vwap", bars=minutes)

    def returns(self):
        return self.portal.get_simple_transform(self.asset, "returns")