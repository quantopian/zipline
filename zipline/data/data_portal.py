from datetime import datetime
import bcolz
import sqlite3
from logbook import Logger

import numpy as np
import pandas as pd

from zipline.utils import tradingcalendar
from zipline.finance.trading import TradingEnvironment

from zipline.utils.algo_instance import get_algo_instance
from zipline.utils.math_utils import nanstd, nanmean, nansum

# FIXME anything to do with 2002-01-02 probably belongs in qexec, right/
FIRST_TRADING_MINUTE = pd.Timestamp("2002-01-02 14:31:00", tz='UTC')

# FIXME should this be passed in (is this qexec specific?)?
INDEX_OF_FIRST_TRADING_DAY = 3028

log = Logger('DataPortal')


class DataPortal(object):
    def __init__(self,
                 sim_params=None,
                 benchmark_iter=None, # FIXME hack
                 findata_dir=None,
                 daily_equities_path=None,
                 adjustments_path=None,
                 asset_finder=None,
                 extra_sources=None):
        self.current_dt = None
        self.cur_data_offset = 0

        self.views = {}

        if findata_dir is None:
            raise ValueError("Must provide findata dir!")

        if daily_equities_path is None:
            raise ValueError("Must provide daily equities path!")

        if adjustments_path is None:
            raise ValueError("Must provide adjustments path!")

        if asset_finder is None:
            raise ValueError("Must provide asset finder!")

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

        self.benchmark_iter = benchmark_iter

        self.column_lookup = {
            'open_price': 'open',
            'high': 'high',
            'low': 'low',
            'close_price': 'close',
            'volume': 'volume',
            'price': 'close'
        }

        self.adjustments_conn = sqlite3.connect(self.adjustments_path)

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

        self.sources_map = {}

        self.sim_params = sim_params

        if extra_sources is not None:
            self._handle_extra_sources(extra_sources)

    def _handle_extra_sources(self, sources):
        """
        Extra sources always have a sid column.

        We expand the given data (by forward filling) to the full range of
        the simulation dates, so that lookup is fast during simulation.
        """
        backtest_days = tradingcalendar.get_trading_days(
            self.sim_params.period_start,
            self.sim_params.period_end
        )

        for source in sources:
            if source.df is None:
                continue

            # reindex the dataframe based on the backtest start/end date
            df = source.df.reindex(
                index=backtest_days,
                method='ffill'
            )

            unique_sids = df.sid.unique()
            for identifier in unique_sids:
                self.sources_map[identifier] = df

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
        if asset in self.sources_map:
            # go find this asset in our custom sources

            # figure out the current date,.  self.cur_data_offset is the #
            # of minutes since 1/2/02 9:30am, so divide by 390 to get the #
            # of days since 1/2/02, then look in the trading calendar.
            date = tradingcalendar.trading_days[INDEX_OF_FIRST_TRADING_DAY +
                                                (self.cur_data_offset / 390)]

            try:
                return self.sources_map[asset].loc[date].loc[column]
            except:
                log.error("Could not find price for asset={0}, date={1},"
                          "column={2}".format(
                              str(asset),
                              str(date),
                              str(column)))

            return None

        asset_int = int(asset)

        if column not in self.column_lookup:
            raise KeyError("Invalid column: " + str(column))

        column_to_use = self.column_lookup[column]
        carray = self._open_minute_file(column_to_use, asset_int)

        if column_to_use == 'volume':
            return carray[self.cur_data_offset]
        else:
            return carray[self.cur_data_offset] * 0.001

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
            field_to_use = self.column_lookup[field]
        except KeyError:
            raise ValueError("Invalid history field: " + str(field))

        if frequency == "1d":
            data = []

            day = end_dt.date()

            # for daily history, we need to get bar_count - 1 complete days,
            # and then a partial day constructed from the minute data of this
            # day.
            day_idx = tradingcalendar.trading_days.searchsorted(day)
            days_for_window = tradingcalendar.trading_days[
                (day_idx - bar_count + 1):(day_idx + 1)]

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

                if days_for_window[-1] > self.asset_end_dates[sid]:
                    # if the last desired day of the window is after the
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
                    all_minutes_for_day = TradingEnvironment.instance().\
                        market_minutes_for_day(pd.Timestamp(day))

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

            df = pd.DataFrame(np.array(data).T,
                              index=days_for_window,
                              columns=sids)

        elif frequency == "1m":
            # get all the minutes for this window
            minutes_for_window = TradingEnvironment.instance().\
                market_minute_window(end_dt, bar_count, step=-1)[::-1]

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
                    sid,
                    field_to_use,
                    modified_minutes_for_window
                )

                if bars_to_prepend != 0:
                    sid_minute_data = np.insert(sid_minute_data, 0,
                                                nans_to_prepend)

                data.append(sid_minute_data)

            df = pd.DataFrame(np.array(data).T,
                              index=minutes_for_window,
                              columns=sids)

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
            field
        )

        return return_data

    def _apply_all_adjustments(self, data, sid, dts, field):
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

            data *= 0.001

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
            self.asset_start_dates[sid]), INDEX_OF_FIRST_TRADING_DAY)

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
        data = daily_data[field][start_index:(end_index + 1)]

        # put data into the right slot into return_data
        if window_offset < 0:
            return_array[abs(window_offset):(bar_count + 1)] = data
        else:
            return_array[0:len(data)] = data

        self._apply_all_adjustments(
            return_array,
            sid,
            days_in_window,
            field
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
        if sid not in adjustments_dict:
            adjustments_for_sid = self.adjustments_conn.execute(
                "SELECT effective_date, ratio FROM %s WHERE sid = ?" %
                table_name, [sid]).fetchall()

            adjustments_dict[sid] = [[pd.Timestamp(adjustment[0],
                                                   unit='s',
                                                   tz='UTC'),
                                      adjustment[1]]
                                     for adjustment in
                                     adjustments_for_sid]

        return adjustments_dict[sid]

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
        return self.portal.get_simple_transform(self.asset, "mavg",
                                                bars=minutes)

    def stddev(self, minutes):
        return self.portal.get_simple_transform(self.asset, "stddev",
                                                bars=minutes)

    def vwap(self, minutes):
        return self.portal.get_simple_transform(self.asset, "vwap",
                                                bars=minutes)

    def returns(self):
        return self.portal.get_simple_transform(self.asset, "returns")
