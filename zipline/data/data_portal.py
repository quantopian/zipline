from datetime import datetime
import bcolz
import sqlite3
from logbook import Logger

import numpy as np
import pandas as pd

from zipline.utils import tradingcalendar

from zipline.utils.algo_instance import get_algo_instance
from zipline.utils.math_utils import nanstd, nanmean, nansum

# FIXME anything to do with 2002-01-02 probably belongs in qexec, right/
FIRST_TRADING_MINUTE = pd.Timestamp("2002-01-02 14:31:00", tz='UTC')

# FIXME should this be passed in (is this qexec specific?)?
INDEX_OF_FIRST_TRADING_DAY = 3028

log = Logger('DataPortal')


class DataPortal(object):
    def __init__(self,
                 env,
                 sim_params=None,
                 benchmark_iter=None,  # FIXME hack
                 minutes_equities_path=None,
                 daily_equities_path=None,
                 adjustments_path=None,
                 asset_finder=None,
                 extra_sources=None,
                 sid_path_func=None):
        self.env = env
        self.current_dt = None
        self.current_day = sim_params.period_start
        self.cur_data_offset = 0

        self.views = {}

        if minutes_equities_path is None and daily_equities_path is None:
            raise ValueError("Must provide at least one of minute or "
                             "daily data path!")

        # if adjustments_path is None:
        #     raise ValueError("Must provide adjustments path!")
        #
        # if asset_finder is None:
        #     raise ValueError("Must provide asset finder!")

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

        self.benchmark_iter = benchmark_iter

        self.column_lookup = {
            'open': 'open',
            'open_price': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'close_price': 'close',
            'volume': 'volume',
            'price': 'close'
        }

        if adjustments_path is not None:
            self.adjustments_conn = sqlite3.connect(adjustments_path)
        else:
            self.adjustments_conn = None

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
        if self.sim_params is not None:
            self.data_frequency = self.sim_params.data_frequency

        if extra_sources is not None:
            self._handle_extra_sources(extra_sources)

        self.sid_path_func = sid_path_func

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
            sid_group = source.df.groupby(['sid'])

            for identifier in unique_sids:
                self.sources_map[identifier] = df

                # get this identifier's earliest date
                earliest_date_idx = sid_group.indices[identifier][0]
                earliest_date = df.index[earliest_date_idx]

                last_date_idx = sid_group.indices[identifier][-1]
                last_date = df.index[min(len(df.index) - 1, last_date_idx)]

                self.asset_start_dates[identifier] = earliest_date
                self.asset_end_dates[identifier] = last_date

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

    def get_current_price_data(self, asset, column, dt=None):
        day_to_use = dt or self.current_day

        if asset in self.sources_map:
            # go find this asset in our custom sources
            try:
                # TODO: Change to index both dt and column at once.
                return self.sources_map[asset].loc[day_to_use].loc[column]
            except:
                log.error(
                    "Could not find price for asset={0}, current_day={1},"
                    "column={2}".format(
                        str(asset),
                        str(self.current_day),
                        str(column)))

            return None

        if column not in self.column_lookup:
            raise KeyError("Invalid column: " + str(column))

        asset_int = int(asset)
        column_to_use = self.column_lookup[column]

        if self.data_frequency == "daily":
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
                # dt was passed in, so calculate the offset.
                # = (390 * number of trading days since 1/2/2002) +
                #   (index of minute in day)

                # FIXME is this expensive?
                if not self.env.is_market_hours(dt):
                    dt = self.env.previous_market_minute(dt)

                given_day = pd.Timestamp(dt.date(), tz='utc')
                day_index = tradingcalendar.trading_days.searchsorted(
                    given_day) - INDEX_OF_FIRST_TRADING_DAY
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

                while result == 0 and minute_offset_to_use > start_day_offset:
                    minute_offset_to_use -= 1
                    result = carray[minute_offset_to_use]

            if column_to_use != 'volume':
                return result * 0.001
            else:
                return result

    def _get_daily_data(self, asset_int, column, dt):
        dt = pd.Timestamp(dt.date(), tz='utc')
        daily_data, daily_attrs = self._open_daily_file()

        # find the start index in the daily file for this asset
        asset_file_index = daily_attrs['first_row'][str(asset_int)]

        # find when the asset started trading
        # TODO: only access this info once.
        calendar = daily_attrs['calendar']
        asset_data_start_date = \
            pd.Timestamp(
                calendar[daily_attrs['calendar_offset'][str(asset_int)]],
                tz='UTC')

        if dt < asset_data_start_date:
            raise ValueError(
                "Cannot fetch daily data for {0} for {1} "
                "because it only started trading on {2}!".
                format(
                    str(asset),
                    str(dt),
                    str(asset_data_start_date)
                )
            )

        trading_days = tradingcalendar.trading_days

        # figure out how many days it's been between now and when this
        # asset starting trading
        window_offset = trading_days.searchsorted(dt) - \
            trading_days.searchsorted(asset_data_start_date)

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
            return raw_value * 0.001
        else:
            return raw_value

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

            df = pd.DataFrame(np.array(data).T,
                              index=days_for_window,
                              columns=sids)

        elif frequency == "1m":
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
        if self.adjustments_conn is None:
            return []

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

    def is_currently_alive(self, name):
        if name not in self.sources_map:
            name = int(name)

        if name not in self.asset_start_dates:
            asset = self.asset_finder.retrieve_asset(name)
            self.asset_start_dates[name] = asset.start_date
            self.asset_end_dates[name] = asset.end_date

        return (self.current_day >= self.asset_start_dates[name] and
                self.current_day <= self.asset_end_dates[name])

    def _get_asset_start_date(self, sid):
        if sid not in self.asset_start_dates:
            asset = self.asset_finder.retrieve_asset(sid)
            self.asset_start_dates[sid] = asset.start_date
            self.asset_end_dates[sid] = asset.end_date

        return self.asset_start_dates[sid]


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)
