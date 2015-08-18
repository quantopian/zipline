import bcolz
import sqlite3

import numpy as np
import pandas as pd

import copy

from qexec.sources.findata import create_asset_finder

from zipline.utils import tradingcalendar

FINDATA_DIR = "/Users/jean/repo/findata/by_sid"
#DAILY_EQUITIES_PATH = findata.daily_equities_path(host_settings.findata_dir, MarketData.max_day())
DAILY_EQUITIES_PATH = "/Users/jean/repo/findata/findata/equity.dailies/2015-08-03/equity_daily_bars.bcolz"
ADJUSTMENTS_PATH = "/Users/jean/repo/findata/findata/adjustments/2015-08-03/adjustments.db"


class DataPortal(object):
    def __init__(self, algo):
        self.current_dt = None
        self.cur_data_offset = 0

        self.views = {}
        self.algo = algo

        self.carrays = {
            'opens': {},
            'highs': {},
            'lows': {},
            'closes': {},
            'volumes': {},
            'sid': {},
            'dt': {},
        }

        # hack
        if self.algo is not None:
            self.benchmark_iter = iter(self.algo.benchmark_iter)

        self.column_lookup = {
            'opens': 'opens',
            'highs': 'highs',
            'lows': 'lows',
            'closes': 'closes',
            'close': 'closes',
            'volumes': 'volumes',
            'volume': 'volumes',
            'open_price': 'opens',
            'close_price': 'closes'
        }

        self.adjustments_conn = sqlite3.connect(ADJUSTMENTS_PATH)

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
            self.daily_equities_data = bcolz.open(DAILY_EQUITIES_PATH)
            self.daily_equities_attrs = self.daily_equities_data.attrs

        return self.daily_equities_data, self.daily_equities_attrs

    def get_current_price_data(self, asset, column):
        asset_int = int(asset)
        path = "{0}/{1}.bcolz".format(FINDATA_DIR, asset_int)

        if column not in self.column_lookup:
            raise KeyError("Invalid column: " + str(column))

        column_to_use = self.column_lookup[column]

        try:
            carray = self.carrays[column_to_use][path]
        except KeyError:
            carray = self.carrays[column_to_use][path] = bcolz.carray(
                rootdir=path + "/" + column_to_use, mode='r')

        adjusted_dt = int(self.current_dt / 1e9)

        split_ratio = self._get_split_ratio(
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

    def get_history_window(self, sids, start_dt, bar_count, frequency, field):
        """
        Public API method that returns a dataframe containing the requested
        history window.

        Parameters
        ---------
        sids : list
            The sids whose data is desired.

        bar_count: int
            The number of bars desired.

        frequency: string
            "daily" or "minute"

        field: string
            "open", "high", "low", "close", "volume"

        Returns
        -------
        A dataframe containing the requested data.
        """
        if frequency == "daily":
            data = []
            for sid in sids:
                data.append(self._get_daily_window_for_sid(sid, start_dt,
                                                           bar_count, field))

            start_day_idx = tradingcalendar.trading_days.searchsorted(start_dt)
            days = tradingcalendar.trading_days[start_day_idx:(start_day_idx + bar_count)]

            return pd.DataFrame(np.array(data).T,
                                index=days,
                                columns=sids)

        # else:
        #     return self._get_minute_history_window(sids, bar_count, field)


    def _get_daily_window_for_sid(self, sid, start_dt, bar_count, field):
        """
        Internal methods that gets a window of adjusted daily data for a sid
        and specified date  range.  Used to support the history API method for
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

        Returns
        -------
        A numpy array with requested values.  Any missing slots filled with
        nan.

        """

        if field == "open_price":
            field_to_use = "open"
        elif field == "close_price":
            field_to_use = "close"
        else:
            field_to_use = field

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

        # create an np.array of size bar_count
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
        window_offset = (trading_days.searchsorted(start_dt) -
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
        data = daily_data[field_to_use][start_index:end_index]

        # put data into the right slot into return_data
        if window_offset < 0:
            return_array[abs(window_offset):bar_count] = data
        else:
            return_array[0:bar_count] = data

        self._apply_adjustments_to_window(
            self._get_adjustment_list(
                sid, self.splits_dict_immutable, "SPLITS"),
            return_array,
            trading_days,
            first_trading_day_to_use,
            window_offset,
            field_to_use != 'volume'
        )

        if field_to_use != 'volume':
            self._apply_adjustments_to_window(
                self._get_adjustment_list(
                    sid, self.mergers_dict_immutable, "MERGERS"),
                return_array,
                trading_days,
                first_trading_day_to_use,
                window_offset,
                True
            )

            return_array *= 0.001

        return return_array

    @staticmethod
    def _apply_adjustments_to_window(adjustments_list, window,
                                     trading_days, first_trading_day,
                                     window_offset, multiply):
        """
        Internal method that adjusts the values in the given window using the
        given adjustments.

        Parameters
        ----------
        adjustments_list : list
            A list of [timestamp, cumulative_ratio] objects representing the
            adjustments to be applied.  Its contents will be modified
            in this method.

        window: np.array
            The data that will be adjusted in-place.

        trading_days: pandas.tseries.index.DatetimeIndex
            A list of our trading days.  Passed in here to avoid looking it up
            again.

        first_trading_day: int
            The position of the first trading day for this asset in
            trading_days.  It is the maximum of the asset's first trade date
            and 1/1/2002.

        window_offset: int
            The number of days between first_trading_day and the date of the
            first slot in window.

        multiply: bool
            Whether to apply the adjustment by multiplying.  If false, the
            adjustment is applied by dividing.

        Returns
        -------
        None.  The data (in the window param) is adjusted in place.

        """

        for adjustment in adjustments_list:
            # convert the timestamp of the split into the # of days since
            # first_trading_day_to_use. then subtract window_offset, leaving
            # us with each split's "timestamp" being a relative
            # offset, in # of days, from start_dt.
            # For example, if start_dt is 6/20/08, then from the example data
            # above:
            # [[-2, 0.125, [378, 0.25], [502, 0.5]]

            # FIXME make this work with negative offsets!!
            adjustment_offset = trading_days.searchsorted(adjustment[0]) - \
                                first_trading_day
            adjustment[0] = adjustment_offset - window_offset

        bar_count = len(window)

        # only keep adjustments whose day index is between 0 and bar_count.
        # if the day index is negative, that adj has happened already and
        # we don't care.
        # if the day index is greater than bar_count, that adj is outside
        # this window and we also don't care.
        adjustments_list = [adj for adj in adjustments_list if adj[0] >= 0
                            and adj[0] < bar_count]

        # we're left with the adjustments that happened during this window.
        # iterate over them and apply them to the appropriate part of the
        # window.
        applied_index = 0
        while len(adjustments_list) > 0:
            adj = adjustments_list.pop(0)
            if multiply:
                window[applied_index:adj[0]] *= adj[1]
            else:
                window[applied_index:adj[0]] /= adj[0]

            applied_index = adj[0] + 1


# TODO HISTORY TEST CASES
# 1) window is entirely after ipo
# 2) window starts before ipo
# 3) window ends after security stops trading
# 4) split/merger inside window
# 5) missing data inside window
# 6) ffill behavior



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
            The data that will be adjusted in-place.

        trading_days: pandas.tseries.index.DatetimeIndex
            A list of our trading days.  Passed in here to avoid looking it up
            again.

        first_trading_day: int
            The position of the first trading day for this asset in
            trading_days.  It is the maximum of the asset's first trade date
            and 1/1/2002.

        window_offset: int
            The number of days between first_trading_day and the date of the
            first slot in window.

        multiply: bool
            Whether to apply the adjustment by multiplying.  If false, the
            adjustment is applied by dividing.

        Returns
        -------
        None.  The data (in the window param) is adjusted in place.

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
            # that is greater than the current time, and get the multipler for
            # that range.

            calculated_list = []
            multiplier = 1
            for adj_info in reversed(adjustments_for_sid):
                multiplier *= adj_info[1]
                calculated_list.insert(0, [pd.Timestamp(adj_info[0], unit='s'),
                                           multiplier])

            adjustments_dict[sid] = calculated_list

        return copy.deepcopy(adjustments_dict[sid])


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


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)
