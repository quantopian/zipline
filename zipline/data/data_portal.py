import bcolz
import sqlite3

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

        self.splits_dict = {}
        self.split_multipliers = {}

        self.mergers_dict = {}
        self.mergers_multipliers = {}

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

        split_ratio = self.get_adjustment_ratio(
            asset_int, adjusted_dt, self.splits_dict,
            self.split_multipliers, "SPLITS")

        mergers_ratio = self.get_adjustment_ratio(
            asset_int, adjusted_dt, self.mergers_dict,
            self.mergers_multipliers, "MERGERS")

        if column_to_use == 'volume':
            return carray[self.cur_data_offset] / split_ratio
        else:
            return carray[self.cur_data_offset] * 0.001 * split_ratio * \
                mergers_ratio

    # For each adjustment type (split, mergers) we keep two dictionaries
    # around:
    # - ADJUSTMENTTYPE_dict: dictionary of sid to a list of future adjustments
    # - ADJUSTMENTTYPE_multipliers: dictionary of sid to the current multiplier
    #
    # Each time we're asked to get a ratio:
    # - if this is the first time we've been asked for this adjustment/sid
    #   pair, we query the data from ADJUSTMENTS_PATH and store it in
    #   ADJUSTMENTTYPE_dict. We get the initial ratio by multiplying all the
    #   ratios together (since we always present pricing data with an as-of
    #   date of today). We then fast-forward to the desired date by dividing
    #   the initial ratio by any encountered adjustments.  The ratio is stored
    #   in ADJUSTMENTTYPE_multipliers.
    # - now that we have the current ratio as well as the current date;
    #   - if there are no adjustments left, just return 1.
    #   - else if the next adjustment's date is in the future, return the
    #     current ratio.
    #   - else apply the next adjustment for this sid, and remove it from
    #     ADJUSTMENTTYPE_dict[sid].  Save the new current ratio in
    #     ADJUSTMENTTYPE_multipliers, and return that.
    def get_adjustment_ratio(self, sid, dt, adjustments_dict, multiplier_dict,
                             table_name):
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
        # For now use benchamrk iterator, and assume this is only called
        # once a day.
        return next(self.benchmark_iter).returns


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)
