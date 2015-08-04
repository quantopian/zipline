import bcolz
import os
import pandas as pd

#FINDATA_DIR = os.getenv("FINDATA_DIR")
FINDATA_DIR = "/Users/jean/repo/findata/by_sid"


class DataPortal(object):

    def __init__(self, algo):
        self.data_start_date = pd.Timestamp('2002-01-02', tz='UTC')
        # TODO: Need to make this work.
        self.data_start_index = 0
        self.views = {}
        self.algo = algo
        self.current_bcolz_handle = None
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

        self.cur_data_offset = 0

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

    def get_current_price_data(self, asset, column):
        path = "{0}/{1}.bcolz".format(FINDATA_DIR, int(asset))

        if column not in self.column_lookup:
            raise KeyError("Invalid column: " + str(column))

        column_to_use = self.column_lookup[column]

        try:
            carray = self.carrays[column_to_use][path]
        except KeyError:
            carray = self.carrays[column_to_use][path] = bcolz.carray(
                rootdir=path + "/" + column_to_use, mode='r')

        if column_to_use == 'volume':
            return carray[self.cur_data_offset]
        else:
            return carray[self.cur_data_offset] * 0.001

    def get_equity_price_view(self, asset):
        try:
            view = self.views[asset]
        except KeyError:
            view = self.views[asset] = DataPortalSidView(asset, self)

        return view

    def get_benchmark_returns_for_day(self, day):
        # For now use benchamrk iterator, and assume this is only called
        # once a day.
        return 0
        #return next(self.benchmark_iter).returns


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)
