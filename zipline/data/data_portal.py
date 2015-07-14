import bcolz
import os
import pandas as pd

FINDATA_DIR = os.getenv("FINDATA_DIR")


class DataPortal(object):

    def __init__(self, algo):
        self.data_start_date = pd.Timestamp('2002-01-02', tz='UTC')
        # TODO: Need to make this work.
        self.data_start_index = 0
        self.views = {}
        self.algo = algo
        self.current_bcolz_handle = None
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
        self.benchmark_iter = iter(self.algo.benchmark_iter)

        self.cur_data_offset = 0

    def get_current_price_data(self, asset, column):
        path = "{0}/{1}.bcolz".format(FINDATA_DIR, int(asset))
        try:
            carray = self.carrays[column][path]
        except KeyError:
            carray = self.carrays[column][path] = bcolz.carray(
                rootdir=path + "/" + column, mode='r')

        if column == 'volume':
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
        return next(self.benchmark_iter).returns


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)
