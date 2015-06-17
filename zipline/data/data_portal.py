import bcolz
import numpy as np
import os

FINDATA_DIR = os.getenv("FINDATA_DIR")


class DataPortal(object):

    def __init__(self, algo):
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

    def get_current_price_data(self, asset, column):
        dt = self.algo.datetime
        path = "{0}/{1}/{2}/{3}_equity-minutes.bcolz".format(
            FINDATA_DIR,
            str(dt.year),
            str(dt.month).zfill(2),
            str(dt.date()))
        try:
            carray = self.carrays[column][path]
        except KeyError:
            carray = self.carrays[column][path] = bcolz.carray(
                rootdir=path + "/" + column, mode='r')
        try:
            sid_carray = self.carrays['sid'][path]
            dt_carray = self.carrays['dt'][path]
        except KeyError:
            sid_carray = self.carrays['sid'][path] = bcolz.carray(
                rootdir=path + "/" + 'sid', mode='r')
            dt_carray = self.carrays['dt'][path] = bcolz.carray(
                rootdir=path + "/" + 'dt', mode='r')
        where_sid = np.where(sid_carray[:] == int(asset))
        where_dt = np.where(dt_carray[:] == int(dt.value / 10e8))
        ix = np.intersect1d(where_sid, where_dt)
        price = carray[ix]
        return price

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
