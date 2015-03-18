from zipline.dataverse.dataverse import BaseDataverse

from .history import BacktestHistoryContainer
from .bardata import BacktestBarData, BacktestSIDData


class BacktestDataverse(BaseDataverse):
    history_container_class = BacktestHistoryContainer
    siddata_class = BacktestSIDData
    bardata_class = BacktestBarData

    def __init__(self):
        self.current_data = self.bardata_class(dataverse=self)
        self.datetime = None

    def asset_finder(self, sid):
        return sid

    def get_history_container(self, *args, **kwargs):
        kwargs['dataverse'] = self
        return self.history_container_class(*args, **kwargs)

    def get_source(self, source, overwrite_sim_params=True):
        self.raw_source = source
        source = super(BacktestDataverse, self).get_source(source)
        self.source = source
        return source

    def update_universe(self, event):
        # we do not update universe i.e. BarData
        pass

    def pre_simulation(self):
        # prime the BarData class with the proper sids
        for sid in self.source.sids:
            sid = self.asset_finder(sid)
            self.current_data.get_default(sid)

    def on_dt_changed(self, dt):
        self.datetime = dt

    def get_sid_data(self, sid, name):
        pass
