import zipline.protocol as zp
from zipline.dataverse.dataverse import BaseDataverse
from zipline.history.history_container import HistoryContainer


class BacktestHistoryContainer(HistoryContainer):
    def get_history(self, history_spec, algo_dt):
        return super(BacktestHistoryContainer, self).get_history(history_spec,
                                                           algo_dt)

class BacktestDataverse(BaseDataverse):
    history_container_class = BacktestHistoryContainer

class SIDData(object):
    def __init__(self, sid, dataverse=None, initial_values=None):
        self.sid = sid
        self.dataverse = dataverse
        self.obj = zp.SIDData(sid)
        if initial_values:
            self.obj.update(initial_values)

    def update(self, *args, **kwargs):
        return self.obj.update(*args, **kwargs)

    @property
    def datetime(self):
        return self.obj.update

    def get(self, name, default=None):
        return self.obj.get(name, default)

    def __getitem__(self, name):
        return self.obj[name]

    def __setitem__(self, name, value):
        self.obj[name] = value

    def __len__(self):
        return len(self.obj)

    def __contains__(self, name):
        return name in self.obj

    def __repr__(self):
        return repr(self.obj)

    def mavg(self, days):
        return self.obj.mavg(days)

    def stddev(self, days):
        return self.obj.stddev(days)

    def vwap(self, days):
        return self.obj.vwap(days)

    def returns(self):
        return self.obj.returns()
