import zipline.protocol as zp
from zipline.dataverse.dataverse import BaseDataverse
from zipline.history.history_container import (
    HistoryContainer,
    HistoryContainerDelta
)


class HistoryFrame(object):
    def __init__(self):
        pass


class BacktestHistoryContainer(HistoryContainer):
    def __init__(self, *args, **kwargs):
        self.dataverse = kwargs.pop('dataverse')
        # need to make this cache easier to update
        # maybe move to chunking?
        source = self.dataverse.source
        values = source.values
        self.values = values
        self.source = source
        self.get_loc = source.major_axis.get_loc
        self.cache = {}

    def get_history(self, history_spec, algo_dt):
        # do this by regions
        loc = self.get_loc(algo_dt)
        start = max(loc - history_spec.bar_count, 0)
        sl = slice(start, loc)

        try:
            data = self.cache[history_spec]
            values = self.values
            vals = values[:, sl]
            block = data._data.blocks[0]
            if vals.flags.f_contiguous != block.values.flags.f_contiguous:
                vals = vals.T
            data._data.blocks[0].values = vals
            data._item_cache.clear()
        except KeyError:
            data = self.source.iloc[:, sl]
            # data = HistoryFrame(data)
            self.cache[history_spec] = data
        return data

    def ensure_spec(self, spec, dt, bar_data):
        """
        Ensure that this container has enough space to hold the data for the
        given spec. This returns a HistoryContainerDelta to represent the
        changes in shape that the container made to support the new
        HistorySpec.
        """
        updated = {}
        return HistoryContainerDelta(**updated)

    def update(self, data, algo_dt):
        pass


class BacktestDataverse(BaseDataverse):
    history_container_class = BacktestHistoryContainer

    def get_history_container(self, *args, **kwargs):
        kwargs['dataverse'] = self
        return self.history_container_class(*args, **kwargs)

    def get_source(self, source, overwrite_sim_params=True):
        self.source = source
        return super(BacktestDataverse, self).get_source(source)

    def update_universe(self, event):
        pass


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
