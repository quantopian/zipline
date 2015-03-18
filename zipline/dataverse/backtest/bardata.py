import zipline.protocol as zp

from six import iteritems, iterkeys


class BacktestSIDData(object):
    def __init__(self, sid, dataverse, initial_values=None):
        self.sid = sid
        self.dataverse = dataverse
        self.obj = zp.SIDData(sid)
        if initial_values:
            self.obj.update(initial_values)

    def update(self, *args, **kwargs):
        return self.obj.update(*args, **kwargs)

    @property
    def datetime(self):
        return self.obj.dt

    def __getattr__(self, name):
        try:
            return self.dataverse.get_sid_data(self.sid, name):
        except:
            return getattr(self.obj, name)


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


class BacktestBarData(zp.BarData):
    def __init__(self, data=None, siddata_class=BacktestSIDData,
                 dataverse=None):
        if dataverse is None:
            raise TypeError("dataverse cannot be None for BacktestBarData")
        self._dataverse = dataverse
        self._data = data or {}
        self._contains_override = None
        self._siddata_class = siddata_class

    _contains_override_ = None

    @property
    def _contains_override(self):
        return self._contains_override_

    @_contains_override.setter
    def _contains_override(self, value):
        self._contains_override_ = value
        self._keys_cache_ = None

    _keys_cache = None

    @property
    def _keys_cache(self):
        if self._keys_cache_ is None:
            keys = self._data
            if self._contains_override:
                keys = list(filter(self._contains_override, keys))
            self._keys_cache_ = keys
        return self._keys_cache_

    def __contains__(self, name):
        return name in self._keys_cache

    def get_default(self, name):
        try:
            sid_data = self[name]
        except KeyError:
            sid_data = self[name] = self._siddata_class(name, self._dataverse)
        return sid_data

    def __setitem__(self, name, value):
        self._data[name] = value
        self._keys_cache_ = None

    def __getitem__(self, name):
        return self._data[name]

    def __delitem__(self, name):
        del self._data[name]
        self._keys_cache_ = None

    def __iter__(self):
        keys_cache = self._keys_cache
        for sid, data in iteritems(self._data):
            # Allow contains override to filter out sids.
            if sid in keys_cache:
                if len(data):
                    yield sid

    def iterkeys(self):
        # Allow contains override to filter out sids.
        keys_cache = self._keys_cache
        return (sid for sid in iterkeys(self._data) if sid in keys_cache)

    def itervalues(self):
        return (value for _sid, value in self.iteritems())

    def iteritems(self):
        keys_cache = self._keys_cache
        return ((sid, value) for sid, value
                in iteritems(self._data)
                if sid in keys_cache)

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self._data)
