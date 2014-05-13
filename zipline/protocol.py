#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six import iteritems, iterkeys

from . utils.protocol_utils import Enum

# Datasource type should completely determine the other fields of a
# message with its type.
DATASOURCE_TYPE = Enum(
    'AS_TRADED_EQUITY',
    'MERGER',
    'SPLIT',
    'DIVIDEND',
    'TRADE',
    'TRANSACTION',
    'ORDER',
    'EMPTY',
    'DONE',
    'CUSTOM',
    'BENCHMARK',
    'COMMISSION'
)


class Event(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__ = initial_values

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __delitem__(self, name):
        delattr(self, name)

    def keys(self):
        return self.__dict__.keys()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "Event({0})".format(self.__dict__)


class Order(Event):
    pass


class Portfolio(object):

    def __init__(self):
        self.capital_used = 0.0
        self.starting_cash = 0.0
        self.portfolio_value = 0.0
        self.pnl = 0.0
        self.returns = 0.0
        self.cash = 0.0
        self.positions = Positions()
        self.start_date = None
        self.positions_value = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Portfolio({0})".format(self.__dict__)


class Position(object):

    def __init__(self, sid):
        self.sid = sid
        self.amount = 0
        self.cost_basis = 0.0  # per share
        self.last_sale_price = 0.0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return "Position({0})".format(self.__dict__)


class Positions(dict):

    def __missing__(self, key):
        pos = Position(key)
        self[key] = pos
        return pos


class SIDData(object):

    def __init__(self, initial_values=None):
        if initial_values:
            self.__dict__ = initial_values

    @property
    def datetime(self):
        """
        Provides an alias from data['foo'].datetime -> data['foo'].dt

        `datetime` was previously provided by adding a seperate `datetime`
        member of the SIDData object via a generator that wrapped the incoming
        data feed and added the field to each equity event.

        This alias is intended to be temporary, to provide backwards
        compatibility with existing algorithms, but should be considered
        deprecated, and may be removed in the future.
        """
        return self.dt

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, name):
        return name in self.__dict__

    def __repr__(self):
        return "SIDData({0})".format(self.__dict__)


class BarData(object):
    """
    Holds the event data for all sids for a given dt.

    This is what is passed as `data` to the `handle_data` function.

    Note: Many methods are analogues of dictionary because of historical
    usage of what this replaced as a dictionary subclass.
    """

    def __init__(self):
        self._data = {}
        self._contains_override = None

    def __contains__(self, name):
        if self._contains_override:
            if self._contains_override(name):
                return name in self._data
            else:
                return False
        else:
            return name in self._data

    def has_key(self, name):
        """
        DEPRECATED: __contains__ is preferred, but this method is for
        compatibility with existing algorithms.
        """
        return name in self

    def __setitem__(self, name, value):
        self._data[name] = value

    def __getitem__(self, name):
        return self._data[name]

    def __delitem__(self, name):
        del self._data[name]

    def __iter__(self):
        for sid, data in iteritems(self._data):
            # Allow contains override to filter out sids.
            if sid in self:
                if len(data):
                    yield sid

    def iterkeys(self):
        # Allow contains override to filter out sids.
        return (sid for sid in iterkeys(self._data) if sid in self)

    def keys(self):
        # Allow contains override to filter out sids.
        return list(self.iterkeys())

    def itervalues(self):
        return (value for sid, value in iteritems(self))

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        return ((sid, value) for sid, value
                in iteritems(self._data)
                if sid in self)

    def items(self):
        return list(iteritems(self))

    def __len__(self):
        return len(self.keys())
