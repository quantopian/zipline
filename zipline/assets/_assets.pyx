# cython: embedsignature=True
#
# Copyright 2015 Quantopian, Inc.
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

"""
Cythonized Asset object.
"""
cimport cython
from cpython.number cimport PyNumber_Index
from cpython.object cimport (
    Py_EQ,
    Py_NE,
    Py_GE,
    Py_LE,
    Py_GT,
    Py_LT,
)
from cpython cimport bool

import numpy as np
from numpy cimport int64_t
import warnings
cimport numpy as np

from zipline.utils.calendars import get_calendar


# IMPORTANT NOTE: You must change this template if you change
# Asset.__reduce__, or else we'll attempt to unpickle an old version of this
# class
CACHE_FILE_TEMPLATE = '/tmp/.%s-%s.v7.cache'


cdef class Asset:

    cdef readonly int sid
    # Cached hash of self.sid
    cdef int sid_hash

    cdef readonly object symbol
    cdef readonly object asset_name

    cdef readonly object start_date
    cdef readonly object end_date
    cdef public object first_traded
    cdef readonly object auto_close_date

    cdef readonly object exchange
    cdef readonly object exchange_full

    _kwargnames = frozenset({
        'sid',
        'symbol',
        'asset_name',
        'start_date',
        'end_date',
        'first_traded',
        'auto_close_date',
        'exchange',
        'exchange_full',
    })

    def __init__(self,
                 int sid, # sid is required
                 object exchange, # exchange is required
                 object symbol="",
                 object asset_name="",
                 object start_date=None,
                 object end_date=None,
                 object first_traded=None,
                 object auto_close_date=None,
                 object exchange_full=None):

        self.sid = sid
        self.sid_hash = hash(sid)
        self.symbol = symbol
        self.asset_name = asset_name
        self.exchange = exchange
        self.exchange_full = (exchange_full if exchange_full is not None
                              else exchange)
        self.start_date = start_date
        self.end_date = end_date
        self.first_traded = first_traded
        self.auto_close_date = auto_close_date

    def __int__(self):
        return self.sid

    def __index__(self):
        return self.sid

    def __hash__(self):
        return self.sid_hash

    def __richcmp__(x, y, int op):
        """
        Cython rich comparison method.  This is used in place of various
        equality checkers in pure python.
        """
        cdef int x_as_int, y_as_int

        try:
            x_as_int = PyNumber_Index(x)
        except (TypeError, OverflowError):
            return NotImplemented

        try:
            y_as_int = PyNumber_Index(y)
        except (TypeError, OverflowError):
            return NotImplemented

        compared = x_as_int - y_as_int

        # Handle == and != first because they're significantly more common
        # operations.
        if op == Py_EQ:
            return compared == 0
        elif op == Py_NE:
            return compared != 0
        elif op == Py_LT:
            return compared < 0
        elif op == Py_LE:
            return compared <= 0
        elif op == Py_GT:
            return compared > 0
        elif op == Py_GE:
            return compared >= 0
        else:
            raise AssertionError('%d is not an operator' % op)

    def __repr__(self):
        if self.symbol:
            return '%s(%d [%s])' % (type(self).__name__, self.sid, self.symbol)
        else:
            return '%s(%d)' % (type(self).__name__, self.sid)

    cpdef __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.exchange,
                                 self.symbol,
                                 self.asset_name,
                                 self.start_date,
                                 self.end_date,
                                 self.first_traded,
                                 self.auto_close_date,
                                 self.exchange_full))

    cpdef to_dict(self):
        """
        Convert to a python dict.
        """
        return {
            'sid': self.sid,
            'symbol': self.symbol,
            'asset_name': self.asset_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'first_traded': self.first_traded,
            'auto_close_date': self.auto_close_date,
            'exchange': self.exchange,
            'exchange_full': self.exchange_full,
        }

    @classmethod
    def from_dict(cls, dict_):
        """
        Build an Asset instance from a dict.
        """
        return cls(**dict_)

    def is_alive_for_session(self, session_label):
        """
        Returns whether the asset is alive at the given dt.

        Parameters
        ----------
        session_label: pd.Timestamp
            The desired session label to check. (midnight UTC)

        Returns
        -------
        boolean: whether the asset is alive at the given dt.
        """
        cdef int64_t ref_start
        cdef int64_t ref_end

        ref_start = self.start_date.value
        ref_end = self.end_date.value

        return ref_start <= session_label.value <= ref_end

    def is_exchange_open(self, dt_minute):
        """
        Parameters
        ----------
        dt_minute: pd.Timestamp (UTC, tz-aware)
            The minute to check.

        Returns
        -------
        boolean: whether the asset's exchange is open at the given minute.
        """
        calendar = get_calendar(self.exchange)
        return calendar.is_open_on_minute(dt_minute)


cdef class Equity(Asset):

    property security_start_date:
        """
        DEPRECATION: This property should be deprecated and is only present for
        backwards compatibility
        """
        def __get__(self):
            warnings.warn("The security_start_date property will soon be "
            "retired. Please use the start_date property instead.",
            DeprecationWarning)
            return self.start_date

    property security_end_date:
        """
        DEPRECATION: This property should be deprecated and is only present for
        backwards compatibility
        """
        def __get__(self):
            warnings.warn("The security_end_date property will soon be "
            "retired. Please use the end_date property instead.",
            DeprecationWarning)
            return self.end_date

    property security_name:
        """
        DEPRECATION: This property should be deprecated and is only present for
        backwards compatibility
        """
        def __get__(self):
            warnings.warn("The security_name property will soon be "
            "retired. Please use the asset_name property instead.",
            DeprecationWarning)
            return self.asset_name


cdef class Future(Asset):

    cdef readonly object root_symbol
    cdef readonly object notice_date
    cdef readonly object expiration_date
    cdef readonly object tick_size
    cdef readonly float multiplier

    _kwargnames = frozenset({
        'sid',
        'symbol',
        'root_symbol',
        'asset_name',
        'start_date',
        'end_date',
        'notice_date',
        'expiration_date',
        'auto_close_date',
        'first_traded',
        'exchange',
        'tick_size',
        'multiplier',
        'exchange_full',
    })

    def __init__(self,
                 int sid, # sid is required
                 object exchange, # exchange is required
                 object symbol="",
                 object root_symbol="",
                 object asset_name="",
                 object start_date=None,
                 object end_date=None,
                 object notice_date=None,
                 object expiration_date=None,
                 object auto_close_date=None,
                 object first_traded=None,
                 object tick_size="",
                 float multiplier=1.0,
                 object exchange_full=None):

        super().__init__(
            sid,
            exchange,
            symbol=symbol,
            asset_name=asset_name,
            start_date=start_date,
            end_date=end_date,
            first_traded=first_traded,
            auto_close_date=auto_close_date,
            exchange_full=exchange_full,
        )
        self.root_symbol = root_symbol
        self.notice_date = notice_date
        self.expiration_date = expiration_date
        self.tick_size = tick_size
        self.multiplier = multiplier

        if auto_close_date is None:
            if notice_date is None:
                self.auto_close_date = expiration_date
            elif expiration_date is None:
                self.auto_close_date = notice_date
            else:
                self.auto_close_date = min(notice_date, expiration_date)

    cpdef __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.exchange,
                                 self.symbol,
                                 self.root_symbol,
                                 self.asset_name,
                                 self.start_date,
                                 self.end_date,
                                 self.notice_date,
                                 self.expiration_date,
                                 self.auto_close_date,
                                 self.first_traded,
                                 self.tick_size,
                                 self.multiplier,
                                 self.exchange_full))

    cpdef to_dict(self):
        """
        Convert to a python dict.
        """
        super_dict = super(Future, self).to_dict()
        super_dict['root_symbol'] = self.root_symbol
        super_dict['notice_date'] = self.notice_date
        super_dict['expiration_date'] = self.expiration_date
        super_dict['tick_size'] = self.tick_size
        super_dict['multiplier'] = self.multiplier
        return super_dict


def make_asset_array(int size, Asset asset):
    cdef np.ndarray out = np.empty([size], dtype=object)
    out.fill(asset)
    return out
