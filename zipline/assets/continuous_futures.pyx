# cython: embedsignature=True
#
# Copyright 2016 Quantopian, Inc.
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
Cythonized ContinuousFutures object.
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

from numpy import empty
from numpy cimport long_t, int64_t
import warnings

from zipline.utils.calendars import get_calendar


cdef class ContinuousFuture:
    """
    Represents a specifier for a chain of future contracts, where the
    coordinates for the chain are:
    root_symbol : str
        The root symbol of the contracts.
    offset : int
        The distance from the primary chain.
        e.g. 0 specifies the primary chain, 1 the secondary, etc.
    roll_style : str
        How rolls from contract to contract should be calculated.
        Currently supports 'calendar'.

    Instances of this class are exposed to the algorithm.
    """

    cdef readonly long_t sid
    # Cached hash of self.sid
    cdef long_t sid_hash

    cdef readonly object root_symbol
    cdef readonly int offset
    cdef readonly object roll_style

    cdef readonly object start_date
    cdef readonly object end_date

    cdef readonly object exchange

    _kwargnames = frozenset({
        'sid',
        'root_symbol',
        'offset',
        'start_date',
        'end_date',
        'exchange',
    })

    def __init__(self,
                 long_t sid, # sid is required
                 object root_symbol,
                 int offset,
                 object roll_style,
                 object start_date,
                 object end_date,
                 object exchange):

        self.sid = sid
        self.sid_hash = hash(sid)
        self.root_symbol = root_symbol
        self.roll_style = roll_style
        self.offset = offset
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date

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
        cdef long_t x_as_int, y_as_int

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

    def __str__(self):
        return '%s(%d [%s, %s, %s])' % (
            type(self).__name__,
            self.sid,
            self.root_symbol,
            self.offset,
            self.roll_style
        )

    def __repr__(self):
        attrs = ('root_symbol', 'offset', 'roll_style')
        tuples = ((attr, repr(getattr(self, attr, None)))
                  for attr in attrs)
        strings = ('%s=%s' % (t[0], t[1]) for t in tuples)
        params = ', '.join(strings)
        return 'ContinuousFuture(%d, %s)' % (self.sid, params)

    cpdef __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.root_symbol,
                                 self.start_date,
                                 self.end_date,
                                 self.offset,
                                 self.roll_style,
                                 self.exchange))

    cpdef to_dict(self):
        """
        Convert to a python dict.
        """
        return {
            'sid': self.sid,
            'root_symbol': self.root_symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'offset': self.offset,
            'roll_style': self.roll_style,
            'exchange': self.exchange,
        }

    @classmethod
    def from_dict(cls, dict_):
        """
        Build an ContinuousFuture instance from a dict.
        """
        return cls(**dict_)

    def is_alive_for_session(self, session_label):
        """
        Returns whether the continuous future is alive at the given dt.

        Parameters
        ----------
        session_label: pd.Timestamp
            The desired session label to check. (midnight UTC)

        Returns
        -------
        boolean: whether the continuous is alive at the given dt.
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
        boolean: whether the continuous futures's exchange is open at the
        given minute.
        """
        calendar = get_calendar(self.exchange)
        return calendar.is_open_on_minute(dt_minute)


cdef class OrderedContracts(object):
    """
    A container for aligned values of a future contract chain, in sorted order
    of their occurrence.
    Used to get answers about contracts in relation to their auto close
    dates and start dates.

    The number of contracts for a given root symbol is ~250,
    which is why search by comparison over the range of contracts is
    used. At this size, this is faster than using an index or np.searchsorted.

    Members
    -------
    root_symbol : str
        The root symbol of the future contract chain.
    contract_sids : long[:]
        The contract sids in sorted order of occurrence.
    start_dates : long[:]
        The start dates of the contracts in the chain.
        Corresponds by index with contract_sids.
    auto_close_dates : long[:]
        The auto close dates of the contracts in the chain.
        Corresponds by index with contract_sids.

    Instances of this class are used by the simulation engine, but not
    exposed to the algorithm.
    """

    cdef readonly object root_symbol
    cdef int _size
    cdef readonly long_t[:] contract_sids
    cdef readonly long_t[:] start_dates
    cdef readonly long_t[:] auto_close_dates

    def __init__(self,
                 object root_symbol,
                 long_t[:] contract_sids,
                 long_t[:] start_dates,
                 long_t[:] auto_close_dates):
        self._size = len(contract_sids)
        self.root_symbol = root_symbol
        self.contract_sids = contract_sids
        self.start_dates = start_dates
        self.auto_close_dates = auto_close_dates
    
    cpdef long_t contract_before_auto_close(self, long_t dt_value):
        """
        Get the contract with next upcoming auto close date.
        """
        cdef Py_ssize_t i, auto_close_date
        for i, auto_close_date in enumerate(self.auto_close_dates):
            if auto_close_date > dt_value:
                break
        return self.contract_sids[i]

    cpdef long_t contract_at_offset(self, long_t sid, Py_ssize_t offset):
        """
        Get the sid which is the given sid plus the offset distance.
        An offset of 0 should be reflexive.
        """
        cdef Py_ssize_t i
        cdef long_t[:] sids
        sids = self.contract_sids
        for i in range(self._size):
            if sid == sids[i]:
                return sids[i + offset]

    cpdef long_t[:] active_chain(self, long_t starting_sid, long_t dt_value):
        cdef Py_ssize_t left, right, i, j
        cdef long_t[:] sids, start_dates
        left = 0
        right = self._size
        sids = self.contract_sids
        start_dates = self.start_dates

        for i in range(self._size):
            if starting_sid == sids[i]:
                left = i
                break

        for j in range(i, self._size):
            if start_dates[j] > dt_value:
                right = j
                break

        return sids[left:right]
