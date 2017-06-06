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

from functools import partial

from numpy import array, empty, iinfo
from numpy cimport long_t, int64_t
from pandas import Timestamp
import warnings

from zipline.utils.calendars import get_calendar


def delivery_predicate(codes, contract):
    # This relies on symbols that are construct following a pattern of
    # root symbol + delivery code + year, e.g. PLF16
    # This check would be more robust if the future contract class had
    # a 'delivery_month' member.
    delivery_code = contract.symbol[-3]
    return delivery_code in codes

march_cycle_delivery_predicate = partial(delivery_predicate,
                                         set(['H', 'M', 'U', 'Z']))

CHAIN_PREDICATES = {
    'ME': march_cycle_delivery_predicate,
    'PL': partial(delivery_predicate, set(['F', 'J', 'N', 'V'])),
    'PA': march_cycle_delivery_predicate,

    # The majority of trading in these currency futures is done on a
    # March quarterly cycle (Mar, Jun, Sep, Dec) but contracts are
    # listed for the first 3 consecutive months from the present day. We
    # want the continuous futures to be composed of just the quarterly
    # contracts.
    'JY': march_cycle_delivery_predicate,
    'CD': march_cycle_delivery_predicate,
    'AD': march_cycle_delivery_predicate,
    'BP': march_cycle_delivery_predicate,

    # Gold and silver contracts trade on an unusual specific set of months.
    'GC': partial(delivery_predicate, set(['G', 'J', 'M', 'Q', 'V', 'Z'])),
    'XG': partial(delivery_predicate, set(['G', 'J', 'M', 'Q', 'V', 'Z'])),
    'SV': partial(delivery_predicate, set(['H', 'K', 'N', 'U', 'Z'])),
    'YS': partial(delivery_predicate, set(['H', 'K', 'N', 'U', 'Z'])),
}

ADJUSTMENT_STYLES = {'add', 'mul', None}


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

    cdef readonly object adjustment

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
                 object exchange,
                 object adjustment=None):

        self.sid = sid
        self.sid_hash = hash(sid)
        self.root_symbol = root_symbol
        self.roll_style = roll_style
        self.offset = offset
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date
        self.adjustment = adjustment


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
        return '%s(%d [%s, %s, %s, %s])' % (
            type(self).__name__,
            self.sid,
            self.root_symbol,
            self.offset,
            self.roll_style,
            self.adjustment,
        )

    def __repr__(self):
        attrs = ('root_symbol', 'offset', 'roll_style', 'adjustment')
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


cdef class ContractNode(object):

    cdef readonly object contract
    cdef public object prev
    cdef public object next

    def __init__(self, contract):
        self.contract = contract
        self.prev = None
        self.next = None

    def __rshift__(self, offset):
        i = 0
        curr = self
        while i < offset and curr is not None:
            curr = curr.next
            i += 1
        return curr

    def __lshift__(self, offset):
        i = 0
        curr = self
        while i < offset and curr is not None:
            curr = curr.prev
            i += 1
        return curr


cdef class OrderedContracts(object):
    """
    A container for aligned values of a future contract chain, in sorted order
    of their occurrence.
    Used to get answers about contracts in relation to their auto close
    dates and start dates.

    Members
    -------
    root_symbol : str
        The root symbol of the future contract chain.
    contracts : deque
        The contracts in the chain in order of occurrence.
    start_dates : long[:]
        The start dates of the contracts in the chain.
        Corresponds by index with contract_sids.
    auto_close_dates : long[:]
        The auto close dates of the contracts in the chain.
        Corresponds by index with contract_sids.
    future_chain_predicates : dict
        A dict mapping root symbol to a predicate function which accepts a contract
    as a parameter and returns whether or not the contract should be included in the
    chain.

    Instances of this class are used by the simulation engine, but not
    exposed to the algorithm.
    """

    cdef readonly object root_symbol
    cdef readonly object _head_contract
    cdef readonly dict sid_to_contract
    cdef readonly int64_t _start_date
    cdef readonly int64_t _end_date
    cdef readonly object chain_predicate

    def __init__(self, object root_symbol, object contracts, object chain_predicate=None):

        self.root_symbol = root_symbol

        self.sid_to_contract = {}

        self._start_date = iinfo('int64').max
        self._end_date = 0

        if chain_predicate is None:
            chain_predicate = lambda x: True

        self._head_contract = None
        prev = None
        while contracts:
            contract = contracts.popleft()

            # It is possible that the first contract in our list has a start
            # date on or after its auto close date. In that case the contract
            # is not tradable, so do not include it in the chain.
            if prev is None and contract.start_date >= contract.auto_close_date:
                continue

            if not chain_predicate(contract):
                continue

            self._start_date = min(contract.start_date.value, self._start_date)
            self._end_date = max(contract.end_date.value, self._end_date)

            curr = ContractNode(contract)
            self.sid_to_contract[contract.sid] = curr
            if self._head_contract is None:
                self._head_contract = curr
                prev = curr
                continue
            curr.prev = prev
            prev.next = curr
            prev = curr

    cpdef long_t contract_before_auto_close(self, long_t dt_value):
        """
        Get the contract with next upcoming auto close date.
        """
        curr = self._head_contract
        while curr.next is not None:
            if curr.contract.auto_close_date.value > dt_value:
                break
            curr = curr.next
        return curr.contract.sid

    cpdef contract_at_offset(self, long_t sid, Py_ssize_t offset, int64_t start_cap):
        """
        Get the sid which is the given sid plus the offset distance.
        An offset of 0 should be reflexive.
        """
        cdef Py_ssize_t i
        curr = self.sid_to_contract[sid]
        i = 0
        while i < offset:
            if curr.next is None:
                return None
            curr = curr.next
            i += 1
        if curr.contract.start_date.value <= start_cap:
            return curr.contract.sid
        else:
            return None

    cpdef long_t[:] active_chain(self, long_t starting_sid, long_t dt_value):
        curr = self.sid_to_contract[starting_sid]
        cdef list contracts = []

        while curr is not None:
            if curr.contract.start_date.value <= dt_value:
                contracts.append(curr.contract.sid)
            curr = curr.next

        return array(contracts, dtype='int64')

    property start_date:
        def __get__(self):
            return Timestamp(self._start_date, tz='UTC')

    property end_date:
        def __get__(self):
            return Timestamp(self._end_date, tz='UTC')
