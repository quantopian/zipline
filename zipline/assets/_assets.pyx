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

import numpy as np
import warnings
cimport numpy as np

# IMPORTANT NOTE: You must change this template if you change
# Asset.__reduce__, or else we'll attempt to unpickle an old version of this
# class
CACHE_FILE_TEMPLATE = '/tmp/.%s-%s.v4.cache'

cdef class Asset:

    cdef readonly int sid
    # Cached hash of self.sid
    cdef int sid_hash

    cdef readonly object symbol
    cdef readonly object asset_name

    cdef readonly object start_date
    cdef readonly object end_date
    cdef public object first_traded

    cdef readonly object exchange

    def __cinit__(self,
                  int sid, # sid is required
                  object symbol="",
                  object asset_name="",
                  object start_date=None,
                  object end_date=None,
                  object first_traded=None,
                  object exchange="",
                  *args,
                  **kwargs):

        self.sid           = sid
        self.sid_hash      = hash(sid)
        self.symbol        = symbol
        self.asset_name    = asset_name
        self.exchange      = exchange
        self.start_date    = start_date
        self.end_date      = end_date
        self.first_traded  = first_traded

    def __int__(self):
        return self.sid

    def __hash__(self):
        return self.sid_hash

    def __richcmp__(x, y, int op):
        """
        Cython rich comparison method.  This is used in place of various
        equality checkers in pure python.

        <	0
        <=	1
        ==	2
        !=	3
        >	4
        >=	5
        """
        cdef int x_as_int, y_as_int

        if isinstance(x, Asset):
            x_as_int = x.sid
        elif isinstance(x, int):
            x_as_int = x
        else:
            return NotImplemented

        if isinstance(y, Asset):
            y_as_int = y.sid
        elif isinstance(y, int):
            y_as_int = y
        else:
            return NotImplemented

        compared = x_as_int - y_as_int

        # Handle == and != first because they're significantly more common
        # operations.
        if op == 2:
            # Equality
            return compared == 0
        elif op == 3:
            # Non-equality
            return compared != 0
        elif op == 0:
            # <
            return compared < 0
        elif op == 1:
            # <=
            return compared <= 0
        elif op == 4:
            # >
            return compared > 0
        elif op == 5:
            # >=
            return compared >= 0

    def __str__(self):
        if self.symbol:
            return 'Asset(%d [%s])' % (self.sid, self.symbol)
        else:
            return 'Asset(%d)' % self.sid

    def __repr__(self):
        attrs = ('symbol', 'asset_name', 'exchange',
                 'start_date', 'end_date', 'first_traded')
        tuples = ((attr, repr(getattr(self, attr, None)))
                  for attr in attrs)
        strings = ('%s=%s' % (t[0], t[1]) for t in tuples)
        params = ', '.join(strings)
        return 'Asset(%d, %s)' % (self.sid, params)

    cpdef __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.symbol,
                                 self.asset_name,
                                 self.start_date,
                                 self.end_date,
                                 self.first_traded,
                                 self.exchange,))

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
            'exchange': self.exchange,
        }

    @classmethod
    def from_dict(cls, dict_):
        """
        Build an Asset instance from a dict.
        """
        return cls(**dict_)


cdef class Equity(Asset):

    def __str__(self):
        if self.symbol:
            return 'Equity(%d [%s])' % (self.sid, self.symbol)
        else:
            return 'Equity(%d)' % self.sid

    def __repr__(self):
        attrs = ('symbol', 'asset_name', 'exchange',
                 'start_date', 'end_date', 'first_traded')
        tuples = ((attr, repr(getattr(self, attr, None)))
                  for attr in attrs)
        strings = ('%s=%s' % (t[0], t[1]) for t in tuples)
        params = ', '.join(strings)
        return 'Equity(%d, %s)' % (self.sid, params)

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
    cdef readonly float contract_multiplier

    def __cinit__(self,
                  int sid, # sid is required
                  object symbol="",
                  object root_symbol="",
                  object asset_name="",
                  object start_date=None,
                  object end_date=None,
                  object notice_date=None,
                  object expiration_date=None,
                  object first_traded=None,
                  object exchange="",
                  float contract_multiplier=1):

        self.root_symbol         = root_symbol
        self.notice_date         = notice_date
        self.expiration_date     = expiration_date
        self.contract_multiplier = contract_multiplier

        # Assign the expiration as the end_date if end_date is not explicit
        if self.end_date is None:
            self.end_date = expiration_date

    def __str__(self):
        if self.symbol:
            return 'Future(%d [%s])' % (self.sid, self.symbol)
        else:
            return 'Future(%d)' % self.sid

    def __repr__(self):
        attrs = ('symbol', 'root_symbol', 'asset_name', 'exchange',
                 'start_date', 'end_date', 'first_traded', 'notice_date',
                 'expiration_date', 'contract_multiplier')
        tuples = ((attr, repr(getattr(self, attr, None)))
                  for attr in attrs)
        strings = ('%s=%s' % (t[0], t[1]) for t in tuples)
        params = ', '.join(strings)
        return 'Future(%d, %s)' % (self.sid, params)

    cpdef __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.symbol,
                                 self.root_symbol,
                                 self.asset_name,
                                 self.start_date,
                                 self.end_date,
                                 self.notice_date,
                                 self.expiration_date,
                                 self.first_traded,
                                 self.exchange,
                                 self.contract_multiplier,))

    cpdef to_dict(self):
        """
        Convert to a python dict.
        """
        super_dict = super(Future, self).to_dict()
        super_dict['root_symbol'] = self.root_symbol
        super_dict['notice_date'] = self.notice_date
        super_dict['expiration_date'] = self.expiration_date
        super_dict['contract_multiplier'] = self.contract_multiplier
        return super_dict


def make_asset_array(int size, Asset asset):
    cdef np.ndarray out = np.empty([size], dtype=object)
    out.fill(asset)
    return out
