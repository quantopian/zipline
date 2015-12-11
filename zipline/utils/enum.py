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

from ctypes import (
    Structure,
    c_ubyte,
    c_uint,
    c_ulong,
    c_ulonglong,
    c_ushort,
    sizeof,
)

import numpy as np
import pandas as pd
from six.moves import range


_inttypes_map = {
    sizeof(t) - 1: t for t in {
        c_ubyte,
        c_uint,
        c_ulong,
        c_ulonglong,
        c_ushort
    }
}
_inttypes = list(
    pd.Series(_inttypes_map).reindex(
        range(max(_inttypes_map.keys())),
        method='bfill',
    ),
)


def enum(option, *options):
    """
    Construct a new enum object.

    Parameters
    ----------
    *options : iterable of str
        The names of the fields for the enum.

    Returns
    -------
    enum
        A new enum collection.

    Examples
    --------
    >>> e = enum('a', 'b', 'c')
    >>> e
    <enum: ('a', 'b', 'c')>
    >>> e.a
    0
    >>> e.b
    1
    >>> e.a in e
    True
    >>> tuple(e)
    (0, 1, 2)

    Notes
    -----
    Identity checking is not guaranteed to work with enum members, instead
    equality checks should be used. From CPython's documentation:

    "The current implementation keeps an array of integer objects for all
    integers between -5 and 256, when you create an int in that range you
    actually just get back a reference to the existing object. So it should be
    possible to change the value of 1. I suspect the behaviour of Python in
    this case is undefined. :-)"
    """
    options = (option,) + options
    rangeob = range(len(options))

    try:
        inttype = _inttypes[int(np.log2(len(options) - 1)) // 8]
    except IndexError:
        raise OverflowError(
            'Cannot store enums with more than sys.maxsize elements, got %d' %
            len(options),
        )

    class _enum(Structure):
        _fields_ = [(o, inttype) for o in options]

        def __iter__(self):
            return iter(rangeob)

        def __contains__(self, value):
            return 0 <= value < len(options)

        def __repr__(self):
            return '<enum: %s>' % (
                ('%d fields' % len(options))
                if len(options) > 10 else
                repr(options)
            )

    return _enum(*rangeob)
