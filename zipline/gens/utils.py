#
# Copyright 2012 Quantopian, Inc.
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


import pytz
import numbers

from hashlib import md5
from datetime import datetime
from zipline.protocol import (
    DATASOURCE_TYPE,
    Event
)


def hash_args(*args, **kwargs):
    """Define a unique string for any set of representable args."""
    arg_string = '_'.join([str(arg) for arg in args])
    kwarg_string = '_'.join([str(key) + '=' + str(value)
                             for key, value in kwargs.iteritems()])
    combined = ':'.join([arg_string, kwarg_string])

    hasher = md5()
    hasher.update(combined)
    return hasher.hexdigest()


def create_trade(sid, price, amount, datetime, source_id="test_factory"):

    trade = Event()

    trade.source_id = source_id
    trade.type = DATASOURCE_TYPE.TRADE
    trade.sid = sid
    trade.dt = datetime
    trade.price = price
    trade.close = price
    trade.open = price
    trade.low = price * .95
    trade.high = price * 1.05
    trade.volume = amount
    trade.TRANSACTION = None

    return trade


def assert_datasource_protocol(event):
    """Assert that an event meets the protocol for datasource outputs."""

    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE

    # Done packets have no dt.
    if not event.type == DATASOURCE_TYPE.DONE:
        assert isinstance(event.dt, datetime)
        assert event.dt.tzinfo == pytz.utc


def assert_trade_protocol(event):
    """Assert that an event meets the protocol for datasource TRADE outputs."""
    assert_datasource_protocol(event)

    assert event.type == DATASOURCE_TYPE.TRADE
    assert isinstance(event.sid, int)
    assert isinstance(event.price, numbers.Real)
    assert isinstance(event.volume, numbers.Integral)
    assert isinstance(event.dt, datetime)


def assert_datasource_unframe_protocol(event):
    """Assert that an event is valid output of zp.DATASOURCE_UNFRAME."""
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE


def assert_sort_protocol(event):
    """Assert that an event is valid input to zp.FEED_FRAME."""
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE


def assert_sort_unframe_protocol(event):
    """Same as above."""
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE


def assert_merge_protocol(tnfm_ids, message):
    """Merge should output an ndict with a field for each id
    in its transform set."""
    assert set(tnfm_ids) == set(message.keys())
