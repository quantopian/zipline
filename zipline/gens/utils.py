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

from collections import OrderedDict
from hashlib import md5
from datetime import datetime
from itertools import izip_longest
from zipline import ndict
from zipline.protocol import DATASOURCE_TYPE


def mock_raw_event(sid, dt):
    event = {
        'sid': sid,
        'dt': dt,
        'price': 1.0,
        'volume': 1
    }
    return event


def mock_done(id):
    return ndict({
            'dt': "DONE",
            "source_id": id,
            'tnfm_id': id,
            'tnfm_value': None,
            'type': DATASOURCE_TYPE.DONE
    })

done_message = mock_done


def alternate(g1, g2):
    """Specialized version of roundrobin for just 2 generators."""
    for e1, e2 in izip_longest(g1, g2):
        if e1 is not None:
            yield e1
        if e2 is not None:
            yield e2


def roundrobin(sources, namestrings):
    """
    Takes N generators, pulling one element off each until all inputs
    are empty.
    """
    assert len(sources) == len(namestrings)
    mapping = OrderedDict(zip(namestrings, sources))

    # While our generators have not been exhausted, pull elements
    while mapping.keys() != []:
        for namestring, source in mapping.iteritems():
            try:
                message = source.next()
                # allow sources to yield None to avoid blocking.
                if message:
                    yield message
            except StopIteration:
                yield done_message(namestring)
                del mapping[namestring]


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

    row = ndict({
        'source_id': source_id,
        'type': DATASOURCE_TYPE.TRADE,
        'sid': sid,
        'dt': datetime,
        'price': price,
        'close': price,
        'open': price,
        'low': price * .95,
        'high': price * 1.05,
        'volume': amount
    })
    return row


def assert_datasource_protocol(event):
    """Assert that an event meets the protocol for datasource outputs."""

    assert isinstance(event, ndict)
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE

    # Done packets have no dt.
    if not event.type == DATASOURCE_TYPE.DONE:
        assert isinstance(event.dt, datetime)
        assert event.dt.tzinfo == pytz.utc


def assert_trade_protocol(event):
    """Assert that an event meets the protocol for datasource TRADE outputs."""
    assert_datasource_protocol(event)

    assert isinstance(event, ndict)
    assert event.type == DATASOURCE_TYPE.TRADE
    assert isinstance(event.sid, int)
    assert isinstance(event.price, numbers.Real)
    assert isinstance(event.volume, numbers.Integral)
    assert isinstance(event.dt, datetime)


def assert_datasource_unframe_protocol(event):
    """Assert that an event is valid output of zp.DATASOURCE_UNFRAME."""
    assert isinstance(event, ndict)
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE
    assert 'dt' in event


def assert_sort_protocol(event):
    """Assert that an event is valid input to zp.FEED_FRAME."""
    assert isinstance(event, ndict)
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE
    assert 'dt' in event


def assert_sort_unframe_protocol(event):
    """Same as above."""
    assert isinstance(event, ndict)
    assert isinstance(event.source_id, basestring)
    assert event.type in DATASOURCE_TYPE
    assert 'dt' in event


def assert_transform_protocol(event):
    """Transforms should return an ndict to be merged by merge."""
    assert isinstance(event, ndict)


def assert_merge_protocol(tnfm_ids, message):
    """Merge should output an ndict with a field for each id
    in its transform set."""
    assert isinstance(message, ndict)
    assert set(tnfm_ids) == set(message.keys())
