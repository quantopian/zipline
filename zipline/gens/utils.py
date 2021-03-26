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


import pytz
import numbers

from hashlib import md5
from datetime import datetime
from zipline.protocol import DATASOURCE_TYPE


def hash_args(*args, **kwargs):
    """Define a unique string for any set of representable args."""
    arg_string = "_".join([str(arg) for arg in args])
    kwarg_string = "_".join(
        [str(key) + "=" + str(value) for key, value in kwargs.items()]
    )
    combined = ":".join([arg_string, kwarg_string])

    hasher = md5()
    hasher.update(combined)
    return hasher.hexdigest()


def assert_datasource_protocol(event):
    """Assert that an event meets the protocol for datasource outputs."""

    assert event.type in DATASOURCE_TYPE

    # Done packets have no dt.
    if not event.type == DATASOURCE_TYPE.DONE:
        assert isinstance(event.dt, datetime)
        assert event.dt.tzinfo == pytz.utc


def assert_trade_protocol(event):
    """Assert that an event meets the protocol for datasource TRADE outputs."""
    assert_datasource_protocol(event)

    assert event.type == DATASOURCE_TYPE.TRADE
    assert isinstance(event.price, numbers.Real)
    assert isinstance(event.volume, numbers.Integral)
    assert isinstance(event.dt, datetime)


def assert_datasource_unframe_protocol(event):
    """Assert that an event is valid output of zp.DATASOURCE_UNFRAME."""
    assert event.type in DATASOURCE_TYPE
