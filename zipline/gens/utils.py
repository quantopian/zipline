import pytz
import numbers

from hashlib import md5
from datetime import datetime

from zipline import ndict
from zipline.protocol import DATASOURCE_TYPE


def stringify_args(*args, **kwargs):
    """Define a unique string for any set of representable args."""
    arg_string = '_'.join([str(arg) for arg in args])
    kwarg_string = '_'.join([str(key) + '=' + str(value) for key, value in kwargs.iteritems()])
    combined = ':'.join([arg_string, kwarg_string])
    
    hasher = md5()
    hasher.update(combined)
    return hasher.hexdigest()
    

def assert_datasource_protocol(event):
    """Assert that an event meets the protocol for datasource outputs."""

    assert isinstance(event, ndict)
    assert isinstance(event.source_id, basestring)
    assert isinstance(event.dt, datetime)
    assert event.dt.tzinfo == pytz.utc
    assert event.type in DATASOURCE_TYPE

def assert_trade_protocol(event):
    """Assert that an event meets the protocol for datasource TRADE outputs."""
    assert_datasource_protocol(event)

    assert isinstance(event, ndict)
    assert event.type == DATASOURCE_TYPE.TRADE
    assert isinstance(event.sid, int)
    assert isinstance(event.price, numbers.Real)
    assert isinstance(event.volume, numbers.Integral)
    
