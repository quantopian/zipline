import pytz
import numbers

from hashlib import md5
from datetime import datetime, timedelta
from itertools import izip_longest
from zipline import ndict
from zipline.protocol import DATASOURCE_TYPE

def mock_raw_event(sid, dt):
    event = {
        'sid'    : sid,
        'dt'     : dt,
        'price'  : 1.0,
        'volume' : 1
    }
    return event

def date_gen(start = datetime(2012, 6, 6, 0), delta = timedelta(minutes = 1), n = 100):
    return (start + i * delta for i in xrange(n))

def alternate(g1, g2):
    for e1, e2 in izip_longest(g1, g2):
        if e1 != None:
            yield e1
        if e2 != None:
            yield e2

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
    assert event.has_key('dt')
    
def assert_feed_protocol(event):
    pass
