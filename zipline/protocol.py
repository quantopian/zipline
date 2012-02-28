"""
The messaging protocol for Zipline.

Asserts are in place because any protocol error corresponds to a
programmer error so we want it to fail fast and in an obvious way
so it doesn't happen again. ZeroMQ follows the same philosophy.

Notes
=====

Msgpack
-------
Msgpack is the fastest seriaization protocol in Python at the
moment. Its 100% C is typically orders of magnitude faster than
json and pickle making it awesome for ZeroMQ.

You can only serialize Python structural primitives: strings,
numeric types, dicts, tuples and lists. Any any recursive
combinations of these.

Basically every basestring in Python corresponds to valid
msgpack message since the protocol is highly error tolerant.
Just keep in mind that if you ever unpack a raw msgpack string
make sure it looks like what you intend and/or catch ValueError
and TypeError exceptions.

It also has the nice benefit of never invoking ``eval`` ( unlike
json and pickle) which is a major security boon since it is
impossible to arbitrary code for evaluation through messages.

UltraJSON
---------
For anything going to the browser UltraJSON is the fastest
serializer, its mostly C as well.

The same domain of serialization as msgpack applies: Python
structural primitives. It also has the additional constraint
that anything outside of UTF8 can cause serious problems, so if
you have a strong desire to JSON encode ancient Sanskrit
( admit it, we all do ), just say no.

"""

import msgpack
import numbers
import datetime
import pytz
import zipline.util as qutil
#import ujson
#import ultrajson_numpy

from ctypes import Structure, c_ubyte

def Enum(*options):
    """
    Fast enums are very important when we want really tight zmq
    loops. These are probably going to evolve into pure C structs
    anyways so might as well get going on that.
    """
    class cstruct(Structure):
        _fields_ = [(o, c_ubyte) for o in options]
    return cstruct(*range(len(options)))

def FrameExceptionFactory(name):
    """
    Exception factory with a closure around the frame class name.
    """
    class InvalidFrame(Exception):
        def __init__(self, got):
            self.got = got
        def __str__(self):
            return "Invalid {framcls} Frame: {got}".format(
                framecls = name,
                got = self.got,
            )

class namedict(object):
    """
    So that you can use:

        foo.BAR
        -- or --
        foo['BAR']

    For more complex strcuts use collections.namedtuple:
    """

    def __init__(self, dct=None):
        if(dct):
            self.__dict__.update(dct)
    
    def __setitem__(self, key, value):
        """Required for use by pymongo as_class parameter to find."""
        if(key == '_id'):
            self.__dict__['id'] = value
        else:
            self.__dict__[key] = value
    
    def merge(self, other_nd):
        assert isinstance(namedict, other_nd)
        self.__dict__.update(other_nd.__dict__)

# ================
# Control Protocol
# ================

INVALID_CONTROL_FRAME = FrameExceptionFactory('CONTROL')

CONTROL_PROTOCOL = Enum(
    'INIT'      , # 0 - req
    'INFO'      , # 1 - req
    'STATUS'    , # 2 - req
    'SHUTDOWN'  , # 3 - req
    'KILL'      , # 4 - req

    'OK'        , # 5 - rep
    'DONE'      , # 6 - rep
    'EXCEPTION' , # 7 - rep
)

def CONTROL_FRAME(id, status):
    assert isinstance(basestring, id)
    assert isinstance(int, status)

    return msgpack.dumps(tuple([id, status]))

def CONTORL_UNFRAME(msg):
    assert isinstance(basestring, msg)

    try:
        id, status = msgpack.loads(msg)
        assert isinstance(basestring, id)
        assert isinstance(int, status)

        return id, status
    except TypeError:
        raise INVALID_CONTROL_FRAME(msg)
    except ValueError:
        raise INVALID_CONTROL_FRAME(msg)
    #except AssertionError:
        #raise INVALID_CONTROL_FRAME(msg)

# ==================
# Heartbeat Protocol
# ==================

# These encode the msgpack equivelant of 1 and 2. The heartbeat
# frame should only be 1 byte on the wire.

HEARTBEAT_PROTOCOL = namedict({
    'REQ' : b'\x01',
    'REP' : b'\x02',
})

# ==================
# Component State
# ==================

COMPONENT_STATE = Enum(
    'OK'        , # 0
    'DONE'      , # 1
    'EXCEPTION' , # 2
)

# ==================
# Datasource Protocol
# ==================

INVALID_DATASOURCE_FRAME = FrameExceptionFactory('ORDER')

def DATASOURCE_FRAME(ds_id, ds_type, payload):
    """
    wraps any datasource payload with id and type, so that unpacking may choose the write
    UNFRAME for the payload.
    ::ds_id:: an identifier that is unique to the datasource in the context of a component host (e.g. Simulator
    ::ds_type:: a string denoting the datasource type. Must be on of::
        TRADE
        (others to follow soon)
    ::payload:: a msgpack string carrying the payload for the frame
    """
    assert isinstance(ds_id, basestring)
    assert isinstance(ds_type, basestring)
    assert isinstance(payload, basestring) 
    return msgpack.dumps(tuple([ds_id, ds_type, payload]))

def DATASOURCE_UNFRAME(msg):
    """
    extracts payload, and calls correct UNFRAME method based on the datasource type passed along
    returns a dict containing at least::
        - source_id
        - type
    other properties are added based on the datasource type::
        - TRADE::
            - sid - int security identifier
            - price - float
            - volume - int 
            - dt - a datetime object
    """
    try:
        ds_id, ds_type, payload = msgpack.loads(msg)
        if(ds_type == "TRADE"):
            result = {'source_id' : ds_id, 'type' : ds_type}
            result.update(TRADE_UNFRAME(payload))
            return namedict(result)
        else:
            raise INVALID_DATASOURCE_FRAME(msg)
            
    except TypeError:
        raise INVALID_DATASOURCE_FRAME(msg)
    except ValueError:
        raise INVALID_DATASOURCE_FRAME(msg)
        
# ==================
# Feed Protocol
# ==================
INVALID_FEED_FRAME = FrameExceptionFactory('FEED')

def FEED_FRAME(event):
    """
    :event: a nameddict with at least::
        - source_id 
        - type
    """
    assert isinstance(event, namedict)
    source_id = event.source_id
    ds_type = event.type
    pack_date(event)
    del(event.__dict__['dt'])
    payload = event.__dict__
    return msgpack.dumps(payload)
    
def FEED_UNFRAME(msg):
    try:
        payload = msgpack.loads(msg)
        #TODO: anything we can do to assert more about the content of the dict?
        assert isinstance(payload, dict)
        rval = namedict(payload)
        unpack_date(rval)
        return namedict(rval)
    except TypeError:
        raise INVALID_TRADE_FRAME(msg)
    except ValueError:
        raise INVALID_TRADE_FRAME(msg)
        
# ==================
# Transform Protocol
# ==================
INVALID_TRANSFORM_FRAME = FrameExceptionFactory('TRANSFORM')

def TRANSFORM_FRAME(name, value):
    """
    :event: a nameddict with at least::
        - source_id 
        - type
    """
    assert isinstance(name, basestring)
    assert value != None
    return msgpack.dumps(tuple([name, value]))
    
def TRANSFORM_UNFRAME(msg):
    try:
        name, value = msgpack.loads(msg)
        #TODO: anything we can do to assert more about the content of the dict?
        assert isinstance(name, basestring)
        assert payload.has_key('value')
        return namedict({name : value})
    except TypeError:
        raise INVALID_TRANSFORM_FRAME(msg)
    except ValueError:
        raise INVALID_TRANSFORM_FRAME(msg)
        

# ==================
# Merge Protocol
# ==================
INVALID_MERGE_FRAME = FrameExceptionFactory('MERGE')

def MERGE_FRAME(event):
    """
    :event: a nameddict with at least::
        - source_id 
        - type
    """
    assert isinstance(event, namedict)
    source_id = event.source_id
    ds_type = event.type
    payload = event.__dict__
    return msgpack.dumps(payload)
    
def MERGE_UNFRAME(msg):
    try:
        payload = msgpack.loads(msg)
        #TODO: anything we can do to assert more about the content of the dict?
        assert isinstance(payload, dict)
        return namedict(payload)
    except TypeError:
        raise INVALID_TRADE_FRAME(msg)
    except ValueError:
        raise INVALID_TRADE_FRAME(msg)

    
# ==================
# Finance Protocol
# ==================

INVALID_ORDER_FRAME = FrameExceptionFactory('ORDER')
INVALID_TRADE_FRAME = FrameExceptionFactory('TRADE')

# ==================
# Trades
# ==================

def TRADE_FRAME(event):
    """:event: should be a namedict with::
            - ds_id     -- the datasource id sending this trade out
            - sid       -- the security id
            - price     -- float of the price printed for the trade
            - volume    -- int for shares in the trade
            
    """
    assert isinstance(event, namedict)
    assert isinstance(event.sid, int)
    assert isinstance(event.price, float)
    assert isinstance(event.volume, int)
    pack_date(event)
    payload = msgpack.dumps(tuple([event.sid, event.price, event.volume, event.epoch, event.micros]))
    return DATASOURCE_FRAME(ds_id, "TRADE", payload)
    
def TRADE_UNFRAME(msg):
    try:
        sid, price, volume, epoch, micros = msgpack.loads(msg)
        assert isinstance(sid, int)
        assert isinstance(price, float)
        assert isinstance(volume, int)
        assert isinstance(epoch, numbers.Integral)
        assert isinstance(micros, numbers.Integral)
        rval = namedict({'sid' : sid, 'price' : price, 'volume' : volume, 'dt' : dt, 'epoch' : epoch, 'micros' : micros})
        unpack_date(rval)
        return rval
    except TypeError:
        raise INVALID_TRADE_FRAME(msg)
    except ValueError:
        raise INVALID_TRADE_FRAME(msg)

# =========
# Orders
# =========

def ORDER_FRAME(sid, amount):
    assert isinstance(sid, int)
    assert isinstance(amount, int) #no partial shares...   
    return msgpack.dumps(tuple([sid, amount]))
    

def ORDER_UNFRAME(msg):
    try:
        sid, amount = msgpack.loads(msg)
        assert isinstance(sid, int)
        assert isinstance(amount, int)

        return sid, amount
    except TypeError:
        raise INVALID_ORDER_FRAME(msg)
    except ValueError:
        raise INVALID_ORDER_FRAME(msg)
        
# =================
# Date Helpers
# =================

def pack_date(event):    
    assert isinstance(event.dt, datetime.datetime)
    assert event.dt.tzinfo == pytz.utc #utc only please
    epoch = long(dt.strftime('%s'))
    event['epoch'] = epoch
    event['micros'] = event.dt.microsecond
    del(event.__dict__['dt'])
    return event

def unpack_date(payload):
    assert isinstance(payload['epoch'], numbers.Integral)
    assert isinstance(payload['micros'], numbers.Integral)
    dt = datetime.datetime.fromtimestamp(payload['epoch'])
    dt.replace(microsecond = payload['micros'], tzinfo = pytz.utc)
    del(payload['epoch'])
    del(payload['micros'])
    payload['dt'] = dt
    return payload