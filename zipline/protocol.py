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
    So that you can use::

        foo.BAR
        -- or --
        foo['BAR']

    For more complex strcuts use collections.namedtuple:
    """

    def __init__(self, dct):
        self.__dict__.update(dct)

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
