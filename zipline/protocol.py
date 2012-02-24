#import msgpack
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

HEARTBEAT_PROTOCOL = {
    'REQ' : '\x01',
    'REP' : '\x02',
}
