import copy
import pandas
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
            return "Invalid {framecls} Frame: {got}".format(
                framecls = name,
                got = self.got,
            )

    return InvalidFrame

class namedict(object):
    """

    Namedicts are dict like objects that have fields accessible by attribute lookup
    as well as being indexable and iterable::

        HEARTBEAT_PROTOCOL = namedict({
            'REQ' : b'\x01',
            'REP' : b'\x02',
        })

        HEARTBEAT_PROTOCOL.REQ # syntactic sugar
        HEARTBEAT_PROTOCOL.REP # oh suga suga

    For more complex structs use collections.namedtuple:
    """

    def __init__(self, dct=None):
        if(dct):
            self.__dict__.update(dct)

    def __setitem__(self, key, value):
        """
        Required for use by pymongo as_class parameter to find.
        """
        if(key == '_id'):
            self.__dict__['id'] = value
        else:
            self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def as_dict(self):
        # shallow copy is O(n)
        return copy.copy(self.__dict__)

    def delete(self, key):
        del(self.__dict__[key])

    def merge(self, other_nd):
        assert isinstance(other_nd, namedict)
        self.__dict__.update(other_nd.__dict__)

    def __repr__(self):
        return "namedict: " + str(self.__dict__)

    def __eq__(self, other):
        # !!!!!!!!!!!!!!!!!!!!
        # !!!! DANGEROUS !!!!!
        # !!!!!!!!!!!!!!!!!!!!
        return other != None and self.__dict__ == other.__dict__

    def has_attr(self, name):
        return self.__dict__.has_key(name)
        
    def as_series(self):
        s = pandas.Series(self.__dict__)
        s.name = self.sid
        return s
