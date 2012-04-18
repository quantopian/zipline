import copy
import pandas
from ctypes import Structure, c_ubyte
from collections import MutableMapping
from itertools import izip

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

class namedict(MutableMapping):
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

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return self.__dict__.iterkeys()

    def __len__(self):
        return len(self.__dict__)

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

class ndict(MutableMapping):
    """
    Xtreme Namedicts 2.0

    Ndicts are dict like objects that have fields accessible by attribute
    lookup as well as being indexable and iterable. Done right
    this time.
    """

    def __init__(self, dct=None):
        self.__internal = dict()
        self.cls = frozenset(dir(self))

        if dct:
            self.__internal.update(dct)

    # Abstact Overloads
    # -----------------

    def __setitem__(self, key, value):
        """
        Required for use by pymongo as_class parameter to find.
        """
        if key == '_id':
            self.__internal['id'] = value
        else:
            self.__internal[key] = value


    def __getattr__(self, key):
        if key in self.cls:
            return self.__dict__[key]
        else:
            return self.__internal[key]

    def __getitem__(self, key):
        return self.__internal[key]

    def __delitem__(self, key):
        del self.__internal[key]

    def __iter__(self):
        return self.__internal.iterkeys()

    def __len__(self, key):
        return len(self.__internal)

    # Compatability with namedicts
    # ----------------------------

    # for compat, not the Python way to do things though...
    # Deprecated, use builtin ``del`` operator.
    delete = __delitem__

    def has_attr(self, key):
        """
        Deprecated, use builtin ``in`` operator.
        """
        return self.__contains__(key)

    # Custom Methods
    # --------------

    def as_dataframe(self):
        """
        Return the representation as a Pandas dataframe.
        """
        d = pandas.DataFrame(self.__internal)
        return d

    def as_series(self):
        """
        Return the representation as a Pandas time series.
        """
        s = pandas.Series(self.__internal)
        s.name = self.sid
        return s

    def as_dict(self):
        """
        Return the representation as a vanilla Python dict.
        """
        # shallow copy is O(n)
        return copy.copy(self.__internal)

    def merge(self, other_nd):
        """
        Merge in place with another ndict.
        """
        assert isinstance(other_nd, ndict)
        self.__internal.update(other_nd.__internal)

    def __repr__(self):
        return "namedict: " + str(self.__internal)

    # Faster dictionary comparison?
    #def __eq__(self, other):
        #assert isinstance(other, ndict)

        #keyeq = set(self.keys()) == set(other.keys())

        #if not keyeq:
            #return False

        #for i, j in izip(self.itervalues(), other.itervalues()):
            #if i != j:
                #return False

        #return True
