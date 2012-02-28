import uuid
import copy
import atexit
import pickle

from datetime import datetime
from collections import defaultdict

from UserDict import DictMixin

class Snapshot(object, DictMixin):
    """
    A snapshot in time of a history container.
    """

    def __init__(self, state, version, ts):
        self.version = version
        self.timestamp = ts
        self._state = state

    def keys(self):
        return self._state.keys()

    def values(self):
        return self._state.values()

    def items(self):
        return self._state.items()

    def __getitem__(self, key):
        return self._state.__getitem__(key)

    def has_key(self, key):
        return self._state.has_key(key)

    def copy(self):
        return copy.copy(self._state)

class History(object, DictMixin):
    """
    A duck-typed dictionary that tracks its time evolution.

    Worth noting this not a particuarly high-performance
    data structure due to the copious amount of copying going on.
    """

    def __init__(self, default=None):
        if default:
            initial = defaultdict(default)
        else:
            initial = {}

        self.version = 0
        self.changeset = [('CREATE', None)]
        self.current = Snapshot(initial, version=self.version, ts=datetime.now())
        self._history = [self.current]

    def items(self, version=-1):
        return self._history[version].items()

    def keys(self, version=-1):
        return self._history[version].keys()

    def rollback(self, version):
        pass

    def event(self, tup):
        self.changeset.append(tup)

    def __getitem__(self, key, version=-1):
        return self._history[version].__getitem__(key)

    def __setitem__(self, key, val):
        if self.current.has_key(key):
            self.changeset.append(('CHANGE', key))
        else:
            self.changeset.append(('ADD', key))

        state = self.current.copy()
        state[key] = val

        self.version += 1
        self.current = Snapshot(state, self.version, datetime.now())
        self._history.append(self.current)

    def __delitem__(self, key):
        self.changeset.append(('REMOVE', key))

        state = self.current.copy()
        del state[key]

        self.version += 1
        self.current = Snapshot(state, self.version, datetime.now())
        self._history.append(self.current)

    def history(self):
        for change in self.changeset:
            print change

    def __repr__(self):
        return ':'.join(['historical', self.current._state.__repr__()])

SocketHistory = History()
ContextHistory = History()

def patch_zmq(_zmq=None):
    """
    Monkey patch zeromq to allow for socket tracking.
    """
    if _zmq:
        zmq = _zmq
    else:
        import zmq

    _Context = zmq.Context
    _Socket = zmq.Socket

    class TrackedSocket(zmq.Socket):

        def __init__(self, context, socket_type):
            self.context = context
            self.uuid = str(uuid.uuid4())
            SocketHistory[self.uuid] = self
            _Socket.__init__(self, context, socket_type)

        def connect(self, address):
            SocketHistory.event(('CONNECT', self.uuid, address))
            _Socket.connect(self, address)

        def bind(self, address):
            SocketHistory.event(('BIND', self.uuid, address))
            _Socket.bind(self, address)

        def close(self, *args, **kwargs):
            del SocketHistory[self.uuid]
            _Socket.close(self, *args, **kwargs)

        def setsockopt(self, option, optval):
            if option == zmq.IDENTITY:
                old = SocketHistory[self.uuid]
                SocketHistory[optval] = old
                del SocketHistory[self.uuid]
                self.uuid = optval

            _Socket.setsockopt(self, option, optval)

    class TrackedContext(zmq.Context):

        def __init__(self, *args, **kwargs):
            self.sockets = {}
            _Context.__init__(self, *args, **kwargs)
            self.uuid = str(uuid.uuid4())
            ContextHistory[self.uuid] = self

        def socket(self, socket_type):
            sock = TrackedSocket(self, socket_type)
            ContextHistory.event(('EMBED', self.uuid, sock.uuid))
            self.sockets[sock.uuid] = sock
            return sock

        def name(self, name):
            """
            Name the context. Is a superset of the vanilla pyzmq
            API.
            """
            old = ContextHistory[self.context.uuid]
            ContextHistory[name] = old
            del ContextHistory[self.context.uuid]
            self.uuid = name

        def term(self, *args, **kwargs):
            for uid, sock in self.sockets.iteritems():
                if not sock.closed:
                    del SocketHistory[sock.uuid]
            del ContextHistory[self.uuid]
            _Context.term(self, *args, **kwargs)

        def destroy(self, *args, **kwargs):
            ContextHistory.event(('DESTROY', self.uuid))
            _Context.destroy(self, *args, **kwargs)

    zmq.Context = TrackedContext
    zmq.Socket = TrackedSocket
    return TrackedContext, TrackedSocket

def track_to_file(f):
    def write_track():
        pickle.dump(SocketHistory.changeset, file(f, 'wb+'))
    atexit.register(write_track)
