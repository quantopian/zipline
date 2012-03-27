"""
Misc ZeroMQ utilities.
"""
import gevent
import msgpack
from gevent_zeromq import zmq

from contextlib import closing

class ZmqDone(Exception):

    def __init__(self, socket, frame):
        self.ident = socket.identity
        self.frame = str(frame)

    def __str__(self):
        return 'Socket ( %s ) finished with frame ( %s )' % \
            ( self.ident, self.frame )

class zs(object):
    """
    A wrapper for the *very* common pattern of reading from a
    upstream socket until you get a DONE or EXCEPTION frame.

        # Eliminates all the boilerplate serialization logic
        # and error handling cases into 3 lines.

        halts = (ERROR_FRAME, CLOSE_FRAME)
        stream = zs(socket, halts)

        stream.on_error(YouFailAtFailing)

        for msg in stream:
            print msg

    """

    def __init__(self, socket, halts, srl=msgpack):
        self._socket = socket
        self.exc_case = halts[0]
        self.done_case = halts[1]

        self.loads = srl.loads
        self.halt_method = 'exception'
        self.exception = ZmqDone
        self.function = None

    def __iter__(self):
        self.last = msg = self.loads(self._socket.recv())

        if msg == self.exc_case:
            return self.halt()

        if msg == self.done_case:
            raise StopIteration

        yield msg

    def last(self):
        return self.last

    def halt(self):
        if self.halt_method == 'exception':
            raise self.exception
        elif self.halt_method == 'function':
            return self.function()

    def on_error(self, callee):

        if isinstance(callee, Exception):
            self.halt_method = 'exception'
            self.exception = callee
        else:
            self.halt_method = 'function'
            self.function = callee

def ZmqConsole(sock_typ, socket_addr, sock_conn=None, context=None):
    """
    A utility to drop into a ZeroMQ pdb console and inspect
    messages as they come through. If you just want to pipe to
    stdout, don't use this.
    """

    context = context or zmq.Context.instance()
    socket = context.socket(zmq.PULL)
    socket.bind(socket_addr)

    def console():
        while True:
            msg = socket.recv_pyobj()
            print msg
            import pdb; pdb.set_trace()

    return gevent.spawn(console)
