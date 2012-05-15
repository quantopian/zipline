"""
This is somewhat legally ambigious, since it technically
hasn't been merged in gevent_zeromq but given that the
author issued it as a Pull Request on a MIT project,
indicates that its probably fine to use. ~Steve
"""

import zmq
from zmq import *

from zmq.core.poll import Poller as _original_Poller

import gevent
from gevent import select
from gevent_zeromq.core import _Socket

def patch_poller(self):
    zmq.Poller = _Poller

class _Poller(_original_Poller):
    """
    Replacement for :class:`zmq.core.Poller`

    Ensures that the greened Poller below is used in calls
    to :meth:`zmq.core.Poller.poll`.
    """

    def _get_descriptors(self):
        """
        Returns three elements tuple with socket descriptors ready for
        gevent.select
        """
        rlist = []
        wlist = []
        xlist = []

        for socket, flags in self.sockets.items():
            if isinstance(socket, _Socket):
                fd = socket.getsockopt(FD)
            elif isinstance(socket, int):
                fd = socket
            elif hasattr(socket, 'fileno'):
                try:
                    fd = int(socket.fileno())
                except:
                    raise ValueError('fileno() must return an valid integer fd')
            else:
                raise TypeError("Socket must be a 0MQ socket, an integer fd or \
                    have a fileno() method: %r" % socket)

            if flags & POLLIN: rlist.append(fd)
            if flags & POLLOUT: wlist.append(fd)
            if flags & POLLERR: xlist.append(fd)

        return (rlist, wlist, xlist)

    def poll(self, timeout=-1):
        """Overridden method to ensure that the green version of Poller is used

        Behaves the same as :meth:`zmq.core.Poller.poll`
        """

        if timeout is None:
            timeout = -1

        timeout = int(timeout)
        if timeout < 0:
            timeout = -1

        rlist = None
        wlist = None
        xlist = None

        if timeout > 0:
            tout = gevent.Timeout.start_new(timeout/1000.0)

        try:
            # Loop until timeout or events available
            while True:
                events = super(_Poller, self).poll(0)
                if events or timeout == 0:
                    return events

                # wait for activity on sockets in a green way
                if not rlist and not wlist and not xlist:
                    rlist, wlist, xlist = self._get_descriptors()

                try:
                    select.select(rlist, wlist, xlist)
                except gevent.select.error, ex:
                    raise ZMQError(*ex.args)

        except gevent.Timeout, t:
            if t is not tout:
                raise
            return []
        finally:
           if timeout > 0:
               tout.cancel()
