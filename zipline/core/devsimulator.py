"""
Simulator hosts all the components necessary to execute a simulation.
See :py:method""
"""

import logbook

log = logbook.Logger('Dev Simulator')

DEPRECATION_WARNING = """
WARNING WARNING WARNING
THE DEVSIMULATOR IS DEPRECATED, IT WILL NOT BEHAVE LIKE ANY OTHER
SYSTEM USED IN TESTS OR IN PRODUCTION
"""


class AddressAllocator(object):
    """
    Produces a iterator of 10000 sockets to allocate as needed.
    Emulates the API of Qexec's socket allocator.
    """

    def __init__(self, ns):
        self.idx = 0
        self.sockets = [
            'tcp://127.0.0.1:%s' % (10000 + n)
            for n in xrange(ns)
        ]

    def lease(self, n):
        sockets = self.sockets[self.idx: self.idx + n]
        self.idx += n
        return sockets

    def reaquire(self, *conn):
        pass
