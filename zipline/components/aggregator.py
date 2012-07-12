"""
Abstract base class for Feed and Merge.

   Component
       |
   Aggregate
       |
      / \
  Feed   Merge

"""
import logbook

import zipline.protocol as zp
from zipline.core.component import Component
from zipline.core.controlled import do_handle_control_events
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE
from zipline.transitions import WorkflowMeta
from zipline.utils.protocol_utils import Enum


log = logbook.Logger('Aggregate')

# =================
# State Transitions
# =================

INIT, READY, DRAINING = AGGREGATE_STATES = \
Enum( 'INIT', 'READY', 'DRAINING')

AGGREGATE_TRANSITIONS = dict(
    do_start = (-1    , INIT)     ,
    do_run   = (INIT  , READY)    ,
    do_drain = (READY , DRAINING) ,
)

# =========
# Component
# =========

class Aggregate(Component):
    """
    Abstract superclass to Merge & Feed. Acts on two sockets

        - pull_socket
        - feed_socket

    Both use ``sources`` for buffering.

    Feed and Merge define these differently.
    """

    abstract = True
    __metaclass__ = WorkflowMeta

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    def add_source(self, source_id):
        self.sources[source_id] = []

    # -------------
    # Core Methods
    # -------------

    def do_work(self):
        # wait for synchronization reply from the host
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        # ----------------
        # Control Dispatch
        # ----------------
        do_handle_control_events(self, socks)

        # -------------
        # Work Dispatch
        # -------------
        if socks.get(self.pull_socket) == self.zmq.POLLIN:
            message = self.pull_socket.recv()

            if message == str(CONTROL_PROTOCOL.DONE):
                self.ds_finished_counter += 1

                if len(self.sources) == self.ds_finished_counter:
                    # Drain any remaining messages in the buffer
                    log.debug("Draining Feed")

                    self.state = DRAINING

                    self.drain()
                    self.signal_done()
            else:
                try:
                    event = self.unframe(message)
                except zp.INVALID_DATASOURCE_FRAME as exc:
                    # Error deserializing
                    return self.signal_exception(exc)

                try:
                    self.append(event)

                    if not (self.is_full() or self.draining):
                        event = self.next()

                        if event:
                            self.send(event)
                        else:
                            pass

                except zp.INVALID_DATASOURCE_FRAME as exc:
                    # Invalid message
                    return self.signal_exception(exc)

    # -------------
    # Flow Control
    # -------------

    def drain(self):
        """
        Send all messages in the buffer.
        """
        while self.pending_messages() > 0:
            event = self.next()
            if event:
                self.send(event)

    def send(self, event):
        """
        Send the (chronologically) next message in the buffer.
        """
        self.feed_socket.send(self.frame(event), self.zmq.NOBLOCK)
        self.sent_counters[event.source_id] += 1
        self.sent_count += 1

    def is_full(self):
        """
        Indicates whether the buffer has messages in buffer for all
        un-DONE, blocking sources.
        """
        for source_id, events in self.sources.iteritems():
            if len(events) == 0:
                return False
        return True

    def pending_messages(self):
        """
        Returns the count of all events from all sources in the
        buffer.
        """
        total = 0
        for events in self.sources.itervalues():
            total += len(events)
        return total

    def __len__(self):
        """
        Buffer's length is same as internal map holding separate
        sorted arrays of events keyed by source id.
        """
        return len(self.sources)
