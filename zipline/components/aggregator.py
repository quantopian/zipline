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
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE, \
    CONTROL_FRAME, CONTROL_UNFRAME
from zipline.utils.protocol_utils import Enum
from zipline.transitions import WorkflowMeta

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

class Aggregate(Component):
    """
    Abstract superclass to Merge & Feed. Acts on two sockets

        - pull_socket
        - feed_socket

    Both use ``data_buffer`` for buffering.

    Feed and Merge define these differently.
    """

    abstract = True
    __metaclass__ = WorkflowMeta

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    # -------------
    # Core Methods
    # -------------

    def do_work(self):
        # wait for synchronization reply from the host
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        # TODO: Abstract this out, maybe on base component
        if socks.get(self.control_in) == self.zmq.POLLIN:
            msg = self.control_in.recv()
            event, payload = CONTROL_UNFRAME(msg)

            # -- Heartbeat --
            if event == CONTROL_PROTOCOL.HEARTBEAT:
                # Heart outgoing
                heartbeat_frame = CONTROL_FRAME(
                    CONTROL_PROTOCOL.OK,
                    payload
                )
                self.control_out.send(heartbeat_frame)

            # -- Soft Kill --
            elif event == CONTROL_PROTOCOL.SHUTDOWN:
                self.signal_done()
                self.shutdown()

            # -- Hard Kill --
            elif event == CONTROL_PROTOCOL.KILL:
                self.kill()


        if socks.get(self.pull_socket) == self.zmq.POLLIN:
            message = self.pull_socket.recv()

            if message == str(CONTROL_PROTOCOL.DONE):
                self.ds_finished_counter += 1

                if len(self.data_buffer) == self.ds_finished_counter:
                    #drain any remaining messages in the buffer
                    log.debug("Draining Feed")
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
                    self.send_next()
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
        self.state = DRAINING
        while self.pending_messages() > 0:
            self.send_next()

    def send_next(self):
        """
        Send the (chronologically) next message in the buffer.
        """
        if not (self.is_full() or self.draining):
            return

        event = self.next()

        if event:
            self.feed_socket.send(self.frame(event), self.zmq.NOBLOCK)
            self.sent_counters[event.source_id] += 1
            self.sent_count += 1

    def is_full(self):
        """
        Indicates whether the buffer has messages in buffer for all
        un-DONE, blocking sources.
        """
        for source_id, events in self.data_buffer.iteritems():
            if len(events) == 0:
                return False
        return True

    def pending_messages(self):
        """
        Returns the count of all events from all sources in the
        buffer.
        """
        total = 0
        for events in self.data_buffer.itervalues():
            total += len(events)
        return total

    def add_source(self, source_id):
        """
        Add a data source to the buffer.
        """
        self.data_buffer[source_id] = []

    def __len__(self):
        """
        Buffer's length is same as internal map holding separate
        sorted arrays of events keyed by source id.
        """
        return len(self.data_buffer)
