import logging
from collections import Counter

from zipline.core.component import Component
import zipline.protocol as zp

from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE, \
    CONTROL_FRAME, CONTROL_UNFRAME

LOGGER = logging.getLogger('ZiplineLogger')

class Feed(Component):
    """
    Connects to N PULL sockets, publishing all messages received to a PUB
    socket.  Published messages are guaranteed to be in chronological order
    based on message property dt.  Expects to be instantiated in one execution
    context (thread, process, etc) and run in another.
    """

    def init(self):
        self.sent_count             = 0
        self.received_count         = 0
        self.draining               = False
        self.ds_finished_counter    = 0

        # Depending on the size of this, might want to use a data
        # structure with better asymptotics.
        self.data_buffer            = {}

        # source_id -> integer count
        self.sent_counters          = Counter()
        self.recv_counters          = Counter()

    @property
    def get_id(self):
        return "FEED"

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    # -------------
    # Core Methods
    # -------------

    def open(self):
        self.pull_socket = self.bind_data()
        self.feed_socket = self.bind_feed()

    def do_work(self):
        # wait for synchronization reply from the host
        socks = dict(self.poll.poll(self.heartbeat_timeout)) 

        # TODO: Abstract this out, maybe on base component
        if self.control_in in socks and socks[self.control_in] == self.zmq.POLLIN:
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


        if self.pull_socket in socks and socks[self.pull_socket] == self.zmq.POLLIN:
            message = self.pull_socket.recv()

            if message == str(CONTROL_PROTOCOL.DONE):
                self.ds_finished_counter += 1

                if len(self.data_buffer) == self.ds_finished_counter:
                    #drain any remaining messages in the buffer
                    LOGGER.debug("draining feed")
                    self.drain()
                    self.signal_done()
            else:
                try:
                    event = self.unframe(message)
                # deserialization error
                except zp.INVALID_DATASOURCE_FRAME as exc:
                    return self.signal_exception(exc)

                try:
                    self.append(event)
                    self.send_next()

                # Invalid message
                except zp.INVALID_DATASOURCE_FRAME as exc:
                    return self.signal_exception(exc)

    def unframe(self, msg):
        return zp.DATASOURCE_UNFRAME(msg)

    def frame(self, event):
        return zp.FEED_FRAME(event)

    # -------------
    # Flow Control
    # -------------

    def drain(self):
        """
        Send all messages in the buffer.
        """
        self.draining = True
        while self.pending_messages() > 0:
            self.send_next()

    def send_next(self):
        """
        Send the (chronologically) next message in the buffer.
        """
        if not (self.is_full() or self.draining):
            return

        event = self.next()
        if(event != None):
            self.feed_socket.send(self.frame(event), self.zmq.NOBLOCK)
            self.sent_counters[event.source_id] += 1
            self.sent_count += 1

    def append(self, event):
        """
        Add an event to the buffer for the source specified by
        source_id.
        """
        self.data_buffer[event.source_id].append(event)
        self.recv_counters[event.source_id] += 1
        self.received_count += 1

    def next(self):
        """
        Get the next message in chronological order.
        """
        if not(self.is_full() or self.draining):
            return

        cur_source = None
        earliest_source = None
        earliest_event = None
        #iterate over the queues of events from all sources 
        #(1 queue per datasource)
        for events in self.data_buffer.values():
            if len(events) == 0:
                continue
            cur_source = events
            first_in_list = events[0]
            if first_in_list.dt == None:
                #this is a filler event, discard
                events.pop(0)
                continue

            if (earliest_event == None) or (first_in_list.dt <= earliest_event.dt):
                earliest_event = first_in_list
                earliest_source = cur_source

        if earliest_event != None:
            return earliest_source.pop(0)

    def is_full(self):
        """
        Indicates whether the buffer has messages in buffer for
        all un-DONE, blocking sources.
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
        for events in self.data_buffer.values():
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
