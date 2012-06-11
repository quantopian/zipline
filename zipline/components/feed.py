import logging
from collections import Counter

from zipline.core.component import Component
from zipline.components.aggregator import Aggregate
from zipline.utils.protocol_utils import Enum
import zipline.protocol as zp
from zipline.transitions import WorkflowMeta

from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE, \
    CONTROL_FRAME, CONTROL_UNFRAME

LOGGER = logging.getLogger('ZiplineLogger')

# FSM
# ---

INIT, READY, DRAINING = FEED_STATES = \
Enum( 'INIT', 'READY', 'DRAINING')

state_transitions = dict(
    do_start = (-1    , INIT)     ,
    do_run   = (INIT  , READY)    ,
    do_drain = (READY , DRAINING) ,
)

class Feed(Aggregate):
    """
    Connects to N PULL sockets, publishing all messages received to a
    PUB socket. Published messages are guaranteed to be in chronological
    order based on message property dt. Expects to be instantiated in
    one execution context (thread, process, etc) and run in another.
    """

    __metaclass__ = WorkflowMeta

    states = list(FEED_STATES)
    transitions = state_transitions
    initial_state = -1

    def init(self):
        self.sent_count             = 0
        self.received_count         = 0
        self.draining               = False
        self.ds_finished_counter    = 0

        self.data_buffer            = {}

        # source_id -> integer count
        self.sent_counters          = Counter()
        self.recv_counters          = Counter()

        self.state = INIT

    @property
    def get_id(self):
        return "FEED"

    # -------
    # Sockets
    # -------

    def open(self):
        self.pull_socket = self.bind_data()
        self.feed_socket = self.bind_feed()

    # -------
    # Framing
    # -------

    def unframe(self, msg):
        return zp.DATASOURCE_UNFRAME(msg)

    def frame(self, event):
        return zp.FEED_FRAME(event)

    # -------------
    # Flow Control
    # -------------

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

        # is_full and draining defined in aggregator
        if not(self.is_full() or self.draining):
            return

        cur_source = None
        earliest_source = None
        earliest_event = None
        #iterate over the queues of events from all sources
        #(1 queue per datasource)
        for events in self.data_buffer.itervalues():
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
