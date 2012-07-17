import logbook
from collections import defaultdict, Counter

from zipline.components.aggregator import Aggregate, \
    AGGREGATE_STATES, AGGREGATE_TRANSITIONS
import zipline.protocol as zp

log = logbook.Logger('Feed')

# =========
# Component
# =========

class Feed(Aggregate):
    """
    Connects to N PULL sockets, publishing all messages received to a
    PUB socket. Published messages are guaranteed to be in chronological
    order based on message property dt. Expects to be instantiated in
    one execution context (thread, process, etc) and run in another.
    """

    states = list(AGGREGATE_STATES)
    transitions = AGGREGATE_TRANSITIONS
    initial_state = -1

    def init(self):
        self.sent_count             = 0
        self.received_count         = 0
        self.ds_finished_counter    = 0

        self.sources = defaultdict(list)

        # source_id -> integer count
        self.sent_counters          = Counter()
        self.recv_counters          = Counter()

        self.state = AGGREGATE_STATES.INIT

    @property
    def get_id(self):
        return "FEED"

    @property
    def draining(self):
        return self.state == AGGREGATE_STATES.DRAINING

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
        self.sources[event.source_id].append(event)
        self.recv_counters[event.source_id] += 1
        self.received_count += 1

    def next(self):
        """
        Get the next message in chronological order.
        """

        # TODO: this is redundant to the guard in aggregator.
        # is_full and draining defined in aggregator
        if not(self.is_full() or self.draining):
            return

        earliest_source = None
        earliest_event = None
        # iterate over the queues of source from all sources
        # (1 queue per datasource)

        for source in self.sources.itervalues():
            if len(source) == 0:
                continue

            head = source[0]

            if head.dt == None:
                #this is a filler event, discard
                source.pop(0)
                continue

            if (earliest_event == None) or (head.dt <= earliest_event.dt):
                earliest_event = head
                earliest_source = source

        if earliest_event != None:
            return earliest_source.pop(0)
        else:
            return False
