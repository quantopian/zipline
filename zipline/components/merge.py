import zipline.protocol as zp
from zipline.components.aggregator import Aggregate, \
    AGGREGATE_STATES, AGGREGATE_TRANSITIONS

from collections import defaultdict, Counter

class Merge(Aggregate):
    """
    Merges multiple streams of events into single messages.
    """

    states = list(AGGREGATE_STATES)
    transitions = AGGREGATE_TRANSITIONS
    initial_state = -1

    def init(self):
        self.sent_count             = 0
        self.received_count         = 0
        self.draining               = False
        self.ds_finished_counter    = 0

        self.sources = defaultdict(list)

        # source_id -> integer count
        self.sent_counters          = Counter()
        self.recv_counters          = Counter()

    @property
    def get_id(self):
        return "MERGE"

    # -------
    # Sockets
    # -------

    def open(self):
        self.pull_socket = self.bind_merge()
        self.feed_socket = self.bind_result()

    # -------
    # Framing
    # -------

    def unframe(self, msg):
        return zp.TRANSFORM_UNFRAME(msg)

    def frame(self, event):
        return zp.MERGE_FRAME(event)

    # ---------
    # Data Flow
    # ---------

    def append(self, event):
        """
        :param event: a ndict with one entry. key is the name of the
        transform, value is the transformed value.
        Add an event to the buffer for the source specified by
        source_id.
        """

        self.sources[event.keys()[0]].append(event)
        self.received_count += 1

    def next(self):
        """Get the next merged message from the feed buffer."""
        if not (self.is_full() or self.draining):
            return

        if self.pending_messages() == 0:
            return

        #get the raw event from the passthrough transform.
        passthrough = self.sources[zp.TRANSFORM_TYPE.PASSTHROUGH]
        result = passthrough.pop(0).PASSTHROUGH

        for source, events in self.sources.iteritems():
            if source == zp.TRANSFORM_TYPE.PASSTHROUGH:
                continue
            if len(events) > 0:
                cur = events.pop(0)
                result.merge(cur)
        return result
