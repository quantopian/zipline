from feed import Feed

import zipline.protocol as zp
from zipline.protocol import COMPONENT_TYPE

from collections import Counter

class Merge(Feed):
    """
    Merges multiple streams of events into single messages.
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
        return "MERGE"

    @property
    def get_type(self):
        return COMPONENT_TYPE.CONDUIT

    def open(self):
        self.pull_socket = self.bind_merge()
        self.feed_socket = self.bind_result()

    def next(self):
        """Get the next merged message from the feed buffer."""
        if not (self.is_full() or self.draining):
            return

        if self.pending_messages() == 0:
            return

        #get the raw event from the passthrough transform.
        result = self.data_buffer[zp.TRANSFORM_TYPE.PASSTHROUGH].pop(0).PASSTHROUGH
        for source, events in self.data_buffer.iteritems():
            if source == zp.TRANSFORM_TYPE.PASSTHROUGH:
                continue
            if len(events) > 0:
                cur = events.pop(0)
                result.merge(cur)
        return result

    def unframe(self, msg):
        return zp.TRANSFORM_UNFRAME(msg)

    def frame(self, event):
        return zp.MERGE_FRAME(event)

    def append(self, event):
        """
        :param event: a ndict with one entry. key is the name of the 
        transform, value is the transformed value.
        Add an event to the buffer for the source specified by
        source_id.
        """

        self.data_buffer[event.keys()[0]].append(event)
        self.received_count += 1

