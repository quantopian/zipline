from feed import Feed

import zipline.protocol as zp
from zipline.protocol import COMPONENT_TYPE

# TODO: By Liskov merge must *be* a feed, don't believe this is
# the case.

class Merge(Feed):
    """
    Merges multiple streams of events into single messages.
    """

    def __init__(self):
        Feed.__init__(self)

        self.init()

    def init(self):
        pass

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

