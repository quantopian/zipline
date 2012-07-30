"""
Generator version of Merge.
"""

from collections import deque

from zipline import ndict
from zipline.gens.utils import hash_args, \
    assert_merge_protocol


def merge(stream_in, tnfm_ids):
    """
    A generator that takes a generator and a list of source_ids. We
    maintain an internal queue for each id in source_ids. Once we
    have a message from every queue, we pop an event from each queue
    and merge them together into an event.  We raise an error if we
    do not receive the same number of events from all sources.
    """

    assert isinstance(tnfm_ids, list)

    # Set up an internal queue for each expected source.
    tnfms = {}
    for id in tnfm_ids:
        assert isinstance(id, basestring), "Bad source_id %s" % id
        tnfms[id] = deque()

    # Process incoming streams.
    for message in stream_in:
        assert isinstance(message, tuple), \
            "Bad message in merge: %s" %message
        assert len(message) == 2
        id, value = message
        assert id in tnfm_ids, \
            "Message from unexpected tnfm: %s, %s" % (id, tnfm_ids)
        assert isinstance(value, ndict), "Bad message in merge: %s" %message

        tnfms[id].append(value)

        # Only pop messages when we have a pending message from
        # all datasources. Stop if all sources have signalled done.

        while ready(tnfms) and not done(tnfms):
            message = merge_one(tnfms)
            assert_merge_protocol(tnfm_ids, message)
            yield message

    # We should have only a done message left in each queue.
    for queue in tnfms.itervalues():
        assert len(queue) == 1, "Bad queue in merge on exit: %s" % queue
        assert queue[0].dt == "DONE", \
            "Bad last message in merge on exit: %s" % queue

def merge_one(sources):
    output = ndict()
    for key, queue in sources.iteritems():
        new_xform = ndict({key: queue.popleft()})
        output.merge(new_xform)
    return output


#TODO: This is replicated in sort.  Probably should be one source file.
def ready(sources):
    """
    Feed is ready when every internal queue has at least one message. Note that
    this include DONE messages, so done(sources) is True only if ready(sources).
    """
    assert isinstance(sources, dict)
    return all( (queue_is_ready(source) for source in sources.itervalues()) )

def queue_is_ready(queue):
    assert isinstance(queue, deque)
    return len(queue) > 0

def done(sources):
    """Feed is done when all internal queues have only a "DONE" message."""
    assert isinstance(sources, dict)
    return all( (queue_is_done(source) for source in sources.itervalues()) )

def queue_is_done(queue):
    assert isinstance(queue, deque)
    if len(queue) == 0:
        return False
    if queue[0].dt == "DONE":
        assert len(queue) == 1, "Message after DONE in date_sort: %s" % queue
        return True
    else:
        return False
