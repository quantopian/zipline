"""
Generator version of Feed.
"""

import pytz
import logbook
import pymongo
import types

from pymongo import ASCENDING
from datetime import datetime, timedelta
from collections import deque, defaultdict

from zipline import ndict
from zipline.gens.utils import hash_args, assert_datasource_protocol, \
    assert_trade_protocol, assert_datasource_unframe_protocol, \
    assert_feed_protocol

import zipline.protocol as zp

def FeedGen(stream_in, source_ids):
    """
    A generator that takes a generator and a list of source_ids.  We
    maintain an internal queue for each id in source_ids. While we
    have messages pending from all sources, we pull the earliest
    message and yield it.  
    """
    
    assert isinstance(source_ids, list)

    # Set up an internal queue for each expected source.
    sources = {}
    for id in source_ids:
        assert isinstance(id, basestring), "Bad source_id %s" % source_id
        sources[id] = deque()

    # Process incoming streams.
    for message in stream_in:
        # Incoming messages should be the output of DATASOURCE_UNFRAME.
        assert_datasource_unframe_protocol(message), \
            "Bad message in FeedGen: %s" % message

        # Only allow messages from sources we expect.
        assert message.source_id in sources, "Unexpected source: %s" % message

        sources[message.source_id].append(message)
        
        # Only pop messages when we have a pending message from
        # all datasources. Stop if all sources have signalled done.

        while full(sources) and not done(sources):
            message = pop_oldest(sources)
            assert_feed_protocol(message)
            yield message
    
    import nose.tools; nose.tools.set_trace()
    # We should have only a done message left in each queue.    
    for queue in sources.itervalues():
        assert len(queue) == 1, "Bad queue in FeedGen on exit: %s" % queue
        assert queue[0].dt == "DONE", \
            "Bad last message in FeedGen on exit: %s" % queue

def full(sources):
    """
    Feed is full when every internal queue has at least one message. Note that
    this include DONE messages, so done(sources) is True only if full(sources).
    """
    assert isinstance(sources, dict)
    return all( (queue_is_full(source) for source in sources.itervalues()) )

def queue_is_full(queue):
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
        assert len(queue) == 1, "Message after DONE in FeedGen: %s" % queue
        return True
    else:
        return False

def pop_oldest(sources):
    
    oldest_event = None
        
    # Iterate over the dict, checking internal queues for the oldest
    # pending event.

    for queue in sources.itervalues():
        current_event = queue[0]
        # Skip queues that are done.
        if current_event.dt == "DONE":
            continue
        # Any event is older than nothing.
        elif oldest_event == None:
            oldest_event = current_event
        # Keep the older event.  Break ties by source_id. This will
        # trip an assert if we have duplicate sources.
        else:
            oldest_event = older(oldest_event, current_event)
            
    # Pop the oldest event we found from its queue and return it.
    return sources[oldest_event.source_id].popleft()

# Return the event with the older timestamp.  Break ties by source_id.
def older(oldest, current):
    assert isinstance(oldest, ndict)
    assert isinstance(oldest, ndict)
    
    # Try to compare by dt.
    if oldest.dt < current.dt:
        return oldest
    elif oldest.dt > current.dt:
        return current
    # Break ties by source_id.
    elif oldest.source_id < current.source_id:
        return oldest
    elif oldest.source_id > current.source_id:
        return current
    else:
        assert False, "Duplicate event"
