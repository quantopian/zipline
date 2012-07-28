from zipline.gens.utils import roundrobin
from zipline.gens.feed import FeedGen


def PreTransformLayer(sources, source_ids):
    """
    A generator that takes a tuple of sources and a list ids, piping
    their output into a feed_gen.
    """
    stream_in = roundrobin(*sources)
    return FeedGen(stream_in, source_ids)

def TransformLayer(feed_stream, tnfms):
    """ """
    pass

    
    
