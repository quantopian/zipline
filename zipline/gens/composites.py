from itertools import tee, chain

from zipline.gens.utils import roundrobin, done_message
from zipline.gens.sort import date_sort
from zipline.gens.merge import merge
from zipline.gens.transform import StatefulTransform

def date_sorted_sources(*sources):
    """
    Takes an iterable of sources, generating namestrings and
    piping their output into date_sort.
    """

    for source in sources:
        assert iter(source), "Source %s not iterable" % source
        assert source.__class__.__dict__.has_key('get_hash'), "No get_hash"

    # Get name hashes to pass to date_sort.
    names = [source.get_hash() for source in sources]

    # Convert the list of generators into a flat stream by pulling
    # one element at a time from each.
    stream_in = roundrobin(sources, names)

    # Guarantee the flat stream will be sorted by date, using
    # source_id as tie-breaker, which is fully deterministic (given
    # deterministic string representation for all args/kwargs)

    return date_sort(stream_in, names)

def merged_transforms(sorted_stream, *transforms):
    """
    A generator that takes the expected output of a date_sort, pipes
    it through a given set of transforms, and runs the results
    through a merge to output a unified stream. tnfms should be a
    list of pointers to generator functions. tnfm_args should be a
    list of tuples, representing the arguments to be passed to each
    transform.  tnfm_kwargs should be a list of dictionaries
    representing keyword arguments to each transform.
    """
    for transform in transforms:
        assert isinstance(transform, StatefulTransform)
        transform.merged = True
        transform.sequential = False
        
    # Generate expected hashes for each transform
    namestrings = [tnfm.get_hash() for tnfm in transforms]

    # Create a copy of the stream for each transform.
    split = tee(sorted_stream, len(transforms))

    # Package a stream copy with each StatefulTransform instance.
    bundles = zip(transforms, split)

    # Convert the copies into transform streams.
    tnfm_gens = [tnfm.transform(stream) for tnfm, stream in bundles]

    # Roundrobin the outputs of our transforms to create a single flat
    # stream.
    to_merge = roundrobin(tnfm_gens, namestrings)

    # Pipe the stream into merge.
    merged = merge(to_merge, namestrings)

    dt_aliased = alias_dt(merged)
    # Return the merged events.
    return add_done(dt_aliased)

def sequential_transforms(stream_in, *transforms):
    """
    Apply each transform in transforms sequentially to each event in stream_in.
    Each transform application will add a new entry indexed to the transform's
    hash string.
    """

    assert isinstance(transforms, (list, tuple))

    for tnfm in transforms:
        tnfm.sequential = True
        tnfm.merged = False
    
    # Recursively apply all transforms to the stream.
    stream_out = reduce(lambda stream, tnfm: tnfm.transform(stream),
                        transforms,
                        stream_in)

    dt_aliased = alias_dt(stream_out)
    return add_done(dt_aliased)

def alias_dt(stream_in):
    """
    Alias the dt field to datetime on each message.
    """
    for message in stream_in:
        message['datetime'] = message['dt']
        yield message

# Add a done message to a stream.
def add_done(stream_in):
    return chain(stream_in, [done_message('Composite')])
