import datetime
from itertools import tee, starmap
from collections import namedtuple

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.utils import roundrobin, hash_args
from zipline.gens.sort import date_sort
from zipline.gens.merge import merge
from zipline.gens.transform import stateful_transform

SourceBundle = namedtuple("SourceBundle", ['source', 'args', 'kwargs'])
TransformBundle = namedtuple("TransformBundle", ['tnfm', 'args', 'kwargs'])

def date_sorted_sources(bundles):
    """
    Takes an iterable of SortBundles, generating namestrings and initialized datasources
    for each before piping them into a date_sort.
    """
    assert isinstance(bundles, (list, tuple))
    for bundle in bundles:
        assert isinstance(bundle, SourceBundle)

    # Calculate namestring hashes to pass to date_sort.
    names = [bundle.source.__name__ + hash_args(*bundle.args, **bundle.kwargs)
             for bundle in bundles]

    # Pass each source its arguments.
    source_gens = [bundle.source(*bundle.args, **bundle.kwargs)
                   for bundle in bundles]
    
    # Convert the list of generators into a flat stream by pulling
    # one element at a time from each.
    stream_in = roundrobin(source_gens, names)
    
    # Guarantee the flat stream will be sorted by date, using source_id as
    # tie-breaker, which is fully deterministic (given deterministic string 
    # representation for all args/kwargs)
    return date_sort(stream_in, names)


def merged_transforms(sorted_stream, bundles):
    """
    A generator that takes the expected output of a date_sort, pipes it
    through a given set of transforms, and runs the results throught a
    merge to output a unified stream. tnfms should be a list of
    pointers to generator functions. tnfm_args should be a list of
    tuples, representing the arguments to be passed to each transform.
    tnfm_kwargs should be a list of dictionaries representing keyword
    arguments to each transform.
    """
    # Generate expected hashes for each transform
    namestrings = [bundle.tnfm.__name__ + hash_args(*bundle.args, **bundle.kwargs)
                   for bundle in bundles]

    # Create a copy of the stream for each transform.
    split = tee(sorted_stream, len(bundles))
    # Package a stream copy with each bundle 
    tnfms_with_streams = zip(split, bundles)

    # Convert the copies into transform streams.
    tnfm_gens = [
        stateful_transform(
            stream_copy, 
            bundle.tnfm, 
            *bundle.args, 
            **bundle.kwargs
        )
        for stream_copy, bundle in tnfms_with_streams
    ]

    # Roundrobin the outputs of our transforms to create a single flat stream.
    to_merge = roundrobin(tnfm_gens, namestrings)

    # Pipe the stream into merge.
    merged = merge(to_merge, namestrings)
    # Return the merged events.
    return merged
