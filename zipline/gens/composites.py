import datetime
from itertools import tee, starmap
from collections import namedtuple

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.utils import roundrobin, hash_args
from zipline.gens.sort import date_sort
from zipline.gens.merge import merge
from zipline.gens.transform import stateful_transform

SortBundle = namedtuple("SortBundle", ['source', 'args', 'kwargs'])
MergeBundle = namedtuple("MergeBundle", ['stream', 'tnfm', 'args', 'kwargs'])

def date_sorted_sources(sources, source_args, source_kwargs):
    """
    Takes a list of generator functions, a list of tuples of positional arguments,
    and a list of dictionaries of keyword arguments.  Packages up all arguments
    and passes them into a date_sort.
    """
    assert len(sources) == len(source_args) == len(source_kwargs)
    # Package up sources and arguments.
    
    # Create a generator of SortBundle objects to be turned into
    # namestrings and generator objects.
    bundle_gen = starmap(SortBundle, zip(sources, source_args, source_kwargs))

    # Load the results of the generator into a tuple so that the
    # results can be used twice (once in namestring comprehension,
    # once in the generator comprehension for intialized sources.
    bundles = tuple(bundle_gen)

    # Calculate namestring hashes to pass to date_sort.
    names = [bundle.source.__name__ + hash_args(*bundle.args, **bundle.kwargs)
             for bundle in bundles]
    # Pass each source its arguments.
    initialized = [bundle.source(*bundle.args, **bundle.kwargs)
                        for bundle in bundles]

    # Convert the list of generators into a flat stream by pulling
    # one element at a time from each.
    stream_in = roundrobin(*initialized)
    
    # Guarantee the flat stream will be sorted by date, using source_id as
    # tie-breaker, which is fully deterministic (given deterministic string 
    # representation for all args/kwargs)
    return date_sort(stream_in, names)


def merged_transforms(sorted_stream, tnfms, tnfm_args, tnfm_kwargs):
    """
    A generator that takes the expected output of a date_sort, pipes it
    through a given set of transforms, and runs the results throught a
    merge to output a unified stream. tnfms should be a list of
    pointers to generator functions. tnfm_args should be a list of
    tuples, representing the arguments to be passed to each transform.
    tnfm_kwargs should be a list of dictionaries representing keyword
    arguments to each transform.
    """

    # We should have as many sets of args as we have transforms.
    assert len(tnfms) == len(tnfm_args) == len(tnfm_kwargs)

    # Create a copy of the stream for each transform.
    split = tee(sorted_stream, len(tnfms))

    # Package each transform with a stream copy and set of args.  Use a list
    # so that we can re-use this for calculating hashes.
    bundle_gen = starmap(MergeBundle, zip(split, tnfms, tnfm_args, tnfm_kwargs))

    bundles = tuple(bundle_gen)
    # list comprehension to create transform generators from
    # bundles
    tnfm_gens = [
        stateful_transform(
            bundle.stream, 
            bundle.tnfm, 
            *bundle.args, 
            **bundle.kwargs
        )
        for bundle in bundles]

    # Generate expected hashes for each transform
    hashes = [bundle.tnfm.__name__ + hash_args(*bundle.args, **bundle.kwargs)
              for bundle in bundles]

    # Roundrobin the outputs of our transforms to create a single flat stream.
    to_merge = roundrobin(*tnfm_gens)

    # Pipe the stream into merge.
    merged = merge(to_merge, hashes)
    return merged_transforms
