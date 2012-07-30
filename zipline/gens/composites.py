import datetime
from itertools import tee

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.utils import roundrobin, hash_args
from zipline.gens.sort import date_sort
from zipline.gens.merge import merge
from zipline.gens.transform import stateful_transform

def date_sorted_sources(sources, source_args, source_kwargs):
    """
    Takes a list of generator functions, a list of tuples of positional arguments,
    and a list of dictionaries of keyword arguments.  Packages up all arguments
    and passes them into a date_sort.
    """
    assert len(sources) == len(source_args) == len(source_kwargs)
    # Package up sources and arguments.
    arg_bundles = zip(sources, source_args, source_kwargs)

    # Calculate namestring hashes to pass to date_sort.
    namestrings = [source.__name__ + hash_args(*args, **kwargs)
                   for source, args, kwargs in arg_bundles]
    # Pass each source its arguments.
    initialized = tuple(source(*args, **kwargs)
                        for source, args, kwargs in arg_bundles)

    stream_in = roundrobin(*initialized)
    return date_sort(stream_in, namestrings)


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
    bundles = zip(split, tnfms, tnfm_args, tnfm_kwargs)

    # list comprehension to create transform generators from
    # bundles
    tnfm_gens = [stateful_transform(stream, tnfm, *args, **kwargs)
                 for stream, tnfm, args, kwargs in bundles]

    # Generate expected hashes for each transform
    hashes = [tnfm.__name__ + hash_args(*args, **kwargs)
              for _, tnfm, args, kwargs in bundles]

    # Roundrobin the outputs of our transforms to create a single flat stream.
    to_merge = roundrobin(*tnfm_gens)

    # Pipe the stream into merge.
    merged = merge(to_merge, hashes)
    return merged_transforms
