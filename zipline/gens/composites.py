import datetime
from itertools import tee

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.utils import roundrobin, hash_args
from zipline.gens.feed import FeedGen
from zipline.gens.merge import MergeGen
from zipline.gens.transform import StatefulTransformGen

def PreTransformLayer(sources, source_args, source_kwargs):
    """
    Takes a list of generator functions, a list of tuples of positional arguments,
    and a list of dictionaries of keyword arguments.  Packages up all arguments
    and passes them into a FeedGen.
    """
    assert len(sources) == len(source_args) == len(source_kwargs)
    # Package up sources and arguments.
    arg_bundles = zip(sources, source_args, source_kwargs)
    
    # Calculate namestring hashes to pass to FeedGen.
    namestrings = [source.__name__ + hash_args(*args, **kwargs)
                   for source, args, kwargs in arg_bundles]
    # Pass each source its arguments.
    initialized = tuple(source(*args, **kwargs)
                        for source, args, kwargs in arg_bundles)
    
    stream_in = roundrobin(*initialized)
    return FeedGen(stream_in, namestrings)


def TransformLayer(feed_stream, tnfms, tnfm_args, tnfm_kwargs):
    """ 
    A generator that takes the expected output of a FeedGen, pipes it
    through a given set of transforms, and runs the results throught a
    MergeGen to output a unified stream. tnfms should be a list of
    pointers to generator functions. tnfm_args should be a list of
    tuples, representing the arguments to be passed to each transform.
    tnfm_kwargs should be a list of dictionaries representing keyword
    arguments to each transform.
    """

    # We should have as many sets of args as we have transforms.
    assert len(tnfms) == len(tnfm_args) == len(tnfm_kwargs)

    # Create a copy of the stream for each transform.
    split = tee(feed_stream, len(tnfms))
    
    # Package each stream copy with a transform and set of args.  Use a list
    # so that we can re-use this for calculating hashes.
    bundles = zip(split, tnfms, tnfm_args, tnfm_kwargs)
 
    tnfm_gens = [StatefulTransformGen(stream, tnfm, *args, **kwargs)
                 for stream, tnfm, args, kwargs in bundles]
    
    # Generate expected hashes for each transform
    hashes = [tnfm.__name__ + hash_args(*args, **kwargs) 
              for _, tnfm, args, kwargs in bundles]
              
    # Roundrobin the outputs of our transforms to create a single flat stream.
    to_merge = roundrobin(*tnfm_gens)
    
    # Pipe the stream into MergeGen.
    merged = MergeGen(to_merge, hashes)
    return merged

if __name__ == "__main__":
    from zipline.gens.transform import MovingAverage, Passthrough
    
    import nose.tools; nose.tools.set_trace()
    source = SpecificEquityTrades
    feed_out = PreTransformLayer((source,), ((),), ({},))

    transforms = [MovingAverage, Passthrough]
    args = [(datetime.timedelta(days = 1), ['price']), ()]
    kwargs = [{}, {}]
    
    tlayer = TransformLayer(feed_out, transforms, args, kwargs)
    
    
    
