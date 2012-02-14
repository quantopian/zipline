"""
Transforms
==========

Transforms provide re-useable components for stream processing. All
Transforms expect to receive data events from zipline.core.DataFeed
asynchronously via zeromq. Each transform is designed to run in independent 
process space, independently of all other transforms, to allow for parallel
computation. 

Each transform must maintain the state necessary to calculate the transform of 
each new feed events. 

To simplify the consumption of feed and transform data events, this module
also provides the TransformsMerge class. TransformsMerge initializes as set of 
transforms and subscribes to their output. Each feed event is then combined with
all the transforms of that event into a single new message.

"""