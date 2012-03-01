"""
Contains the various deployable topologies of ziplines.

This is mostly hardcoded at the moment but as the topologies
becomes more sophisiticated this logic will be the primary
router of sockets.

Ontology of Stream Processing
=============================

Source
******

A producer of data. The data could be in a datastore, coming from a
socket, etc. To access this data, we pull from the source. Sources increase the
total amount of data flowing through the system.  Sources are generally not
pure since they involve IO.

Sink
****

A consumer of data. Basic examples would be a sum function (adding up a
stream of numbers fed in), a datastore sink, a socket etc. We push data
into a sink. When / If a sink completes processing, it may return some
value that exists outside of the system.

Sinks decrease the total amount of information flowing through the system.

Conduit
*******

A transformer of data. We push data into a conduit.  Similar to a sink,
but instead of returning a single value at the end, a conduit can
return multiple outputs every time it is pushed to. The returned values
remain in the system.

Conduits may or may not be pure, it is usefull to distinguish between the
two since pure conduits have a variety of nice properties under composition

"""

from zipline.protocol import  COMPONENT_TYPE

class Topology(object):
    pass

class DiamondTopology(Topology):
    """
    Exposes a feed, merge, and passthrough bypass::

                                 +--------+
                     +---------->|        |---------------+
                     |           +--------+               |
                     |                                    v
    +---+----+   +---+----+      +--------+           +--------+    +---+----+
    |        +-->|        +----->|        |---------->|        |--->|        |
    +---+----+   +---+----+      +--------+           +--------+    +---+----+
                     |                                    ^
                     |           +--------+               |
                     +---------->|        |---------------+
                     |           +--------+               |
                     |                                    |
                     +------------passthru----------------+

    """

    flow = {
        'flow'        : COMPONENT_TYPE.SOURCE  ,
        'serializers' : COMPONENT_TYPE.CONDUIT ,
        'transforms'  : COMPONENT_TYPE.CONDUIT ,
        'merges'      : COMPONENT_TYPE.CONDUIT ,
        'clients'     : COMPONENT_TYPE.SINK    ,
    }

    def __init__(self):
        self.sources     = []
        self.serializers = []
        self.transforms  = []
        self.merges      = []
        self.clients     = []
