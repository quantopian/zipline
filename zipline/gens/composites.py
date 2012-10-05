#
# Copyright 2012 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain

from zipline.gens.utils import roundrobin, done_message
from zipline.gens.sort import date_sort


def date_sorted_sources(*sources):
    """
    Takes an iterable of sources, generating namestrings and
    piping their output into date_sort.
    """

    for source in sources:
        assert iter(source), "Source %s not iterable" % source
        assert hasattr(source, 'get_hash'), "No get_hash"

    # Get name hashes to pass to date_sort.
    names = [source.get_hash() for source in sources]

    # Convert the list of generators into a flat stream by pulling
    # one element at a time from each.
    stream_in = roundrobin(sources, names)

    # Guarantee the flat stream will be sorted by date, using
    # source_id as tie-breaker, which is fully deterministic (given
    # deterministic string representation for all args/kwargs)

    return date_sort(stream_in, names)


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
