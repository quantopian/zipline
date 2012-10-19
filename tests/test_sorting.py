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

import pytz

from unittest2 import TestCase
from itertools import chain, izip_longest
from datetime import datetime, timedelta
from collections import deque

from zipline import ndict
from zipline.gens.sort import (
    date_sort,
    ready,
    done,
    queue_is_ready,
    queue_is_done
)
from zipline.gens.utils import alternate, done_message
from zipline.sources import SpecificEquityTrades
from zipline.gens.composites import date_sorted_sources


class HelperTestCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_individual_queue_logic(self):
        queue = deque()
        # Empty queues are neither done nor ready.
        assert not queue_is_ready(queue)
        assert not queue_is_done(queue)

        queue.append(to_dt('foo'))
        assert queue_is_ready(queue)
        assert not queue_is_done(queue)

        queue.appendleft(to_dt('DONE'))
        assert queue_is_ready(queue)

        # Checking done when we have a message after done will trip an assert.
        self.assertRaises(AssertionError, queue_is_done, queue)

        queue.pop()
        assert queue_is_ready(queue)
        assert queue_is_done(queue)

    def test_pop_logic(self):
        sources = {}
        ids = ['a', 'b', 'c']
        for id in ids:
            sources[id] = deque()

        assert not ready(sources)
        assert not done(sources)

        # All sources must have a message to be ready/done
        sources['a'].append(to_dt("datetime"))
        assert not ready(sources)
        assert not done(sources)
        sources['a'].pop()

        for id in ids:
            sources[id].append(to_dt("datetime"))

        assert ready(sources)
        assert not done(sources)

        for id in ids:
            sources[id].appendleft(to_dt("DONE"))

        # ["DONE", message] will trip an assert in queue_is_done.
        assert ready(sources)
        self.assertRaises(AssertionError, done, sources)

        for id in ids:
            sources[id].pop()

        assert ready(sources)
        assert done(sources)


class DateSortTestCase(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def run_date_sort(self, event_stream, expected, source_ids):
        """
        Take a list of events, their source_ids, and an expected sorting.
        Assert that date_sort's output agrees with expected.
        """
        sort_out = date_sort(event_stream, source_ids)
        for m1, m2 in izip_longest(sort_out, expected):
            assert m1 == m2

    def test_single_source(self):

        # Just using the built-in defaults.  See
        # zipline.sources.py
        source = SpecificEquityTrades()
        expected = list(source)
        source.rewind()
        # The raw source doesn't handle done messaging, so we need to
        # append a done message for sort to work properly.
        with_done = chain(source, [done_message(source.get_hash())])
        self.run_date_sort(with_done, expected, [source.get_hash()])

    def test_multi_source(self):

        filter = [2, 3]
        args_a = tuple()
        kwargs_a = {
            'count': 100,
            'sids': [1, 2, 3],
            'start': datetime(2012, 1, 3, 15, tzinfo=pytz.utc),
            'delta': timedelta(minutes=6),
            'filter': filter
        }
        source_a = SpecificEquityTrades(*args_a, **kwargs_a)

        args_b = tuple()
        kwargs_b = {
            'count': 100,
            'sids': [2, 3, 4],
            'start': datetime(2012, 1, 3, 15, tzinfo=pytz.utc),
            'delta': timedelta(minutes=5),
            'filter': filter
        }
        source_b = SpecificEquityTrades(*args_b, **kwargs_b)

        all_events = list(chain(source_a, source_b))

        # The expected output is all events, sorted by dt with
        # source_id as a tiebreaker.
        expected = sorted(all_events, comp)
        source_ids = [source_a.get_hash(), source_b.get_hash()]

        # Generating the events list consumes the sources. Rewind them
        # for testing.
        source_a.rewind()
        source_b.rewind()

        # Append a done message to each source.
        with_done_a = chain(source_a, [done_message(source_a.get_hash())])
        with_done_b = chain(source_b, [done_message(source_b.get_hash())])

        interleaved = alternate(with_done_a, with_done_b)

        # Test sort with alternating messages from source_a and
        # source_b.
        self.run_date_sort(interleaved, expected, source_ids)

        source_a.rewind()
        source_b.rewind()
        with_done_a = chain(source_a, [done_message(source_a.get_hash())])
        with_done_b = chain(source_b, [done_message(source_b.get_hash())])

        sequential = chain(with_done_a, with_done_b)

        # Test sort with all messages from a, followed by all messages
        # from b.

        self.run_date_sort(sequential, expected, source_ids)

    def test_sort_composite(self):

        filter = [1, 2]

        #Set up source a. One hour between events.
        args_a = tuple()
        kwargs_a = {
            'count': 100,
            'sids': [1],
            'start': datetime(2012, 6, 6, 0),
            'delta': timedelta(hours=1),
            'filter': filter
        }
        source_a = SpecificEquityTrades(*args_a, **kwargs_a)

        #Set up source b. One day between events.
        args_b = tuple()
        kwargs_b = {
            'count': 50,
            'sids': [2],
            'start': datetime(2012, 6, 6, 0),
            'delta': timedelta(days=1),
            'filter': filter
        }
        source_b = SpecificEquityTrades(*args_b, **kwargs_b)

        #Set up source c. One minute between events.
        args_c = tuple()
        kwargs_c = {
            'count': 150,
            'sids': [1, 2],
            'start': datetime(2012, 6, 6, 0),
            'delta': timedelta(minutes=1),
            'filter': filter
        }
        source_c = SpecificEquityTrades(*args_c, **kwargs_c)
        # Set up source d. This should produce no events because the
        # internal sids don't match the filter.
        args_d = tuple()
        kwargs_d = {
            'count': 50,
            'sids': [3],
            'start': datetime(2012, 6, 6, 0),
            'delta': timedelta(minutes=1),
            'filter': filter
        }
        source_d = SpecificEquityTrades(*args_d, **kwargs_d)
        sources = [source_a, source_b, source_c, source_d]
        hashes = [source.get_hash() for source in sources]

        sort_out = date_sorted_sources(*sources)

        # Read all the values from sort and assert that they arrive in
        # the correct sorting with the expected hash values.
        to_list = list(sort_out)
        copy = to_list[:]

        # We should have 300 events (100 from a, 150 from b, 50 from c)
        assert len(to_list) == 300

        for e in to_list:
            # All events should match one of our expected source_ids.
            assert e.source_id in hashes
            # But none of them should match source_d.
            assert e.source_id != source_d.get_hash()

        # The events should be sorted by dt, with source_id as tiebreaker.
        expected = sorted(copy, comp)

        assert to_list == expected


def compare_by_dt_source_id(x, y):
    if x.dt < y.dt:
        return -1
    elif x.dt > y.dt:
        return 1

    elif x.source_id < y.source_id:
        return -1
    elif x.source_id > y.source_id:
        return 1
    else:
        return 0

#Alias for ease of use
comp = compare_by_dt_source_id


def to_dt(msg):
    return ndict({'dt': msg})
