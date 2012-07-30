import os

import uuid
import msgpack
import pytz

from unittest2 import TestCase
from pymongo import Connection, ASCENDING
from itertools import izip, izip_longest, permutations, cycle, chain
from datetime import datetime, timedelta
from collections import deque

from zipline import ndict
from zipline.gens.sort import date_sort, ready, done, queue_is_ready,queue_is_done,\
    pop_oldest
from zipline.gens.utils import hash_args, assert_datasource_protocol,\
    assert_trade_protocol, alternate
from zipline.gens.tradegens import date_gen, SpecificEquityTrades
from zipline.gens.composites import date_sorted_sources

import zipline.protocol as zp

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

    def run_date_sort(self, events, expected, source_ids):
        """
        Take a list of events, their source_ids, and an expected sorting.
        Assert that date_sort's output agrees with expected.
        """
        sort_gen = date_sort(events, source_ids)
        l = list(sort_gen)
        assert l == expected
        
    def test_single_source(self):
        source_ids = ['a']
        # 100 events, increasing by a minute at a time.
        type = zp.DATASOURCE_TYPE.TRADE
        dates = list(date_gen(count = 100))
        dates.append("DONE")
        
        # [('a', date1, type), ('a', date2, type), ... ('a', "DONE", type)]
        event_args = zip(cycle(source_ids), iter(dates), cycle([type]))
        
        # Turn event_args into proper events.
        events = [mock_data_unframe(*args) for args in event_args]
        
        # We don't expected Feed to yield the last event.
        expected = events[:-1]

        event_gen = (e for e in events)
        
        self.run_date_sort(event_gen, expected, source_ids)
    
    def test_multi_source(self):
        source_ids = ['a', 'b']
        type = zp.DATASOURCE_TYPE.TRADE

        # Set up source 'a'. Outputs 20 events with 2 minute deltas.
        delta_a = timedelta(minutes = 2)
        dates_a = list(date_gen(delta = delta_a, count = 20))
        dates_a.append("DONE")

        events_a_args = zip(cycle(['a']), iter(dates_a), cycle([type]))
        events_a = [mock_data_unframe(*args) for args in events_a_args]        
        
        # Set up source 'b'. Outputs 10 events with 1 minute deltas.
        delta_b = timedelta(minutes = 1)
        dates_b = list(date_gen(delta = delta_b, count = 10))
        dates_b.append("DONE")

        events_b_args = zip(cycle(['b']), iter(dates_b), cycle([type]))
        events_b = [mock_data_unframe(*args) for args in events_b_args]
        
        # The expected output is all non-DONE events in both a and b,
        # sorted first by dt and then by source_id.
        non_dones = events_a[:-1] + events_b[:-1]
        expected = sorted(non_dones, compare_by_dt_source_id)

        # Alternating between a and b.
        interleaved = alternate(iter(events_a), iter(events_b))
        self.run_date_sort(interleaved, expected, source_ids)

        # All of a, then all of b.

        sequential = chain(iter(events_a), iter(events_b))
        self.run_date_sort(sequential, expected, source_ids)

    def test_sorted_sources(self):
        
        filter = [1,2]
        #Set up source a. One hour between events.
        args_a = tuple()
        kwargs_a = {'sids'   : [1,2,3,4],
                    'start'  : datetime(2012,6,6,0),
                    'delta'  : timedelta(hours = 1),
                    'filter' : filter
        }
        #Set up source b. One day between events.       
        args_b = tuple()
        kwargs_b = {'sids'   : [1,2,3,4],
                    'start'  : datetime(2012,6,6,0),
                    'delta'  : timedelta(days = 1),
                    'filter' : filter
        }
        #Set up source c. One minute between events.
        args_c = tuple()
        kwargs_c = {'sids'   : [1,2,3,4],
                    'start'  : datetime(2012,6,6,0),
                    'delta'  : timedelta(minutes = 1),
                    'filter' : filter
        }
        # Set up source d. This should produce no events because the
        # internal sids don't match the filter.
        args_d = tuple()
        kwargs_d = {'sids'   : [3,4],
                    'start'  : datetime(2012,6,6,0),
                    'delta'  : timedelta(minutes = 1),
                    'filter' : filter
        }
        
        sources = (SpecificEquityTrades,) * 4
        source_args = (args_a, args_b, args_c, args_d)
        source_kwargs = (kwargs_a, kwargs_b, kwargs_c, kwargs_d)
        
        # Generate our expected source_ids.
        zip_args = zip(source_args, source_kwargs)
        expected_ids = ["SpecificEquityTrades" + hash_args(*args, **kwargs)
                        for args, kwargs in zip_args]
        
        # Pipe our sources into sort.
        sort_out = date_sorted_sources(sources, source_args, source_kwargs)
        
        # Read all the values from sort and assert that they arrive in
        # the correct sorting with the expected hash values.
        to_list = list(sort_out)
        copy = to_list[:]
        for e in to_list:
            # All events should match one of our expected source_ids.
            assert e.source_id in expected_ids
            # But none of them should match source_d.
            assert e.source_id != hash_args(*args_d, **kwargs_d)

        expected = sorted(copy, compare_by_dt_source_id)
        assert to_list == expected
    
def mock_data_unframe(source_id, dt, type):
    event = ndict()
    event.source_id = source_id
    event.dt = dt
    event.type = type
    return event

def to_dt(val):
    return ndict({'dt': val})

def compare_by_dt_source_id(x,y):
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
