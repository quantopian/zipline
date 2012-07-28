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
from zipline.gens.feed import FeedGen, full, done, queue_is_full,queue_is_done,\
    pop_oldest
from zipline.gens.utils import hash_args, assert_datasource_protocol,\
    assert_trade_protocol, alternate
from zipline.gens.tradegens import date_gen, SpecificEquityTrades
from zipline.gens.composites import PreTransformLayer

import zipline.protocol as zp

class FeedHelperTestCase(TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_individual_queue_logic(self):
        queue = deque()
        # Empty queues are neither done nor full.
        assert not queue_is_full(queue)
        assert not queue_is_done(queue)
        
        queue.append(to_dt('foo'))
        assert queue_is_full(queue)
        assert not queue_is_done(queue)

        
        queue.appendleft(to_dt('DONE'))
        assert queue_is_full(queue)

        # Checking done when we have a message after done will trip an assert.
        self.assertRaises(AssertionError, queue_is_done, queue)

        queue.pop()
        assert queue_is_full(queue)
        assert queue_is_done(queue)
        
    def test_pop_logic(self):
        sources = {}
        ids = ['a', 'b', 'c']
        for id in ids:
            sources[id] = deque()
        
        assert not full(sources)
        assert not done(sources)

        # All sources must have a message to be full/done
        sources['a'].append(to_dt("datetime"))
        assert not full(sources)
        assert not done(sources)
        sources['a'].pop()

        for id in ids:
            sources[id].append(to_dt("datetime"))
        
        assert full(sources)
        assert not done(sources)

        for id in ids:
            sources[id].appendleft(to_dt("DONE"))
            
        # ["DONE", message] will trip an assert in queue_is_done.
        assert full(sources)
        self.assertRaises(AssertionError, done, sources)

        for id in ids:
            sources[id].pop()

        assert full(sources)
        assert done(sources)
            
class FeedGenTestCase(TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def run_FeedGen(self, events, expected, source_ids):
        """
        Take a list of events, their source_ids, and an expected sorting.
        Assert that FeedGen's output agrees with expected.
        """
        feed_gen = FeedGen(events, source_ids)
        l = list(feed_gen)
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
        
        self.run_FeedGen(event_gen, expected, source_ids)
    
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
        self.run_FeedGen(interleaved, expected, source_ids)

        # All of a, then all of b.

        sequential = chain(iter(events_a), iter(events_b))
        self.run_FeedGen(sequential, expected, source_ids)
    
    def test_full_feed_layer(self):
        filter = [1,2]
        
        source_a = SpecificEquityTrades(sids = [1,2,3,4],
                                        start = datetime(2012,6,6,0),
                                        delta = timedelta(minutes=1),
                                        filter = filter
        )
        id_a = "SpecificEquityTradesd175237b28d2f52df208c97cf4af896e"

        # Change the internal sid list to give us a different hash.
        source_b = SpecificEquityTrades(sids = [1,2,3,5],
                                        start = datetime(2012,6,6,0),
                                        delta = timedelta(minutes=1),
                                        filter = filter
        )
        
        id_b = 'SpecificEquityTrades2bf2c2d6d01d4dbfc0b2818438ea8151'

        # Change the internal sid list to give us a different hash.
        source_c = SpecificEquityTrades(sids = [1,2,3,6],
                                        start = datetime(2012,6,6,0),
                                        delta = timedelta(minutes=1),
                                        filter = filter
        )
        id_c = 'SpecificEquityTrades16f7437db2d14e5373ef20025f49a3fe'

        sources = (source_a, source_b, source_c)
        source_ids = [id_a, id_b, id_c]
        
        feed_out = PreTransformLayer(sources, source_ids)
        l = list(feed_out)
    
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
        
        
        
        
