import os

import uuid
import msgpack
import pytz

from unittest2 import TestCase
from pymongo import Connection, ASCENDING
from itertools import izip, izip_longest
from datetime import datetime, timedelta

from zipline.gens.mongods import create_pymongo_iterator, MongoTradeHistoryGen
from zipline.gens.utils import stringify_args, assert_datasource_protocol,\
    assert_trade_protocol, mock_raw_event

import zipline.protocol as zp

mongo_conn_args = {
    'mongodb_host':  'localhost',
    'mongodb_port':  27017,
}

class TempMongo(object):

    def __enter__(self):
        self.conn = Connection(mongo_conn_args['mongodb_host'],
                                mongo_conn_args['mongodb_port'])

        temp_id = 'qexec_test_id'

        self.db = self.conn[temp_id]

        return self

    def __exit__(self, type, value, traceback):
        self.conn.drop_database(self.db.name)

class TestMongoDataGenerator(TestCase):
    
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_create_pymongo_iterator(self):
        
        with TempMongo() as temp_mongo:
            db = temp_mongo.db
            coll = db.test
            coll.ensure_index([('dt', ASCENDING), ('sid', ASCENDING)])
            
            for i in xrange(100):
                # sid = 1, dt ranging from 0 to 99
                coll.insert(mock_raw_event(1, i))
            
            start_date = 20
            end_date = 50
            filter = {'sid' : [1]}
            args = (coll, filter, start_date, end_date)
            
            cursor = create_pymongo_iterator(*args)
            # We filter to only get dt's between 20 and 50
            expected = (mock_raw_event(1, i) for i in xrange(20, 51))
            
            # Assert that our iterator returns the expected values.
            for cursor_event, expected_event in izip_longest(cursor, expected):
                del cursor_event['_id']
                # Easiest way to convert unicode to strings.
                cursor_event = msgpack.loads(msgpack.dumps(cursor_event))
                assert expected_event.keys() == cursor_event.keys()
                assert expected_event.values() == cursor_event.values()
    
    def test_MongoTradeHistoryGen(self):
        
        with TempMongo() as temp_mongo:
            db = temp_mongo.db
            coll = db.test
            coll.ensure_index([('dt', ASCENDING), ('sid', ASCENDING)])
            
            start_date = datetime(year = 2012,month=6,day=5,hour=0)
            delta = timedelta(hours = 1)

            for i in xrange(100):
                # sid = 1, dt's increasing an hour at a time from start
                time = start_date + i * delta
                coll.insert(mock_raw_event(1, time))

            # Halfway through the events we added to db.
            end_date = start_date + delta * 50

            filter = {'sid' : [1]}
            args = (coll, filter, start_date, end_date)
            db_gen = MongoTradeHistoryGen(*args)
            
            expected_times = (start_date + i * delta for i in xrange(51))
            expected_events = (mock_raw_event(1, t) for t in expected_times)
            
            # DB events should match the expected events for price, dt, volume,
            # and sid. They should also conform to the trade frame protocol.

            for db, expected in izip_longest(db_gen, expected_events):
                expected['dt'] = expected['dt'].replace(tzinfo = pytz.utc)
                # Check that our output meets the trade protocol.
                assert_trade_protocol(db)

                # Check that our output matches expectations
                for field in iter(['sid', 'dt', 'price', 'volume']):
                    assert db[field] == expected[field]
                
                # Expected output of stringify_args:
                assert db['source_id'] == \
                    'MongoTradeHistoryGen983a27fd0710414239a5cde71ef5a8fc'
                
            
