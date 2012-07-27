"""
Generator-style DataSource that loads from MongoDB.
"""

import pytz
import logbook
import pymongo

from pymongo import ASCENDING
from datetime import datetime, timedelta

from zipline import ndict
from zipline.gens.utils import hash_args, assert_datasource_protocol, \
    assert_trade_protocol

import zipline.protocol as zp

def MongoTradeHistoryGen(collection, filter, start_date, end_date):
    """A generator that takes a pymongo Collection object, a list of
    filters, a start date and an end_date and yields ndicts containing
    the results of a query to its collection with the given filter,
    start, and end.  The output is also packaged with a unique
    source_id string for downstream sorting
    """
    
    assert isinstance(collection, pymongo.collection.Collection)
    assert isinstance(filter, dict)
    assert isinstance(start_date, (datetime))
    assert isinstance(end_date, (datetime))
    
    # Set up internal iterator.  This outputs raw dictionaries.
    iterator = create_pymongo_iterator(collection, filter, start_date, end_date)

    # Create unique identifier string that can be used to break
    # sorting ties deterministically
    argstring = hash_args(collection, filter, start_date, end_date)
    source_id = "MongoTradeHistoryGen" + argstring

    # All datasources
    for event in iterator:
        # Construct a new event that fulfills the datasource protocol.
        event['type'] = zp.DATASOURCE_TYPE.TRADE
        event['dt'] = event['dt'].replace(tzinfo=pytz.utc)
        event['source_id'] = source_id

        payload = ndict(event)
        assert_trade_protocol(payload)
        yield payload
        
def create_pymongo_iterator(collection, filter, start_date, end_date):
    """
    Returns an iterator that spits out raw objects loaded from a
    MongoDB collection.  

    See the comments on :py:class:`zipline.messaging.DataSource` 
    for expected content of filter.
    """
    log = logbook.Logger("MongoDBQuery")

    # Object that will hold our database query.
    spec = {}

    # add the filters from the algorithm.
    for name, value in filter.iteritems():

        # Add the list of sids that we care about.
        if name == 'sid':
            assert isinstance(value, list)
            sid_range = {'sid':{'$in':value}}
            spec.update(sid_range)
            
    # limit the data to the date range [start, end], inclusive
    date_range = {'dt':{'$gte': start_date, '$lte': end_date}}
    spec.update(date_range)

    fields = ['sid','price','volume','dt']

    # In our collection, load all objects matching spec.  Of those
    # objects, get only the fields matching fields, and return the
    # loaded objects sorted by dt from least to greatest.
    
    cursor = collection.find(
        fields   = fields,
        spec     = spec,
        sort     = [("dt",ASCENDING)],
        slave_ok = True
    )

    # Optimize the cursor sort to query in 'dt' and 'sid' order.
    cursor = cursor.hint([('dt', ASCENDING),('sid', ASCENDING)])

    # Set up the iterator
    iterator = iter(cursor)
    log.info("MongoDataSource iterator ready")
    
    return iterator
        
    
        
        
    
            
