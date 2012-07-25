"""
Generator-style DataSource that loads from MongoDB.
"""

import pytz
import logbook

from pymongo import ASCENDING

from zipline import ndict
from zipline.gens.utils import stringify_args

import zipline.protocol as zp

def create_pymongo_iterator(self, collection, filter, start_date, end_date):
    """
    See the comments on :py:class:`zipline.messaging.DataSource` for
    expected content of self.filter. Spec must adhere to that definition.
    Returns an iterator that spits out raw objects loaded from MongoDB.
    """
    log = logbook.Logger("MongoDBQuery")

    # Object that will hold our database query.
    spec = {}

    # add the filters from the algorithm.
    for name, value in filter.iteritems():

        # Add the list of sids that we care about.
        if name == 'sid':
            assert isinstance(value, sid)
            sid_range = {'sid':{'$in':value}}
            spec.update(sid_range)
            
    # limit the data to the date range [start, end], inclusive
    date_range = {'dt':{'$gte':self.start, '$lte':self.end}}
    spec.update(date_range)

    fields = ['sid','price','volume','dt']

    # In our collection, load all objects matching spec.  Of those
    # objects, get only the fields matching fields, and return the
    # loaded objects sorted by dt from least to greatest.

    cursor = self.collection.find(
        fields   = fields,
        spec     = spec,
        sort     = [("dt",ASCENDING)],
        slave_ok = True
    )

    # Optimize the cursor sort to query in 'dt' and 'sid' order.
    cursor = cursor.hint([('dt', ASCENDING),('sid', ASCENDING)])

    # Set up the iterator
    iterator = iter(self.cursor)
    log.info("MongoDataSource iterator ready")
    return iterator


def MongoTradeHistoryGen(collection, filter, start_date, end_date):
    
    iterator = create_pymongo_iterator(collection, filter, start_date, end_date)
    source_id = "MongoTradeHistoryGen" + stringify_args(collection, filter, start_date, end_date)

    for event in iterator:
        
    
            
