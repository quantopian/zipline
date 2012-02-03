import datetime
import zmq
import pymongo
import pymongo.json_util
import json
import pytz
import copy
import multiprocessing
import logging
import random
from pymongo import ASCENDING, DESCENDING

from backtest.util import *

from qbt_server import * #connect_db

class DataSource(object):
    def __init__(self, feed, source_id):
        self.source_id              = source_id
        self.logger                 = logging.getLogger()
        self.feed                   = feed
        self.sync                   = FeedSync(self.feed, str(source_id))
        self.data_address           = self.feed.data_address
        self.logger.info("data address is {ds}".format(ds=self.feed.data_address))        
        self.cur_event = None

    def start(self):
        self.proc = multiprocessing.Process(target=self.run)
        self.proc.start()
        
        
    def open(self):    
        self.logger.info("starting data source:{source_id} on {addr}".format(source_id=self.source_id, addr=self.feed.data_address))
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_socket.connect(self.data_address)
        
        #signal we are ready
        self.sync.confirm()
    
    def run(self):    
        try:    
            self.open()
            self.send_all()
            self.close()    
        except Exception as err:
            self.logger.error(err, "Unexpected failure running datasource - {name}.".format(name=name))
            
    def send(self, event):
        event['s'] = self.source_id             
        event['type'] = 'event'
        self.data_socket.send(json.dumps(event))
    
    def close(self):
        done_msg = {}
        done_msg['type'] = 'DONE'
        done_msg['s'] = self.source_id
        self.data_socket.send(json.dumps(done_msg))   
        self.data_socket.close()
        self.context.term()
        self.logger.info("finished processing data source")

class EquityMinuteTrades(DataSource):
    
    def __init__(self, sid, feed, source_id):
        self.sid = sid
        self.connection, self.db    = connect_db()
        DataSource.__init__(self, feed, source_id)
               
     
    def send_all(self):   
        eventQS = self.db.equity.trades.minute.find(fields=["sid","price","volume","dt"],
                                     spec={"sid":self.sid},
                                     sort=[("dt",ASCENDING)],
                                     slave_ok=True)
        self.logger.info("found {count} events".format(count=eventQS.count()))
        
        for doc in eventQS:
            doc_dt = doc['dt'].replace(tzinfo = pytz.utc)
            doc_dt_str = format_date(doc_dt)
            event = copy.copy(doc)
            event['dt'] = doc_dt_str
            del(event['_id'])
            self.send(event)
            
        
        
class RandomEquityTrades(DataSource):
    
    def __init__(self, sid, feed, source_id, count):
        DataSource.__init__(self, feed, source_id)
        self.count = count
        self.sid = sid
        
    def send_all(self):
        trade_start = datetime.datetime.now()
        minute = datetime.timedelta(minutes=1)
        price = random.uniform(5.0,50.0)
        
        for i in range(self.count):
            price = price + random.uniform(-0.05,0.05)
            event = {'sid':self.sid, 'dt':format_date(trade_start + (minute * i)),'price':price, 'volume':random.randrange(100,10000,100)}
            self.send(event)
          
   
       
        
        
