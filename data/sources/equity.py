import datetime
import zmq
import pymongo
import pymongo.json_util
import json
import pytz
import copy
import multiprocessing
import logging
from pymongo import ASCENDING, DESCENDING

from backtest.util import *


class EquityMinuteTrades(object):
    
    def __init__(self, sid, db, data_address, sync_address, source_id):
        self.sid = sid
        self.db = db
        self.source_id      = source_id
        self.logger         = logging.getLogger()
        self.sync_address   = sync_address
        self.data_address   = data_address
        self.logger.info("data address is {ds}".format(ds=data_address))
        
        self.cur_event = None
        
    def start(self):
        self.proc = multiprocessing.Process(target=self.run)
        self.proc.start()
               
    def run(self):
        self.logger.info("starting data source:{sid} on {addr}".format(sid=self.sid, addr=self.data_address))
        self.context = zmq.Context()
        
        #synchronize with feed
        sync_socket = self.context.socket(zmq.REQ)
        sync_socket.connect(self.sync_address)
        # send a synchronization request to the feed
        sync_socket.send('')
        # wait for synchronization reply from the feed
        sync_socket.recv()
        sync_socket.close()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_socket.connect(self.data_address)
        
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
            event['s'] = self.source_id 
            event['type'] = 'event'
            del(event['_id'])
            
            #send this event to feed address
            #self.logger.info("sending {event}".format(event=event))
            self.data_socket.send(json.dumps(event))
        
        done_msg = {}
        done_msg['type'] = 'DONE'
        done_msg['s'] = self.source_id
        self.data_socket.send(json.dumps(done_msg))   
        self.data_socket.close()
        self.context.term()
        self.logger.info("finished processing data source")
        
        
            
   
       
        
        