import datetime
import zmq
import pymongo
import pymongo.json_util
import json
import pytz
import copy
from pymongo import ASCENDING, DESCENDING

from backtest.util import *


class EquityMinuteTrades(object):
    
    def __init__(self, sid, db, data_socket, control_socket, source_id, logger):
        self.sid = sid
        self.db = db
        self.source_id      = source_id
        self.logger         = logger
        self.control_socket = control_socket
        self.data_socket = data_socket
        self.logger.info("data socket is {ds}".format(ds=data_socket))
        self.logger.info("control socket is {cs}".format(cs=control_socket))
        
        
    def run2(self):
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        #self.data_source = self.context.socket(zmq.PUSH)
        #self.data_source.connect(data_socket)
        
        #create the control subscription
        self.qbt_control = self.context.socket(zmq.SUB)
        self.qbt_control.connect(self.control_socket)
        self.qbt_control.setsockopt(zmq.SUBSCRIBE, '')
        
        while True:
            self.logger.info("about to call receive")
            try:
                message = self.qbt_control.recv()
                self.logger.info("received message: {msg}".format(msg=message))
            except zmq.ZMQError as err:
                if err.errno != zmq.EAGAIN:
                    raise err
            
            
        
    def run(self):
        
        self.logger.info("starting data source:{sid}".format(sid=self.sid))
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        self.data_source = self.context.socket(zmq.PUSH)
        self.data_source.connect(self.data_socket)
        
        #create the control subscription
        self.qbt_control = self.context.socket(zmq.SUB)
        self.qbt_control.connect(self.control_socket)
        self.qbt_control.setsockopt(zmq.SUBSCRIBE, '')
        
        eventQS = self.db.equity.trades.minute.find(fields=["sid","price","volume","dt"],
                                     spec={"sid":self.sid},
                                     sort=[("dt",ASCENDING)],
                                     slave_ok=True)
        self.logger.info("found {count} events".format(count=eventQS.count()))
        control_dt_str = None
        
        for doc in eventQS:
            doc_dt = doc['dt'].replace(tzinfo = pytz.utc)
            doc_dt_str = format_date(doc_dt)
            event = copy.copy(doc)
            event['dt'] = doc_dt_str
            event['s'] = self.source_id #s is for source
            del(event['_id'])
            
            #wait for a control message if our current event is ahead of the control time
            #if our current event is in the past wrt to control time, keep sending messages.
            if(control_dt_str == None or event['dt'] > control_dt_str):
                try:               
                    self.logger.info("about to call receive")
                    control_dt_str = self.qbt_control.recv()
                    self.logger.info("received message: {msg}".format(msg=control_dt_str))
                except zmq.ZMQError:
                    self.logger.info("we got an error on receive")
                    continue
                    
            #send this event to qbt
            self.logger.info("sending {event}".format(event=event))
            self.data_source.send(json.dumps(event))
   
       
        
        