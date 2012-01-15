"""
QBT - Quantopian Backtest
====================================

qbt runs backtests using multiple processes and zeromq messaging for communication and coordination. 

Backtest is the primary process. It maintains both server and client sockets:
zmq sockets for internal processing::

    - data sink,        ZMQ.REQ. Port = port_start + 1
        - backtest will connect to socket, and then spawn one process per datasource, passing the data sink url as a startup arg. Each
          datasource process will bind to the socket, and start processing 
        - backtest is responsible for merging the data events from all sources into a serialized stream and relaying it to the 
          aggregators, merging agg results, and transmitting consolidated stream to event feed.
    - agg source,       ZMQ.PUSH. Port = port_start + 2
    - agg sink,         ZMQ.PULL. Port = port_start + 3
    - control source,   ZMQ.PUB.  Port = port_start + 4
        - all child processes must subscribe to this socket. Control commands:
            - START -- begin processing
            - TIME  -- current simulated time in backtest
            - KILL  -- exit immediately
            
zmq sockets for backtest clients:
=================================
    - orders sink,      ZMQ.RESP. Port = port_start + 5
        - backtest will connect (can you bind?) to this socket and await orders from the client. Order data will be processed against the streaming datafeed.
    - event feed,       ZMQ.RESP. Port = port_start + 6
        - backtest will bind to this socket and respond to requests from client for more data. Response data will be the queue of events that
          transpired since the last request.
           
    
"""
import copy
import multiprocessing
import zmq

from backtest.util import *
from data.sources.equity import *



CONTROLLER_PORT     =  9000
DATA_SINK_PORT      = 10000
DATA_FEED_PORT      = 30000

class Backtest(object):
    
    def __init__(self, db, logger):
        self.logger = logger
        self.db = db
        self.data_workers = {}
        
        
    def start_data_workers(self):
        """Start a sub-process for each datasource."""
        
        emt1 = EquityMinuteTrades(133, self.db, DATA_SINK_PORT, 1, self.logger)
        data_workers[1] = emt1
        emt1.start()
        #emt2 = EquityMinuteTrades(134, self.db, self.data_socket, self.controller_socket, 2, self.logger)
        #emt2.start()
        self.logger.info("ds processes launched")
       
    def run(self):   
        # Prepare our context and sockets
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        #see: http://zguide.zeromq.org/py:taskwork2
        self.data_socket = "tcp://127.0.0.1:{port}".format(port=DATA_SINK_PORT)
        self.data_sink = self.context.socket(zmq.PULL)
        self.data_sink.bind(self.data_socket)
        
        #create the controller publishing socket.
        self.controller_socket = "tcp://127.0.0.1:{port}".format(port=CONTROLLER_PORT) 
        #self.controller = self.context.socket(zmq.PUB)
        #self.controller.bind(self.controller_socket)
        
        
        #create the merged dataset feed socket
        self.data_feed = self.context.socket(zmq.PUSH)
        self.data_feed.bind("tcp://127.0.0.1:{port}".format(port=DATA_FEED_PORT))
        
        self.last_event_time = None
        self.data_workers = []
        
        last_dt = None
        event_q = []
        #data workers start in independent processes, connecting to the data_socket
        self.start_data_workers()
        while True:
            try:
                #self.logger.info("about to receive")
                message = self.data_sink.recv()
                event = json.loads(message)
                last_dt = event['dt']
                
                event_q.append(event)
                event_q = sorted(event_q, key=lambda event: event['dt'])
                cur_event = event_q.pop(0)
                self.data_feed.send(json.dumps(cur_event))
                
                #this signals loop at datasource should proceed... all sources will send all events as fast as possible.
                self.data_sink.send(str(last_dt))
            except zmq.ZMQError as err:
                
                #EAGAIN indicates recv doesn't have messages now, and datasources are sending 
                #so only throw if we have an error other than EAGAIN.
                if err.errno != zmq.EAGAIN:
                     raise err #real error, throw back to caller
                #self.logger.info("we received an error")
                break
                
            #no events in q at this point means we've processed all the data in all the sources!
            #if(len(event_q) == 0):
            #    return
                
            #event_q = sorted(event_q, key=lambda event: event['dt'])    
            # we have the most recent message from each source, so we can only 
            # be sure the first uninterrupted streak of messages from the same source 
            # in the queue are most recent.
            #while len(event_q) > 0:
            #    cur_event = event_q.pop(0)
            #    self.data_feed.send(json.dumps(cur_event))
            #    last_dt = event_q[0]['dt']
      