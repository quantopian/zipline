"""
qbt runs backtests using multiple processes and zeromq messaging. 

Backtest is the primary process. It maintains both server and client sockets:
zmq sockets for internal processing: 
====================================
    - data sink,        ZMQ.PULL. Port = port_start + 1
        - backtest will connect to socket, and then spawn one process per datasource, passing the data sink url as a startup arg. Each
          datasource process will bind to the socket, and start processing 
        - backtest is responsible for merging the data events from all sources into a serialized stream and relaying it to the 
          aggregators, merging agg results, and transmitting consolidated stream to event feed.
    - agg source,       ZMQ.PUSH. Port = port_start + 2
    - agg sink,         ZMQ.PULL. Port = port_start + 3
    - control source,   ZMQ.PUB.  Port = port_start + 4
        - all child processes must subscribe to this socket. Control commands:
            - START -- begin processing
            - KILL  -- exit immediately
            
zmq sockets for backtest clients:
=================================
    - orders sink,      ZMQ.RESP. Port = port_start + 5
        - backtest will connect (can you bind?) to this socket and await orders from the client. Order data will be processed against the streaming datafeed.
    - event feed,       ZMQ.RESP. Port = port_start + 6
        - backtest will bind to this socket and respond to requests from client for more data. Response data will be the queue of events that
          transpired since the last request.
           
    
"""
import multiprocessing
import zmq

class Backtest(object):
    
    def __init__(self, port_start=10000):
        