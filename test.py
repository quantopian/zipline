import zmq
import logging
import tornado
from data.sources.equity import *
from data.feed import *
from qbt_server import connect_db

def datafeed():
    connection, db = connect_db()
    logger = logging.getLogger()
    feed = DataFeed(db, logger)
    logger.info("starting the feed")
    feed.run()
    
    
def sub():
    context = zmq.Context()
    controller = context.socket(zmq.SUB)
    controller.connect("tcp://127.0.0.1:10099")    
    controller.setsockopt(zmq.SUBSCRIBE, '')
    while True:
        try:
            message = controller.recv()
            print message
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                raise err
                
def pub():
    context = zmq.Context()
    controller = context.socket(zmq.PUB)
    controller.bind("tcp://127.0.0.1:10099")    
    while True:
        controller.send("HELLO3")
        

     
if __name__ == "__main__":
    tornado.options.parse_command_line()
    datafeed()              
     