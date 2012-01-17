import zmq
import logging
import tornado
from data.sources.equity import *
from data.feed import *
from data.transforms import Merge, MovingAverage

from qbt_server import connect_db
from qbt_client import BacktestClient

def datafeed():
    connection, db = connect_db()
    logger = logging.getLogger()
    feed = DataFeed(db, 1) #one merge, two moving averages.
    feed_proc = multiprocessing.Process(target=feed.run)
    feed_proc.start()
    
    #config = {}
    #config['name'] = '**merged feed**'
    #config['transforms'] = [{'name':'mavg1', 'class':'MovingAverage', 'hours':1},{'name':'mavg2', 'class':'MovingAverage', 'hours':2}]
    
    #result_address = "tcp://127.0.0.1:20202"
    
    #mavg = MovingAverage(feed.feed_address, result_address, feed.sync_address, config['transforms'][0])
    #mavg.run()
    
    #merger = Merge(feed.feed_address, result_address, feed.sync_address, config)
    #merger_proc = multiprocessing.Process(target=merger.run)
    #merger_proc.start() 
    
    client = BacktestClient(feed.feed_address, feed.sync_address)
    client.run()
    
def pubsub():
    proc1 = multiprocessing.Process(target=sub)
    proc2 = multiprocessing.Process(target=pub)
    proc1.start()
    proc2.start()
    
def sub():
    context = zmq.Context()
    controller = context.socket(zmq.PULL)
    controller.connect("tcp://127.0.0.1:10099")    
    #controller.setsockopt(zmq.SUBSCRIBE, '')
    while True:
        try:
            message = controller.recv()
            print message
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                raise err
                
def pub():
    context = zmq.Context()
    controller = context.socket(zmq.PUSH)
    controller.bind("tcp://127.0.0.1:10099")    
    while True:
        controller.send("HELLO3")
        

     
if __name__ == "__main__":
    tornado.options.parse_command_line()
    datafeed()              
     