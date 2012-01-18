import zmq
import logging
import tornado
from data.sources.equity import *
from data.feed import *
from data.transforms import MergedTransformsFeed, MovingAverage

from qbt_server import connect_db
from qbt_client import TestClient

def datafeed():
    connection, db = connect_db()
    logger = logging.getLogger()
    feed = DataFeed(db, 2) 
    #feed.run()
    feed_proc = multiprocessing.Process(target=feed.run)
    feed_proc.start()
    
    config = {}
    config['name'] = '**merged feed**'
    config['transforms'] = [{'name':'mavg1', 'class':'MovingAverage', 'hours':1},{'name':'mavg2', 'class':'MovingAverage', 'hours':2}]
    
    result_address = "tcp://127.0.0.1:20202"
    
    mavg = MovingAverage(feed.feed_address, result_address, feed.sync_address, config['transforms'][0])
    mavg_proc = multiprocessing.Process(target=mavg.run)
    logger.info("about to launch moving average")
    mavg_proc.start()
    
    #merger = Merge(feed.feed_address, result_address, feed.sync_address, config)
    #merger.run()
    #merger_proc = multiprocessing.Process(target=merger.run)
    #merger_proc.start() 
    
    
    #subscribe a client directly to the consolidated feed
    #client = TestClient(feed.feed_address, feed.sync_address)
    
    logger.info("starting the client")
    #subscribe a client to the transformed feed
    client = TestClient(result_address, feed.sync_address, bind=True)
    client.run()
    
    logger.info("feed has {pending} messages".format(pending=feed.data_buffer.pending_messages()))
    assert(feed.data_buffer.pending_messages() == 0)
    
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
     