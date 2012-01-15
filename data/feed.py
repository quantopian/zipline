
from data.sources.equity import *

class DataFeed(object):
    
    def __init__(self, db, logger):
        self.logger = logger
        self.db = db
        self.data_workers = {}
        self.data_address = "tcp://127.0.0.1:{port}".format(port=10101)
        self.sync_address = "tcp://127.0.0.1:{port}".format(port=10102)
        
    def start_data_workers(self):
        """Start a sub-process for each datasource."""
        
        # Socket to receive signals
        syncservice = self.context.socket(zmq.REP)
        syncservice.bind(self.sync_address)
        
        
        emt1 = EquityMinuteTrades(133, self.db, self.data_address, self.sync_address, 1, self.logger)
        self.data_workers[1] = emt1
        emt1.start()
        emt2 = EquityMinuteTrades(134, self.db, self.data_address, self.sync_address, 2, self.logger)
        self.data_workers[2] = emt1
        emt2.start()
        
        workers = 0
        while workers < len(self.data_workers):
            # wait for synchronization request
            msg = syncservice.recv()
            # send synchronization reply
            syncservice.send('')
            workers += 1
        
        syncservice.close()
        
        self.logger.info("{count} ds processes launched".format(count=workers))
       
    def run(self):   
        # Prepare our context and sockets
        self.context = zmq.Context()
        
        #create the data sink. Based on http://zguide.zeromq.org/py:tasksink2 
        #see: http://zguide.zeromq.org/py:taskwork2
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind(self.data_address)
        
        counter = 0
        
        self.start_data_workers()
        
        while True:
            message = self.data_socket.recv()
            event = json.loads(message)
            #self.logger.info(message)
            counter = counter + 1
            if(event['type'] == "DONE"):
                source = event['s']
                if(self.data_workers.has_key(source)):
                    del(self.data_workers[source])
                if(len(self.data_workers) == 0):
                    break
                
        self.logger.info("Collected {n} messages".format(n=counter))
        self.data_socket.close()
        self.context.term()