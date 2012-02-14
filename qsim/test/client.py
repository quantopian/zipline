import zmq
import json
import qsim.util as qutil

class TestClient(object):
    
    def __init__(self, utest, expected_msg_count=0):
        self.address            = None
        self.sync               = None
        self.received_count     = 0
        self.expected_msg_count = expected_msg_count
        self.error              = False
        self.utest              = utest
        self.data_feed          = None
        self.context            = None
        
    def run(self):
        
        try:
            qutil.LOGGER.info("running the client")
            self.context = zmq.Context()
        
            self.data_feed = self.context.socket(zmq.PULL)

            qutil.LOGGER.info("connecting to {address}".format(address=self.address))
            self.data_feed.connect(self.address)
        
            self.sync.open()
        
            self.poller = zmq.Poller()
            self.poller.register(self.data_feed, zmq.POLLIN)
            
            qutil.LOGGER.info("Starting the client loop")

            prev_dt = None
            while self.sync.confirm():
                socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.
                if self.data_feed in socks and socks[self.data_feed] == zmq.POLLIN:   
                    msg = self.data_feed.recv()
                    if(msg == "DONE"):
                        qutil.LOGGER.info("DONE!")
                        break
                    self.received_count += 1
                    event = json.loads(msg)
                    if(prev_dt != None):
                        if(not event['dt'] >= prev_dt):
                            raise Exception("Message out of order: {date} after {prev}".format(date=event['dt'], prev=prev_dt))
            
                    prev_dt = event['dt']
                    if(self.received_count % 100 == 0):
                        qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
            
            qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
            self.data_feed.close()
            self.sync.close()
        except:
            self.error = True
            qutil.LOGGER.exception("**********************Error in test client.")
        finally:
            self.context.destroy()
        
        self.utest.assertEqual(self.expected_msg_count, self.received_count, 
                        "The client should have received ({n}) the same number of messages as the feed sent ({m})."
                            .format(n=self.received_count, m=self.expected_msg_count))