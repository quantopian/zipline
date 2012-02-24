import json
import zipline.util as qutil
import zipline.messaging as qmsg

from zipline.protocol import CONTROL_PROTOCOL

class TestClient(qmsg.Component):

    def __init__(self, utest, expected_msg_count=0):
        qmsg.Component.__init__(self)
        self.received_count     = 0
        self.expected_msg_count = expected_msg_count
        self.utest              = utest
        self.prev_dt            = None

    @property
    def get_id(self):
        return "TEST_CLIENT"

    def open(self):
        self.data_feed, self.poller = self.connect_result()

    def do_work(self):
        socks = dict(self.poller.poll(2000)) #timeout after 2 seconds.

        if self.data_feed in socks and socks[self.data_feed] == self.zmq.POLLIN:   
            msg = self.data_feed.recv()

            if msg == str(CONTROL_PROTOCOL.DONE):
                qutil.LOGGER.info("Client is DONE!")
                self.signal_done()
                self.utest.assertEqual(self.expected_msg_count, self.received_count, 
                                "The client should have received ({n}) the same number of messages as the feed sent ({m})."
                                    .format(n=self.received_count, m=self.expected_msg_count))
                return

            self.received_count += 1
            event = json.loads(msg)
            if(self.prev_dt != None):
                if(not event['dt'] >= self.prev_dt):
                    raise Exception("Message out of order: {date} after {prev}".format(date=event['dt'], prev=prev_dt))

            self.prev_dt = event['dt']
            if(self.received_count % 100 == 0):
                qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
