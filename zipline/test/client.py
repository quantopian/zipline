import json
import zipline.util as qutil
import zipline.messaging as qmsg

from zipline.finance.trading import TradeSimulationClient
import zipline.protocol as zp

class TestClient(qmsg.Component):
    """no-op client - Just connects to the merge and counts messages. compares received message count to the expected count."""

    def __init__(self, expected_msg_count=0):
        qmsg.Component.__init__(self)
        self.received_count     = 0
        self.expected_msg_count = expected_msg_count
        self.prev_dt            = None
        self.heartbeat_timeout = 2000

    @property
    def get_id(self):
        return "TEST_CLIENT"

    def open(self):
        self.data_feed = self.connect_result()

    def do_work(self):
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        if self.data_feed in socks and socks[self.data_feed] == self.zmq.POLLIN:   
            msg = self.data_feed.recv()

            if msg == str(zp.CONTROL_PROTOCOL.DONE):
                qutil.LOGGER.info("Client is DONE!")
                self.signal_done()
                if(self.expected_msg_count > 0):
                    assert self.received_count == self.expected_msg_count
                return

            self.received_count += 1
            event = zp.MERGE_UNFRAME(msg)
            if(self.prev_dt != None):
                if(not event.dt >= self.prev_dt):
                    raise Exception("Message out of order: {date} after {prev}".format(date=event['dt'], prev=prev_dt))

            self.prev_dt = event.dt
            if(self.received_count % 100 == 0):
                qutil.LOGGER.info("received {n} messages".format(n=self.received_count))

class TestTradingClient(TradeSimulationClient):
    
    def __init__(self, count):
        TradeSimulationClient.__init__(self)
        self.count = count
        self.incr = 0
    
    def handle_events(self, event_queue):
        #place an order for 100 shares of sid:133
        if(self.incr < self.count):
            self.order(133, 100)
            self.incr += 1
    
