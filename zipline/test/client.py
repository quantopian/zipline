#import logging
import ujson as json

import zipline.util as qutil
import zipline.messaging as qmsg
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE

#qlogger = logging.getLogger('qexec')

class TestClient(qmsg.Component):

    def __init__(self, expected_msg_count=0):
        qmsg.Component.__init__(self)

        # Generally shouldn't reference unit tests here.
        self.utest              = utest
        self.expected_msg_count = expected_msg_count
        self.init()

    def init(self):
        self.received_count = 0
        self.prev_dt        = None

    @property
    def get_id(self):
        return "TEST_CLIENT"

    @property
    def get_type(self):
        return COMPONENT_TYPE.SINK

    def open(self):
        self.data_feed = self.connect_result()

    def do_work(self):
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        if self.control_in in socks and socks[self.control_in] == self.zmq.POLLIN:
            msg = self.control_in.recv()

        if self.data_feed in socks and socks[self.data_feed] == self.zmq.POLLIN:
            msg = self.data_feed.recv()
            #logger.info('msg:' + str(msg))

            if msg == str(CONTROL_PROTOCOL.DONE):
                qutil.LOGGER.info("Client is DONE!")
                self.signal_done()
                return

            self.received_count += 1

            try:
                event = json.loads(msg)

            # JSON deserialization error
            except ValueError as exc:
                return self.signal_exception(exc)

            if self.prev_dt != None:
                if not event['dt'] >= self.prev_dt:
                    raise Exception(
                        "Message out of order: {date} after {prev}".format(
                            date = event['dt'], prev = self.prev_dt
                        )
                    )
            else:
                self.prev_dt = event['dt']

            if self.received_count % 100 == 0:
                qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
