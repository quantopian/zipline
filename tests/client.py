import logging

import zipline.protocol as zp
from zipline.core.component import Component
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE

LOGGER = logging.getLogger('ZiplineLogger')

class TestClient(Component):

    def init(self):
        self.received_count     = 0
        self.prev_dt            = None

        self.result_streams     = []

        # Maximum outgoing result streams, really shouldn't ever
        # need more than 1.
        self.max_outgoing       = 5

    @property
    def get_id(self):
        return "TEST_CLIENT"

    @property
    def get_type(self):
        return COMPONENT_TYPE.SINK

    def open(self):
        self.data_feed = self.connect_result()

    def result_stream(self, zmq_socket, context=None):
        """
        Asynchronously grab a socket to stream results out on.
        """
        ctx = context or zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.bind(zmq_socket)

        # Add
        self.result_streams.append( sock )

    def do_work(self):
        socks = dict(self.poll.poll(self.heartbeat_timeout))

        if socks.get(self.control_in) == self.zmq.POLLIN:
            msg = self.control_in.recv()

        if socks.get(self.data_feed) == self.zmq.POLLIN:
            msg = self.data_feed.recv()
            #logger.info('msg:' + str(msg))

            if msg == str(CONTROL_PROTOCOL.DONE):
                LOGGER.info("Client is DONE!")
                self.signal_done()
                return

            self.received_count += 1

            try:
                event = self.unframe(msg)

            # deserialization error
            except zp.INVALID_MERGE_FRAME as exc:
                return self.signal_exception(exc)

            if self.prev_dt != None:
                if not event['dt'] >= self.prev_dt:
                    raise Exception(
                        "Message out of order: {date} after {prev}".format(
                            date = event['dt'], prev = self.prev_dt
                        )
                    )
            else:
                self.prev_dt = event.dt

            if self.received_count % 100 == 0:
                LOGGER.info("received {n} messages".format(n=self.received_count))

    def unframe(self, msg):
        return zp.MERGE_UNFRAME(msg)
