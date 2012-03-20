from gevent_zeromq import zmq

import zipline.util as qutil
import zipline.messaging as qmsg
import zipline.protocol as zp
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_TYPE
from zipline.finance.trading import TradeSimulationClient

class TestClient(qmsg.Component):

    def __init__(self):
        qmsg.Component.__init__(self)
        self.init()

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
                qutil.LOGGER.info("received {n} messages".format(n=self.received_count))
            
    def unframe(self, msg):
        return zp.MERGE_UNFRAME(msg)


class TestAlgorithm():
    
    def __init__(self, sid, amount, order_count):
        self.count = order_count
        self.sid = sid
        self.amount = amount
        self.incr = 0
        self.done = False
        self.order = None
        
    def set_order(self, order_callable):
        self.order = order_callable
        
    def handle_frame(self, frame):
        for dt, s in frame.iteritems():     
            data = {}
            data.update(s)
            event = zp.namedict(data)
            #place an order for 100 shares of sid:133
            if self.incr < self.count:
                self.order(self.sid, self.amount)
                self.incr += 1
                
class NoopAlgorithm(object):
        
    def set_order(self, order_callable):
        pass
    
    def handle_frame(self, frame):
        pass
        