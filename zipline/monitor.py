import time
import gevent
import itertools
# pyzmq
import zmq
# gevent_zeromq
import gevent_zeromq
# zmq_ctypes
#import zmq_ctypes

from protocol import CONTROL_PROTOCOL, CONTROL_FRAME, \
    CONTROL_UNFRAME, CONTROL_STATES, INVALID_CONTROL_FRAME

from gpoll import _Poller as GeventPoller

# Roll Call ( Discovery )
# -----------------------
#
#       Controller ( 'foo', 'bar', 'fizz', 'pop' )
#    ------------------
#    |     |     |     |
#  +---+
#  | 0 |   ?     ?     ?
#  +---+
#    |
#   IDENTITY: foo
#   get message: PROTOCOL.HEARTBEAT
#   reply with PROTOCOL.OK
#
#       Controller topology = ( 'foo', 'bar', 'fizz', 'pop' )
#       'foo' in topology = YES ->
#           track 'foo'
#    ------------------
#    |     |     |     |
#  +---+
#  | 1 |   ?     ?     ?
#  +---+

# Heartbeating
# ------------
#
#       Controller ( time = 2.717828 )
#    ------------------
#    |     |     |     |
#  +---+ +---+ +---+ +---+
#  | 0 | | 0 | | 0 | | 0 |
#  +---+ +---+ +---+ +---+
#    |
#   IDENTITY: foo
#   get message: time = 2.717828
#   reply with [ foo, 2.71828 ]
#
#       Controller ( foo.status = OK )
#    ------------------
#    |     |     |     |
#  +---+ +---+ +---+ +---+
#  | 1 | | 0 | | 0 | | 0 |
#  +---+ +---+ +---+ +---+
#    |
#  Controller tracks this node as good
#  for this heartbeat

# Shutdown
# --------
#
#       Controller ( state = RUNNING )
#    ------------------
#    |     |     |     |
#  +---+ +---+ +---+ +---+
#  | 1 | | 1 | | 1 | | 1 |
#  +---+ +---+ +---+ +---+
#    |
#   IDENTITY: foo
#   send [ DONE ]

#       Controller ( state = SHUTDOWN )
#       Controller topology.remove('foo')
#    ------------------
#          |     |     |
#  +---+ +---+ +---+ +---+
#  |   | | 1 | | 1 | | 1 |
#  +---+ +---+ +---+ +---+
#    |
#   IDENTITY: foo
#   yield, stop sending messages

# Termination
# ------------
#
#       Controller ( state = TERMINATE )
#    ------------------
#    |     |     |     |
#  +---+ +---+ +---+ +---+
#  | 1 | | 1 | | 1 | | 1 |
#  +---+ +---+ +---+ +---+
#    |
#   get message PROTOCOL.KILL

#       Controller ( state = TERMINATE )
#    ------------------
#    |     |     |     |
#  +---+ +---+ +---+ +---+
#  | 0 | | 0 | | 0 | | 0 |
#  +---+ +---+ +---+ +---+

class UnknownChatter(Exception):
    def __init__(self, name):
        self.named = name
    def __str__(self):
        return """Component calling itself "%s" talking on unexpected channel"""\
            % self.named

class Controller(object):
    """
    A N to M messaging system for inter component communication.

    :param pub_socket: Socket to publish messages, the starting
                       point of :func message_listener: .

    :param route_socket: Socket to listen for status updates for
                         the individual components.
                         :func message_sender: .

    :param logging: Logging interface for tracking broker state
        Defaults to None

    Topology is the set of components we expect to show up.
    States are the transitions the sytems go through. The
    simplest is from RUNNING -> NOT RUNNING .

    Usage::

        controller = Controller(
            'tcp://127.0.0.1:5000',
            'tcp://127.0.0.1:5001',
        )

        # typically you'd want to run this async to your main
        # program since it blocks indefinetely.
        controller.manage(
            [ TOPOLOGY ]
            [ STATES ]
        )

    """

    debug = False
    period = 1

    def __init__(self, pub_socket, route_socket, logging = None):

        self.context = None
        self.zmq = None
        self.zmq_poller = None

        polling = False

        self.polling = polling

        self.tracked = set()
        self.responses = set()

        self.ctime = 0
        self.tic = time.time()
        self.freeform = False

        self.associated = []

        self.pub_socket = pub_socket
        self.route_socket = route_socket

        if logging:
            self.logging = logging
        else:
            import util as qutil
            self.logging = qutil.LOGGER

    def init_zmq(self, flavor):

        assert self.zmq_flavor in ['thread', 'mp', 'green']

        if flavor == 'mp':
            self.zmq = zmq
            self.context = self.zmq.Context()
            self.zmq_poller = self.zmq.Poller
            return
        if flavor == 'thread':
            self.zmq = zmq
            self.context = self.zmq.Context.instance()
            self.zmq_poller = self.zmq.Poller
            return
        if flavor == 'green':
            self.zmq = gevent_zeromq.zmq
            self.context = self.zmq.Context.instance()
            self.zmq_poller = GeventPoller
            return
        if flavor == 'pypy':
            self.zmq = zmq
            self.context = self.zmq.Context.instance()
            self.zmq_poller = self.zmq.Poller
            return

    def manage(self, topology, states=None, context=None):
        """
        Give the controller a set set of components to manage and
        a set of state transitions for the entire system.
        """

        # A freeform topology is where we heartbeat with anything
        # that shows up.
        if topology == 'freeform':
            self.freeform = True
            self.topology = frozenset([])
        else:
            self.freeform = False
            self.topology = frozenset(topology)

        default_states = [
            CONTROL_STATES.RUNNING,
            CONTROL_STATES.SHUTDOWN,
            CONTROL_STATES.TERMINATE,
        ]

        self.states = states or default_states
        self.polling = True

        # Start off in RUNNING, state
        self.state = self.states[0]

    def run(self):
        self.init_zmq(self.zmq_flavor)

        try:
            return self._poll() # use a python loop
        except KeyboardInterrupt:
            self.logging.info('Shutdown event loop')

    def log_status(self):
        """
        Snapshot of the tracked components at every period.
        """
        #self.logging.info("[Controller] Tracking : %s" % ([c for c in self.tracked],))
        pass

    # -------------
    # Publications
    # -------------

    def send_heart(self):
        heartbeat_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.HEARTBEAT,
            str(self.ctime)
        )
        self.pub.send(heartbeat_frame)

    def send_hardkill(self):
        kill_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.KILL,
            ''
        )
        self.pub.send(kill_frame)

    def send_softkill(self):
        soft_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.SHUTDOWN,
            ''
        )
        self.pub.send(soft_frame)

    # -----------
    # Event Loops
    # -----------

    def _poll(self):

        self.pub = self.context.socket(self.zmq.PUB)
        self.pub.bind(self.pub_socket)

        self.router = self.context.socket(self.zmq.ROUTER)
        self.router.bind(self.route_socket)

        self.associated.extend([self.pub, self.router])

        poller = self.zmq.Poller()
        poller.register(self.router, self.zmq.POLLIN)

        buffer = []

        for i in itertools.count(0):
            self.log_status()
            self.responses = set()

            self.ctime = time.time()
            self.send_heart()

            while self.polling:
                # Reset the responses for this cycle

                socks = dict(poller.poll(self.period))
                tic = time.time()

                if tic - self.ctime > self.period:
                    break

                if self.router in socks and socks[self.router] == self.zmq.POLLIN:
                    rawmessage = self.router.recv()

                    if rawmessage:
                        buffer.append(rawmessage)

                    try:
                        if not self.router.getsockopt(self.zmq.RCVMORE):
                            self.handle_recv(buffer[:])
                            buffer = []
                    except INVALID_CONTROL_FRAME:
                        self.logging.error('Invalid frame', rawmessage)
                        pass

            self.beat()

            if self.zmq_flavor == 'green':
                gevent.sleep(0)

            if not self.polling:
                break

        # After loop exits
        self.terminated = True

    def beat(self):

        # These the set overloaded operations
        # A & B ~ set.intersection
        # A - B ~ set.difference

        # * good - Components we are currently tracking and who just sent
        #          us back the right response.
        # * bad - Components we are currently tracking but who did not
        #         send us back a response.
        # * new - Components we haven't heard from yet, but sent back the
        #         right response.

        good = self.tracked   & self.responses
        bad  = self.tracked   - good
        new  = self.responses - good

        for component in new:
            self.new(component)

        for component in bad:
            self.fail(component)

    # ------------------
    # Component Handlers
    # ------------------

    # The various "states of being that a component can inform us
    # of
    def new(self, component):
        self.logging.info('[Controller] Alive "%s" ' % component)

        if component in self.topology or self.freeform:
            self.tracked.add(component)
        else:
            # Some sort of socket collision has occured, this is
            # a very bad failure mode.
            raise UnknownChatter(component)

    def fail(self, component):
        self.logging.info('[Controller] Component "%s" timed out' % component)
        self.tracked.remove(component)

    def done(self, component):
        # TODO: This will be what we ship off to vbench at some
        # point...
        # print component finished at self.ctime
        self.logging.info('[Controller] Component "%s" done.' % component)

    def exception(self, component, failure):
        self.logging.error('Component "%s" in exception state' % component)

    # -----------------
    # Protocol Handling
    # -----------------

    def handle_recv(self, msg):
        """
        Check for proper framing at the transport layer.
        Seperates the proper frames from anything else that might
        be coming over the wire. Which shouldn't happen ...  right?
        """
        identity = msg[0]
        id, status = CONTROL_UNFRAME(msg[1])

        # A component is telling us its alive:
        if id is CONTROL_PROTOCOL.OK:

            if status == str(self.ctime):
                self.responses.add(identity)
            else:
                # Otherwise its something weird and we don't know
                # what to do so just say so
                self.logging.error("Weird stuff happened: %s" % msg)

        # A component is telling us it failed, and how
        if id is CONTROL_PROTOCOL.EXCEPTION:
            self.exception(identity, status)

        # A component is telling us its done with work and won't
        # be talking to us anymore
        if id is CONTROL_PROTOCOL.DONE:
            self.done(identity)

    # -------------------
    # Hooks for Endpoints
    # -------------------

    # These are all connects so no complex allocation logic is
    # needed. Dealers and Subscribers can all come and go as a
    # function of time without impacting flow of the whole
    # system.

    def message_sender(self, identity, context = None):
        """
        Spin off a socket used for sending messages to this
        controller.
        """

        if not context:
            context = self.zmq.Context.instance()

        s = context.socket(zmq.DEALER)
        s.setsockopt(zmq.IDENTITY, identity)
        s.connect(self.route_socket)

        self.associated.append(s)
        return s

    def message_listener(self, context = None):
        """
        Spin off a socket used for receiving messages from this
        controller.
        """

        if not context:
            context = self.zmq.Context.instance()

        s = context.socket(zmq.SUB)
        s.connect(self.pub_socket)
        s.setsockopt(zmq.SUBSCRIBE, '')

        self.associated.append(s)
        return s

    def shutdown(self, hard=False, soft=True, context=None):

        if not self.polling:
            return

        self.polling = False

        assert hard or soft, """ Must specify kill hard or soft """

        if hard:
            self.state = CONTROL_STATES.SHUTDOWN

            self.logging.info('[Controller] Hard Shutdown')

            #for asoc in self.associated:
                #asoc.close()

        if soft:
            self.state = CONTROL_STATES.TERMINATE

            self.logging.info('[Controller] Soft Shutdown')
            self.send_softkill()

            #for asoc in self.associated:
                #asoc.close()

if __name__ == '__main__':

    print 'Running on ',\
        'tcp://127.0.0.1:5000', \
        'tcp://127.0.0.1:5001',

    controller = Controller(
        'tcp://127.0.0.1:5000',
        'tcp://127.0.0.1:5001',
    )
    controller.manage(
        'freeform',
        []
    )
    controller.run('green')
