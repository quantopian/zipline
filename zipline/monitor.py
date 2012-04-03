import time
import gevent

from gevent_zeromq import zmq
import util as qutil

from protocol import CONTROL_PROTOCOL, CONTROL_FRAME, \
    CONTROL_UNFRAME, CONTROL_STATES

# TODO: print -> qutil.LOGGER.info

# When you're cold and waiting for the train... draw ascii art!
#
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

        self._ctx = None
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
            self.dologging = True
        else:
            self.logging = False
            self.dologging = False

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

        if not context:
            self._ctx = zmq.Context.instance()
        else:
            self._ctx = context

    def run(self, flavor):
        """
        Abstracts
        """
        assert flavor in ['thread', 'mp', 'green']
        return self._poll() # use a python loop

    def log_status(self):
        print "[Controller] Tracking : %s" % ([c for c in self.tracked],)
        gevent.spawn_later(5, self.log_status)

    def _poll(self):

        self.pub = self._ctx.socket(zmq.PUB)
        self.pub.bind(self.pub_socket)

        self.router = self._ctx.socket(zmq.ROUTER)
        self.router.bind(self.route_socket)

        self.associated.extend([self.pub, self.router])

        # Spin the coroutines
        gevent.spawn(self.log_status)
        gevent.spawn_raw(self.recv_hearts_router)

        while self.polling:
            self.beat()
            gevent.sleep(self.period)

            try:
                msg = self.pull.recv()
                self.pub.send(msg)
            except KeyboardInterrupt:
                self.polling = False
                break
            except zmq.ZMQError:
                self.polling = False
                break
            except Exception as e:
                # Its common to wrap these in wildcard exceptions so
                # that we don't loose messages, ever
                if self.logging:
                    self.logging.error(str(e))
                continue

        # After loop exits
        self.terminated = True

    def beat(self):
        toc = time.time()

        self.ctime += toc - self.tic
        self.tic = toc
        self.message = str(self.ctime)

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

        # Reset the responses for this cycle
        self.responses = set()

        gevent.spawn_raw(self.pub.send, str(self.ctime))

    # ------------------
    # Component Handlers
    # ------------------

    # The various "states of being that a component can inform us
    # of
    def new(self, component):
        print '[Controller] Tracking "%s" ' % component

        if component in self.topology or self.freeform:
            self.tracked.add(component)
        else:
            # Some sort of socket collision has occured, this is
            # a very bad failure mode.
            raise UnknownChatter(component)

    def fail(self, component):
        print '[Controller] Component "%s" timed out' % component
        self.tracked.remove(component)

    def done(self, component):
        # TODO: This will be what we ship off to vbench at some
        # point...
        # print component finished at self.ctime
        pass

    def exception(self, component, failure):
        pass

    # --------
    # IO Loops
    # --------

    def recv_hearts(self):
        buffer = []

        while True:
            message = self.router.recv(zmq.NOBLOCK, copy=False)

            if message:
                buffer.append(message)
            if not self.router.rcvmore():
                break

        gevent.spawn_later(0.001, self.handle_pong, buffer)

    def recv_hearts_router(self):
        buffer = []

        while True:
            message = self.router.recv()

            if message:
                buffer.append(message)

            if not self.router.getsockopt(zmq.RCVMORE):
                gevent.spawn_raw(self.handle_recv, buffer[:])
                buffer = []

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
                print "Weird stuff happened: %s" % msg[1]

        # A component is telling us it failed, and how
        if id is CONTROL_PROTOCOL.EXCEPTION:
            self.exception(identity, status)

        # A component is telling us its done with work and won't
        # be talking to us anymore
        if id is CONTROL_PROTOCOL.DONE:
            self.done(identity, status)

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
            context = zmq.Context.instance()

        s = context.socket(zmq.DEALER)
        s.connect(self.route_socket)
        s.setsockopt(zmq.IDENTITY, identity)

        self.associated.append(s)
        return s

    def message_listener(self, context = None):
        """
        Spin off a socket used for receiving messages from this
        controller.
        """

        if not context:
            context = zmq.Context.instance()

        s = context.socket(zmq.SUB)
        s.connect(self.pub_socket)
        s.setsockopt(zmq.SUBSCRIBE, '')

        self.associated.append(s)
        return s

    def shutdown(self, hard=False, soft=True, context=None):
        self.polling = False

        assert hard or soft, """ Must specify kill hard or soft """

        if not context:
            context = zmq.Context.instance()

        if hard:
            self.state = CONTROL_STATES.SHUTDOWN

            print '[Controller] Hard Shutdown'

            #for asoc in self.associated:
                #asoc.close()

        if soft:
            self.state = CONTROL_STATES.TERMINATE

            print '[Controller] Soft Shutdown'

            #for asoc in self.associated:
                #asoc.close()

    def destroy(self):
        """
        Manual cleanup.
        """
        self.shutdown()

    #def __del__(self):
        #self.shutdown()

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
