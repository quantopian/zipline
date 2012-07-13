import os
import zmq
import sys
import time
import gevent
import itertools
import logbook
import gevent_zeromq
from setproctitle import setproctitle
from signal import SIGHUP, SIGINT

from collections import OrderedDict, Counter

from zipline.utils.gpoll import _Poller as GeventPoller
from zipline.protocol import CONTROL_PROTOCOL, CONTROL_FRAME, \
    CONTROL_UNFRAME, CONTROL_STATES, INVALID_CONTROL_FRAME \

from zipline.utils.protocol_utils import ndict

INIT, SOURCES_READY, RUNNING, TERMINATE = CONTROL_STATES

CONTROLLER_TRANSITIONS = frozenset([
    (-1            , INIT),
    (INIT          , SOURCES_READY),
    (SOURCES_READY , RUNNING),

    (INIT          , TERMINATE), # pseudo failure mode
    (SOURCES_READY , TERMINATE), # pseudo failure mode
    (RUNNING       , TERMINATE),
])

class UnknownChatter(Exception):
    def __init__(self, name):
        self.named = name
    def __str__(self):
        return "Component calling itself '%s' talking on unexpected channel"
            % self.named


log = logbook.Logger('Controller')

# The scalars determining the timing of the monitor behavior for
# the system.

PARAMETERS = ndict(dict(
    GENERATIONAL_PERIOD        = 10, #seconds
    ALLOWED_SKIPPED_HEARTBEATS = 10,
    ALLOWED_INVALID_HEARTBEATS = 3,
    PRESTART_HEARBEATS         = 3,
    SOURCES_START_HEARTBEATS   = 3,
    SYSTEM_TIMEOUT             = 50,
))

class Controller(object):
    """
    A N to M messaging system for inter component communication.

    :param pub_socket: Socket to publish messages, the starting
                       point of :func message_listener: .

    :param route_socket: Socket to listen for status updates for
                         the individual components.
                         :func message_sender: .

    """

    # Turn on debug for verbose logging of the system.
    debug = True
    period = PARAMETERS.GENERATIONAL_PERIOD

    def __init__(self, pub_socket, route_socket, devel=True):

        self.devel      = devel
        self.nosignals  = False
        self.context    = None
        self.zmq        = None
        self.zmq_poller = None

        self.running = False
        self.alive = False
        self.tracked = set()
        self.finished = set()

        self.responses = set()

        self.ctime    = 0
        self.tic      = time.time()
        self.freeform = False
        self._state   = -1

        self.associated = []

        self.pub_socket   = pub_socket
        self.route_socket = route_socket

        self.error_replay = OrderedDict()

        self.missed_beats = Counter()

        log.warn("Running Controller in development mode, will ONLY synchronize start.")

    def init_zmq(self, flavor):

        assert self.zmq_flavor in ['thread', 'mp', 'green']

        if flavor == 'mp':
            self.zmq        = zmq
            self.context    = self.zmq.Context()
            self.zmq_poller = self.zmq.Poller

            if self.devel:
                log.warning("USING DEVELOPMENT MODE IN MP CONTEXT NOT RECOMMENDED")
            return
        if flavor == 'thread':
            self.zmq        = zmq
            self.context    = self.zmq.Context.instance()
            self.zmq_poller = self.zmq.Poller
            return
        if flavor == 'green':
            self.zmq        = gevent_zeromq.zmq
            self.context    = self.zmq.Context.instance()
            self.zmq_poller = GeventPoller
            return
        if flavor == 'pypy':
            self.zmq        = zmq
            self.context    = self.zmq.Context.instance()
            self.zmq_poller = self.zmq.Poller
            return

    def manage(self, topology):
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
        self.alive = True

    @property
    def state(self):
        #log.info('returned %s' % self._state)
        return self._state

    @state.setter
    def state(self, new):
        old = self._state

        if (old, new) in CONTROLLER_TRANSITIONS:
            self._state = new
            log.info("State Transition : %s -> %s" % (old, self._state))
        else:
            raise RuntimeError("Invalid State Transition : %s -> %s" %(old, new))

    def run(self):
        self.running = True
        self.init_zmq(self.zmq_flavor)
        setproctitle('Monitor')

        self.state = CONTROL_STATES.INIT

        # Interpreter SIDE EFFECT
        # -----------------------
        # The last breathe of the interpreter will assume that we've
        # failed unless we specify otherwise.
        if not self.devel:
            sys.exitfunc = self.signal_interrupt
        # We overload this if ( and only if ) the topology exits
        # cleanly. This prevents failure modes where the monitor
        # dies.

        try:
            return self._poll() # use a python loop
        except KeyboardInterrupt:
            log.info('Shutdown event loop')

    def log_status(self):
        """
        Snapshot of the tracked components at every period.
        """
        #log.info("Tracking component : %s" % ([c for c in self.tracked],))
        pass

    def replay_errors(self):
        """
        Replay the errors in the order they were reported to the
        controller.
        """
        return [ a for a in sorted(self.replay_errors.keys())]

    # -------------
    # Publications
    # -------------

    def send_go(self):
        go_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.GO,
            ''
        )
        self.pub.send(go_frame)

    def send_heart(self):
        if not self.running:
            return

        heartbeat_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.HEARTBEAT,
            str(self.ctime)
        )
        self.pub.send(heartbeat_frame)

    def send_hardkill(self):
        if not self.running:
            return

        kill_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.KILL,
            ''
        )
        self.pub.send(kill_frame)

    def send_softkill(self):
        if not self.running:
            return

        soft_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.SHUTDOWN,
            ''
        )
        self.pub.send(soft_frame)

    # -----------
    # Event Loops
    # -----------

    def _poll(self):

        assert self.route_socket
        assert self.pub_socket

        assert self.topology,\
        """"Must define topology to monitor, call setup_controller() on
        your Zipline. """

        # -- Publish --
        # =============
        self.pub = self.context.socket(self.zmq.PUB)
        self.pub.bind(self.pub_socket)
        self.pub.setsockopt(zmq.LINGER, 0)

        # -- Router --
        # =============
        self.router = self.context.socket(self.zmq.ROUTER)
        self.router.bind(self.route_socket)
        self.router.setsockopt(zmq.LINGER, 0)

        poller = self.zmq.Poller()
        poller.register(self.router, self.zmq.POLLIN)
        #poller.register(self.cancel, self.zmq.POLLIN)

        self.associated += [self.pub, self.router]

        # TODO: actually do this
        self.state = CONTROL_STATES.SOURCES_READY
        self.state = CONTROL_STATES.RUNNING

        buffer = []

        # ===================
        # Heartbeat Iteration
        # ===================

        for i in itertools.count(0):
            self.log_status()

            # Reset the responses for this cycle
            self.responses = set()

            # broadcast the heartbeat packet
            self.ctime = time.time()
            self.send_heart()

            # ==============
            # Hearbeat Cycle
            # ==============

            # Wait the responses
            checktime = self.ctime
            while self.alive:

                socks = dict(poller.poll(0))
                tic = time.time()

                # We break out of this loop if the time between
                # sending and receiving the heartbeat is more
                # than our poll period.
                # if tic - self.ctime > self.period:
                #    break

                if socks.get(self.router) == self.zmq.POLLIN:
                    rawmessage = self.router.recv()

                    if rawmessage:
                        buffer.append(rawmessage)
                    try:
                        if not self.router.getsockopt(self.zmq.RCVMORE):
                            self.handle_recv(buffer[:])
                            buffer = []
                            #checktime = time.time()

                    except INVALID_CONTROL_FRAME:
                        log.error('Invalid frame', rawmessage)
                        pass

                if tic - checktime > self.period:
                    log.info("heartbeat loop timedout: %s" % (tic - checktime))
                    log.info(repr(self.responses))
                    break

            # ================
            # Heartbeat Stats
            # ================

            complete = self.beat()

            # ================
            # Topology Status
            # ================

            # Has the entire topology told us its DONE
            done = len(self.finished) == len(self.topology)

            # Has the entire topology shown up to the party
            complete = len(self.tracked) == len(self.topology)

            if complete:
                self.send_go()

                # If we're running in development stop here
                # because our responsibilites are over. The
                # zipline will either run to completion or die,
                # monitor doesn't care anymore because its all
                # threads.

                if self.devel:
                    log.warn("Shutting down Controller because in devel mode")
                    #sys.exitfunc = lambda: None
                    self.shutdown(soft=True)

            log.info('Heartbeat (%s, %s)' % (done, complete))

            # ================
            # Exit Strategies
            # ================

            if self.zmq_flavor == 'green':
                gevent.sleep(0)

            # Will also fall out of loop when done, if using
            # non-freeform topology
            if done:
                log.info('Entire topology exited cleanly')
                self.shutdown(soft=True)

                # Noop exit func
                #sys.exitfunc = lambda: None

                # Send SIGHUP to buritto
                self.signal_hangup()

            if not self.alive:
                log.info('Breaking out of Monitor Loop')
                break

    def signal_hangup(self):
        """
        A clean exit, inform the burrito ( and arbiter ) that
        we're good. The topology exited cleanly and we can prove
        it.
        """
        if not self.nosignals:
            ppid = os.getppid()
            log.warning("Sending SIGHUP")
            os.kill(ppid, SIGHUP)
        else:
            log.warning("Would SIGHUP here, but disabled")

    def signal_interrupt(self):
        """
        Send a SIGINT in the error mode that the monitor's
        interpreter exits.  If the monitor dies the system is
        considered a failure.
        """
        if not self.nosignals:
            ppid = os.getpid()
            os.kill(ppid, SIGINT)
        else:
            log.warning("Would SIGINT here, but disabled")

    def beat(self):
        """
        The tracking logic of the system. It's the "stethoscope"
        that inspects to the heartbeats in a generation and
        infers the state of the system from the responses.
        """

        # These the set overloaded operations
        # A & B ~ set.intersection
        # A - B ~ set.difference

        # * good - Components we are currently tracking and who just sent
        #          us back the right response.
        # * bad - Components we are currently tracking but who did not
        #         send us back a response.
        # * new - Components we haven't heard from yet, but sent back the
        #         right response.
        # * finished - Components we were tracking but have now
        #         finished, when this set goes to zero this
        #         triggers the end of the topology.

        good = self.tracked   & self.responses
        bad  = self.tracked   - good - self.finished
        new  = self.responses - good - self.finished

        missing = self.topology - self.tracked - self.finished

        for component in new:
            self.new(component)

            if self.debug:
                log.info('New component %r' % component)

        for component in bad:
            self.fail(component)


        for component in missing:

            if self.debug:
                log.info('Missing component %r' % component)

        if self.debug:

            for component in self.tracked:
                if component not in self.topology:
                    log.info('Uninvited component %r' % component)

    # --------------
    # Init Handlers
    # --------------

    def new_universal(self):
        pass

    # The various "states of being that a component can inform us
    # of
    def new(self, component):
        if self.state is CONTROL_STATES.TERMINATE:
            return

        if component in self.finished:
            #log.info("Got heartbeat from supposedly finished component")
            return

        log.info('Now Tracking "%s" ' % component)

        universal = self.new_universal
        init_handlers = {}

        if component in (self.topology - self.finished) or self.freeform:
            init_handlers.get(component, universal)()
            self.tracked.add(component)
        else:
            # Some sort of socket collision has occurred, this is
            # a very bad failure mode.
            raise UnknownChatter(component)

    # ------------------
    # Epic Fail Handling
    # ------------------

    def fail_universal(self):
        # TODO: this requires higher order functionality
        log.error('System in exception state, shutting down')
        self.shutdown(soft=True)

    def fail(self, component):
        if self.state is CONTROL_STATES.TERMINATE:
            return

        universal = self.fail_universal
        fail_handlers = { }

        if component in (self.topology - self.finished) or self.freeform:
            log.warning('Component "%s" missed heartbeat' % component)
            self.tracked.remove(component)
            fail_handlers.get(component, universal)()

    # -------------------
    # Completion Handling
    # -------------------

    def done(self, component):
        self.finished.add(component)
        self.tracked.discard(component)
        log.info('Component "%s" finished.' % component)

    # --------------
    # Error Handling
    # --------------

    def exception_universal(self):
        """
        Shutdown the system on failure.
        """
        log.error('System in exception state, shutting down')
        self.shutdown(soft=True)

    def exception(self, component, failure):
        universal = self.exception_universal
        exception_handlers = { }

        if component in self.topology or self.freeform:
            self.error_replay[(component, time.time())] = failure
            log.error('Component in exception state: %s' % component)
            log.error(str(failure))

            exception_handlers.get(component, universal)()
        else:
            raise UnknownChatter(component)

    # -----------------
    # Protocol Handling
    # -----------------

    def handle_recv(self, msg):
        """
        Check for proper framing at the transport layer.
        Seperates the proper frames from anything else that might
        be coming over the wire.
        """

        identity = msg[0] # identity of the socket
        id, status = CONTROL_UNFRAME(msg[1])

        # I'm alive, condemned to be a free process in the cold
        # cold dark absurd Zipline universe.
        if id is CONTROL_PROTOCOL.READY:
            self.responses.add(identity)
            return

        # The heartbeat love song between a component and the
        # controller
        if id is CONTROL_PROTOCOL.OK:

            if status == str(self.ctime):
                # Go to your bosom; knock there, and ask your heart what
                # it doth know...
                self.responses.add(identity)
            elif float(status) < self.ctime:
                # False face must hide what the false heart doth know.
                log.warning('Delayed heartbeat received: %s' % msg)
            elif float(status) > self.ctime:
                # Pre-emptive heartbeat from the component
                # log.info("pre-emptive pong: %s" % msg)
                self.responses.add(identity)
            else:
                # Otherwise its something weird and we don't know
                # what to do so just say so, probably line noise
                # from ZeroMQ

                # What's in a name? that which we call a rose...
                log.error("Weird heartbeat packet happened: %s" % msg)
            return

        # A component is telling us it failed, and how
        if id is CONTROL_PROTOCOL.EXCEPTION:
            self.exception(identity, status)
            return

        # A component is telling us its done with work and won't
        # be talking to us anymore
        if id is CONTROL_PROTOCOL.DONE:
            self.done(identity)
            return

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

    def do_error_replay(self):
        for (component, time), error in self.error_replay.iteritems():
            log.info('Component Log for -- %s --:\n%s' % (component, error))

    def shutdown(self, hard=False, soft=True):

        assert hard or soft, """ Must specify kill hard or soft """

        if self.state is CONTROL_STATES.TERMINATE:
            return

        self.alive = False

        if hard and not self.devel:
            self.state = CONTROL_STATES.TERMINATE
            log.info('Hard Shutdown')

        if soft and not self.devel:
            self.state = CONTROL_STATES.TERMINATE
            log.info('Soft Shutdown')
            self.send_softkill()
