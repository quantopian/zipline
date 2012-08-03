import os
import zmq
import sys
import time
import itertools
import logbook
from setproctitle import setproctitle
from signal import SIGHUP, SIGINT

from collections import OrderedDict, Counter

from zipline.protocol import (
    CONTROL_PROTOCOL,
    CONTROL_FRAME,
    CONTROL_UNFRAME,
    CONTROL_STATES,
    INVALID_CONTROL_FRAME
)

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
        return """Component calling itself "%s" talking on unexpected channel""" % self.named


log = logbook.Logger('Monitor')

# The scalars determining the timing of the monitor behavior for
# the system.

PARAMETERS = ndict(dict(
    # time Monitor will wait for a heartbeat, in seconds
    GENERATIONAL_PERIOD        = 20,
    # time Component will wait for GO and for a heartbeat before
    # timing out.
    MAX_COMPONENT_WAIT         = 25,
    ALLOWED_SKIPPED_HEARTBEATS = 10,
    ALLOWED_INVALID_HEARTBEATS = 3,
    PRESTART_HEARBEATS         = 3,
    SOURCES_START_HEARTBEATS   = 3,
    SYSTEM_TIMEOUT             = 50,
))

class Monitor(object):
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

    def __init__(
        self,
        pub_socket,
        route_socket,
        exception_socket,
        send_sighup=False):

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

        self.pub_socket         = pub_socket
        self.route_socket       = route_socket
        self.exception_socket   = exception_socket

        self.missed_beats = Counter()

        # start with an empty topology
        self.topology = set([])

        self.send_sighup = send_sighup
        if self.send_sighup:
            log.info("Request to send sighup/sigint")


    def init_zmq(self):
        self.zmq        = zmq
        self.context    = self.zmq.Context()
        self.zmq_poller = self.zmq.Poller
        return

    def add_to_topology(self, component_id):
        add = set([component_id, "FORK-" + component_id])
        self.topology.update(add)

    def freeze_topology(self):
        if isinstance(self.topology, frozenset):
            return
        # we've been incrementally adding components.
        # time to freeze.
        self.manage(self.topology)

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
        self.freeze_topology()
        self.running = True
        self.init_zmq()
        setproctitle('Monitor')

        self.state = CONTROL_STATES.INIT

        # TODO: keep the exitfunc? the corresponding override on clean
        # exit is commented out currently.
        #
        # Interpreter SIDE EFFECT
        # -----------------------
        # The last breathe of the interpreter will assume that we've
        # failed unless we specify otherwise.
        log.info('registering exit function')
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

        # -- Exception Out --
        # ===================
        self.ex_out = self.context.socket(self.zmq.PUSH)
        self.ex_out.connect(self.exception_socket)

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

            initializing = len(self.tracked) == 0 and len(self.finished) == 0

            # Wait the responses
            while self.alive:

                socks = dict(poller.poll(0))
                tic = time.time()

                if socks.get(self.router) == self.zmq.POLLIN:
                    rawmessage = self.router.recv()

                    if rawmessage:
                        buffer.append(rawmessage)
                    try:
                        if not self.router.getsockopt(self.zmq.RCVMORE):
                            self.handle_recv(buffer[:])
                            buffer = []

                    except INVALID_CONTROL_FRAME:
                        log.error('Invalid frame', rawmessage)
                        pass

                # We break out of this loop if the time between
                # sending and receiving the heartbeat is more
                # than our poll period.

                if tic - self.ctime > self.period:
                    log.info("heartbeat loop timedout: %s" % (tic - self.ctime))
                    log.info(repr(self.responses))
                    break

                # if this is the first time heartbeating, break
                # out early if we get everything tracked no need
                # to hold out for the full heartbeat.
                if initializing and not self.freeform:
                    if len(self.responses) == len(self.topology):
                        log.info("breaking out of initial heartbeat")
                        break

                # Has the entire topology told us its DONE
                done = len(self.finished) == len(self.topology)
                if done:
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

            log.info('Heartbeat (%s, %s)' % (done, complete))

            # ================
            # Exit Strategies
            # ================

            # Will also fall out of loop when done, if using
            # non-freeform topology
            if done:
                log.info('Entire topology exited cleanly')
                self.shutdown()

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
        if not self.send_sighup:
            log.warning("Skipping SIGHUP")
            return
        ppid = os.getppid()
        log.warning("Sending SIGHUP")
        os.kill(ppid, SIGHUP)

    def signal_interrupt(self):
        """
        Send a SIGINT in the error mode that the monitor's
        interpreter exits.  If the monitor dies the system is
        considered a failure.
        """
        if not self.send_sighup:
            log.warning("Skipping SIGINT")
            return
        ppid = os.getpid()
        log.warning("Sending SIGINT")
        os.kill(ppid, SIGINT)

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
            self.timed_out(component)

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

    def timed_out(self, component):
        if self.state is CONTROL_STATES.TERMINATE:
            return

        if component in (self.topology - self.finished) or self.freeform:
            log.warning('Component "%s" missed heartbeat' % component)
            # we treat a time out as a severe failure, and
            # conduct a rapid shutdown
            self.kill()


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
    def exception(self, component, exception_data):
        log.error('Component in exception state: %s. Shutting down system and sending exception data to listeners.'\
            % component)
        # Send the exception message out to listeners.
        self.ex_out.send(exception_data)
        # An exception in one component is treated as a hard
        # failure, and we conduct a rapid shutdown.
        self.kill()

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
            # status should be a msgpack emitted from
            # EXCEPTION_FRAME
            try:
                exception_data = status
                self.exception(identity, exception_data)
            except:
                # if an exception occurs when we try to handle
                # the exception, signal the parent that we need
                # to go down
                # TODO: should we attempt to call self.exception?
                log.exception("Unexpected exception sending exception data")
                self.kill()

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

    def kill(self):
        """Aggressively exit the whole zipline.
        """
        if self.state is CONTROL_STATES.TERMINATE:
            return

        log.info('Hard Shutdown')
        self.send_hardkill()
        self.state = CONTROL_STATES.TERMINATE
        self.alive = False
        # send burrito an interrupt, instructing it to kill all
        # child processes assocated with this zipline.
        time.sleep(3)
        self.signal_interrupt()

    def shutdown(self):

        if self.state is CONTROL_STATES.TERMINATE:
            return

        log.info('Soft Shutdown')
        self.send_softkill()
        self.state = CONTROL_STATES.TERMINATE
        self.alive = False
