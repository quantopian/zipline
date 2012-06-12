"""
Contains the base class for all components.
"""

import os
import sys
import uuid
import time
import socket
import logbook
import traceback
import humanhash
from setproctitle import setproctitle

# pyzmq
import zmq
# gevent_zeromq
import gevent_zeromq
# zmq_ctypes
#import zmq_ctypes

from zipline.utils.gpoll import _Poller as GeventPoller
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_STATE, \
    COMPONENT_FAILURE, CONTROL_FRAME

log = logbook.Logger('Component')

from zipline.exceptions import ComponentNoInit
from zipline.transitions import WorkflowMeta

# LOGBOOK - embed PID in log output

class Component(object):

    """
    Base class for components. Defines the the base messaging
    interface for components.

    :param addresses: a dict of name_string -> zmq port address strings.
                      Must have the following entries

    :param sync_address: socket address used for synchronizing the start of
                         all workers, heartbeating, and exit notification
                         will be used in REP/REQ sockets. Bind is always on
                         the REP side.

    :param data_address: socket address used for data sources to stream
                         their records. Will be used in PUSH/PULL sockets
                         between data sources and a Feed. Bind will always
                         be on the PULL side (we always have N producers and
                         1 consumer)

    :param feed_address: socket address used to publish consolidated feed
                         from serialization of data sources
                         will be used in PUB/SUB sockets between Feed and
                         Transforms. Bind is always on the PUB side.

    :param merge_address: socket address used to publish transformed
                          values.  will be used in PUSH/PULL from many
                          transforms to one Merge Bind will always be on
                          the PULL side (we always have N producers and
                          1 consumer)

    :param result_address: socket address used to publish merged data
                           source feed and transforms to clients will be
                           used in PUB/SUB from one Merge to one or many
                           clients. Bind is always on the PUB side.

    bind/connect methods will return the correct socket type for each
    address.

    """

    # ------------
    # Construction
    # ------------

    abstract = True
    #__metaclass__ = WorkflowMeta

    def __init__(self, *args, **kwargs):
        self.zmq               = None
        self.context           = None
        self.addresses         = None

        self.out_socket        = None
        self.killed            = False
        self.controller        = None
        # timeout after a full minute
        self.heartbeat_timeout = 60 *1000
        self.state_flag        = COMPONENT_STATE.OK
        self.error_state       = COMPONENT_FAILURE.NOFAILURE
        self.on_done           = None

        self._exception        = None
        self.fail_time         = None
        self.start_tic         = None
        self.stop_tic          = None
        self.note              = None
        self.confirmed         = False

        # Humanhashes make this way easier to debug because they stick
        # in your mind unlike a 32 byte string of random hex.
        self.guid = uuid.uuid4()
        self.huid = humanhash.humanize(self.guid.hex)

        # This is where component specific constructors should be
        # defined. Arguments passed to init are threaded through.
        self.init(*args, **kwargs)

    def init(self):
        """
        Subclasses should override this to extend the setup for the
        class. Shouldn't have side effects.
        """
        raise ComponentNoInit(self.__class__)


    # ------------
    # Core Methods
    # ------------

    def open(self):
        """
        Open the connections needed to start doing work.
        """
        raise NotImplementedError

    def ready(self):
        """
        Return ``True`` if and only if the component has finished
        execution.
        """
        return self.state_flag in [COMPONENT_STATE.DONE, \
            COMPONENT_STATE.EXCEPTION]

    def successful(self):
        """
        Return ``True`` if and only if the component has finished
        execution successfully, that is, without raising an error.
        """
        return self.state_flag == COMPONENT_STATE.DONE and not \
            self.exception

    @property
    def exception(self):
        """
        Holds the exception that the component failed on, or ``None`` if
        the component has not failed.
        """
        return self._exception

    def do_work(self):
        raise NotImplementedError

    def init_zmq(self, flavor):
        """
        ZMQ in all flavors. Have it your way.

            mp     - Distinct contexts | pyzmq
            thread - Same context      | pyzmq
            green  - Same context      | gevent_zeromq
            pypy   - Same context      | zmq_ctypes

        """

        if flavor == 'mp':
            self.zmq = zmq
            self.context = self.zmq.Context()
            self.zmq_poller = self.zmq.Poller
            # The the process title so you can watch it in top
            setproctitle(self.__class__.__name__)
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

        raise Exception("Unknown ZeroMQ Flavor")

    def _run(self):
        self.start_tic = time.time()

        self.done       = False # TODO: use state flag
        self.sockets    = []

        self.init_zmq(self.zmq_flavor)

        self.setup_poller()

        self.open()
        self.setup_sync()
        self.setup_control()

        self.loop()
        self.shutdown()

        self.stop_tic = time.time()

    def run(self, catch_exceptions=True):
        """
        Run the component.

        Optionally takes an argument to catch and log all exceptions
        raised during execution ues this with care since it makes it
        very hard to debug since it mucks up your stacktraces.
        """

        if catch_exceptions:
            try:
                self._run()
            except Exception as exc:
                exc_info = sys.exc_info()
                self.signal_exception(exc)

                # Reraise the exception
                raise exc_info[0], exc_info[1], exc_info[2]
            finally:

                self.shutdown()
                self.teardown_sockets()

    def working(self):
        """
        Controls when the work loop will start and end

        If we encounter an exception or signal done exit.

        Overload for higher order behavior.
        """
        return (not self.done)

    def loop(self, lockstep=True):
        """
        Loop to do work while we still have work to do.
        """
        while self.working():
            self.confirm()
            self.do_work()

    def confirm(self):
        """
        Send a synchronization request to the host.
        """
        if not self.confirmed:
            # TODO: proper framing
            self.sync_socket.send(self.get_id + ":RUN")

            self.receive_sync_ack() # blocking
            self.confirmed = True

    def runtime(self):
        if self.ready() and self.start_tic and self.stop_tic:
            return self.stop_tic - self.start_tic

    # ----------------------------
    #  Cleanup & Modes of Failure
    # ----------------------------

    def teardown_sockets(self):
        """
        Close all zmq sockets safely. This is universal, no matter where
        this is running it will need the sockets closed.
        """
        #close all the sockets
        for sock in self.sockets:
            sock.close()

    def shutdown(self):
        """
        Clean shutdown.

        Tear down after normal operation.
        """
        if self.on_done:
            self.on_done()

    def kill(self):
        """
        Unclean shutdown.

        Tear down ( fast ) as a mode of failure in the simulation or on
        service halt.
        """
        raise NotImplementedError

    # ----------------------
    #  Internal Maintenance
    # ----------------------

    def signal_exception(self, exc=None, scope=None):
        """
        This is *very* important error tracking handler.

        Will inform the system that the component has failed and how it
        has failed.
        """

        if scope == 'algo':
            self.error_state = COMPONENT_FAILURE.ALGOEXCEPT
        else:
            self.error_state = COMPONENT_FAILURE.HOSTEXCEPT

        self.state_flag = COMPONENT_STATE.EXCEPTION
        # mark the time of failure so we can track the failure
        # progogation through the system.

        self.stop_tic = time.time()

        self._exception = exc
        exc_type, exc_value, exc_traceback = sys.exc_info()
        trace = '\n>>>'.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        sys.stdout.write(trace)

        if hasattr(self, 'control_out'):
            exception_frame = CONTROL_FRAME(
                CONTROL_PROTOCOL.EXCEPTION,
                trace
            )
            self.control_out.send(exception_frame)

        #LOGGER.exception("Unexpected error in run for {id}.".format(id=self.get_id))

    def signal_done(self):
        """
        Notify down stream components that we're done.
        """

        self.state_flag = COMPONENT_STATE.DONE

        if self.out_socket:
            self.out_socket.send(str(CONTROL_PROTOCOL.DONE))

        #notify host we're done
        # TODO: proper framing
        self.sync_socket.send(self.get_id + ":" + str(CONTROL_PROTOCOL.DONE))

        #notify controller we're done
        done_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.DONE,
            ''
        )
        self.control_out.send(done_frame)

        self.receive_sync_ack()
        #notify internal work look that we're done
        self.done = True # TODO: use state flag

        #log.info("[%s] DONE" % self.get_id)

    # -----------
    #  Messaging
    # -----------

    def setup_poller(self):
        """
        Setup the poller used for multiplexing the incoming data
        handling sockets.
        """

        # Initializes the poller class specified by the flavor of
        # ZeroMQ. Either zmq.Poller or gpoll.Poller .
        self.poll = self.zmq_poller()

    def receive_sync_ack(self):
        """
        Wait for synchronization reply from the host.

        DEPRECATED, left in for compatability for now.
        """

        socks = dict(self.sync_poller.poll(self.heartbeat_timeout))
        if self.sync_socket in socks and socks[self.sync_socket] == self.zmq.POLLIN:
            message = self.sync_socket.recv()
        #else:
            #raise Exception("Sync ack timed out on response for {id}".format(id=self.get_id))

    def bind_data(self):
        return self.bind_pull_socket(self.addresses['data_address'])

    def connect_data(self):
        return self.connect_push_socket(self.addresses['data_address'])

    def bind_feed(self):
        return self.bind_pub_socket(self.addresses['feed_address'])

    def connect_feed(self):
        return self.connect_sub_socket(self.addresses['feed_address'])

    def bind_merge(self):
        return self.bind_pull_socket(self.addresses['merge_address'])

    def connect_merge(self):
        return self.connect_push_socket(self.addresses['merge_address'])

    def bind_result(self):
        return self.bind_pub_socket(self.addresses['result_address'])

    def connect_result(self):
        return self.connect_sub_socket(self.addresses['result_address'])

    def bind_pull_socket(self, addr):
        pull_socket = self.context.socket(self.zmq.PULL)
        pull_socket.bind(addr)
        self.poll.register(pull_socket, self.zmq.POLLIN)

        self.sockets.append(pull_socket)

        return pull_socket

    def connect_push_socket(self, addr):
        push_socket = self.context.socket(self.zmq.PUSH)
        push_socket.connect(addr)
        #push_socket.setsockopt(self.zmq.LINGER,0)
        self.sockets.append(push_socket)
        self.out_socket = push_socket

        return push_socket

    def bind_pub_socket(self, addr):
        pub_socket = self.context.socket(self.zmq.PUB)
        pub_socket.bind(addr)
        #pub_socket.setsockopt(self.zmq.LINGER,0)
        self.out_socket = pub_socket

        return pub_socket

    def connect_sub_socket(self, addr):
        sub_socket = self.context.socket(self.zmq.SUB)
        sub_socket.connect(addr)
        sub_socket.setsockopt(self.zmq.SUBSCRIBE,'')
        self.sockets.append(sub_socket)

        self.poll.register(sub_socket, self.zmq.POLLIN)

        return sub_socket

    def setup_control(self):
        """
        Set up the control socket. Used to monitor the overall status
        of the simulation and to forcefully tear down the simulation in
        case of a failure.
        """

        # Allow for the possibility of not having a controller,
        # possibly the zipline devsimulator may not want this.
        if not self.controller:
            return

        self.control_out = self.controller.message_sender(
            identity = self.get_id,
            context  = self.context,
        )

        self.control_in = self.controller.message_listener(
            context = self.context
        )

        self.poll.register(self.control_in, self.zmq.POLLIN)
        self.sockets.extend([self.control_in, self.control_out])

    def setup_sync(self):
        """
        Setup the sync socket and poller. ( Connect )

        DEPRECATED, left in for compatability for now.
        """

        #LOGGER.debug("Connecting sync client for {id}".format(id=self.get_id))

        self.sync_socket = self.context.socket(self.zmq.REQ)
        self.sync_socket.connect(self.addresses['sync_address'])
        #self.sync_socket.setsockopt(self.zmq.LINGER,0)

        self.sync_poller = self.zmq_poller()
        self.sync_poller.register(self.sync_socket, self.zmq.POLLIN)

        self.sockets.append(self.sync_socket)

    # -----------
    # FSM Actions
    # -----------

    #@property
    #def state(self):
        #if not hasattr(self, '_state'):
            #self._state = self.initial_state
        #else:
            #return self._state

    #@state.setter
    #def state(self, new):
        #if not hasattr(self, '_state'):
            #self._state = self.initial_state

        #old = self._state

        #if (old, new) in self.workflow:
            #self._state = new
        #else:
            #raise RuntimeError("Invalid State Transition : %s -> %s" %(old, new))

    # ---------------------
    # Description and Debug
    # ---------------------

    def extern_logger(self):
        """
        Pipe logs out to a provided logging interface.
        """
        pass

    def setup_extern_logger(self):
        """
        Pipe logs out to a provided logging interface.
        """
        pass

    @property
    def get_id(self):
        """
        The descriptive name of the component.
        """
        # Prevents the bug that Thomas ran into
        raise NotImplementedError

    @property
    def get_type(self):
        """
        The data flow type of the component.

        - ``SOURCE``
        - ``CONDUIT``
        - ``SINK``

        """
        raise NotImplementedError

    @property
    def get_pure(self):
        """
        Describes whehter this component purely functional, i.e. for a
        given set of inputs is it guaranteed to always give the same
        output . Components that are side-effectful are, generally, not
        pure.
        """
        return False

    def debug(self):
        """
        Debug information about the component.
        """
        return {
            'id'         : self.get_id          ,
            'huid'       : self.huid            ,
            'host'       : socket.gethostname() ,
            'pid'        : os.getpid()          ,
            'memaddress' : hex(id(self))        ,
            'ready'      : self.successful()    ,
            'succesfull' : self.ready()         ,
        }

    def __len__(self):
        """
        Some components overload this for debug purposes
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Return a usefull string representation of the component to
        indicate its type, unique identifier, and computational context
        identifier name.
        """

        return "<{name} {uuid} at {host} {pid} {pointer}>".format(
            name    = self.get_id          ,
            uuid    = self.guid            ,
            host    = socket.gethostname() ,
            pid     = os.getpid()          ,
            pointer = hex(id(self))        ,
        )
