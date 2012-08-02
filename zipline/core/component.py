"""
Contains the base class for all components.
"""

import os
import sys
import uuid
import time
import socket
import logbook
import humanhash
import multiprocessing
from setproctitle import setproctitle
from collections import namedtuple


# pyzmq
import zmq

from zipline.gens.zmqgen import gen_from_poller

from zipline.core.monitor import PARAMETERS

from zipline.protocol import (
    CONTROL_PROTOCOL,
    COMPONENT_STATE,
    CONTROL_FRAME,
    CONTROL_UNFRAME,
    EXCEPTION_FRAME
)


log = logbook.Logger('Component')

class KillSignal(Exception):
    def __init__(self):
        pass

ComponentSocketArgs = namedtuple('ComponentSocketArgs',['uri','style','bind'])

class Component(object):

    # ------------
    # Construction
    # ------------

    def __init__(self,
            generator,
            monitor,
            socket_uri,
            frame,
            unframe
            ):

        # -----------------
        # Generator
        # -----------------
        self.generator              = generator
        self.frame                  = frame
        self.component_id           = self.generator.get_hash()

        # lock for waiting on monitor "GO"
        self.waiting                = None

        # -----------------
        # ZMQ properties
        # -----------------
        self.in_socket_args         = ComponentSocketArgs(
                                        uri = socket_uri,
                                        style = zmq.PULL,
                                        bind = False
                                      )
        self.out_socket_args        = ComponentSocketArgs(
                                        uri = socket_uri,
                                        style = zmq.PUSH,
                                        bind = True
                                      )
        self.zmq                    = None
        self.context                = None
        self.out_socket             = None
        self.in_socket              = None
        self.monitor                = monitor
        self.unframe                = unframe
        self.prefix                 = ""

        # register two components with the monitor
        monitor.add_to_topology(self.component_id)
        monitor.add_to_topology("FORK-"+self.component_id)

        # TODO: state_flag is deprecated, remove
        self.state_flag             = COMPONENT_STATE.OK

        # track time of last ping we received from monitor
        self.last_ping              = None

        # Humanhashes make this way easier to debug because they stick
        # in your mind unlike a 32 byte string of random hex.
        self.guid                   = uuid.uuid4()
        self.huid                   = humanhash.humanize(self.guid.hex)


    # ------------
    # Core Methods
    # ------------


    def _run_out(self):
        """
        The main component loop. This is wrapped inside a
        exception reporting context inside of run.

        The core logic of the all components is run here.
        """
        # The process title so you can watch it in top, ps.
        setproctitle(self.generator.__class__.__name__)
        self.prefix = "FORK-"

        log.info("Start %r" % self)
        log.info("Pid %s" % os.getpid())
        log.info("Group %s" % os.getpgrp())

        self.open()

        self.signal_ready()
        self.lock_ready()
        self.wait_ready()

        # -----------------------
        # YOU SHALL NOT PASS!!!!!
        # -----------------------
        # ... until the monitor signals GO

        for event in self.generator:
            self.heartbeat()
            event.source_id = self.get_id
            msg = self.frame(event)
            self.out_socket.send(msg)

        self.signal_done()

    def _run_in(self):
        self.open(send=False)
        self.signal_ready()
        self.lock_ready()
        self.wait_ready()
         # -----------------------
        # YOU SHALL NOT PASS!!!!!
        # -----------------------
        # ... until the monitor signals GO

        # return the generator
        for event in gen_from_poller(self.poll, self.in_socket, self.unframe):
            event.source_id = self.get_id
            yield event

        self.signal_done()

    def run_safe(self, func):
        """
        Run a function that is assumed to include wait_ready and
        heartbeat. Used to wrap fork_generator and consume_gen.
        """
        try:
            return func()
        except Exception as exc:
            if not isinstance(exc, KillSignal):
                self.signal_exception(exc)
            else:
                # if we get a kill signal, forcibly close all the
                # sockets.
                self.teardown_sockets()
        finally:
            log.info("Exiting %r" % self)


    def _launch(self):
        # first, start the generator in its own process. Once
        # Monitor says "go", Events from the generator will be
        # FRAME'd and PUSH'd to self.socket_uri.
        proc = multiprocessing.Process(
                    target=self.run_safe,
                    args=(self._run_out,)
                )
        proc.start()

        # Start the poller-generator, which will PULL messages
        # from self.sockiet_uri, UNFRAME'd them, and yield them.
        return self.run_safe(self._run_in)

    def __iter__(self):
        return self._launch()

    # ----------------------------
    #  Cleanup & Modes of Failure
    # ----------------------------

    def teardown_sockets(self):
        """
        Close all zmq sockets safely. This is universal, no matter where
        this is running it will need the sockets closed.
        """
        log.warn("{id} closing all sockets".format(id=self.get_id))
        #close all the sockets
        for sock in self.sockets:
            sock.close()

    def kill(self):
        """
        Unclean shutdown.

        Tear down ( fast ) as a mode of failure in the simulation or on
        service halt.
        """
        raise KillSignal()

    def signal_exception(self, exc=None, scope=None):
        """
        All exceptions inside any component should boil back to
        this handler.

        Will inform the system that the component has failed and how it
        has failed.
        """
        self.state_flag = COMPONENT_STATE.EXCEPTION
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # if a downstream component fails, this component may try
        # sending when there are zero connections to the socket,
        # which will raise ZMQError(EAGAIN). So, it doesn't make
        # sense to relay this exception to Monitor and the rest
        # of the zipline.
        if isinstance(exc, zmq.ZMQError) and exc.errno == zmq.EAGAIN:
            log.warn("{id} raised a ZMQError(EAGAIN) not relaying"\
                    .format(id=self.get_id))
            return

        # sys.stdout.write(trace)
        log.exception("Unexpected error in run for {id}.".format(id=self.get_id))

        try:
            log.info('{id} sending exception to monitor'\
                .format(id=self.get_id))
            msg = EXCEPTION_FRAME(
                    exc_traceback,
                    exc_type.__name__,
                    exc_value.message
                )

            exception_frame = CONTROL_FRAME(
                CONTROL_PROTOCOL.EXCEPTION,
                msg
            )
            self.control_out.send(exception_frame, self.zmq.NOBLOCK)
            # The monitor should relay the exception back
            # to all zipline components. Wait here until the
            # notice arrives, and we can assume other zipline
            # components have broken out of their message
            # loops.
            for i in xrange(PARAMETERS.MAX_COMPONENT_WAIT):
                self.heartbeat(timeout=1000)
                log.warn("{id} never heard back from monitor."\
                        .format(id=self.get_id))

        except KillSignal:
            log.info("{id} received confirmation from monitor"\
                        .format(id=self.get_id))
        except:
            log.exception("Exception waiting for monitor reply")



    # ----------------------
    #  Internal Maintenance
    # ----------------------

    def lock_ready(self):
        """
        Unlock the component, topology is now ready to run.
        """
        self.waiting = True

    def unlock_ready(self):
        """
        Unlock the component, topology is still pending.
        """
        self.waiting = False

    def wait_ready(self):
        # Implicit side-effect of unlocking the component iff
        # the GO message is received from the monitor level.
        # This then unlocks the barrier and proceeds to the
        # do_work state.

        # Poll on a subset of the control protocol while we exist
        # in the locked quasimode. Respond to HEARTBEAT and GO
        # messages.

        start_wait = time.time()

        while self.waiting:
            socks = dict(self.poll.poll(0))

            assert self.control_in, \
                    'Component does not have a control_in socket'

            if socks.get(self.control_in) == zmq.POLLIN:

                msg = self.control_in.recv()
                event, payload = CONTROL_UNFRAME(msg)

                # ====
                #  Go
                # ====

                # A distributed lock from the monitor to ensure
                # synchronized start.

                if event == CONTROL_PROTOCOL.HEARTBEAT:
                    heartbeat_frame = CONTROL_FRAME(
                        CONTROL_PROTOCOL.OK,
                        payload
                    )
                    self.control_out.send(heartbeat_frame)
                    log.info('Prestart Heartbeat ' + self.get_id)

                elif event == CONTROL_PROTOCOL.GO:
                    # Side effectful call from the monitor to unlock
                    # and begin doing work only when the entire topology
                    # of the system beings to come online
                    log.info('Unlocking ' + self.__class__.__name__)
                    self.unlock_ready()

                # =========
                # Soft Kill
                # =========

                # Try and clean up properly and send out any reports or
                # data that are done during a clean shutdown. Inform the
                # monitor that we're done.
                elif event == CONTROL_PROTOCOL.SHUTDOWN:
                    self.signal_done()
                    break

                # =========
                # Hard Kill
                # =========

                # Just exit.
                elif event == CONTROL_PROTOCOL.KILL:
                    self.kill()
                    break

            elif time.time() - start_wait > PARAMETERS.MAX_COMPONENT_WAIT:
                log.info('No go signal from monitor, %s exiting' \
                    % self.__class__.__name__)
                self.kill()
                break

    def heartbeat(self, timeout=0):
        # wait for synchronization reply from the host
        socks = dict(self.poll.poll(timeout))

        # ----------------
        # Control Dispatch
        # ----------------
        assert self.control_in, 'Component does not have a control_in socket'

        if socks.get(self.control_in) == zmq.POLLIN:
            msg = self.control_in.recv()
            event, payload = CONTROL_UNFRAME(msg)

            # ===========
            #  Heartbeat
            # ===========

            # The monitor will send out a single number packed in
            # a CONTROL_FRAME with ``heartbeat`` event every
            # (n)-seconds. The component then has n seconds to
            # respond to it. If not then it will be considered as
            # malfunctioning or maybe CPU bound.

            if event == CONTROL_PROTOCOL.HEARTBEAT:
                # Heart outgoing
                heartbeat_frame = CONTROL_FRAME(
                    CONTROL_PROTOCOL.OK,
                    payload
                )

                self.last_ping = float(payload)
                # Echo back the heartbeat identifier to tell the
                # monitor that this component is still alive and
                # doing work
                self.control_out.send(heartbeat_frame)


            # =========
            # Soft Kill
            # =========

            # Try and clean up properly and send out any reports or
            # data that are done during a clean shutdown. Inform the
            # monitor that we're done.
            elif event == CONTROL_PROTOCOL.SHUTDOWN:
                self.signal_done()

            # =========
            # Hard Kill
            # =========

            # Just exit.
            elif event == CONTROL_PROTOCOL.KILL:
                self.kill()

        # In case we didn't receive a ping, send a pre-emptive
        # pong to the monitor.
        elif self.last_ping and time.time() - self.last_ping > 1:
            # send a ping ahead of schedule
            pre_pong = time.time()
            heartbeat_frame = CONTROL_FRAME(
                    CONTROL_PROTOCOL.OK,
                    str(pre_pong)
                )

            # Echo back the heartbeat identifier to tell the
            # monitor that this component is still alive and
            # doing work
            self.control_out.send(heartbeat_frame, self.zmq.NOBLOCK)
            self.last_ping = pre_pong
        elif self.last_ping and \
                time.time() - self.last_ping > PARAMETERS.MAX_COMPONENT_WAIT:
            # monitor is gone without sending the shutdown
            # signal, do a hard exit.
            self.kill()


    def signal_ready(self):
        log.info(self.__class__.__name__ + ' is ready')
        frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.READY,
            ''
        )
        self.control_out.send(frame)

    def signal_done(self):
        """
        Notify down stream components that we're done.
        """

        self.state_flag = COMPONENT_STATE.DONE
        # notify internal work loop that we're done
        self.done = True # TODO: use state flag

        if self.out_socket:
            msg = zmq.Message(str(CONTROL_PROTOCOL.DONE))
            self.out_socket.send(msg)


        # notify monitor we're done
        done_frame = CONTROL_FRAME(
            CONTROL_PROTOCOL.DONE,
            ''
        )

        self.control_out.send(done_frame)
        log.info("[%s] sent control done" % self.get_id)

        # there is a narrow race condition where we finish just
        # after the Monitor accepts our prior heartbeat, but just
        # before the next one is sent. So, we hang around for one
        # last heartbeat, and wait an unusually long time.
        # TODO: decided if this is really necessary.
        # self.heartbeat(timeout=5000)

    # -----------
    #  Messaging
    # -----------

    def open(self, send=True):
        """
        Open the connections needed to start doing work.
        Perform any setup that must be done within process.
        """
        self.sockets    = []
        self.zmq = zmq
        self.context = self.zmq.Context()
        self.poll = self.zmq.Poller()

        self.setup_control()

        if send:
            self.out_socket = self.open_socket(self.out_socket_args)
            self.sockets.extend([self.out_socket])
        else:
            self.in_socket = self.open_socket(self.in_socket_args)
            self.sockets.extend([self.in_socket])

    def open_socket(self, sock_args):
        if sock_args.bind:
            return self.bind_socket(sock_args)
        else:
            return self.connect_socket(sock_args)

    def bind_socket(self, sock_args):
        if sock_args.style == zmq.PULL:
            return self.bind_pull_socket(sock_args.uri)
        if sock_args.style == zmq.PUSH:
            return self.bind_push_socket(sock_args.uri)
        if sock_args.style == zmq.PUB:
            return self.bind_pub_socket(sock_args.uri)

        raise Exception("Invalid socket arguments")

    def connect_socket(self, sock_args):
        if sock_args.style == zmq.PULL:
            return self.connect_pull_socket(sock_args.uri)
        if sock_args.style == zmq.PUSH:
            return self.connect_push_socket(sock_args.uri)
        if sock_args.style == zmq.SUB:
            return self.connect_sub_socket(sock_args.uri)

        raise Exception("Invalid socket arguments")

    def bind_push_socket(self, addr):
        push_socket = self.context.socket(self.zmq.PUSH)
        push_socket.bind(addr)
        self.out_socket = push_socket
        self.sockets.append(push_socket)

        return push_socket

    def connect_pull_socket(self, addr):
        pull_socket = self.context.socket(self.zmq.PULL)
        pull_socket.connect(addr)
        self.sockets.append(pull_socket)
        self.poll.register(pull_socket, self.zmq.POLLIN)

        return pull_socket


    def bind_pull_socket(self, addr):
        pull_socket = self.context.socket(self.zmq.PULL)
        pull_socket.bind(addr)
        self.poll.register(pull_socket, self.zmq.POLLIN)

        self.sockets.append(pull_socket)

        return pull_socket

    def connect_push_socket(self, addr):
        push_socket = self.context.socket(self.zmq.PUSH)
        push_socket.connect(addr)
        self.sockets.append(push_socket)
        self.out_socket = push_socket

        return push_socket

    def bind_pub_socket(self, addr):
        pub_socket = self.context.socket(self.zmq.PUB)
        pub_socket.bind(addr)
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
        self.control_out = self.monitor.message_sender(
            identity = self.get_id,
            context  = self.context,
        )

        self.control_in = self.monitor.message_listener(
            context = self.context
        )

        self.poll.register(self.control_in, self.zmq.POLLIN)
        self.sockets.extend([self.control_in, self.control_out])

    # ---------------------
    # Description and Debug
    # ---------------------

    @property
    def get_id(self):
        """
        The time invariant name for this component.
        Must be unique within this zipline.
        """
        return self.prefix + self.component_id

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
            'successful' : self.ready()         ,
        }

    def __repr__(self):
        """
        Return a useful string representation of the component to
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
