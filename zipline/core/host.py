import logging
import datetime

from component import Component

from zipline.transforms import BaseTransform
from zipline.components import Feed, Merge, PassthroughTransform, \
    DataSource
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_STATE

LOGGER = logging.getLogger('ZiplineLogger')

class ComponentHost(Component):
    """
    Components that can launch multiple sub-components, synchronize their
    start, and then wait for all components to be finished.
    """

    def __init__(self, addresses):
        Component.__init__(self)
        self.addresses     = addresses
        self.running       = False

        self.init()

    def init(self):
        assert hasattr(self, 'zmq_flavor'), """
        You must specify a flavor of ZeroMQ for all
        ComponentHost subclasses. """

        # Component Registry, keyed by get_id
        # ----------------------
        self.components     = {}
        # ----------------------
        # Internal Registry, keyed by guid
        self._components     = {}
        # ----------------------

        self.sync_register  = {}
        self.timeout        = datetime.timedelta(seconds=60)

        self.feed           = Feed()
        self.merge          = Merge()
        self.passthrough    = PassthroughTransform()
        self.controller     = None

        #register the feed and the merge
        self.register_components([self.feed, self.merge, self.passthrough])

    def register_controller(self, controller):
        """
        Add the given components to the registry. Establish
        communication with them.
        """
        if self.controller != None:
            raise Exception("There can be only one!")

        self.controller = controller
        self.controller.zmq_flavor = self.zmq_flavor

        # Propogate the controller to all the subcomponents
        for component in self.components.itervalues():
            component.controller = controller

    def register_components(self, components):
        """
        Add the given components to the registry. Establish
        communication with them.
        """
        assert isinstance(components, list)
        for component in components:

            component.addresses     = self.addresses
            component.controller    = self.controller

            # Hosts share their zmq flavor with hosted components
            component.zmq_flavor    = self.zmq_flavor

            self._components[component.guid] = component
            self.components[component.get_id] = component
            self.sync_register[component.get_id] = datetime.datetime.utcnow()

            if isinstance(component, DataSource):
                self.feed.add_source(component.get_id)
            if isinstance(component, BaseTransform):
                self.merge.add_source(component.get_id)

    def unregister_component(self, component_id):
        del self.components[component_id]
        del self.sync_register[component_id]

    def setup_sync(self):
        """
        Setup the sync socket and poller. ( Bind )
        """
        LOGGER.debug("Connecting sync server.")

        self.sync_socket = self.context.socket(self.zmq.REP)
        self.sync_socket.bind(self.addresses['sync_address'])

        self.sync_poller = self.zmq_poller()
        self.sync_poller.register(self.sync_socket, self.zmq.POLLIN)

        self.sockets.append(self.sync_socket)

    def open(self):
        for component in self.components.values():
            self.launch_component(component)
        self.launch_controller()

    def is_running(self):
        """
        DEPRECATED, left in for compatability for now.
        """

        cur_time = datetime.datetime.utcnow()

        if len(self.components) == 0:
            LOGGER.info("Component register is empty.")
            return False

        return True

    def loop(self, lockstep=True):

        while self.is_running():
            # wait for synchronization request at start, and DONE at end.
            # don't timeout.
            socks = dict(self.sync_poller.poll()) 

            if self.sync_socket in socks and socks[self.sync_socket] == self.zmq.POLLIN:
                msg = self.sync_socket.recv()

                try:
                    parts = msg.split(':')
                    sync_id, status = parts
                except ValueError as exc:
                    self.signal_exception(exc)

                if status == str(CONTROL_PROTOCOL.DONE): # TODO: other way around
                    LOGGER.debug("{id} is DONE".format(id=sync_id))
                    self.unregister_component(sync_id)
                    self.state_flag = COMPONENT_STATE.DONE
                else:
                    self.sync_register[sync_id] = datetime.datetime.utcnow()

                #qutil.LOGGER.info("confirmed {id}".format(id=msg))
                # send synchronization reply
                self.sync_socket.send('ack', self.zmq.NOBLOCK)

    # ------------------
    # Simulation Control
    # ------------------

    def launch_controller(self, controller):
        raise NotImplementedError

    def launch_component(self, component):
        raise NotImplementedError

    def teardown_component(self, component):
        raise NotImplementedError


