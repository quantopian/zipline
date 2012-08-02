import os
import sys
import logbook

from zipline.transforms import BaseTransform
from zipline.components import Feed, Merge, PassthroughTransform, \
    DataSource
from zipline.protocol import CONTROL_PROTOCOL, COMPONENT_STATE

log = logbook.Logger('Topology')

class ComponentHost(object):
    """
    Components that can launch multiple sub-components, synchronize
    their start, and then wait for all components to be finished.
    """

    def __init__(self, addresses):
        self.addresses     = addresses
        self.running       = False

        # Component Registry, keyed by unique string
        # ----------------------
        self.components     = {}
        # ----------------------
        # Internal Registry, keyed by guid
        self._components     = {}
        # ----------------------

        self.exception      = None

        self.feed           = Feed()
        self.merge          = Merge()
        self.passthrough    = PassthroughTransform()
        self.controller     = None

        self.register_components([self.feed, self.merge, self.passthrough])

    def _run(self):
        self.open()

    def run(self, catch_exceptions=True):
        """
        Run the host.
        """
        log.info('===== PARENT PID: %s' % os.getppid())
        log.info('===== PID: %s' % os.getpid())

        self.open()
        #self.shutdown()

    def shutdown(self, ensure_clean=True):
        raise NotImplementedError

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

            if isinstance(component, DataSource):
                self.feed.add_source(component.get_id)
            if isinstance(component, BaseTransform):
                self.merge.add_source(component.get_id)

    def unregister_component(self, component_id):
        del self.components[component_id]

    @property
    def pids(self):
        return [proc.pid for proc in self.subprocesses]

    def open(self):
        assert hasattr(self, 'zmq_flavor'), \
        """ You must specify a flavor of ZeroMQ for all Topology
        subclasses. """


        log.info('== Roll Call ==')
        log.info('Monitor')

        self.launch_controller()

        for component in self.components.itervalues():
            log.info(component)

        log.info('== End Roll Call ==')

        for component in self.components.itervalues():
            self.launch_component(component)


    def is_running(self):
        """
        DEPRECATED, left in for compatability for now.
        """

        if len(self.components) == 0:
            log.info("Component register is empty.")
            return False

        return True

    def ready(self):
        return True

    # ------------------
    # Simulation Control
    # ------------------

    # Overloaded by simulator

    def launch_controller(self, controller):
        raise NotImplementedError

    def launch_component(self, component):
        raise NotImplementedError
