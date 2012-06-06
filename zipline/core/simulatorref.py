"""

The reference simulator for all of Quantopian infastructure.

If a subclass does not conform to the API it will fail at
compiletime.

Subclasses:

    - (partial) zipline.devsimulator.Simulator
    - ( full ) qexec.executor.simulator.ProcessSimulator
    - ( full ) qexec.executor.simulator.ThreadSimulator
    - ( full ) qexec.executor.simulator.GreenletSimulator

"""

import abc
from zipline.core.host import ComponentHost

class SimulatorBase(ComponentHost):

    __metaclass__ = abc.ABCMeta

    def __init__(self, addresses):
        """
        Initailizes the simulator.
        """
        ComponentHost.__init__(self, addresses)

    @abc.abstractproperty
    def get_id(self):
        """Human readable name of the simulator."""
        return "Reference Simulator"

    @abc.abstractmethod
    def launch_component(self, component):
        """ Launch an indvidiaul component in the simulation. """
        raise NotImplementedError

    @abc.abstractmethod
    def launch_controller(self):
        """ Launch the controller for the simulation. """
        raise NotImplementedError

    @abc.abstractmethod
    def simulate(self):
        """ Run a simulation. """
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self):
        """ Normal shutdown procedure. """
        raise NotImplementedError

    def cancel(self):
        """ Soft shutdown """
        self.controller.shutdown(soft=True)

    def kill(self):
        """ Hard shutdown """
        self.controller.shutdown(hard=True)

    # Extension Methods
    # -----------------
    # Provided by some simulators, those that do not will degrade
    # gracefully.

    # - ``did_clean_shutdown``
    # - ``point_of_failure``
    # - ``launch_debugger``

    def did_clean_shutdown(self):
        """
        Returns True if all the subcomponents in the simulation yielded
        cleanly.
        """
        return False

    def point_of_failure(self):
        """ Returns the point of failure of the code.  """
        failures = [
            c for c in self._components.values()
            if c.exception
        ]

        # Sort by failure time so we can follow the failure
        # through the system.
        return sorted(failures, key=lambda c: c.fail_time)

    def launch_debugger(self):
        """
        Launches a remote debug shell in the context of the failed component.
        """
        pass
