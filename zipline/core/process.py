"""
The process simulator. Each component in a separate
multiprocessing.process.
"""

import logbook
import multiprocessing
from zipline.core.host import ComponentHost

log = logbook.Logger('Process Simulator')

class ProcessSimulator(ComponentHost):
    """
    The process simulator.
    """

    zmq_flavor = 'mp'

    def __init__(self, addresses):
        ComponentHost.__init__(self, addresses)
        self.subprocesses = []
        self.running = False
        self.mapping = {}

    def define(self, key, val):
        """
        Returns the mapping between a component and its
        pid.
        """
        self.mapping[key] = val

    @property
    def get_id(self):
        return 'Multiprocess Simulator'

    # =========
    # Launchers
    # =========
    #
    # invoked by the host's open()

    def launch_controller(self):
        proc = multiprocessing.Process(target=self.monitor.run)
        proc.start()
        self.con = proc

        # Process specific
        self.monitor_process = proc
        self.mapping[proc.pid] = 'Monitor'

    def launch_component(self, component):
        proc = multiprocessing.Process(target=component.run)
        proc.start()
        self.subprocesses.append(proc)

        self.mapping[proc.pid] = component.get_id
        return proc

    def simulate(self):
        """
        Kick off the simulation
        """
        self.run()

    def did_clean_shutdown(self):
        cleanly = not any([s.is_alive() for s in self.subprocesses])
        if not cleanly:
            for process in self.subprocesses:
                if process.is_alive():
                    log.error('Failed to Yield', self.mapping[process.pid])
        return cleanly

    def shutdown(self, ensure_clean=True):
        """
        Shutdown the simulation.
        """
        for component in self.components.itervalues():
            component.shutdown()

        for process in self.subprocesses:
            process.join(timeout=1)
            process.terminate()

        self.monitor.shutdown(soft=True)
        self.running = False

        self.con.terminate()

        if ensure_clean:
            assert self.did_clean_shutdown()
