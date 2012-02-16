"""
Provides simulated data feed services...
"""
import multiprocessing
import json
import copy
import threading

import zipline.util as qutil
import zipline.messaging as qmsg

class SimulatorBase(qmsg.ComponentHost):
    """
    Simulator coordinates the launch and communication of source, feed, transform, and merge components.
    """
    
    def __init__(self, addresses, gevent_needed=False):
        """
        """    
        qmsg.ComponentHost.__init__(self, addresses, gevent_needed)
                                   
    def simulate(self):
        self.run()
        
    def get_id(self):
        return "Simulator"
        
class ThreadSimulator(SimulatorBase):
    
    def __init__(self, addresses):
        SimulatorBase.__init__(self, addresses)
        
    def launch_component(self, component):
        qutil.LOGGER.info("starting {name}".format(name=component.get_id()))
        thread = threading.Thread(target=component.run)
        thread.start()
        return thread
    
class ProcessSimulator(SimulatorBase):        
    
    def __init__(self, addresses):
        SimulatorBase.__init__(self, addresses)
        
    def launch_component(self, component):
        qutil.LOGGER.info("starting {name}".format(name=component.get_id()))
        proc = multiprocessing.Process(target=component.run)
        proc.start()
        return proc

