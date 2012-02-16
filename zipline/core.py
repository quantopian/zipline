"""
Provides simulated data feed services...
"""
import multiprocessing
from gevent_zeromq import zmq
import json
import copy
import threading
import datetime
import atexit

import zipline.util as qutil
import zipline.messaging as qmsg

class SimulatorBase(ComponentHost):
    """
    Simulator coordinates the launch and communication of source, feed, transform, and merge components.
    """
    
    def __init__(self, addresses):
        """
        """                               
        self.feed                   = ParallelBuffer()
        self.merge                  = MergedParallelBuffer()
        
        #workaround for defect in threaded use of strptime: http://bugs.python.org/issue11108
        qutil.parse_date("2012/02/13-10:04:28.114")
        
        #register the feed and the merge
        self.register_components([self.feed, self.merge])
                 
    def simulate(self):
        self.launch_component(self)
        
class ThreadSimulator(SimulatorBase):
    
    def __init__(self, sources, transforms, client, feed=None, merge=None):
        SimulatorBase.__init__(self, sources, transforms, client, feed, merge)
        
    def launch_component(self, name, component):
        qutil.LOGGER.info("starting {name}".format(name=name))
        thread = threading.Thread(target=component.run)
        thread.start()
        return thread
    
class ProcessSimulator(SimulatorBase):        
    
    def __init__(self, sources, transforms, client, feed=None, merge=None):
        SimulatorBase.__init__(self, sources, transforms, client, feed, merge)
    
    def launch_component(self, name, component):
        qutil.LOGGER.info("starting {name}".format(name=name))
        proc = multiprocessing.Process(target=component.run)
        proc.start()
        return proc

