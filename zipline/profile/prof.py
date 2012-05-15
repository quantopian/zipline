"""

Viscosity - Tools for benchmarking ZeroMQ data flow.

"""

import time as timer
import logging
import pycounters
from contextlib import contextmanager, nested
from pycounters import base
from pycounters.shortcuts import frequency, time
from pycounters import shortcuts, reporters, start_auto_reporting, register_reporter
from pycounters import shortcuts,reporters,report_value, output_report, \
counters, register_counter, _reporting_decorator_context_manager

JSONFile = "counters.json"

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

reporter = reporters.JSONFileReporter(output_file=JSONFile)
logreport = reporters.LogReporter(logger)
register_reporter(logreport)
register_reporter(reporter)

class timecontext:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        cntr = base.GLOBAL_REGISTRY.get_counter(self.name, throw=False)
        if not cntr:
            counter = counters.AverageTimeCounter(self.name)
            register_counter(counter)
        self.tic = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            shortcuts.value(self.name, timer.time() - self.tic)

class ttimecontext:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        counter = base.GLOBAL_REGISTRY.get_counter(self.name, throw=False)

        if not counter:
            counter = counters.EventCounter(self.name)
            counter.value = 0
            register_counter(counter)

        self.counter = counter
        self.tic = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            val = (timer.time() - self.tic)
            if not self.counter.value:
                self.counter.value = long(0.0)
            self.counter.value += val

class occurancecontext:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        cntr = base.GLOBAL_REGISTRY.get_counter(self.name, throw=False)
        if not cntr:
            cntr = counters.TotalCounter(self.name)
            counter = counters.TotalCounter(self.name)
            register_counter(counter)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shortcuts.value(self.name, 1)

if __name__ == '__main__':

    with timecontext('average time'):
        for i in xrange(5):
            x = [2] * 1000
            timer.sleep(0.01)

    with occurancecontext('totalcount'):
        for i in xrange(5):
            x = [2] * 1000

    with ttimecontext('total time'):
        for i in xrange(5):
            x = [2] * 1000
            timer.sleep(1)

    pycounters.output_report()
