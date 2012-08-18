import signal

from pprint import pprint as pp
from numbers import Number
from logbook import Logger

class Timeout(Exception):
    
    def __init__(self, frame):
        self.frame = frame
        

class timeout(object):
    """
    Decorator to make a function raise TimeoutException if it spends
    more than a specified number of seconds executing.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        assert isinstance(seconds, Number), "Failed to specify a timeout."
        assert seconds > 0, "Timeout must be greater than 0"

    def handler(self, signum, frame):
        raise Timeout(frame)

    def __call__(self, fn):

        def wrapped(*args, **kwargs):
            # Set the alarm.
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.seconds)
            try:
                outval = fn(*args, **kwargs)

            # Deactivate the alarm once we're done so that the
            # decorator doesn't have unexpected side-effects later.
            # Note that this will still raise TimeoutException if the
            # call to fn takes too long.
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)

            # Return the value of fn if it finished before the alarm.  This
            # won't execute if the Timeout was raised.
            return outval
        return wrapped
    
class heartbeat(object):
    """
    Decorator to perform pseudo-heartbeat checks on a single-threaded
    function. Calls frame_handler on the current stack frame of the 
    decorated function every ``interval`` seconds.  After ``max_interval``
    intervals, raises MaxHeartBeats
    """

    def __init__(self, interval, max_intervals, frame_handler=None):
        self.count = 0
        self.interval = interval
        self.max_intervals = max_intervals
        self.frame_handler = frame_handler
        
    def handler(self, signum, frame):
        self.count += 1
        if self.frame_handler:
            self.frame_handler(frame)
            
        if self.count > self.max_intervals:
            raise Timeout(frame)

    def __call__(self, fn):
        def wrapped(*args, **kwargs):
            # Set a timer to call our handler every N seconds. 
            signal.signal(signal.SIGALRM, self.handler)
            signal.setitimer(signal.ITIMER_REAL, self.interval, self.interval)
            try:
                outval = fn(*args, **kwargs)

            # Deactivate the timer once we're done so that the
            # decorator doesn't have unexpected side-effects later.
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)

            # Return the value of fn if it finished without tripping
            # an exception.  This won't execute if the Timeout or any
            # other exception was raised by self.handle.
            return outval
        return wrapped
   
if __name__ == "__main__":
    import time
    
    def pframe_g(frame):
        print frame.f_globals

    @heartbeat(1, 10, pframe_g)
    def foo():
        for i in xrange(10000):
            time.sleep(.1)
            print i
    foo()
