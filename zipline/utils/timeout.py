import signal

from functools import wraps

from pprint import pprint as pp
from numbers import Number
from logbook import Logger

class TimeoutException(Exception):
    
    def __init__(self, frame, message=''):
        self.frame = frame
        self.message = message

# TODO: fix code replication here.
        
class Timeout(object):
    """
    Utility to make a function raise TimeoutException if it spends
    more than a specified number of seconds executing. Can be used
    as a decorator to apply a static timeout to a function, or as 
    a context manager to dynamically add a timeout to a code block.
    """
    
    def __init__(self, seconds, message=''):
        self.seconds = seconds
        self.message = message
        assert isinstance(seconds, Number), "Failed to specify a timeout."
        assert seconds > 0, "Timeout must be greater than 0"

    def handler(self, signum, frame):
        raise TimeoutException(frame, self.message)

    def __call__(self, fn):
        
        @wraps(fn)
        def call_fn_with_timeout(*args, **kwargs):
            # Set the alarm, saving any handler that existed previously.
            signal.signal(signal.SIGALRM, self.handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds, 0)
            try:
                outval = fn(*args, **kwargs)

            # Deactivate the alarm once we're done so that the
            # decorator doesn't have unexpected side-effects later.
            # Note that this will still raise Timeout if the
            # call to fn takes too long.
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)

            # Return the value of fn if it finished before the alarm.  This
            # won't execute if the Timeout was raised.
            return outval
        return call_fn_with_timeout

    def __enter__(self):
        # Set the alarm on entrance.
        signal.signal(signal.SIGALRM, self.handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds, 0)
            
    def __exit__(self, type, value, traceback):
        # Deactivate the alarm on exit. This will re-raise
        # any exceptions raised inside the with block.
        signal.signal(signal.SIGALRM, self.handler)
        signal.setitimer(signal.ITIMER_REAL, 0, 0)
    
class Heartbeat(object):
    """
    Utility to perform pseudo-heartbeat checks on a single-threaded
    function. Calls frame_handler on the current stack frame of the 
    wrapped function every ``interval`` seconds.  After ``max_interval``
    intervals, raises Timeout.  Can be used either as a decorator or
    a context manager.
    """
    def __init__(self, 
                 interval, 
                 max_intervals, 
                 frame_handler=None, 
                 timeout_message=''):

        self.interval = interval
        self.max_intervals = max_intervals
        self.frame_handler = frame_handler
        self.timeout_message = timeout_message
        self.count = 0
        
    def handler(self, signum, frame):
        self.count += 1
        if self.frame_handler:
            self.frame_handler(self.count, frame)
            
        if self.count >= self.max_intervals:
            raise TimeoutException(frame, self.timeout_message)

    def __call__(self, fn):

        @wraps(fn)
        def call_fn_with_heartbeat(*args, **kwargs):
            # Set a timer to call our handler every ``interval`` seconds. 
            signal.signal(signal.SIGALRM, self.handler)
            signal.setitimer(signal.ITIMER_REAL, self.interval, self.interval)
            try:
                outval = fn(*args, **kwargs)

            finally:
                # Deactivate the timer once we're done so that the
                # decorator doesn't have unexpected side-effects later.
                signal.setitimer(signal.ITIMER_REAL, 0, 0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                self.count = 0

            # Return the value of fn if it finished without tripping
            # an exception.  This won't execute if the Timeout or any
            # other exception was raised by self.handle.
            return outval
        return call_fn_with_heartbeat
    
    def __enter__(self):
        # Set a timer to call our handler every N seconds. 
        self.count = 0
        signal.signal(signal.SIGALRM, self.handler)
        signal.setitimer(signal.ITIMER_REAL, self.interval, self.interval)
        
    def __exit__(self, type, value, traceback):
        # Turn off the timer on exit.  This will re-raise any exception raised
        # during execution of the with-block
        self.count = 0
        signal.setitimer(signal.ITIMER_REAL, 0, 0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
