from functools import wraps
from signal    import signal

class delayed_signals(object):
    """
    Utility to temporary intercept one or more signals while a function or code
    block is executed, restore their signal handlers at the end of execution,
    and invoke them if the signals were in fact received during execution.

    Can be used either as a decorator or a context manager.

    Pass in an iterable of signals to intercept.
    """

    def handler(self, signum, frame=None):
        self.got.append({'signum': signum, 'frame': frame})

    def __init__(self, signals):
        self.signals = signals
        self.handlers = {}
        self.got = []

    def __enter__(self):
        for signum in self.signals:
            # signal() returns the old signal handler
            self.handlers[signum] = signal(signum, self.handler)

    def __exit__(self, time, value, traceback):
        for signum, handler in self.handlers.items():
            signal(signum, handler)
        for signum, frame in ((i['signum'], i['frame']) for i in self.got):
            self.handlers[signum](signum, frame)

    def __call__(self, fn):
        @wraps(fn)
        def call_fn(*args, **kwargs):
            with self:
                outval = fn(*args, **kwargs)
            return outval
        return call_fn
