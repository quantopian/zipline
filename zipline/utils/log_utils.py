import logbook
from contextlib import contextmanager


log = logbook.Logger("LogUtils")

class redirector(object):
    def __init__(self, logger, name):
        self.logger = logger
        self.buffer = bytes()
        self.name = name

    def write(self, line):
        self.buffer += ''.join(['>>> ', line.strip('\n'), '\n'])

    def flush(self, final=False):
        if not self.buffer:
            return
        out_form = """ [{pipe_name}] \n{buffer}""".format(
            pipe_name = self.name,
            buffer    = self.buffer
        )
        self.logger.error(out_form)
        self.buffer = bytes()

class log_redirector(object):
    def __init__(self, logger):
        self.logger = logger

    def write(self, line):
        #Absorb blank lines from print statements.
        if line =='\n':
            return

        else:
            #TODO: add logic to guarantee we made this
            self.logger.info(line.strip('\n'))

    def flush(self, final=False):
        pass

@contextmanager
def stdout_pipe(logger, pipe_name):
    """
    Pipe stdout and stderr into a python logger interface
    """
    import sys
    orig_fds = sys.stdout, sys.stderr

    sys.stderr = redirector(logger, pipe_name)
    sys.stdout = redirector(logger, pipe_name)

    yield
    sys.stderr.flush()
    sys.stdout.flush()
    sys.stdout, sys.stderr = orig_fds

@contextmanager
def stdout_only_pipe(logger, pipe_name):
    """
    Pipes just stdout into a python logger interface
    """
    import sys
    orig_fd = sys.stdout
    sys.stdout = log_redirector(logger)

    yield
    sys.stdout.flush()
    sys.stdout = orig_fd
