import logbook
import zmq
import pytz
import datetime

from logbook import NOTSET
from logbook.handlers import Handler, FileHandler

from zipline.protocol import LOG_FRAME, LOG_FIELDS, \
    LOG_EXTRA_FIELDS

from contextlib import contextmanager


log = logbook.Logger("LogUtils")

class redirecter(object):
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

class log_redirecter(object):
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

    sys.stderr = redirecter(logger, pipe_name)
    sys.stdout = redirecter(logger, pipe_name)

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
    sys.stdout = log_redirecter(logger)

    yield
    sys.stdout.flush()
    sys.stdout = orig_fd

class ZeroMQLogHandler(Handler):
    """
    A handler that takes messages captured from the user algorithm stdout
    and transforms them into LOG_FRAMES suitable for database storage.
    Setup is similar to logbook.queues.ZeroMQHandler, except we connect
    instead of binding and we extract record fields into a dict.
    """

    def __init__(self, socket=None, level=NOTSET, filter=None, bubble=False,
                 context=None, fds = LOG_FIELDS, extra_fds = LOG_EXTRA_FIELDS):
        Handler.__init__(self, level, filter, bubble)

        try:
            import zmq
        except ImportError:
            raise RuntimeError('The pyzmq library is required for '
                               'the ZeroMQHandler.')
        #: the zero mq context
        self.context = context
        #: the zero mq socket.
        self.socket = socket #self.context.socket(zmq.PUSH)

        #self.uri = uri
        #if uri is not None:
        #    self.socket.connect(uri)

        self.fds = fds
        self.extra_fds = extra_fds

    def export_record(self, record):
        """
        Extract relevant fields from a log record, fiddling with datetime
        fields to make json happy.
        """
        from zipline.utils.date_utils import EPOCH

        #Needed to extract record info from dictionary.
        record.pull_information()

        #Logbook stores record times as datetime objects, which
        #can't be serialized by JSON, so we need to convert to
        #unix epoch representation.


        #Do the same if algo_dt is a datetime object.
        if record.extra.has_key('algo_dt'):
            algo_dt = record.extra['algo_dt']

            if isinstance(algo_dt, datetime.datetime):
               algo_dt = EPOCH(algo_dt.replace(tzinfo = pytz.utc))
               record.extra['algo_dt'] = algo_dt

        data = {}

        #Extract all the fields we care about from LogRecord's internal
        #dictionary.

        for field in iter(self.fds):
            if record.__dict__.has_key(field):
                data[field] = record.__dict__[field]
            else:
                data[field] = None

        for field in iter(self.extra_fds):
            if record.extra.has_key(field):
                data[field] = record.extra[field]
            else:
                data[field] = None

        if data['time']:
            assert isinstance(data['time'], datetime.datetime)

            time = data['time'].replace(tzinfo = pytz.utc)
            #logbook measures time in utc already, no need to convert.
            data['time'] = EPOCH(time)

        return data

    def emit(self, record):
        """Extract relevant fields and send info as JSON over a zmq socket."""
        payload = self.export_record(record)
        self.socket.send(LOG_FRAME(payload))

    def close(self):
        pass
        #self.socket.close()
