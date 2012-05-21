import logging
import uuid

from zipline.utils.logger import configure_logging
from unittest2 import TestCase



class LoggerTestCase(TestCase):

    def setUp(self):
       configure_logging()
       self.LOG = logging.getLogger("ZiplineLogger")

    def test_log(self):
        test_msg = uuid.uuid1().hex
        self.LOG.info(test_msg)
        logfile = open('/var/log/zipline/zipline.log','r')
        last_line = tail(logfile)
        logged_msg = last_line.split(" - ")[1]
        self.assertEqual(test_msg, logged_msg)

def tail( f, window=20 ):
    """
    from
    http://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file- \
            with-python-similar-to-tail
    """
    BUFSIZ = 1024
    f.seek(0, 2)
    bytes = f.tell()
    size = window
    block = -1
    data = []
    while size > 0 and bytes > 0:
        if (bytes - BUFSIZ > 0):
            # Seek back one whole BUFSIZ
            f.seek(block*BUFSIZ, 2)
            # read BUFFER
            data.append(f.read(BUFSIZ))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            data.append(f.read(bytes))
        linesFound = data[-1].count('\n')
        size -= linesFound
        bytes -= BUFSIZ
        block -= 1
    return '\n'.join(''.join(data).splitlines()[-window:])
