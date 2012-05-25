import logging
import uuid

from zipline.utils.logger import configure_logging, tail
from unittest2 import TestCase



class LoggerTestCase(TestCase):

    def setUp(self):
       configure_logging()
       self.LOG = logging.getLogger("ZiplineLogger")

    def test_log(self):
        test_msg = uuid.uuid1().hex
        self.LOG.info(test_msg)
        logfile = open('/var/log/zipline/zipline.log','r')
        with logfile:
            last_line = tail(logfile, window=1)
            logged_msg = last_line.split(" - ")[1]
            self.assertEqual(test_msg, logged_msg)
