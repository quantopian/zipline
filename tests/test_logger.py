import logging
import logbook
import uuid
import zmq

from zipline import ndict

from zipline.utils.logger import configure_logging, tail
from zipline.utils.log_utils import ZeroMQLogHandler

from zipline.utils.test_utils import create_receiver, drain_receiver

from unittest2 import TestCase


class LoggerTestCase(TestCase):

    def setUp(self):
        configure_logging()
        self.LOG = logging.getLogger("ZiplineLogger")

    def test_log(self):
        test_msg = uuid.uuid1().hex
        self.LOG.info(test_msg)
        logfile = open('/var/log/zipline/zipline.log', 'r')
        with logfile:
            last_line = tail(logfile, window=1)
            logged_msg = last_line.split(" - ")[1]
            self.assertEqual(test_msg, logged_msg)

    def test_zmq_handler(self):
        socket_addr = 'tcp://127.0.0.1:10000'
        ctx = zmq.Context()
        socket_push = ctx.socket(zmq.PUSH)
        socket_push.connect(socket_addr)
        recv = create_receiver(socket_addr, ctx)
        zmq_out = ZeroMQLogHandler(
            socket=socket_push,
            filter=lambda r, h: r.channel in ['test zmq logger'],
            context=ctx,
            #bubble=False
        )

        log = logbook.Logger('test zmq logger')
        x = ndict({})
        x.a = 1
        ex = example(133)
        with zmq_out.threadbound():
            log.info(ex.num)

        output, _ = drain_receiver(recv, count=1)
        self.assertEqual(output[-1]['prefix'], 'LOG')
        self.assertTrue(isinstance(output[-1]['payload']['msg'], basestring))


class example(object):

    def __init__(self, num):
        self.num = num
