import os
from   signal    import signal, SIGHUP, SIGINT
import time
from   types     import FrameType
import unittest

from zipline.utils.delayed_signals import delayed_signals

class DelayedSignals(unittest.TestCase):
    def handler(self, signum, frame):
        print "Got signal " + str(signum)
        self.got[signum] = time.time()
        self.assertTrue(isinstance(frame, FrameType))

    def setUp(self):
        signal(SIGHUP, self.handler)
        signal(SIGINT, self.handler)

    def reset(self):
        self.got = {}

    def test_delayed_signals(self):
        self.reset()
        with delayed_signals([SIGHUP]):
            os.kill(os.getpid(), SIGHUP)
            time.sleep(2)
        self.assertTrue(self.got[SIGHUP])
        self.assertTrue(time.time() - self.got[SIGHUP] < 2)

    def test_immediate_signals(self):
        self.reset()
        os.kill(os.getpid(), SIGHUP)
        time.sleep(2)
        self.assertTrue(self.got[SIGHUP])
        self.assertTrue(time.time() - self.got[SIGHUP] > 1)

    def test_multiple_signals(self):
        self.reset()
        with delayed_signals([SIGHUP, SIGINT]):
            os.kill(os.getpid(), SIGINT)
        self.assertFalse(SIGHUP in self.got)
        self.assertTrue(SIGINT in self.got)

    @delayed_signals([SIGHUP])
    def kill_and_sleep(self):
        os.kill(os.getpid(), SIGHUP)
        time.sleep(2)

    def test_decorator(self):
        self.reset()
        self.kill_and_sleep()
        self.assertTrue(SIGHUP in self.got)
        self.assertTrue(time.time() - self.got[SIGHUP] < 2)
