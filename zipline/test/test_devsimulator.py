from unittest2 import TestCase
from zipline.test.test_messaging import SimulatorTestCase
from zipline.test.dummy import ThreadPoolExecutorMixin


class ThreadPoolExecutor(SimulatorTestCase, TestCase):
    
    def test_universe(self):
        # first order logic is working today. Yay!
        self.assertTrue(True != False)
