import unittest
from test.testmessaging import *

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MessagingTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)