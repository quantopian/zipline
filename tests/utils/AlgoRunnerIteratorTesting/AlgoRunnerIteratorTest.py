import unittest
from mock import patch, MagicMock
import pandas as pd;
from zipline.utils.AlgoRunnerIterator import AlgoRunnerIterator
d = {'one' : [1., 2., 3., 4.],'two' : [4., 3., 2., 1.]}
df = pd.DataFrame(d);
 
def mockLoadBars():
    print('This is mocking of load_bars_from_yahoo...');
    return ['Returned', 'from', 'Mock', 'Load'];

class AlgoRunnerIteratorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(AlgoRunnerIteratorTest, cls).setUpClass();
        attrs = {'run.return_value':df};

    @patch('zipline.data.loader.load_bars_from_yahoo')
    @patch('zipline.algorithm.TradingAlgorithm.run', MagicMock(return_value=df))
    def testRunNext(self, test_patch):
        test_patch.return_value = ['Returned', 'from', 'Mock', 'Load']
        print('Testing AlgoRunner...');
        algoRunner = AlgoRunnerIterator('SimulationTimesConfig.csv',
                                         None, None, ['IBM']);
        while algoRunner.hasNext():
            result = algoRunner.algoRunNext()
            self.assertTrue(result.equals(df), 'Wrong value returned');
        self.assertEqual(algoRunner.index, 6, 'Index is not in right value')
