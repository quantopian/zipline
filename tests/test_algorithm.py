from unittest2 import TestCase
from datetime import timedelta

from zipline.utils.test_utils import setup_logger
import zipline.utils.factory as factory
from zipline.test_algorithms import TestRegisterTransformAlgorithm
from zipline.gens.tradegens import SpecificEquityTrades, DataFrameSource
from zipline.gens.mavg import MovingAverage

class TestTransformAlgorithm(TestCase):
    def setUp(self):
        setup_logger(self)
        self.trading_environment = factory.create_trading_environment()
        setup_logger(self)

        trade_history = factory.create_trade_history(
            133,
            [10.0, 10.0, 11.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
            self.trading_environment
        )
        self.source = SpecificEquityTrades(event_list=trade_history)

        self.df_source, self.df = factory.create_test_df_source()

    def test_source_as_input(self):
        algo = TestRegisterTransformAlgorithm(sids=[133])
        algo.run(self.source)
        self.assertEqual(len(algo.sources), 1)
        assert isinstance(algo.sources[0], SpecificEquityTrades)

    def test_multi_source_as_input_no_start_end(self):
        algo = TestRegisterTransformAlgorithm(sids=[133])
        with self.assertRaises(AssertionError):
            algo.run([self.source, self.df_source])

    def test_multi_source_as_input(self):
        algo = TestRegisterTransformAlgorithm(sids=[0, 1, 133])
        algo.run([self.source, self.df_source], start=self.df.index[0], end=self.df.index[-1])
        self.assertEqual(len(algo.sources), 2)

    def test_df_as_input(self):
        algo = TestRegisterTransformAlgorithm(sids=[0, 1])
        algo.run(self.df)
        assert isinstance(algo.sources[0], DataFrameSource)

    def test_transform_registered(self):
        algo = TestRegisterTransformAlgorithm(sids=[133])
        algo.run(self.source)
        assert algo.get_sid_filter() == algo.sids == [133]
        assert 'mavg' in algo.registered_transforms
        assert algo.registered_transforms['mavg']['args'] == (['price'],)
        assert algo.registered_transforms['mavg']['kwargs'] == {'days': 2, 'market_aware': True}
        assert algo.registered_transforms['mavg']['class'] is MovingAverage


