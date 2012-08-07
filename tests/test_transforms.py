from datetime import timedelta
from collections import defaultdict
from unittest2 import TestCase

from zipline.utils.test_utils import setup_logger, teardown_logger

import zipline.utils.factory as factory

from zipline.gens.tradegens import SpecificEquityTrades
from zipline.gens.transform import StatefulTransform
from zipline.gens.vwap import VWAP
from zipline.gens.mavg import MovingAverage
from zipline.gens.returns import Returns
from zipline.lines import SimulatedTrading
from zipline.core.devsimulator import AddressAllocator

allocator = AddressAllocator(1000)

class FinanceTransformsTestCase(TestCase):

    def setUp(self):
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

    def tearDown(self):
        self.log_handler.pop_application()

    def test_vwap(self):
        
        vwap = StatefulTransform(
            VWAP, 
            market_aware = False,
            delta = timedelta(days = 2)
        )
        transformed = list(vwap.transform(self.source))

        # Output values
        tnfm_vals = [message.tnfm_value for message in transformed]
        # "Hand calculated" values.
        expected = [(10.0 * 100) / 100.0,
                    ((10.0 * 100) + (10.0 * 100)) / (200.0),
                    # We should drop the first event here.
                    ((10.0 * 100) + (11.0 * 100)) / (200.0),
                    # We should drop the second event here.
                    ((11.0 * 100) + (11.0 * 300)) / (400.0)] 

        # Output should match the expected.
        assert tnfm_vals == expected
        
    def test_returns(self):
        trade_history = factory.create_trade_history(
            133,
            [10.0, 10.0, 10.0, 11.0],
            [100, 100, 100, 300],
            timedelta(days=1),
        )
        returns = StatefulTransform(
            Returns
        for trade in trade_history:
            returns.update(trade)
        self.assertEqual(returns.returns, .1)

    def test_moving_average(self):

        mavg = StatefulTransform(
            MovingAverage,          
            market_aware = False,
            fields = ['price', 'volume'],
            delta = timedelta(days = 2), 
        ) 
        transformed = list(mavg.transform(self.source))
        # Output values.
        tnfm_prices = [message.tnfm_value.price for message in transformed]
        tnfm_volumes = [message.tnfm_value.volume for message in transformed]

        # "Hand-calculated" values
        expected_prices = [((10.0) / 1.0),
                           ((10.0 + 10.0) / 2.0),
                           # First event should get dropped here.
                           ((10.0 + 11.0) / 2.0),
                           # Second event should get dropped here.
                           ((11.0 + 11.0) / 2.0)]

        expected_volumes = [((100.0) / 1.0),
                           ((100.0 + 100.0) / 2.0),
                            # First event should get dropped here.
                           ((100.0 + 100.0) / 2.0),
                           # Second event should get dropped here.
                           ((100.0 + 300.0) / 2.0)]
        
        assert tnfm_prices == expected_prices
        assert tnfm_volumes == expected_volumes
