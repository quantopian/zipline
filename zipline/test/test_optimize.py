"""Tests for the zipline.finance package"""
import unittest
from unittest2 import TestCase
from nose.tools import timed
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from zipline.optimize.factory import create_updown_trade_source
import zipline.test.factory as factory
import zipline.util as qutil

from zipline.simulator import AddressAllocator, Simulator
from zipline.optimize.algorithms import BuySellAlgorithm
from zipline.finance.trading import TradingEnvironment
from zipline.lines import SimulatedTrading
from zipline.finance.trading import SIMULATION_STYLE

DEFAULT_TIMEOUT = 15 # seconds
EXTENDED_TIMEOUT = 90

allocator = AddressAllocator(1000)

class FinanceTestCase(TestCase):

    leased_sockets = defaultdict(list)

    def setUp(self):
        qutil.configure_logging()
        self.zipline_test_config = {
            'allocator':allocator,
            'sid':133
        }

    @timed(DEFAULT_TIMEOUT)
    def test_buysell(self):
        #generate events
        trade_count = 50
        sid = 133
        base_price = 50
        amplitude = 6
        offset = 0
        self.zipline_test_config['order_count'] = trade_count - 1
        self.zipline_test_config['trade_count'] = trade_count
        self.zipline_test_config['simulation_style'] = \
        SIMULATION_STYLE.FIXED_SLIPPAGE

        trading_environment = factory.create_trading_environment()
        source = factory.create_updown_trade_source(sid,
            trade_count,
            trading_environment,
            base_price,
            amplitude
        )

        prices = np.array([event.price for event in source.event_list])
        max_price_idx = np.where(prices==prices.max())[0]
        min_price_idx = np.where(prices==prices.min())[0]
        self.assertTrue(np.all(max_price_idx % 2 == 1),
            "Maximum prices are not periodic."
        )
        self.assertTrue(np.all(min_price_idx % 2 == 0),
            "Minimum prices are not periodic."
        )
        self.assertEqual(prices.max(), base_price+amplitude/2.,
            "Maximum price does not equal expected maximum price."
        )
        self.assertEqual(prices.min(), base_price-amplitude/2.,
            "Minimum price does not equal expected maximum price."
        )

        algo = BuySellAlgorithm(sid, 100, 0)

        self.zipline_test_config['trade_source'] = source
        self.zipline_test_config['algorithm'] = algo
        self.zipline_test_config['environment'] = trading_environment

        zipline = SimulatedTrading.create_test_zipline(**self.zipline_test_config)
        zipline.simulate(blocking=True)

        orders = np.asarray(algo.orders)
        max_order_idx = np.where(orders==orders.max())[0]
        min_order_idx = np.where(orders==orders.min())[0]

        self.assertTrue(np.all(max_order_idx % 2 == 1),
            "Maximum orders are not periodic."
        )
        self.assertTrue(np.all(min_order_idx % 2 == 0),
            "Minimum orders are not periodic."
        )
        self.assertTrue(np.all(max_order_idx == max_price_idx),
            "Algorithm did not buy when price was going to drop."
        )
        self.assertTrue(np.all(min_order_idx == min_price_idx),
            "Algorithm did not sell when price was going to increase."
        )

    def test_buysell_concave(self):
        #generate events
        trade_count = 6
        sid = 133
        amplitude = 30
        base_price = 50
        self.zipline_test_config['order_count'] = trade_count - 1
        self.zipline_test_config['trade_count'] = trade_count
        self.zipline_test_config['simulation_style'] = \
        SIMULATION_STYLE.FIXED_SLIPPAGE

        #test whether return-function is concave wrt repeats.
        test_offsets = np.arange(-9, 9, 1.)
        supposed_max = np.zeros(len(test_offsets), dtype=bool)
        supposed_max[len(test_offsets) // 2] = True

        compound_returns = np.empty(len(test_offsets))
        ziplines = []
        for i, test_offset in enumerate(test_offsets):
            trading_environment = factory.create_trading_environment()
            source = factory.create_updown_trade_source(sid,
                trade_count,
                trading_environment,
                base_price,
                amplitude
            )

            algo = BuySellAlgorithm(sid, 100, test_offset)
            self.zipline_test_config['algorithm'] = algo
            self.zipline_test_config['trade_source'] = source
            self.zipline_test_config['environment'] = trading_environment
            zipline = SimulatedTrading.create_test_zipline(**self.zipline_test_config)
            zipline.simulate(blocking=True)
            ziplines.append(zipline)
            compound_returns[i] = zipline.get_cumulative_performance()['returns']

        self.assertTrue(np.all(compound_returns[supposed_max] > compound_returns[np.logical_not(supposed_max)]),
            "Maximum compound returns are not where they are supposed to be."
        )

        # test for concavity
        max_idx = np.where(supposed_max)[0][0]
        idx = np.array([max_idx, max_idx])
        for i in range((len(test_offsets)-1)/2):
            # going outwards, returns must decrease
            self.assertTrue(compound_returns[idx[0]-1] < compound_returns[idx[0]],
                "Compound returns are not convex."
            )
            self.assertTrue(compound_returns[idx[1]+1] < compound_returns[idx[1]],
                "Compound returns are not convex."
            )
            idx[0] -= 1
            idx[1] += 1


    def test_optimize(self):
        def simulate(offset):
            #generate events
            trade_count = 3
            sid = 133
            amplitude = 10
            base_price = 50
            self.zipline_test_config['order_count'] = trade_count - 1
            self.zipline_test_config['trade_count'] = trade_count
            self.zipline_test_config['simulation_style'] = \
            SIMULATION_STYLE.FIXED_SLIPPAGE
            trading_environment = factory.create_trading_environment()
            source = create_updown_trade_source(sid,
                trade_count,
                trading_environment,
                base_price,
                amplitude
            )

            algo = BuySellAlgorithm(sid, 100, offset)
            self.zipline_test_config['algorithm'] = algo
            self.zipline_test_config['trade_source'] = source
            self.zipline_test_config['environment'] = trading_environment
            zipline = SimulatedTrading.create_test_zipline(**self.zipline_test_config)
            zipline.simulate(blocking=True)
            zipline.shutdown()
            #function is getting minimized, so have return negative.
            return -zipline.get_cumulative_performance()['returns']

        from scipy import optimize
        opt = optimize.fmin_powell(simulate, 1.5)
        np.testing.assert_almost_equal(opt, 0, 5)
