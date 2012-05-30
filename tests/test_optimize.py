"""Tests for the zipline.finance package"""
import unittest
from unittest2 import TestCase, skip
from nose.tools import timed
from collections import defaultdict
from datetime import datetime, timedelta
import logging

import numpy as np

from zipline.optimize.factory import create_updown_trade_source
import zipline.utils.factory as factory
from zipline.utils.logger import configure_logging

from zipline.core.devsimulator import AddressAllocator, Simulator
from zipline.optimize.algorithms import BuySellAlgorithm
from zipline.optimize.factory import create_predictable_zipline
from zipline.finance.trading import TradingEnvironment
from zipline.lines import SimulatedTrading
from zipline.finance.trading import SIMULATION_STYLE

DEFAULT_TIMEOUT = 15 # seconds
EXTENDED_TIMEOUT = 90

allocator = AddressAllocator(1000)
LOGGER = logging.getLogger('ZiplineLogger')

class TestUpDown(TestCase):
    """This unittest verifies that the BuySellAlgorithm in
    combination with the UpDownSource are suitable for usage in an
    optimization framework.

    """
    leased_sockets = defaultdict(list)

    def setUp(self):
        configure_logging()
        self.zipline_test_config = {
            'allocator':allocator,
            'sid':133
        }

    @timed(DEFAULT_TIMEOUT)
    def test_source_and_orders(self):
        """verify that UpDownSource is having the correct
        behavior and that BuySellAlgorithm places the buy/sell
        orders at the right time. Moreover, establishes that
        UpDownSource and BuySellAlgorithm interact correctly."

        """
        #generate events
        trade_count = 5
        sid = 133
        base_price = 50
        amplitude = 6
        offset = 0

        zipline, config = create_predictable_zipline(
            self.zipline_test_config,
            sid=sid,
            amplitude=amplitude,
            base_price=base_price,
            offset=offset,
            trade_count=5,
            simulate=False
        )

        prices = np.array([event.price for event in config['trade_source'].event_list])
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

        zipline.simulate(blocking=True)

        algo = config['algorithm']

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

    def test_concavity_of_returns(self):
        """verify concave relationship between free parameter and
        returns in certain region around the max. Moreover,
        establishes that the max returns is at the correct value
        (i.e. 0).

        """
        #generate events
        trade_count = 6
        sid = 133
        amplitude = 30
        base_price = 50

        #test whether return-function is concave wrt repeats.
        test_offsets = np.arange(-9, 9, 1.)
        supposed_max = np.zeros(len(test_offsets), dtype=bool)
        supposed_max[len(test_offsets) // 2] = True

        compound_returns = np.empty(len(test_offsets))
        ziplines = []
        for i, offset in enumerate(test_offsets):
            zipline, config = create_predictable_zipline(
                self.zipline_test_config,
                sid=sid,
                amplitude=amplitude,
                base_price=base_price,
                offset=offset,
                trade_count=trade_count,
                simulate=True
            )
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

    #@skip
    def test_optimize(self):
        """verify that gradient descent (Powell's method) can find
        the optimal free parameter under which the BuySellAlgorithm produces
        maximum returns.

        """
        def simulate(offset):
            zipline, config = create_predictable_zipline(
                self.zipline_test_config,
                sid=133,
                amplitude=10,
                base_price=50,
                offset=offset,
                trade_count=5,
                simulate=True
            )
            #function is getting minimized, so have to return negative cum returns.
            return -zipline.get_cumulative_performance()['returns']

        from scipy import optimize
        opt = optimize.fmin_powell(simulate, 1.5)
        np.testing.assert_almost_equal(opt, 0, 5)
