#
# Copyright 2012 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the zipline.finance package"""
from unittest2 import TestCase, skip
from collections import defaultdict

import numpy as np

from zipline.optimize.factory import create_predictable_zipline

from zipline.utils.test_utils import setup_logger, teardown_logger


class TestUpDown(TestCase):
    """This unittest verifies that the BuySellAlgorithm in
    combination with the UpDownSource are suitable for usage in an
    optimization framework.

    """
    leased_sockets = defaultdict(list)

    def setUp(self):
        self.zipline_test_config = {
            'sid': [0],
            'trade_count': 5,
            'amplitude': 30,
            'base_price': 50
        }
        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

    @skip
    def test_source_and_orders(self):
        """verify that UpDownSource is having the correct
        behavior and that BuySellAlgorithm places the buy/sell
        orders at the right time. Moreover, establishes that
        UpDownSource and BuySellAlgorithm interact correctly."

        """

        algo, config = create_predictable_zipline(
            self.zipline_test_config,
            offset=0
        )

        #extract arguments
        base_price = self.zipline_test_config['base_price']
        amplitude = self.zipline_test_config['amplitude']

        prices = config['trade_source'][0].values
        max_price_idx = np.where(prices == prices.max())[0]
        min_price_idx = np.where(prices == prices.min())[0]
        self.assertTrue(np.all(max_price_idx % 2 == 1),
            "Maximum prices are not periodic."
        )
        self.assertTrue(np.all(min_price_idx % 2 == 0),
            "Minimum prices are not periodic."
        )
        self.assertEqual(prices.max(), base_price + amplitude / 2.,
            "Maximum price does not equal expected maximum price."
        )
        self.assertEqual(prices.min(), base_price - amplitude / 2.,
            "Minimum price does not equal expected maximum price."
        )

        stats = algo.run(config['trade_source'])

        self.assertTrue(len(stats) != 0)

        orders = np.asarray(algo.orders)
        max_order_idx = np.where(orders == orders.max())[0]
        min_order_idx = np.where(orders == orders.min())[0]

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

    @skip
    def test_concavity_of_returns(self):
        """verify concave relationship between free parameter and
        returns in certain region around the max. Moreover,
        establishes that the max returns is at the correct value
        (i.e. 0).

        """
        test_offsets = np.arange(-9, 9, 1.)
        #maximum value is expect to be at center, create boolean mask
        #for later extraction
        supposed_max = np.zeros(len(test_offsets), dtype=bool)
        supposed_max[len(test_offsets) // 2] = True

        compound_returns = np.empty(len(test_offsets))
        ziplines = []
        for i, offset in enumerate(test_offsets):
            algo, config = create_predictable_zipline(
                self.zipline_test_config,
                offset=offset,
            )
            results = algo.run(config['trade_source'])
            ziplines.append(algo)

            compound_returns[i] = results.returns.sum()

        self.assertTrue(np.all(
            compound_returns[supposed_max] >
            compound_returns[np.logical_not(supposed_max)]),
            "Maximum compound returns are not where they are supposed to be."
        )

        # test for concavity
        max_idx = np.where(supposed_max)[0][0]
        idx = np.array([max_idx, max_idx])
        for i in range((len(test_offsets) - 1) / 2):
            # going outwards, returns must decrease
            self.assertTrue(compound_returns[idx[0] - 1] <
                            compound_returns[idx[0]],
                "Compound returns are not convex."
            )
            self.assertTrue(compound_returns[idx[1] + 1] <
                            compound_returns[idx[1]],
                "Compound returns are not convex."
            )
            idx[0] -= 1
            idx[1] += 1

    @skip
    def test_optimize(self):
        """verify that gradient descent (Powell's method) can find
        the optimal free parameter under which the BuySellAlgorithm produces
        maximum returns.

        """
        def simulate(offset):
            zipline, config = create_predictable_zipline(
                self.zipline_test_config,
                offset=offset,
            )
            # function is getting minimized,
            # so have to return negative cum returns.
            return -zipline.get_cumulative_performance()['returns']

        from scipy import optimize
        opt = optimize.fmin_powell(simulate, 1.5)
        np.testing.assert_almost_equal(opt, 0, 5)
