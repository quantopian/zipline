#
# Copyright 2014 Quantopian, Inc.
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

from nose_parameterized import parameterized
from unittest import TestCase

from zipline.finance.blotter import Blotter
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)

from zipline.utils.test_utils import(
    setup_logger,
    teardown_logger,
)


class BlotterTestCase(TestCase):

    def setUp(self):
        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

    @parameterized.expand([(MarketOrder(), None, None),
                           (LimitOrder(10), 10, None),
                           (StopOrder(10), None, 10),
                           (StopLimitOrder(10, 20), 10, 20)])
    def test_blotter_order_types(self, style_obj, expected_lmt, expected_stp):

        blotter = Blotter()

        blotter.order(24, 100, style_obj)
        result = blotter.open_orders[24][0]

        self.assertEqual(result.limit, expected_lmt)
        self.assertEqual(result.stop, expected_stp)
