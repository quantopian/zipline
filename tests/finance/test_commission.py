#
# Copyright 2013 Quantopian, Inc.
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

"""
Unit tests for finance.commission
"""
import datetime

import pytz

from unittest import TestCase

from zipline.finance.commission import PerShare, PerTrade
from zipline.finance.slippage import Transaction


class CommissionTestCase(TestCase):

    def test_per_share_commission(self):
        commission_model = PerShare(cost=0.03)

        txn = {
            'sid': 133,
            'amount': 100,
            'dt':  datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
            'price': 3.0,
        }
        transaction = Transaction(**txn)

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 3.0)

        self.assertEquals(commission, exp_commission)

        # cost is an integer
        commission_model = PerShare(cost=2)

        commission = commission_model.calculate(transaction)
        exp_commission = (2.0, 200.0)

        self.assertEquals(commission, exp_commission)

        # cost is zero
        commission_model = PerShare(cost=0.0)

        commission = commission_model.calculate(transaction)
        exp_commission = (0.0, 0.0)

        self.assertEquals(commission, exp_commission)

        # reduced amount of shares
        commission_model = PerShare(cost=0.03)
        transaction.amount = 10

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 0.3)

        self.assertEquals(commission, exp_commission)

        # negative amount of shares
        commission_model = PerShare(cost=0.03)
        transaction.amount = -100

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 3.0)

        self.assertEquals(commission, exp_commission)

    def test_min_per_order_commission(self):
        commission_model = PerShare(cost=0.03, min_per_order=1.0)

        txn = {
            'sid': 133,
            'amount': 100,
            'dt':  datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
            'price': 3.0,
        }
        transaction = Transaction(**txn)

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 3.0)

        self.assertEquals(commission, exp_commission)

        # total comission less than minimum per order
        transaction.amount = 10

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 1.0)

        self.assertEquals(commission, exp_commission)

    def test_per_trade_commission(self):
        commission_model = PerTrade(cost=3.0)

        txn = {
            'sid': 133,
            'amount': 100,
            'dt':  datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
            'price': 3.0,
        }
        transaction = Transaction(**txn)

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 3.0)

        self.assertEquals(commission, exp_commission)

        # negative amount of shares
        transaction.amount = -100

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 3.0)

        self.assertEquals(commission, exp_commission)

        # cost is an integer
        commission_model = PerTrade(cost=3)

        commission = commission_model.calculate(transaction)
        exp_commission = (0.03, 3.0)

        self.assertEquals(commission, exp_commission)

        # cost is zero
        commission_model = PerTrade(cost=0.0)

        commission = commission_model.calculate(transaction)
        exp_commission = (0.0, 0.0)

        self.assertEquals(commission, exp_commission)
