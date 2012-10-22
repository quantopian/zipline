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

"""
Unit tests for finance.slippage
"""
import datetime

import pytz

from unittest2 import TestCase

from zipline.finance.slippage import VolumeShareSlippage

from zipline import ndict


class SlippageTestCase(TestCase):

    def test_volume_share_slippage(self):

        event = ndict(
            {'volume': 200,
             'TRANSACTION': None,
             'type': 4,
             'price': 3.0,
             'datetime': datetime.datetime(
                 2006, 1, 5, 14, 31, tzinfo=pytz.utc),
             'high': 3.15,
             'low': 2.85,
             'sid': 133,
             'source_id': 'test_source',
             'close': 3.0,
             'dt':
             datetime.datetime(2006, 1, 5, 14, 31, tzinfo=pytz.utc),
             'open': 3.0}
        )

        slippage_model = VolumeShareSlippage()

        open_orders = {133: [
            ndict(
                {'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                 'amount': 100,
                 'filled': 0, 'sid': 133})
        ]}

        txn = slippage_model.simulate(
            event,
            open_orders
        )

        expected_txn = {
            'price': float(3.01875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 31, tzinfo=pytz.utc),
            'amount': int(50),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])
