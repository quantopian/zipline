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
Unit tests for finance.slippage
"""
import datetime

import pytz

from unittest import TestCase

from zipline.finance.slippage import VolumeShareSlippage

from zipline.protocol import Event
from zipline.gens.tradesimulation import Order


class SlippageTestCase(TestCase):

    def test_volume_share_slippage(self):

        event = Event(
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
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133})
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
            'sid': int(133),
            'commission': None
        }

        self.assertIsNotNone(txn)

        # TODO: Make expected_txn an Transaction object and ensure there
        # is a __eq__ for that class.
        self.assertEquals(expected_txn, txn.__dict__)

    def test_orders_limit(self):

        events = self.gen_trades()

        slippage_model = VolumeShareSlippage()

        # long, does not trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[2],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        # long, does trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[3],
            open_orders
        )

        expected_txn = {
            'price': float(3.500875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            'amount': int(100),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

        # short, does not trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[0],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        # short, does trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[1],
            open_orders
        )

        expected_txn = {
            'price': float(3.499125),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 32, tzinfo=pytz.utc),
            'amount': int(-100),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

    def test_orders_stop(self):
        events = self.gen_trades()

        slippage_model = VolumeShareSlippage()

        # long, does not trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[2],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        # long, does trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 3.6})
        ]}

        txn = slippage_model.simulate(
            events[3],
            open_orders
        )

        expected_txn = {
            'price': float(3.500875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            'amount': int(100),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

        # short, does not trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[0],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        # short, does trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.4})
        ]}

        txn = slippage_model.simulate(
            events[1],
            open_orders
        )

        expected_txn = {
            'price': float(3.499125),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 32, tzinfo=pytz.utc),
            'amount': int(-100),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

    def test_orders_stop_limit(self):

        events = self.gen_trades()
        slippage_model = VolumeShareSlippage()

        # long, does not trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 4.0,
                'limit': 3.0})
        ]}

        txn = slippage_model.simulate(
            events[2],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        txn = slippage_model.simulate(
            events[3],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        # long, does trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 4.0,
                'limit': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[2],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        txn = slippage_model.simulate(
            events[3],
            open_orders
        )

        expected_txn = {
            'price': float(3.500875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            'amount': int(100),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

        # short, does not trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.0,
                'limit': 4.0})
        ]}

        txn = slippage_model.simulate(
            events[0],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        txn = slippage_model.simulate(
            events[1],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        # short, does trade

        open_orders = {133: [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.0,
                'limit': 3.5})
        ]}

        txn = slippage_model.simulate(
            events[0],
            open_orders
        )

        expected_txn = {}

        self.assertIsNone(txn)

        txn = slippage_model.simulate(
            events[1],
            open_orders
        )

        expected_txn = {
            'price': float(3.499125),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 32, tzinfo=pytz.utc),
            'amount': int(-100),
            'sid': int(133)
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

    def gen_trades(self):
        # create a sequence of trades
        events = [
            Event({
                'volume': 2000,
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
                'open': 3.0
            }),
            Event({
                'volume': 2000,
                'TRANSACTION': None,
                'type': 4,
                'price': 3.5,
                'datetime': datetime.datetime(
                    2006, 1, 5, 14, 32, tzinfo=pytz.utc),
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'source_id': 'test_source',
                'close': 3.5,
                'dt':
                datetime.datetime(2006, 1, 5, 14, 32, tzinfo=pytz.utc),
                'open': 3.0
            }),
            Event({
                'volume': 2000,
                'TRANSACTION': None,
                'type': 4,
                'price': 4.0,
                'datetime': datetime.datetime(
                    2006, 1, 5, 14, 33, tzinfo=pytz.utc),
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'source_id': 'test_source',
                'close': 4.0,
                'dt':
                datetime.datetime(2006, 1, 5, 14, 33, tzinfo=pytz.utc),
                'open': 3.5
            }),
            Event({
                'volume': 2000,
                'TRANSACTION': None,
                'type': 4,
                'price': 3.5,
                'datetime': datetime.datetime(
                    2006, 1, 5, 14, 34, tzinfo=pytz.utc),
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'source_id': 'test_source',
                'close': 3.5,
                'dt':
                datetime.datetime(2006, 1, 5, 14, 34, tzinfo=pytz.utc),
                'open': 4.0
            }),
            Event({
                'volume': 2000,
                'TRANSACTION': None,
                'type': 4,
                'price': 3.0,
                'datetime': datetime.datetime(
                    2006, 1, 5, 14, 35, tzinfo=pytz.utc),
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'source_id': 'test_source',
                'close': 3.0,
                'dt':
                datetime.datetime(2006, 1, 5, 14, 35, tzinfo=pytz.utc),
                'open': 3.5
            })
        ]
        return events
