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

from nose_parameterized import parameterized

import pandas as pd

from zipline.finance.slippage import VolumeShareSlippage

from zipline.protocol import Event, DATASOURCE_TYPE
from zipline.finance.blotter import Order


class SlippageTestCase(TestCase):

    def test_volume_share_slippage(self):
        event = Event(
            {'volume': 200,
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

        open_orders = [
            Order(dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                  amount=100,
                  filled=0,
                  sid=133)
        ]

        orders_txns = list(slippage_model.simulate(
            event,
            open_orders
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        expected_txn = {
            'price': float(3.01875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 31, tzinfo=pytz.utc),
            'amount': int(50),
            'sid': int(133),
            'commission': None,
            'type': DATASOURCE_TYPE.TRANSACTION,
            'order_id': open_orders[0].id
        }

        self.assertIsNotNone(txn)

        # TODO: Make expected_txn an Transaction object and ensure there
        # is a __eq__ for that class.
        self.assertEquals(expected_txn, txn.__dict__)

    def test_orders_limit(self):

        events = self.gen_trades()

        slippage_model = VolumeShareSlippage()

        # long, does not trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]

        orders_txns = list(slippage_model.simulate(
            events[2],
            open_orders
        ))
        self.assertEquals(len(orders_txns), 0)

        # long, does trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]

        orders_txns = list(slippage_model.simulate(
            events[3],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 1)
        txn = orders_txns[0][1]

        expected_txn = {
            'price': float(3.500875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            'amount': int(100),
            'sid': int(133),
            'order_id': open_orders[0].id
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

        # short, does not trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]

        orders_txns = list(slippage_model.simulate(
            events[0],
            open_orders
        ))

        expected_txn = {}

        self.assertEquals(len(orders_txns), 0)

        # short, does trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'limit': 3.5})
        ]

        orders_txns = list(slippage_model.simulate(
            events[1],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

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

    STOP_ORDER_CASES = {
        # Stop orders can be long/short and have their price greater or
        # less than the stop.
        #
        # A stop being reached is conditional on the order direction.
        # Long orders reach the stop when the price is greater than the stop.
        # Short orders reach the stop when the price is less than the stop.
        #
        # Which leads to the following 4 cases:
        #
        #                    |   long   |   short  |
        # | price > stop     |          |          |
        # | price < stop     |          |          |
        #
        # Currently the slippage module acts according to the following table,
        # where 'X' represents triggering a transaction
        #                    |   long   |   short  |
        # | price > stop     |          |     X    |
        # | price < stop     |    X     |          |
        #
        # However, the following behavior *should* be followed.
        #
        #                    |   long   |   short  |
        # | price > stop     |    X     |          |
        # | price < stop     |          |     X    |

        'long | price gt stop': {
            'order': {
                'dt': pd.Timestamp('2006-01-05 14:30', tz='UTC'),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 3.5
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 4.0,
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'close': 4.0,
                'open': 3.5
            },
            'expected': {
                'transaction': {
                    'price': 4.001,
                    'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                    'amount': 100,
                    'sid': 133,
                }
            }
        },
        'long | price lt stop': {
            'order': {
                'dt': pd.Timestamp('2006-01-05 14:30', tz='UTC'),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 3.6
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 3.5,
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'close': 3.5,
                'open': 4.0
            },
            'expected': {
                'transaction': None
            }
        },
        'short | price gt stop': {
            'order': {
                'dt': pd.Timestamp('2006-01-05 14:30', tz='UTC'),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.4
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 3.5,
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'close': 3.5,
                'open': 3.0
            },
            'expected': {
                'transaction': None
            }
        },
        'short | price lt stop': {
            'order': {
                'dt': pd.Timestamp('2006-01-05 14:30', tz='UTC'),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.5
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 3.0,
                'high': 3.15,
                'low': 2.85,
                'sid': 133,
                'close': 3.0,
                'open': 3.0
            },
            'expected': {
                'transaction': {
                    'price': 2.99925,
                    'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                    'amount': -100,
                    'sid': 133,
                }
            }
        },
    }

    @parameterized.expand([
        (name, case['order'], case['event'], case['expected'])
        for name, case in STOP_ORDER_CASES.items()
    ])
    def test_orders_stop(self, name, order_data, event_data, expected):
        order = Order(**order_data)
        event = Event(initial_values=event_data)

        slippage_model = VolumeShareSlippage()

        try:
            _, txn = next(slippage_model.simulate(event, [order]))
        except StopIteration:
            txn = None

        if expected['transaction'] is None:
            self.assertIsNone(txn)
        else:
            self.assertIsNotNone(txn)

            for key, value in expected['transaction'].items():
                self.assertEquals(value, txn[key])

    def test_orders_stop_limit(self):

        events = self.gen_trades()
        slippage_model = VolumeShareSlippage()

        # long, does not trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 4.0,
                'limit': 3.0})
        ]

        orders_txns = list(slippage_model.simulate(
            events[2],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 0)

        orders_txns = list(slippage_model.simulate(
            events[3],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 0)

        # long, does trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': 133,
                'stop': 4.0,
                'limit': 3.5})
        ]

        orders_txns = list(slippage_model.simulate(
            events[2],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 0)

        orders_txns = list(slippage_model.simulate(
            events[3],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        expected_txn = {
            'price': float(3.500875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            'amount': int(100),
            'sid': int(133)
        }

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

        # short, does not trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.0,
                'limit': 4.0})
        ]

        orders_txns = list(slippage_model.simulate(
            events[0],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 0)

        orders_txns = list(slippage_model.simulate(
            events[1],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 0)

        # short, does trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': 133,
                'stop': 3.0,
                'limit': 3.5})
        ]

        orders_txns = list(slippage_model.simulate(
            events[0],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 0)

        orders_txns = list(slippage_model.simulate(
            events[1],
            open_orders
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        expected_txn = {
            'price': float(3.499125),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 32, tzinfo=pytz.utc),
            'amount': int(-100),
            'sid': int(133)
        }

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

    def gen_trades(self):
        # create a sequence of trades
        events = [
            Event({
                'volume': 2000,
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
