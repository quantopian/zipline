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
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Unit tests for finance.slippage
'''
import datetime

import pytz

from nose_parameterized import parameterized

import pandas as pd
from pandas.tslib import normalize_date

from zipline.finance.slippage import VolumeShareSlippage

from zipline.protocol import DATASOURCE_TYPE
from zipline.finance.blotter import Order

from zipline.data.data_portal import DataPortal
from zipline.protocol import BarData
from zipline.testing import tmp_bcolz_minute_bar_reader
from zipline.testing.fixtures import (
    WithDataPortal,
    WithSimParams,
    ZiplineTestCase,
)


class SlippageTestCase(WithSimParams, WithDataPortal, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-05 14:31', tz='utc')
    END_DATE = pd.Timestamp('2006-01-05 14:36', tz='utc')
    SIM_PARAMS_CAPITAL_BASE = 1.0e5
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    SIM_PARAMS_EMISSION_RATE = 'daily'

    ASSET_FINDER_EQUITY_SIDS = (133,)
    ASSET_FINDER_EQUITY_START_DATE = pd.Timestamp('2006-01-05', tz='utc')
    ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp('2006-01-07', tz='utc')
    minutes = pd.DatetimeIndex(
        start=START_DATE,
        end=END_DATE - pd.Timedelta('1 minute'),
        freq='1min'
    )

    @classmethod
    def make_minute_bar_data(cls):
        return {
            133: pd.DataFrame(
                {
                    'open': [3.0, 3.0, 3.5, 4.0, 3.5],
                    'high': [3.15, 3.15, 3.15, 3.15, 3.15],
                    'low': [2.85, 2.85, 2.85, 2.85, 2.85],
                    'close': [3.0, 3.5, 4.0, 3.5, 3.0],
                    'volume': [2000, 2000, 2000, 2000, 2000],
                },
                index=cls.minutes,
            ),
        }

    @classmethod
    def init_class_fixtures(cls):
        super(SlippageTestCase, cls).init_class_fixtures()
        cls.ASSET133 = cls.env.asset_finder.retrieve_asset(133)

    def test_volume_share_slippage(self):
        assets = {
            133: pd.DataFrame(
                {
                    'open': [3.00],
                    'high': [3.15],
                    'low': [2.85],
                    'close': [3.00],
                    'volume': [200],
                },
                index=[self.minutes[0]],
            ),
        }
        days = pd.date_range(
            start=normalize_date(self.minutes[0]),
            end=normalize_date(self.minutes[-1])
        )
        with tmp_bcolz_minute_bar_reader(self.env, days, assets) as reader:
            data_portal = DataPortal(
                self.env,
                equity_minute_reader=reader,
            )

            slippage_model = VolumeShareSlippage()

            open_orders = [
                Order(
                    dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                    amount=100,
                    filled=0,
                    sid=self.ASSET133
                )
            ]

            bar_data = BarData(data_portal,
                               lambda: self.minutes[0],
                               'minute')

            orders_txns = list(slippage_model.simulate(
                bar_data,
                self.ASSET133,
                open_orders,
            ))

            self.assertEquals(len(orders_txns), 1)
            _, txn = orders_txns[0]

            expected_txn = {
                'price': float(3.0001875),
                'dt': datetime.datetime(
                    2006, 1, 5, 14, 31, tzinfo=pytz.utc),
                'amount': int(5),
                'sid': int(133),
                'commission': None,
                'type': DATASOURCE_TYPE.TRANSACTION,
                'order_id': open_orders[0].id
            }

            self.assertIsNotNone(txn)

            # TODO: Make expected_txn an Transaction object and ensure there
            # is a __eq__ for that class.
            self.assertEquals(expected_txn, txn.__dict__)

            open_orders = [
                Order(
                    dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                    amount=100,
                    filled=0,
                    sid=self.ASSET133
                )
            ]

            # Set bar_data to be a minute ahead of last trade.
            # Volume share slippage should not execute when there is no trade.
            bar_data = BarData(data_portal,
                               lambda: self.minutes[1],
                               'minute')

            orders_txns = list(slippage_model.simulate(
                bar_data,
                self.ASSET133,
                open_orders,
            ))

            self.assertEquals(len(orders_txns), 0)

    def test_orders_limit(self):
        slippage_model = VolumeShareSlippage()
        slippage_model.data_portal = self.data_portal

        # long, does not trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[3],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # long, does not trade - impacted price worse than limit price
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[3],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # long, does trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': self.ASSET133,
                'limit': 3.6})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[3],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 1)
        txn = orders_txns[0][1]

        expected_txn = {
            'price': float(3.50021875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            # we ordered 100 shares, but default volume slippage only allows
            # for 2.5% of the volume.  2.5% * 2000 = 50 shares
            'amount': int(50),
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
                'sid': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[0],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # short, does not trade - impacted price worse than limit price
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[0],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # short, does trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': self.ASSET133,
                'limit': 3.4})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[1],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        expected_txn = {
            'price': float(3.49978125),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 32, tzinfo=pytz.utc),
            'amount': int(-50),
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
                    'price': 4.00025,
                    'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                    'amount': 50,
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
                    'price': 2.9998125,
                    'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                    'amount': -50,
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
        data = order_data
        data['sid'] = self.ASSET133
        order = Order(**data)

        assets = {
            133: pd.DataFrame(
                {
                    'open': [event_data['open']],
                    'high': [event_data['high']],
                    'low': [event_data['low']],
                    'close': [event_data['close']],
                    'volume': [event_data['volume']],
                },
                index=[pd.Timestamp('2006-01-05 14:31', tz='UTC')],
            ),
        }
        days = pd.date_range(
            start=normalize_date(self.minutes[0]),
            end=normalize_date(self.minutes[-1])
        )
        with tmp_bcolz_minute_bar_reader(self.env, days, assets) as reader:
            data_portal = DataPortal(
                self.env,
                equity_minute_reader=reader,
            )

            slippage_model = VolumeShareSlippage()

            try:
                dt = pd.Timestamp('2006-01-05 14:31', tz='UTC')
                bar_data = BarData(data_portal,
                                   lambda: dt,
                                   'minute')
                _, txn = next(slippage_model.simulate(
                    bar_data,
                    self.ASSET133,
                    [order],
                ))
            except StopIteration:
                txn = None

            if expected['transaction'] is None:
                self.assertIsNone(txn)
            else:
                self.assertIsNotNone(txn)

                for key, value in expected['transaction'].items():
                    self.assertEquals(value, txn[key])

    def test_orders_stop_limit(self):
        slippage_model = VolumeShareSlippage()
        slippage_model.data_portal = self.data_portal

        # long, does not trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': self.ASSET133,
                'stop': 4.0,
                'limit': 3.0})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[2],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[3],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # long, does not trade - impacted price worse than limit price
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': self.ASSET133,
                'stop': 4.0,
                'limit': 3.5})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[2],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[3],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # long, does trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'sid': self.ASSET133,
                'stop': 4.0,
                'limit': 3.6})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[2],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[3],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        expected_txn = {
            'price': float(3.50021875),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 34, tzinfo=pytz.utc),
            'amount': int(50),
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
                'sid': self.ASSET133,
                'stop': 3.0,
                'limit': 4.0})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[0],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[1],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # short, does not trade - impacted price worse than limit price
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': self.ASSET133,
                'stop': 3.0,
                'limit': 3.5})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[0],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[1],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        # short, does trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'sid': self.ASSET133,
                'stop': 3.0,
                'limit': 3.4})
        ]

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[0],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = BarData(self.data_portal,
                           lambda: self.minutes[1],
                           self.sim_params.data_frequency)

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        expected_txn = {
            'price': float(3.49978125),
            'dt': datetime.datetime(
                2006, 1, 5, 14, 32, tzinfo=pytz.utc),
            'amount': int(-50),
            'sid': int(133)
        }

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])
