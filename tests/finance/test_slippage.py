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

import numpy as np
import pandas as pd
from pandas.tslib import normalize_date
from testfixtures import TempDirectory

from zipline.finance.slippage import VolumeShareSlippage
from zipline.finance.trading import TradingEnvironment, SimulationParameters

from zipline.protocol import DATASOURCE_TYPE
from zipline.finance.blotter import Order

from zipline.data.minute_bars import BcolzMinuteBarReader
from zipline.data.data_portal import DataPortal
from zipline.protocol import BarData
from zipline.testing.core import write_bcolz_minute_data


class SlippageTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()
        cls.env = TradingEnvironment()

        cls.sim_params = SimulationParameters(
            period_start=pd.Timestamp("2006-01-05 14:31", tz="utc"),
            period_end=pd.Timestamp("2006-01-05 14:36", tz="utc"),
            capital_base=1.0e5,
            data_frequency="minute",
            emission_rate='daily',
            env=cls.env
        )

        cls.sids = [133]

        cls.minutes = pd.DatetimeIndex(
            start=pd.Timestamp("2006-01-05 14:31", tz="utc"),
            end=pd.Timestamp("2006-01-05 14:35", tz="utc"),
            freq="1min"
        )

        assets = {
            133: pd.DataFrame({
                "open": np.array([3.0, 3.0, 3.5, 4.0, 3.5]),
                "high": np.array([3.15, 3.15, 3.15, 3.15, 3.15]),
                "low": np.array([2.85, 2.85, 2.85, 2.85, 2.85]),
                "close": np.array([3.0, 3.5, 4.0, 3.5, 3.0]),
                "volume": [2000, 2000, 2000, 2000, 2000],
                "dt": cls.minutes
            }).set_index("dt")
        }

        write_bcolz_minute_data(
            cls.env,
            pd.date_range(
                start=normalize_date(cls.minutes[0]),
                end=normalize_date(cls.minutes[-1])
            ),
            cls.tempdir.path,
            assets
        )

        cls.env.write_data(equities_data={
            133: {
                "start_date": pd.Timestamp("2006-01-05", tz='utc'),
                "end_date": pd.Timestamp("2006-01-07", tz='utc')
            }
        })

        cls.ASSET133 = cls.env.asset_finder.retrieve_asset(133)

        cls.data_portal = DataPortal(
            cls.env,
            equity_minute_reader=BcolzMinuteBarReader(cls.tempdir.path),
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()
        del cls.env

    def test_volume_share_slippage(self):
        tempdir = TempDirectory()

        try:
            assets = {
                133: pd.DataFrame({
                    "open": [3.00],
                    "high": [3.15],
                    "low": [2.85],
                    "close": [3.00],
                    "volume": [200],
                    "dt": [self.minutes[0]]
                }).set_index("dt")
            }

            write_bcolz_minute_data(
                self.env,
                pd.date_range(
                    start=normalize_date(self.minutes[0]),
                    end=normalize_date(self.minutes[-1])
                ),
                tempdir.path,
                assets
            )

            equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

            data_portal = DataPortal(
                self.env,
                equity_minute_reader=equity_minute_reader,
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

        finally:
            tempdir.cleanup()

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
        tempdir = TempDirectory()
        try:
            data = order_data
            data['sid'] = self.ASSET133

            order = Order(**data)

            assets = {
                133: pd.DataFrame({
                    "open": [event_data["open"]],
                    "high": [event_data["high"]],
                    "low": [event_data["low"]],
                    "close": [event_data["close"]],
                    "volume": [event_data["volume"]],
                    "dt": [pd.Timestamp('2006-01-05 14:31', tz='UTC')]
                }).set_index("dt")
            }

            write_bcolz_minute_data(
                self.env,
                pd.date_range(
                    start=normalize_date(self.minutes[0]),
                    end=normalize_date(self.minutes[-1])
                ),
                tempdir.path,
                assets
            )

            equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

            data_portal = DataPortal(
                self.env,
                equity_minute_reader=equity_minute_reader,
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
        finally:
            tempdir.cleanup()

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
