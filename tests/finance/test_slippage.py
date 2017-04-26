#
# Copyright 2017 Quantopian, Inc.
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
from collections import namedtuple
import datetime
from math import sqrt

from nose_parameterized import parameterized
from pandas.tslib import normalize_date
import numpy as np
import pandas as pd
import pytz

from zipline.assets import Equity
from zipline.data.data_portal import DataPortal
from zipline.finance.asset_restrictions import NoRestrictions
from zipline.finance.order import Order
from zipline.finance.slippage import (
    fill_price_worse_than_limit_price,
    MarketImpactBase,
    NO_DATA_VOLATILITY_SLIPPAGE_IMPACT,
    VolatilityVolumeShare,
    VolumeShareSlippage,
)
from zipline.protocol import DATASOURCE_TYPE, BarData
from zipline.testing import (
    create_minute_bar_data,
    tmp_bcolz_equity_minute_bar_reader,
)
from zipline.testing.fixtures import (
    WithCreateBarData,
    WithDataPortal,
    WithSimParams,
    WithTradingEnvironment,
    ZiplineTestCase,
)
from zipline.utils.classproperty import classproperty


TestOrder = namedtuple('TestOrder', 'limit direction')


class SlippageTestCase(WithCreateBarData,
                       WithSimParams,
                       WithDataPortal,
                       ZiplineTestCase):
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

    @classproperty
    def CREATE_BARDATA_DATA_FREQUENCY(cls):
        return cls.sim_params.data_frequency

    @classmethod
    def make_equity_minute_bar_data(cls):
        yield 133, pd.DataFrame(
            {
                'open': [3.0, 3.0, 3.5, 4.0, 3.5],
                'high': [3.15, 3.15, 3.15, 3.15, 3.15],
                'low': [2.85, 2.85, 2.85, 2.85, 2.85],
                'close': [3.0, 3.5, 4.0, 3.5, 3.0],
                'volume': [2000, 2000, 2000, 2000, 2000],
            },
            index=cls.minutes,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(SlippageTestCase, cls).init_class_fixtures()
        cls.ASSET133 = cls.env.asset_finder.retrieve_asset(133)

    def test_equality_and_comparison(self):
        vol1 = VolumeShareSlippage(volume_limit=0.2)
        vol2 = VolumeShareSlippage(volume_limit=0.2)

        self.assertEqual(vol1, vol2)
        self.assertEqual(hash(vol1), hash(vol2))

        self.assertEqual(vol1.__dict__, vol1.asdict())
        self.assertEqual(vol2.__dict__, vol2.asdict())

    def test_fill_price_worse_than_limit_price(self):
        non_limit_order = TestOrder(limit=None, direction=1)
        limit_buy = TestOrder(limit=1.5, direction=1)
        limit_sell = TestOrder(limit=1.5, direction=-1)

        for price in [1, 1.5, 2]:
            self.assertFalse(
                fill_price_worse_than_limit_price(price, non_limit_order)
            )

        self.assertFalse(fill_price_worse_than_limit_price(1, limit_buy))
        self.assertFalse(fill_price_worse_than_limit_price(1.5, limit_buy))
        self.assertTrue(fill_price_worse_than_limit_price(2, limit_buy))

        self.assertTrue(fill_price_worse_than_limit_price(1, limit_sell))
        self.assertFalse(fill_price_worse_than_limit_price(1.5, limit_sell))
        self.assertFalse(fill_price_worse_than_limit_price(2, limit_sell))

    def test_orders_limit(self):
        slippage_model = VolumeShareSlippage()
        slippage_model.data_portal = self.data_portal

        # long, does not trade
        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': 100,
                'filled': 0,
                'asset': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[3],
        )

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
                'asset': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[3],
        )

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
                'asset': self.ASSET133,
                'limit': 3.6})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[3],
        )

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
            'asset': self.ASSET133,
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
                'asset': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

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
                'asset': self.ASSET133,
                'limit': 3.5})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

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
                'asset': self.ASSET133,
                'limit': 3.4})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[1],
        )

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
            'asset': self.ASSET133,
        }

        self.assertIsNotNone(txn)

        for key, value in expected_txn.items():
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
                'asset': self.ASSET133,
                'stop': 4.0,
                'limit': 3.0})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[2],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[3],
        )

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
                'asset': self.ASSET133,
                'stop': 4.0,
                'limit': 3.5})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[2],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[3],
        )

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
                'asset': self.ASSET133,
                'stop': 4.0,
                'limit': 3.6})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[2],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[3],
        )

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
            'asset': self.ASSET133
        }

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])

        # short, does not trade

        open_orders = [
            Order(**{
                'dt': datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                'amount': -100,
                'filled': 0,
                'asset': self.ASSET133,
                'stop': 3.0,
                'limit': 4.0})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[1],
        )

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
                'asset': self.ASSET133,
                'stop': 3.0,
                'limit': 3.5})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[1],
        )

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
                'asset': self.ASSET133,
                'stop': 3.0,
                'limit': 3.4})
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[1],
        )

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
            'asset': self.ASSET133,
        }

        for key, value in expected_txn.items():
            self.assertEquals(value, txn[key])


class VolumeShareSlippageTestCase(WithCreateBarData,
                                  WithSimParams,
                                  WithDataPortal,
                                  ZiplineTestCase):

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

    @classproperty
    def CREATE_BARDATA_DATA_FREQUENCY(cls):
        return cls.sim_params.data_frequency

    @classmethod
    def make_equity_minute_bar_data(cls):
        yield 133, pd.DataFrame(
            {
                'open': [3.00],
                'high': [3.15],
                'low': [2.85],
                'close': [3.00],
                'volume': [200],
            },
            index=[cls.minutes[0]],
        )

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame({
            'sid': [1000],
            'root_symbol': ['CL'],
            'symbol': ['CLF06'],
            'start_date': [cls.ASSET_FINDER_EQUITY_START_DATE],
            'end_date': [cls.ASSET_FINDER_EQUITY_END_DATE],
            'multiplier': [500],
            'exchange': ['CME'],
        })

    @classmethod
    def make_future_minute_bar_data(cls):
        yield 1000, pd.DataFrame(
            {
                'open': [5.00],
                'high': [5.15],
                'low': [4.85],
                'close': [5.00],
                'volume': [100],
            },
            index=[cls.minutes[0]],
        )

    @classmethod
    def init_class_fixtures(cls):
        super(VolumeShareSlippageTestCase, cls).init_class_fixtures()
        cls.ASSET133 = cls.env.asset_finder.retrieve_asset(133)
        cls.ASSET1000 = cls.env.asset_finder.retrieve_asset(1000)

    def test_volume_share_slippage(self):

        slippage_model = VolumeShareSlippage()

        open_orders = [
            Order(
                dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                amount=100,
                filled=0,
                asset=self.ASSET133
            )
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

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
            'asset': self.ASSET133,
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
                asset=self.ASSET133
            )
        ]

        # Set bar_data to be a minute ahead of last trade.
        # Volume share slippage should not execute when there is no trade.
        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[1],
        )

        orders_txns = list(slippage_model.simulate(
            bar_data,
            self.ASSET133,
            open_orders,
        ))

        self.assertEquals(len(orders_txns), 0)

    def test_volume_share_slippage_with_future(self):
        slippage_model = VolumeShareSlippage(volume_limit=1, price_impact=0.3)

        open_orders = [
            Order(
                dt=datetime.datetime(2006, 1, 5, 14, 30, tzinfo=pytz.utc),
                amount=10,
                filled=0,
                asset=self.ASSET1000,
            ),
        ]

        bar_data = self.create_bardata(
            simulation_dt_func=lambda: self.minutes[0],
        )

        orders_txns = list(
            slippage_model.simulate(bar_data, self.ASSET1000, open_orders)
        )

        self.assertEquals(len(orders_txns), 1)
        _, txn = orders_txns[0]

        # We expect to fill the order for all 10 contracts. The volume for the
        # futures contract in this bar is 100, so our volume share is:
        #     10.0 / 100 = 0.1
        # The current price is 5.0 and the price impact is 0.3, so the expected
        # impacted price is:
        #     5.0 + (5.0 * (0.1 ** 2) * 0.3) = 5.015
        expected_txn = {
            'price': 5.015,
            'dt': datetime.datetime(2006, 1, 5, 14, 31, tzinfo=pytz.utc),
            'amount': 10,
            'asset': self.ASSET1000,
            'commission': None,
            'type': DATASOURCE_TYPE.TRANSACTION,
            'order_id': open_orders[0].id,
        }

        self.assertIsNotNone(txn)
        self.assertEquals(expected_txn, txn.__dict__)


class VolatilityVolumeShareTestCase(WithCreateBarData,
                                    WithSimParams,
                                    WithDataPortal,
                                    ZiplineTestCase):

    ASSET_START_DATE = pd.Timestamp('2006-02-10')

    TRADING_CALENDAR_STRS = ('NYSE', 'us_futures')
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

    @classmethod
    def init_class_fixtures(cls):
        super(VolatilityVolumeShareTestCase, cls).init_class_fixtures()
        cls.ASSET = cls.asset_finder.retrieve_asset(1000)

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame({
            'sid': [1000],
            'root_symbol': ['CL'],
            'symbol': ['CLF07'],
            'start_date': [cls.ASSET_START_DATE],
            'end_date': [cls.END_DATE],
            'multiplier': [500],
            'exchange': ['CME'],
        })

    @classmethod
    def make_future_minute_bar_data(cls):
        data = list(
            super(
                VolatilityVolumeShareTestCase, cls,
            ).make_future_minute_bar_data()
        )
        # Make the first month's worth of data NaN to simulate cases where a
        # futures contract does not exist yet.
        data[0][1].loc[:cls.ASSET_START_DATE] = np.NaN
        return data

    def test_calculate_impact_buy(self):
        answer_key = [
            # We ordered 10 contracts, but are capped at 100 * 0.05 = 5
            (91485.500085168125, 5),
            (91486.500085169057, 5),
            (None, None),
        ]
        order = Order(
            dt=pd.Timestamp.now(tz='utc').round('min'),
            asset=self.ASSET,
            amount=10,
        )
        self._calculate_impact(order, answer_key)

    def test_calculate_impact_sell(self):
        answer_key = [
            # We ordered -10 contracts, but are capped at -(100 * 0.05) = -5
            (91485.499914831875, -5),
            (91486.499914830943, -5),
            (None, None),
        ]
        order = Order(
            dt=pd.Timestamp.now(tz='utc').round('min'),
            asset=self.ASSET,
            amount=-10,
        )
        self._calculate_impact(order, answer_key)

    def _calculate_impact(self, test_order, answer_key):
        model = VolatilityVolumeShare(volume_limit=0.05)
        first_minute = pd.Timestamp('2006-03-31 11:35AM', tz='UTC')

        next_3_minutes = self.trading_calendar.minutes_window(first_minute, 3)
        remaining_shares = test_order.open_amount

        for i, minute in enumerate(next_3_minutes):
            data = self.create_bardata(simulation_dt_func=lambda: minute)
            new_order = Order(
                dt=data.current_dt, asset=self.ASSET, amount=remaining_shares,
            )
            price, amount = model.process_order(data, new_order)

            self.assertEqual(price, answer_key[i][0])
            self.assertEqual(amount, answer_key[i][1])

            amount = amount or 0
            if remaining_shares < 0:
                remaining_shares = min(0, remaining_shares - amount)
            else:
                remaining_shares = max(0, remaining_shares - amount)

    def test_calculate_impact_without_history(self):
        model = VolatilityVolumeShare(volume_limit=1)
        minutes = [
            # Start day of the futures contract; no history yet.
            pd.Timestamp('2006-02-10 11:35AM', tz='UTC'),
            # Only a week's worth of history data.
            pd.Timestamp('2006-02-17 11:35AM', tz='UTC'),
        ]

        for minute in minutes:
            data = self.create_bardata(simulation_dt_func=lambda: minute)

            order = Order(dt=data.current_dt, asset=self.ASSET, amount=10)
            price, amount = model.process_order(data, order)

            avg_price = (
                data.current(self.ASSET, 'high') +
                data.current(self.ASSET, 'low')
            ) / 2
            expected_price = \
                avg_price + (avg_price * NO_DATA_VOLATILITY_SLIPPAGE_IMPACT)

            self.assertEqual(price, expected_price)
            self.assertEqual(amount, 10)

    def test_impacted_price_worse_than_limit(self):
        model = VolatilityVolumeShare(volume_limit=0.05)

        # Use all the same numbers from the 'calculate_impact' tests. Since the
        # impacted price is 59805.5, which is worse than the limit price of
        # 59800, the model should return None.
        minute = pd.Timestamp('2006-03-01 11:35AM', tz='UTC')
        data = self.create_bardata(simulation_dt_func=lambda: minute)
        order = Order(
            dt=data.current_dt, asset=self.ASSET, amount=10, limit=59800,
        )
        price, amount = model.process_order(data, order)

        self.assertIsNone(price)
        self.assertIsNone(amount)

    def test_low_transaction_volume(self):
        # With a volume limit of 0.001, and a bar volume of 100, we should
        # compute a transaction volume of 100 * 0.001 = 0.1, which gets rounded
        # down to zero. In this case we expect no amount to be transacted.
        model = VolatilityVolumeShare(volume_limit=0.001)

        minute = pd.Timestamp('2006-03-01 11:35AM', tz='UTC')
        data = self.create_bardata(simulation_dt_func=lambda: minute)
        order = Order(dt=data.current_dt, asset=self.ASSET, amount=10)
        price, amount = model.process_order(data, order)

        self.assertIsNone(price)
        self.assertIsNone(amount)


class MarketImpactTestCase(WithCreateBarData, ZiplineTestCase):

    ASSET_FINDER_EQUITY_SIDS = (1,)

    @classmethod
    def make_equity_minute_bar_data(cls):
        trading_calendar = cls.trading_calendars[Equity]
        return create_minute_bar_data(
            trading_calendar.minutes_for_sessions_in_range(
                cls.equity_minute_bar_days[0],
                cls.equity_minute_bar_days[-1],
            ),
            cls.asset_finder.equities_sids,
        )

    def test_window_data(self):
        session = pd.Timestamp('2006-03-01')
        minute = self.trading_calendar.minutes_for_session(session)[1]
        data = self.create_bardata(simulation_dt_func=lambda: minute)
        asset = self.asset_finder.retrieve_asset(1)

        mean_volume, volatility = MarketImpactBase()._get_window_data(
            data, asset, window_length=20,
        )

        #                            close  volume
        # 2006-01-31 00:00:00+00:00   29.0   119.0
        # 2006-02-01 00:00:00+00:00   30.0   120.0
        # 2006-02-02 00:00:00+00:00   31.0   121.0
        # 2006-02-03 00:00:00+00:00   32.0   122.0
        # 2006-02-06 00:00:00+00:00   33.0   123.0
        # 2006-02-07 00:00:00+00:00   34.0   124.0
        # 2006-02-08 00:00:00+00:00   35.0   125.0
        # 2006-02-09 00:00:00+00:00   36.0   126.0
        # 2006-02-10 00:00:00+00:00   37.0   127.0
        # 2006-02-13 00:00:00+00:00   38.0   128.0
        # 2006-02-14 00:00:00+00:00   39.0   129.0
        # 2006-02-15 00:00:00+00:00   40.0   130.0
        # 2006-02-16 00:00:00+00:00   41.0   131.0
        # 2006-02-17 00:00:00+00:00   42.0   132.0
        # 2006-02-21 00:00:00+00:00   43.0   133.0
        # 2006-02-22 00:00:00+00:00   44.0   134.0
        # 2006-02-23 00:00:00+00:00   45.0   135.0
        # 2006-02-24 00:00:00+00:00   46.0   136.0
        # 2006-02-27 00:00:00+00:00   47.0   137.0
        # 2006-02-28 00:00:00+00:00   48.0   138.0

        # Mean volume is (119 + 138) / 2 = 128.5
        self.assertEqual(mean_volume, 128.5)

        # Volatility is closes.pct_change().std() * sqrt(252)
        reference_vol = pd.Series(range(29, 49)).pct_change().std() * sqrt(252)
        self.assertEqual(volatility, reference_vol)


class OrdersStopTestCase(WithSimParams,
                         WithTradingEnvironment,
                         ZiplineTestCase):

    START_DATE = pd.Timestamp('2006-01-05 14:31', tz='utc')
    END_DATE = pd.Timestamp('2006-01-05 14:36', tz='utc')
    SIM_PARAMS_CAPITAL_BASE = 1.0e5
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    SIM_PARAMS_EMISSION_RATE = 'daily'
    ASSET_FINDER_EQUITY_SIDS = (133,)
    minutes = pd.DatetimeIndex(
        start=START_DATE,
        end=END_DATE - pd.Timedelta('1 minute'),
        freq='1min'
    )

    @classmethod
    def init_class_fixtures(cls):
        super(OrdersStopTestCase, cls).init_class_fixtures()
        cls.ASSET133 = cls.env.asset_finder.retrieve_asset(133)

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
                'stop': 3.5
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 4.0,
                'high': 3.15,
                'low': 2.85,
                'close': 4.0,
                'open': 3.5
            },
            'expected': {
                'transaction': {
                    'price': 4.00025,
                    'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                    'amount': 50,
                }
            }
        },
        'long | price lt stop': {
            'order': {
                'dt': pd.Timestamp('2006-01-05 14:30', tz='UTC'),
                'amount': 100,
                'filled': 0,
                'stop': 3.6
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 3.5,
                'high': 3.15,
                'low': 2.85,
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
                'stop': 3.4
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 3.5,
                'high': 3.15,
                'low': 2.85,
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
                'stop': 3.5
            },
            'event': {
                'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                'volume': 2000,
                'price': 3.0,
                'high': 3.15,
                'low': 2.85,
                'close': 3.0,
                'open': 3.0
            },
            'expected': {
                'transaction': {
                    'price': 2.9998125,
                    'dt': pd.Timestamp('2006-01-05 14:31', tz='UTC'),
                    'amount': -50,
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
        data['asset'] = self.ASSET133
        order = Order(**data)

        if expected['transaction']:
            expected['transaction']['asset'] = self.ASSET133
        event_data['asset'] = self.ASSET133

        assets = (
            (133, pd.DataFrame(
                {
                    'open': [event_data['open']],
                    'high': [event_data['high']],
                    'low': [event_data['low']],
                    'close': [event_data['close']],
                    'volume': [event_data['volume']],
                },
                index=[pd.Timestamp('2006-01-05 14:31', tz='UTC')],
            )),
        )
        days = pd.date_range(
            start=normalize_date(self.minutes[0]),
            end=normalize_date(self.minutes[-1])
        )
        with tmp_bcolz_equity_minute_bar_reader(
                self.trading_calendar, days, assets) as reader:
            data_portal = DataPortal(
                self.env.asset_finder, self.trading_calendar,
                first_trading_day=reader.first_trading_day,
                equity_minute_reader=reader,
            )

            slippage_model = VolumeShareSlippage()

            try:
                dt = pd.Timestamp('2006-01-05 14:31', tz='UTC')
                bar_data = BarData(
                    data_portal,
                    lambda: dt,
                    self.sim_params.data_frequency,
                    self.trading_calendar,
                    NoRestrictions(),
                )

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
