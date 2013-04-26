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
Tests for the zipline.finance package
"""
import itertools
import operator

import pytz

from unittest import TestCase
from datetime import datetime, timedelta

import numpy as np

from nose.tools import timed

import zipline.protocol
from zipline.protocol import Event, DATASOURCE_TYPE

import zipline.utils.factory as factory
import zipline.utils.simfactory as simfactory

from zipline.finance.blotter import Blotter
from zipline.gens.composites import date_sorted_sources

import zipline.finance.trading as trading
from zipline.finance.trading import SimulationParameters

from zipline.finance.performance import PerformanceTracker
from zipline.utils.test_utils import(
    setup_logger,
    teardown_logger,
    assert_single_position
)

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90


class FinanceTestCase(TestCase):

    def setUp(self):
        self.zipline_test_config = {
            'sid': 133,
        }

        setup_logger(self)

    def tearDown(self):
        teardown_logger(self)

    @timed(DEFAULT_TIMEOUT)
    def test_factory_daily(self):
        sim_params = factory.create_simulation_parameters()
        trade_source = factory.create_daily_trade_source(
            [133],
            200,
            sim_params
        )
        prev = None
        for trade in trade_source:
            if prev:
                self.assertTrue(trade.dt > prev.dt)
            prev = trade

    @timed(DEFAULT_TIMEOUT)
    def test_trading_environment(self):
        #holidays taken from: http://www.nyse.com/press/1191407641943.html
        new_years = datetime(2008, 1, 1, tzinfo=pytz.utc)
        mlk_day = datetime(2008, 1, 21, tzinfo=pytz.utc)
        presidents = datetime(2008, 2, 18, tzinfo=pytz.utc)
        good_friday = datetime(2008, 3, 21, tzinfo=pytz.utc)
        memorial_day = datetime(2008, 5, 26, tzinfo=pytz.utc)
        july_4th = datetime(2008, 7, 4, tzinfo=pytz.utc)
        labor_day = datetime(2008, 9, 1, tzinfo=pytz.utc)
        tgiving = datetime(2008, 11, 27, tzinfo=pytz.utc)
        christmas = datetime(2008, 5, 25, tzinfo=pytz.utc)
        a_saturday = datetime(2008, 8, 2, tzinfo=pytz.utc)
        a_sunday = datetime(2008, 10, 12, tzinfo=pytz.utc)
        holidays = [
            new_years,
            mlk_day,
            presidents,
            good_friday,
            memorial_day,
            july_4th,
            labor_day,
            tgiving,
            christmas,
            a_saturday,
            a_sunday
        ]

        for holiday in holidays:
            self.assertTrue(not trading.environment.is_trading_day(holiday))

        first_trading_day = datetime(2008, 1, 2, tzinfo=pytz.utc)
        last_trading_day = datetime(2008, 12, 31, tzinfo=pytz.utc)
        workdays = [first_trading_day, last_trading_day]

        for workday in workdays:
            self.assertTrue(trading.environment.is_trading_day(workday))

    def test_simulation_parameters(self):
        env = SimulationParameters(
            period_start=datetime(2008, 1, 1, tzinfo=pytz.utc),
            period_end=datetime(2008, 12, 31, tzinfo=pytz.utc),
            capital_base=100000,
        )

        self.assertTrue(env.last_close.month == 12)
        self.assertTrue(env.last_close.day == 31)

    @timed(DEFAULT_TIMEOUT)
    def test_sim_params_days_in_period(self):

        #     January 2008
        #  Su Mo Tu We Th Fr Sa
        #         1  2  3  4  5
        #   6  7  8  9 10 11 12
        #  13 14 15 16 17 18 19
        #  20 21 22 23 24 25 26
        #  27 28 29 30 31

        env = SimulationParameters(
            period_start=datetime(2007, 12, 31, tzinfo=pytz.utc),
            period_end=datetime(2008, 1, 7, tzinfo=pytz.utc),
            capital_base=100000,
        )

        expected_trading_days = (
            datetime(2007, 12, 31, tzinfo=pytz.utc),
            # Skip new years
            #holidays taken from: http://www.nyse.com/press/1191407641943.html
            datetime(2008, 1, 2, tzinfo=pytz.utc),
            datetime(2008, 1, 3, tzinfo=pytz.utc),
            datetime(2008, 1, 4, tzinfo=pytz.utc),
            # Skip Saturday
            # Skip Sunday
            datetime(2008, 1, 7, tzinfo=pytz.utc)
        )

        num_expected_trading_days = 5
        self.assertEquals(num_expected_trading_days, env.days_in_period)
        np.testing.assert_array_equal(expected_trading_days,
                                      env.trading_days.tolist())

    @timed(EXTENDED_TIMEOUT)
    def test_full_zipline(self):
        #provide enough trades to ensure all orders are filled.
        self.zipline_test_config['order_count'] = 100
        # making a small order amount, so that each order is filled
        # in a single transaction, and txn_count == order_count.
        self.zipline_test_config['order_amount'] = 25
        # No transactions can be filled on the first trade, so
        # we have one extra trade to ensure all orders are filled.
        self.zipline_test_config['trade_count'] = 101
        full_zipline = simfactory.create_test_zipline(
            **self.zipline_test_config)
        assert_single_position(self, full_zipline)

    # TODO: write tests for short sales
    # TODO: write a test to do massive buying or shorting.

    @timed(DEFAULT_TIMEOUT)
    def test_partially_filled_orders(self):

        # create a scenario where order size and trade size are equal
        # so that orders must be spread out over several trades.
        params = {
            'trade_count': 360,
            'trade_amount': 100,
            'trade_interval': timedelta(minutes=1),
            'order_count': 2,
            'order_amount': 100,
            'order_interval': timedelta(minutes=1),
            # because we placed an order for 100 shares, and the volume
            # of each trade is 100, the simulator should spread the order
            # into 4 trades of 25 shares per order.
            'expected_txn_count': 8,
            'expected_txn_volume': 2 * 100
        }

        self.transaction_sim(**params)

        # same scenario, but with short sales
        params2 = {
            'trade_count': 360,
            'trade_amount': 100,
            'trade_interval': timedelta(minutes=1),
            'order_count': 2,
            'order_amount': -100,
            'order_interval': timedelta(minutes=1),
            'expected_txn_count': 8,
            'expected_txn_volume': 2 * -100
        }

        self.transaction_sim(**params2)

    @timed(DEFAULT_TIMEOUT)
    def test_collapsing_orders(self):
        # create a scenario where order.amount <<< trade.volume
        # to test that several orders can be covered properly by one trade,
        # but are represented by multiple transactions.
        params1 = {
            'trade_count': 6,
            'trade_amount': 100,
            'trade_interval': timedelta(hours=1),
            'order_count': 24,
            'order_amount': 1,
            'order_interval': timedelta(minutes=1),
            # because we placed an orders totaling less than 25% of one trade
            # the simulator should produce just one transaction.
            'expected_txn_count': 24,
            'expected_txn_volume': 24
        }
        self.transaction_sim(**params1)

        # second verse, same as the first. except short!
        params2 = {
            'trade_count': 6,
            'trade_amount': 100,
            'trade_interval': timedelta(hours=1),
            'order_count': 24,
            'order_amount': -1,
            'order_interval': timedelta(minutes=1),
            'expected_txn_count': 24,
            'expected_txn_volume': -24
        }
        self.transaction_sim(**params2)

        # Runs the collapsed trades over daily trade intervals.
        # Ensuring that our delay works for daily intervals as well.
        params3 = {
            'trade_count': 6,
            'trade_amount': 100,
            'trade_interval': timedelta(days=1),
            'order_count': 24,
            'order_amount': 1,
            'order_interval': timedelta(minutes=1),
            'expected_txn_count': 24,
            'expected_txn_volume': 24
        }
        self.transaction_sim(**params3)

    @timed(DEFAULT_TIMEOUT)
    def test_alternating_long_short(self):
        # create a scenario where we alternate buys and sells
        params1 = {
            'trade_count': int(6.5 * 60 * 4),
            'trade_amount': 100,
            'trade_interval': timedelta(minutes=1),
            'order_count': 4,
            'order_amount': 10,
            'order_interval': timedelta(hours=24),
            'alternate': True,
            'complete_fill': True,
            'expected_txn_count': 4,
            'expected_txn_volume': 0  # equal buys and sells
        }
        self.transaction_sim(**params1)

    def transaction_sim(self, **params):
        """ This is a utility method that asserts expected
        results for conversion of orders to transactions given a
        trade history"""

        trade_count = params['trade_count']
        trade_interval = params['trade_interval']
        order_count = params['order_count']
        order_amount = params['order_amount']
        order_interval = params['order_interval']
        expected_txn_count = params['expected_txn_count']
        expected_txn_volume = params['expected_txn_volume']
        # optional parameters
        # ---------------------
        # if present, alternate between long and short sales
        alternate = params.get('alternate')
        # if present, expect transaction amounts to match orders exactly.
        complete_fill = params.get('complete_fill')

        sid = 1
        sim_params = factory.create_simulation_parameters()
        blotter = Blotter()
        price = [10.1] * trade_count
        volume = [100] * trade_count
        start_date = sim_params.first_open

        generated_trades = factory.create_trade_history(
            sid,
            price,
            volume,
            trade_interval,
            sim_params
        )

        if alternate:
            alternator = -1
        else:
            alternator = 1

        order_date = start_date
        for i in xrange(order_count):

            blotter.set_date(order_date)
            blotter.order(sid, order_amount * alternator ** i, None, None)

            order_date = order_date + order_interval
            # move after market orders to just after market next
            # market open.
            if order_date.hour >= 21:
                    if order_date.minute >= 00:
                        order_date = order_date + timedelta(days=1)
                        order_date = order_date.replace(hour=14, minute=30)

        # there should now be one open order list stored under the sid
        oo = blotter.open_orders
        self.assertEqual(len(oo), 1)
        self.assertTrue(sid in oo)
        order_list = oo[sid]
        self.assertEqual(order_count, len(order_list))

        for i in xrange(order_count):
            order = order_list[i]
            self.assertEqual(order.sid, sid)
            self.assertEqual(order.amount, order_amount * alternator ** i)

        tracker = PerformanceTracker(sim_params)

        benchmark_returns = [
            Event({'dt': ret.date,
                   'returns': ret.returns,
                   'type':
                   zipline.protocol.DATASOURCE_TYPE.BENCHMARK,
                   'source_id': 'benchmarks'})
            for ret in trading.environment.benchmark_returns
            if ret.date.date() >= sim_params.period_start.date()
            and ret.date.date() <= sim_params.period_end.date()
        ]

        generated_events = date_sorted_sources(generated_trades,
                                               benchmark_returns)

        # this approximates the loop inside TradingSimulationClient
        transactions = []
        for dt, events in itertools.groupby(generated_events,
                                            operator.attrgetter('dt')):
            for event in events:
                if event.type == DATASOURCE_TYPE.TRADE:

                    txns, _ = blotter.process_trade(event)

                    for txn in txns:
                        transactions.append(txn)
                        tracker.process_event(txn)

                tracker.process_event(event)

        if complete_fill:
            self.assertEqual(len(transactions), len(order_list))

        total_volume = 0
        for i in xrange(len(transactions)):
            txn = transactions[i]
            total_volume += txn.amount
            if complete_fill:
                order = order_list[i]
                self.assertEqual(order.amount, txn.amount)

        self.assertEqual(total_volume, expected_txn_volume)
        self.assertEqual(len(transactions), expected_txn_count)

        cumulative_pos = tracker.cumulative_performance.positions[sid]
        self.assertEqual(total_volume, cumulative_pos.amount)

        # the open orders should now be empty
        oo = blotter.open_orders
        self.assertTrue(sid in oo)
        order_list = oo[sid]
        self.assertEqual(0, len(order_list))
