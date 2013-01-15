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

import collections

import unittest
from nose_parameterized import parameterized
import copy
import random
import datetime
import pytz
import itertools
from operator import attrgetter

import zipline.utils.factory as factory
import zipline.finance.performance as perf
from zipline.utils.protocol_utils import ndict

from zipline.gens.composites import date_sorted_sources

from zipline.finance.trading import TradingEnvironment


class TestPerformance(unittest.TestCase):

    def setUp(self):
        self.onesec = datetime.timedelta(seconds=1)
        self.oneday = datetime.timedelta(days=1)
        self.tradingday = datetime.timedelta(hours=6, minutes=30)

        self.trading_environment, self.dt, self.end_dt = self.create_env()

    def create_env(self, start_dt=None):
        benchmark_returns, treasury_curves = \
            factory.load_market_data()

        if not start_dt:
            for n in range(100):

                random_index = random.randint(
                    0,
                    len(treasury_curves)
                )

                start_dt = treasury_curves.keys()[random_index]
                end_dt = start_dt + datetime.timedelta(days=365)

                now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

                if end_dt <= now:
                    break
        else:
            end_dt = start_dt + datetime.timedelta(days=365)
            now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

        assert end_dt <= now, """
failed to find a date suitable daterange after 100 attempts. please double
check treasury and benchmark data in findb, and re-run the test."""
        assert start_dt < end_dt, "start_dt must be less than end_dt"

        trading_environment = TradingEnvironment(
            benchmark_returns,
            treasury_curves,
            period_start=start_dt,
            period_end=end_dt
        )

        return trading_environment, start_dt, end_dt

    def test_long_position(self):
        """
            verify that the performance period calculates properly for a
            single buy transaction
        """
        #post some trades in the market
        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            self.onesec,
            self.trading_environment
        )

        txn = factory.create_txn(1, 10.0, 100, self.dt + self.onesec)
        pp = perf.PerformancePeriod(1000.0)

        pp.execute_transaction(txn)
        for trade in trades:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(
            pp.period_capital_used,
            -1 * txn.price * txn.amount,
            "capital used should be equal to the opposite of the transaction \
            cost of sole txn in test"
        )

        self.assertEqual(
            len(pp.positions),
            1,
            "should be just one position")

        self.assertEqual(
            pp.positions[1].sid,
            txn.sid,
            "position should be in security with id 1")

        self.assertEqual(
            pp.positions[1].amount,
            txn.amount,
            "should have a position of {sharecount} shares".format(
                sharecount=txn.amount
            )
        )

        self.assertEqual(
            pp.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            pp.positions[1].last_sale_price,
            trades[-1]['price'],
            "last sale should be same as last trade. \
            expected {exp} actual {act}".format(
            exp=trades[-1]['price'],
            act=pp.positions[1].last_sale_price)
        )

        self.assertEqual(
            pp.ending_value,
            1100,
            "ending value should be price of last trade times number of \
            shares in position"
        )

        self.assertEqual(pp.pnl, 100, "gain of 1 on 100 shares should be 100")

    def test_short_position(self):
        """verify that the performance period calculates properly for a \
single short-sale transaction"""
        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11, 10, 9],
            [100, 100, 100, 100, 100, 100],
            self.onesec,
            self.trading_environment
        )

        trades_1 = trades[:-2]

        txn = factory.create_txn(1, 10.0, -100, self.dt + self.onesec)
        pp = perf.PerformancePeriod(1000.0)

        pp.execute_transaction(txn)
        for trade in trades_1:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(
            pp.period_capital_used,
            -1 * txn.price * txn.amount,
            "capital used should be equal to the opposite of the transaction\
             cost of sole txn in test"
        )

        self.assertEqual(
            len(pp.positions),
            1,
            "should be just one position")

        self.assertEqual(
            pp.positions[1].sid,
            txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            pp.positions[1].amount,
            -100,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            pp.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            pp.positions[1].last_sale_price,
            trades_1[-1]['price'],
            "last sale should be price of last trade"
        )

        self.assertEqual(
            pp.ending_value,
            -1100,
            "ending value should be price of last trade times number of \
            shares in position"
        )

        self.assertEqual(pp.pnl, -100, "gain of 1 on 100 shares should be 100")

        # simulate additional trades, and ensure that the position value
        # reflects the new price
        trades_2 = trades[-2:]

        #simulate a rollover to a new period
        pp.rollover()

        for trade in trades_2:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(
            pp.period_capital_used,
            0,
            "capital used should be zero, there were no transactions in \
            performance period"
        )

        self.assertEqual(
            len(pp.positions),
            1,
            "should be just one position"
        )

        self.assertEqual(
            pp.positions[1].sid,
            txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            pp.positions[1].amount,
            -100,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            pp.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            pp.positions[1].last_sale_price,
            trades_2[-1].price,
            "last sale should be price of last trade"
        )

        self.assertEqual(
            pp.ending_value,
            -900,
            "ending value should be price of last trade times number of \
            shares in position")

        self.assertEqual(
            pp.pnl,
            200,
            "drop of 2 on -100 shares should be 200"
        )

        #now run a performance period encompassing the entire trade sample.
        ppTotal = perf.PerformancePeriod(1000.0)

        for trade in trades_1:
            ppTotal.update_last_sale(trade)

        ppTotal.execute_transaction(txn)

        for trade in trades_2:
            ppTotal.update_last_sale(trade)

        ppTotal.calculate_performance()

        self.assertEqual(
            ppTotal.period_capital_used,
            -1 * txn.price * txn.amount,
            "capital used should be equal to the opposite of the transaction \
cost of sole txn in test"
        )

        self.assertEqual(
            len(ppTotal.positions),
            1,
            "should be just one position"
        )
        self.assertEqual(
            ppTotal.positions[1].sid,
            txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            ppTotal.positions[1].amount,
            -100,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            ppTotal.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            ppTotal.positions[1].last_sale_price,
            trades_2[-1].price,
            "last sale should be price of last trade"
        )

        self.assertEqual(
            ppTotal.ending_value,
            -900,
            "ending value should be price of last trade times number of \
            shares in position")

        self.assertEqual(
            ppTotal.pnl,
            100,
            "drop of 1 on -100 shares should be 100"
        )

    def test_covering_short(self):
        """verify performance where short is bought and covered, and shares \
trade after cover"""

        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11, 9, 8, 7, 8, 9, 10],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            self.onesec,
            self.trading_environment
        )

        short_txn = factory.create_txn(
            1,
            10.0,
            -100,
            self.dt + self.onesec
        )

        cover_txn = factory.create_txn(1, 7.0, 100, self.dt + self.onesec * 6)
        pp = perf.PerformancePeriod(1000.0)

        pp.execute_transaction(short_txn)
        pp.execute_transaction(cover_txn)

        for trade in trades:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        short_txn_cost = short_txn.price * short_txn.amount
        cover_txn_cost = cover_txn.price * cover_txn.amount

        self.assertEqual(
            pp.period_capital_used,
            -1 * short_txn_cost - cover_txn_cost,
            "capital used should be equal to the net transaction costs"
        )

        self.assertEqual(
            len(pp.positions),
            1,
            "should be just one position"
        )

        self.assertEqual(
            pp.positions[1].sid,
            short_txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            pp.positions[1].amount,
            0,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            pp.positions[1].cost_basis,
            0,
            "a covered position should have a cost basis of 0"
        )

        self.assertEqual(
            pp.positions[1].last_sale_price,
            trades[-1].price,
            "last sale should be price of last trade"
        )

        self.assertEqual(
            pp.ending_value,
            0,
            "ending value should be price of last trade times number of \
shares in position"
        )

        self.assertEqual(
            pp.pnl,
            300,
            "gain of 1 on 100 shares should be 300"
        )

    def test_cost_basis_calc(self):
        trades = factory.create_trade_history(
            1,
            [10, 11, 11, 12],
            [100, 100, 100, 100],
            self.onesec,
            self.trading_environment
        )

        transactions = factory.create_txn_history(
            1,
            [10, 11, 11, 12],
            [100, 100, 100, 100],
            self.onesec,
            self.trading_environment
        )

        pp = perf.PerformancePeriod(1000.0)

        for txn in transactions:
            pp.execute_transaction(txn)

        for trade in trades:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(
            pp.positions[1].last_sale_price,
            trades[-1].price,
            "should have a last sale of 12, got {val}".format(
                val=pp.positions[1].last_sale_price)
        )

        self.assertEqual(
            pp.positions[1].cost_basis,
            11,
            "should have a cost basis of 11"
        )

        self.assertEqual(
            pp.pnl,
            400
        )

        saleTxn = factory.create_txn(
            1,
            10.0,
            -100,
            self.dt + self.onesec * 4)

        down_tick = factory.create_trade(
            1,
            10.0,
            100,
            trades[-1].dt + self.onesec)

        pp.rollover()

        pp.execute_transaction(saleTxn)
        pp.update_last_sale(down_tick)

        pp.calculate_performance()
        self.assertEqual(
            pp.positions[1].last_sale_price,
            10,
            "should have a last sale of 10, was {val}".format(
                val=pp.positions[1].last_sale_price)
        )

        self.assertEqual(
            round(pp.positions[1].cost_basis, 2),
            11.33,
            "should have a cost basis of 11.33"
        )

        #print "second period pnl is {pnl}".format(pnl=pp2.pnl)
        self.assertEqual(pp.pnl, -800, "this period goes from +400 to -400")

        pp3 = perf.PerformancePeriod(1000.0)

        transactions.append(saleTxn)
        for txn in transactions:
            pp3.execute_transaction(txn)

        trades.append(down_tick)
        for trade in trades:
            pp3.update_last_sale(trade)

        pp3.calculate_performance()
        self.assertEqual(
            pp3.positions[1].last_sale_price,
            10,
            "should have a last sale of 10"
        )

        self.assertEqual(
            round(pp3.positions[1].cost_basis, 2),
            11.33,
            "should have a cost basis of 11.33"
        )

        self.assertEqual(
            pp3.pnl,
            -400,
            "should be -400 for all trades and transactions in period"
        )


class TestPerformanceTracker(unittest.TestCase):

    NumDaysToDelete = collections.namedtuple(
        'NumDaysToDelete', ('start', 'middle', 'end'))

    @parameterized.expand([
        ("Don't delete any events",
         NumDaysToDelete(start=0, middle=0, end=0)),
        ("Delete first day of events",
         NumDaysToDelete(start=1, middle=0, end=0)),
        ("Delete first two days of events",
         NumDaysToDelete(start=2, middle=0, end=0)),
        ("Delete one day of events from the middle",
         NumDaysToDelete(start=0, middle=1, end=0)),
        ("Delete two events from the middle",
         NumDaysToDelete(start=0, middle=2, end=0)),
        ("Delete last day of events",
         NumDaysToDelete(start=0, middle=0, end=1)),
        ("Delete last two days of events",
         NumDaysToDelete(start=0, middle=0, end=2)),
        ("Delete all but one event.",
         NumDaysToDelete(start=2, middle=1, end=2)),
    ])
    def test_tracker(self, parameter_comment, days_to_delete):
        """
        @days_to_delete - configures which days in the data set we should
        remove, used for ensuring that we still return performance messages
        even when there is no data.
        """
        # This date range covers Columbus day,
        # however Columbus day is not a market holiday
        #
        #     October 2008
        # Su Mo Tu We Th Fr Sa
        #           1  2  3  4
        #  5  6  7  8  9 10 11
        # 12 13 14 15 16 17 18
        # 19 20 21 22 23 24 25
        # 26 27 28 29 30 31
        start_dt = datetime.datetime(year=2008,
                                     month=10,
                                     day=9,
                                     tzinfo=pytz.utc)
        end_dt = datetime.datetime(year=2008,
                                   month=10,
                                   day=16,
                                   tzinfo=pytz.utc)

        trade_count = 6
        sid = 133
        price = 10.1
        price_list = [price] * trade_count
        volume = [100] * trade_count
        trade_time_increment = datetime.timedelta(days=1)

        benchmark_returns, treasury_curves = \
            factory.load_market_data()

        trading_environment = TradingEnvironment(
            benchmark_returns,
            treasury_curves,
            period_start=start_dt,
            period_end=end_dt
        )

        trade_history = factory.create_trade_history(
            sid,
            price_list,
            volume,
            trade_time_increment,
            trading_environment,
            source_id="factory1"
        )

        sid2 = 134
        price2 = 12.12
        price2_list = [price2] * trade_count
        trade_history2 = factory.create_trade_history(
            sid2,
            price2_list,
            volume,
            trade_time_increment,
            trading_environment,
            source_id="factory2"
        )
        # 'middle' start of 3 depends on number of days == 7
        middle = 3

        # First delete from middle
        if days_to_delete.middle:
            del trade_history[middle:(middle + days_to_delete.middle)]
            del trade_history2[middle:(middle + days_to_delete.middle)]

        # Delete start
        if days_to_delete.start:
            del trade_history[:days_to_delete.start]
            del trade_history2[:days_to_delete.start]

        # Delete from end
        if days_to_delete.end:
            del trade_history[-days_to_delete.end:]
            del trade_history2[-days_to_delete.end:]

        trading_environment.first_open = \
            trading_environment.calculate_first_open()
        trading_environment.last_close = \
            trading_environment.calculate_last_close()
        trading_environment.capital_base = 1000.0
        trading_environment.frame_index = [
            'sid',
            'volume',
            'dt',
            'price',
            'changed']
        perf_tracker = perf.PerformanceTracker(
            trading_environment
        )

        events = date_sorted_sources(trade_history, trade_history2)

        events = [self.event_with_txn(event, trade_history[0].dt)
                  for event in events]

        # Extract events with transactions to use for verification.
        events_with_txns = [event for event in events if event.TRANSACTION]

        perf_messages = \
            [msg for date, snapshot in
             perf_tracker.transform(
                 itertools.groupby(events, attrgetter('dt')))
             for event in snapshot
             for msg in event.perf_messages]

        end_perf_messages, risk_message = perf_tracker.handle_simulation_end()

        perf_messages.extend(end_perf_messages)

        #we skip two trades, to test case of None transaction
        self.assertEqual(perf_tracker.txn_count, len(events_with_txns))

        cumulative_pos = perf_tracker.cumulative_performance.positions[sid]
        expected_size = len(events_with_txns) / 2 * -25
        self.assertEqual(cumulative_pos.amount, expected_size)

        self.assertEqual(perf_tracker.last_close,
                         perf_tracker.cumulative_risk_metrics.end_date)

        self.assertEqual(len(perf_messages),
                         trading_environment.days_in_period)

    def event_with_txn(self, event, no_txn_dt):
        #create a transaction for all but
        #first trade in each sid, to simulate None transaction
        if event.dt != no_txn_dt:
            txn = ndict({
                'sid': event.sid,
                'amount': -25,
                'dt': event.dt,
                'price': 10.0,
                'commission': 0.50
            })
        else:
            txn = None
        event['TRANSACTION'] = txn

        return event
