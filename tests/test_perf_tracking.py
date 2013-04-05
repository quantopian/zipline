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

import collections

import unittest
from nose_parameterized import parameterized
import datetime
import pytz
import itertools
from operator import attrgetter

import zipline.utils.factory as factory
import zipline.finance.performance as perf
from zipline.finance.slippage import Transaction

from zipline.gens.composites import date_sorted_sources
from zipline.finance.trading import SimulationParameters
import zipline.finance.trading as trading
from zipline.utils.factory import create_random_simulation_parameters

onesec = datetime.timedelta(seconds=1)
oneday = datetime.timedelta(days=1)
tradingday = datetime.timedelta(hours=6, minutes=30)


class TestDividendPerformance(unittest.TestCase):

    def setUp(self):

        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        self.sim_params.capital_base = 10e3

    def test_market_hours_calculations(self):
        with trading.TradingEnvironment():
            # DST in US/Eastern began on Sunday March 14, 2010
            before = datetime.datetime(2010, 3, 12, 14, 30, tzinfo=pytz.utc)
            after = factory.get_next_trading_dt(
                before,
                datetime.timedelta(days=1)
            )
            self.assertEqual(after.hour, 13)

    def test_long_position_receives_dividend(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            # declared date, when the algorithm finds out about
            # the dividend
            events[1].dt,
            # ex_date, when the algorithm is credited with the
            # dividend
            events[1].dt,
            # pay date, when the algorithm receives the dividend.
            events[2].dt
        )

        txn = factory.create_txn(1, 10.0, 100, events[0].dt)
        events[0].TRANSACTION = txn
        events.insert(1, dividend)
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(results[0]['daily_perf']['period_open'], events[0].dt)
        self.assertEqual(
            results[-1]['daily_perf']['period_open'],
            events[-1].dt
        )

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.1, 0.1, 0.1])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.10, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 1000, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [-1000, -1000, 0, 0, 0])
        cash_pos = \
            [event['cumulative_perf']['ending_cash'] for event in results]
        self.assertEqual(cash_pos, [9000, 9000, 10000, 10000, 10000])

    def test_post_ex_long_position_receives_no_dividend(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[1].dt,
            events[2].dt
        )

        events.insert(1, dividend)
        txn = factory.create_txn(1, 10.0, 100, events[3].dt)
        events[3].TRANSACTION = txn
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0, 0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, 0, -1000, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [0, 0, -1000, -1000, -1000])

    def test_selling_before_dividend_payment_still_gets_paid(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[1].dt,
            events[3].dt
        )

        buy_txn = factory.create_txn(1, 10.0, 100, events[0].dt)
        events[0].TRANSACTION = buy_txn
        sell_txn = factory.create_txn(1, 10.0, -100, events[2].dt)
        events[2].TRANSACTION = sell_txn
        events.insert(1, dividend)
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0.1, 0.1])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0.1, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 1000, 1000, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [-1000, -1000, 0, 1000, 1000])

    def test_buy_and_sell_before_ex(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            events[3].dt,
            events[4].dt,
            events[5].dt
        )

        buy_txn = factory.create_txn(1, 10.0, 100, events[1].dt)
        events[1].TRANSACTION = buy_txn
        sell_txn = factory.create_txn(1, 10.0, -100, events[2].dt)
        events[2].TRANSACTION = sell_txn
        events.insert(1, dividend)
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0, 0, 0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, -1000, 1000, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [0, -1000, 0, 0, 0, 0])

    def test_ending_before_pay_date(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[1].dt,
            events[-1].dt + 10*oneday
        )

        buy_txn = factory.create_txn(1, 10.0, 100, events[1].dt)
        events[1].TRANSACTION = buy_txn
        events.insert(1, dividend)
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0.0, 0.0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, -1000, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(
            cumulative_cash_flows,
            [0, -1000, -1000, -1000, -1000]
        )

    def test_short_position_pays_dividend(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[1].dt,
            events[2].dt
        )

        txn = factory.create_txn(1, 10.0, -100, self.dt+oneday)
        events[0].TRANSACTION = txn
        events.insert(0, dividend)
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, -0.1, -0.1, -0.1])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, -0.1, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [1000, 0, -1000, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [1000, 1000, 0, 0, 0])

    def test_no_position_receives_no_dividend(self):
        #post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[1].dt,
            events[2].dt
        )

        events.insert(1, dividend)
        perf_tracker = perf.PerformanceTracker(self.sim_params)
        transformed_events = list(perf_tracker.transform(
            ((event.dt, [event]) for event in events))
        )

        #flatten the list of events
        results = []
        for te in transformed_events:
            for event in te[1]:
                for message in event.perf_messages:
                    results.append(message)

        perf_messages, risk = perf_tracker.handle_simulation_end()
        results.append(perf_messages[0])

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.0, 0.0, 0.0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.0, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [0, 0, 0, 0, 0])


class TestPositionPerformance(unittest.TestCase):

    def setUp(self):
        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

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
            onesec,
            self.sim_params
        )

        txn = factory.create_txn(1, 10.0, 100, self.dt + onesec)
        pp = perf.PerformancePeriod(1000.0)

        pp.execute_transaction(txn)
        for trade in trades:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(
            pp.period_cash_flow,
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
            onesec,
            self.sim_params
        )

        trades_1 = trades[:-2]

        txn = factory.create_txn(1, 10.0, -100, self.dt + onesec)
        pp = perf.PerformancePeriod(1000.0)

        pp.execute_transaction(txn)
        for trade in trades_1:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(
            pp.period_cash_flow,
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
            pp.period_cash_flow,
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
            ppTotal.period_cash_flow,
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
            onesec,
            self.sim_params
        )

        short_txn = factory.create_txn(
            1,
            10.0,
            -100,
            self.dt + onesec
        )

        cover_txn = factory.create_txn(1, 7.0, 100, self.dt + onesec * 6)
        pp = perf.PerformancePeriod(1000.0)

        pp.execute_transaction(short_txn)
        pp.execute_transaction(cover_txn)

        for trade in trades:
            pp.update_last_sale(trade)

        pp.calculate_performance()

        short_txn_cost = short_txn.price * short_txn.amount
        cover_txn_cost = cover_txn.price * cover_txn.amount

        self.assertEqual(
            pp.period_cash_flow,
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
            onesec,
            self.sim_params
        )

        transactions = factory.create_txn_history(
            1,
            [10, 11, 11, 12],
            [100, 100, 100, 100],
            onesec,
            self.sim_params
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
            self.dt + onesec * 4)

        down_tick = factory.create_trade(
            1,
            10.0,
            100,
            trades[-1].dt + onesec)

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

        sim_params = SimulationParameters(
            period_start=start_dt,
            period_end=end_dt
        )

        trade_history = factory.create_trade_history(
            sid,
            price_list,
            volume,
            trade_time_increment,
            sim_params,
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
            sim_params,
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

        sim_params.first_open = \
            sim_params.calculate_first_open()
        sim_params.last_close = \
            sim_params.calculate_last_close()
        sim_params.capital_base = 1000.0
        sim_params.frame_index = [
            'sid',
            'volume',
            'dt',
            'price',
            'changed']
        perf_tracker = perf.PerformanceTracker(
            sim_params
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
                         sim_params.days_in_period)

    def event_with_txn(self, event, no_txn_dt):
        #create a transaction for all but
        #first trade in each sid, to simulate None transaction
        if event.dt != no_txn_dt:
            txn = Transaction(**{
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

    def test_minute_tracker(self):
        """ Tests minute performance tracking."""
        start_dt = trading.environment.exchange_dt_in_utc(
            datetime.datetime(2013, 3, 1, 9, 30))
        end_dt = trading.environment.exchange_dt_in_utc(
            datetime.datetime(2013, 3, 1, 16, 0))

        sim_params = SimulationParameters(
            period_start=start_dt,
            period_end=end_dt,
            emission_rate='minute'
        )
        tracker = perf.PerformanceTracker(sim_params)

        foo_event_1 = factory.create_trade('foo', 10.0, 20, start_dt)
        bar_event_1 = factory.create_trade('bar', 100.0, 200, start_dt)
        txn = Transaction(sid=foo_event_1.sid,
                          amount=-25,
                          dt=foo_event_1.dt,
                          price=10.0,
                          commission=0.50)
        foo_event_1.TRANSACTION = txn

        foo_event_2 = factory.create_trade(
            'foo', 11.0, 20, start_dt + datetime.timedelta(minutes=1))
        bar_event_2 = factory.create_trade(
            'bar', 11.0, 20, start_dt + datetime.timedelta(minutes=1))

        foo_event_3 = factory.create_trade(
            'foo', 12.0, 30, start_dt + datetime.timedelta(minutes=2))

        tracker.process_event(foo_event_1)
        tracker.process_event(bar_event_1)
        messages = tracker.process_event(foo_event_2)
        tracker.process_event(bar_event_2)
        messages += tracker.process_event(foo_event_3)

        self.assertEquals(2, len(messages))

        self.assertEquals(1, len(messages[0]['intraday_perf']['transactions']),
                          "The first message should contain one transaction.")
        # Check that transactions aren't emitted for previous events.
        self.assertEquals(0, len(messages[1]['intraday_perf']['transactions']),
                          "The second message should have no transactions.")
