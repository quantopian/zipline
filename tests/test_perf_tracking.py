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
import heapq
import operator

import unittest
from nose_parameterized import parameterized
import datetime
import pytz
import itertools

import zipline.utils.factory as factory
import zipline.finance.performance as perf
from zipline.finance.slippage import Transaction, create_transaction

from zipline.gens.composites import date_sorted_sources
from zipline.finance.trading import SimulationParameters
from zipline.finance.blotter import Order
import zipline.finance.trading as trading
from zipline.protocol import DATASOURCE_TYPE
from zipline.utils.factory import create_random_simulation_parameters
import zipline.protocol
from zipline.protocol import Event

onesec = datetime.timedelta(seconds=1)
oneday = datetime.timedelta(days=1)
tradingday = datetime.timedelta(hours=6, minutes=30)


def create_txn(sid, price, amount, dt):
    return create_transaction(sid, amount, price, dt, "fakeuid")


def benchmark_events_in_range(sim_params):
    return [
        Event({'dt': ret.date,
               'returns': ret.returns,
               'type':
               zipline.protocol.DATASOURCE_TYPE.BENCHMARK,
               'source_id': 'benchmarks'})
        for ret in trading.environment.benchmark_returns
        if ret.date.date() >= sim_params.period_start.date()
        and ret.date.date() <= sim_params.period_end.date()
    ]


def calculate_results(host, events):

    perf_tracker = perf.PerformanceTracker(host.sim_params)

    all_events = heapq.merge(
        ((event.dt, event) for event in events),
        ((event.dt, event) for event in host.benchmark_events))

    filtered_events = [(date, filt_event) for (date, filt_event)
                       in all_events if date <= events[-1].dt]
    filtered_events.sort(key=lambda x: x[0])
    grouped_events = itertools.groupby(filtered_events, lambda x: x[0])
    results = []

    bm_updated = False
    for date, group in grouped_events:
        for _, event in group:
            perf_tracker.process_event(event)
            if event.type == DATASOURCE_TYPE.BENCHMARK:
                bm_updated = True
        if bm_updated:
            msg = perf_tracker.handle_market_close()
            results.append(msg)
            bm_updated = False
    return results


class TestDividendPerformance(unittest.TestCase):

    def setUp(self):

        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        self.sim_params.capital_base = 10e3

        self.benchmark_events = benchmark_events_in_range(self.sim_params)

    def test_market_hours_calculations(self):
        with trading.TradingEnvironment():
            # DST in US/Eastern began on Sunday March 14, 2010
            before = datetime.datetime(2010, 3, 12, 14, 31, tzinfo=pytz.utc)
            after = factory.get_next_trading_dt(
                before,
                datetime.timedelta(days=1)
            )
            self.assertEqual(after.hour, 13)

    @trading.use_environment(trading.TradingEnvironment())
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

        txn = create_txn(1, 10.0, 100, events[0].dt)
        events.insert(0, txn)
        events.insert(1, dividend)
        results = calculate_results(self, events)

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
        txn = create_txn(1, 10.0, 100, events[3].dt)
        events.insert(4, txn)
        results = calculate_results(self, events)

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

        buy_txn = create_txn(1, 10.0, 100, events[0].dt)
        events.insert(1, buy_txn)
        sell_txn = create_txn(1, 10.0, -100, events[3].dt)
        events.insert(4, sell_txn)
        events.insert(0, dividend)
        results = calculate_results(self, events)

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

        buy_txn = create_txn(1, 10.0, 100, events[1].dt)
        events.insert(1, buy_txn)
        sell_txn = create_txn(1, 10.0, -100, events[3].dt)
        events.insert(3, sell_txn)
        events.insert(1, dividend)
        results = calculate_results(self, events)

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

        pay_date = self.sim_params.first_open
        # find pay date that is much later.
        for i in xrange(30):
            pay_date = factory.get_next_trading_dt(pay_date, oneday)
        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[1].dt,
            pay_date
        )

        buy_txn = create_txn(1, 10.0, 100, events[1].dt)
        events.insert(2, buy_txn)
        events.insert(1, dividend)
        results = calculate_results(self, events)

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
            # declare at open of test
            events[0].dt,
            # ex_date same as trade 2
            events[2].dt,
            events[3].dt
        )

        txn = create_txn(1, 10.0, -100, events[1].dt)
        events.insert(1, txn)
        events.insert(0, dividend)
        results = calculate_results(self, events)

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.0, -0.1, -0.1])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.0, -0.1, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, 1000, 0, -1000, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [0, 1000, 1000, 0, 0])

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
        results = calculate_results(self, events)

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


class TestDividendPerformanceHolidayStyle(TestDividendPerformance):

    # The holiday tests begins the simulation on the day
    # before Thanksgiving, so that the next trading day is
    # two days ahead. Any tests that hard code events
    # to be start + oneday will fail, since those events will
    # be skipped by the simulation.

    def setUp(self):
        self.dt = datetime.datetime(2003, 11, 30, tzinfo=pytz.utc)
        self.end_dt = datetime.datetime(2004, 11, 25, tzinfo=pytz.utc)
        self.sim_params = SimulationParameters(
            self.dt,
            self.end_dt)
        self.benchmark_events = benchmark_events_in_range(self.sim_params)


class TestPositionPerformance(unittest.TestCase):

    def setUp(self):
        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        self.benchmark_events = benchmark_events_in_range(self.sim_params)

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

        txn = create_txn(1, 10.0, 100, self.dt + onesec)
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

        txn = create_txn(1, 10.0, -100, self.dt + onesec)
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

        short_txn = create_txn(
            1,
            10.0,
            -100,
            self.dt + onesec
        )

        cover_txn = create_txn(1, 7.0, 100, self.dt + onesec * 6)
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

        saleTxn = create_txn(
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

        benchmark_events = benchmark_events_in_range(sim_params)

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

        events = [event for event in
                  self.trades_with_txns(events, trade_history[0].dt)]

        # Extract events with transactions to use for verification.
        txns = [event for event in
                events if event.type == DATASOURCE_TYPE.TRANSACTION]

        orders = [event for event in
                  events if event.type == DATASOURCE_TYPE.ORDER]

        all_events = (msg[1] for msg in heapq.merge(
            ((event.dt, event) for event in events),
            ((event.dt, event) for event in benchmark_events)))

        filtered_events = [filt_event for filt_event
                           in all_events if event.dt <= end_dt]
        filtered_events.sort(key=lambda x: x.dt)
        grouped_events = itertools.groupby(filtered_events, lambda x: x.dt)
        perf_messages = []

        for date, group in grouped_events:
            for event in group:
                perf_tracker.process_event(event)
            msg = perf_tracker.handle_market_close()
            perf_messages.append(msg)

        self.assertEqual(perf_tracker.txn_count, len(txns))
        self.assertEqual(perf_tracker.txn_count, len(orders))

        cumulative_pos = perf_tracker.cumulative_performance.positions[sid]
        expected_size = len(txns) / 2 * -25
        self.assertEqual(cumulative_pos.amount, expected_size)

        self.assertEqual(len(perf_messages),
                         sim_params.days_in_period)

    def trades_with_txns(self, events, no_txn_dt):
        for event in events:

            #create a transaction for all but
            #first trade in each sid, to simulate None transaction
            if event.dt != no_txn_dt:
                order = Order(**{
                    'sid': event.sid,
                    'amount': -25,
                    'dt': event.dt
                })
                yield order
                yield event
                txn = Transaction(**{
                    'sid': event.sid,
                    'amount': -25,
                    'dt': event.dt,
                    'price': 10.0,
                    'commission': 0.50,
                    'order_id': order.id
                })
                yield txn
            else:
                yield event

    @trading.use_environment(trading.TradingEnvironment())
    def test_minute_tracker(self):
        """ Tests minute performance tracking."""
        start_dt = trading.environment.exchange_dt_in_utc(
            datetime.datetime(2013, 3, 1, 9, 31))
        end_dt = trading.environment.exchange_dt_in_utc(
            datetime.datetime(2013, 3, 1, 16, 0))

        sim_params = SimulationParameters(
            period_start=start_dt,
            period_end=end_dt,
            emission_rate='minute'
        )
        tracker = perf.PerformanceTracker(sim_params)

        foo_event_1 = factory.create_trade('foo', 10.0, 20, start_dt)
        order_event_1 = Order(**{
                              'sid': foo_event_1.sid,
                              'amount': -25,
                              'dt': foo_event_1.dt
                              })
        bar_event_1 = factory.create_trade('bar', 100.0, 200, start_dt)
        txn_event_1 = Transaction(sid=foo_event_1.sid,
                                  amount=-25,
                                  dt=foo_event_1.dt,
                                  price=10.0,
                                  commission=0.50)
        benchmark_event_1 = Event({
            'dt': start_dt,
            'returns': 1.0,
            'type': DATASOURCE_TYPE.BENCHMARK
        })

        foo_event_2 = factory.create_trade(
            'foo', 11.0, 20, start_dt + datetime.timedelta(minutes=1))
        bar_event_2 = factory.create_trade(
            'bar', 11.0, 20, start_dt + datetime.timedelta(minutes=1))
        benchmark_event_2 = Event({
            'dt': start_dt + datetime.timedelta(minutes=1),
            'returns': 2.0,
            'type': DATASOURCE_TYPE.BENCHMARK
        })

        events = [
            foo_event_1,
            order_event_1,
            benchmark_event_1,
            txn_event_1,
            bar_event_1,
            foo_event_2,
            benchmark_event_2,
            bar_event_2,
        ]

        grouped_events = itertools.groupby(
            events, operator.attrgetter('dt'))

        messages = {}
        for date, group in grouped_events:
            tracker.set_date(date)
            for event in group:
                tracker.process_event(event)
            tracker.handle_minute_close(date)
            msg = tracker.to_dict()
            messages[date] = msg

        self.assertEquals(2, len(messages))

        msg_1 = messages[foo_event_1.dt]
        msg_2 = messages[foo_event_2.dt]

        self.assertEquals(1, len(msg_1['intraday_perf']['transactions']),
                          "The first message should contain one transaction.")
        # Check that transactions aren't emitted for previous events.
        self.assertEquals(0, len(msg_2['intraday_perf']['transactions']),
                          "The second message should have no transactions.")

        self.assertEquals(1, len(msg_1['intraday_perf']['orders']),
                          "The first message should contain one orders.")
        # Check that orders aren't emitted for previous events.
        self.assertEquals(0, len(msg_2['intraday_perf']['orders']),
                          "The second message should have no orders.")

        # Ensure that period_close moves through time.
        # Also, ensure that the period_closes are the expected dts.
        self.assertEquals(foo_event_1.dt,
                          msg_1['intraday_perf']['period_close'])
        self.assertEquals(foo_event_2.dt,
                          msg_2['intraday_perf']['period_close'])
