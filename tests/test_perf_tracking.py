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

from __future__ import division

import pickle
import collections
from datetime import (
    datetime,
    timedelta,
)
import logging
import operator

import unittest
from nose_parameterized import parameterized
import nose.tools as nt
import pytz
import itertools

import pandas as pd
import numpy as np
from six.moves import range, zip

import zipline.utils.factory as factory
import zipline.finance.performance as perf
from zipline.finance.slippage import Transaction, create_transaction
import zipline.utils.math_utils as zp_math
from zipline.assets import EQUITY, FUTURE

from zipline.gens.composites import date_sorted_sources
from zipline.finance.trading import SimulationParameters
from zipline.finance.blotter import Order
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance import trading
from zipline.utils.factory import create_random_simulation_parameters
import zipline.protocol as zp
from zipline.protocol import Event

logger = logging.getLogger('Test Perf Tracking')

onesec = timedelta(seconds=1)
oneday = timedelta(days=1)
tradingday = timedelta(hours=6, minutes=30)

# nose.tools changed name in python 3
if not hasattr(nt, 'assert_count_equal'):
    nt.assert_count_equal = nt.assert_items_equal


def check_perf_period(pp,
                      gross_leverage,
                      net_leverage,
                      long_exposure,
                      longs_count,
                      short_exposure,
                      shorts_count):

    perf_data = pp.to_dict()
    np.testing.assert_allclose(
        gross_leverage, perf_data['gross_leverage'], rtol=1e-3)
    np.testing.assert_allclose(
        net_leverage, perf_data['net_leverage'], rtol=1e-3)
    np.testing.assert_allclose(
        long_exposure, perf_data['long_exposure'], rtol=1e-3)
    np.testing.assert_allclose(
        longs_count, perf_data['longs_count'], rtol=1e-3)
    np.testing.assert_allclose(
        short_exposure, perf_data['short_exposure'], rtol=1e-3)
    np.testing.assert_allclose(
        shorts_count, perf_data['shorts_count'], rtol=1e-3)


def check_account(account,
                  settled_cash,
                  equity_with_loan,
                  total_positions_value,
                  regt_equity,
                  available_funds,
                  excess_liquidity,
                  cushion,
                  leverage,
                  net_leverage,
                  net_liquidation):
    # this is a long only portfolio that is only partially invested
    # so net and gross leverage are equal.

    np.testing.assert_allclose(settled_cash,
                               account['settled_cash'], rtol=1e-3)
    np.testing.assert_allclose(equity_with_loan,
                               account['equity_with_loan'], rtol=1e-3)
    np.testing.assert_allclose(total_positions_value,
                               account['total_positions_value'], rtol=1e-3)
    np.testing.assert_allclose(regt_equity,
                               account['regt_equity'], rtol=1e-3)
    np.testing.assert_allclose(available_funds,
                               account['available_funds'], rtol=1e-3)
    np.testing.assert_allclose(excess_liquidity,
                               account['excess_liquidity'], rtol=1e-3)
    np.testing.assert_allclose(cushion,
                               account['cushion'], rtol=1e-3)
    np.testing.assert_allclose(leverage, account['leverage'], rtol=1e-3)
    np.testing.assert_allclose(net_leverage,
                               account['net_leverage'], rtol=1e-3)
    np.testing.assert_allclose(net_liquidation,
                               account['net_liquidation'], rtol=1e-3)


def create_txn(trade_event, price, amount):
    """
    Create a fake transaction to be filled and processed prior to the execution
    of a given trade event.
    """
    mock_order = Order(trade_event.dt, trade_event.sid, amount, id=None)
    return create_transaction(trade_event, mock_order, price, amount)


def benchmark_events_in_range(sim_params):
    return [
        Event({'dt': dt,
               'returns': ret,
               'type': zp.DATASOURCE_TYPE.BENCHMARK,
               # We explicitly rely on the behavior that benchmarks sort before
               # any other events.
               'source_id': '1Abenchmarks'})
        for dt, ret in trading.environment.benchmark_returns.iteritems()
        if dt.date() >= sim_params.period_start.date() and
        dt.date() <= sim_params.period_end.date()
    ]


def calculate_results(host,
                      trade_events,
                      dividend_events=None,
                      splits=None,
                      txns=None):
    """
    Run the given events through a stripped down version of the loop in
    AlgorithmSimulator.transform.

    IMPORTANT NOTE FOR TEST WRITERS/READERS:

    This loop has some wonky logic for the order of event processing for
    datasource types.  This exists mostly to accomodate legacy tests accomodate
    existing tests that were making assumptions about how events would be
    sorted.

    In particular:

        - Dividends passed for a given date are processed PRIOR to any events
          for that date.
        - Splits passed for a given date are process AFTER any events for that
          date.

    Tests that use this helper should not be considered useful guarantees of
    the behavior of AlgorithmSimulator on a stream containing the same events
    unless the subgroups have been explicitly re-sorted in this way.
    """

    txns = txns or []
    splits = splits or []

    perf_tracker = perf.PerformanceTracker(host.sim_params)

    if dividend_events is not None:
        dividend_frame = pd.DataFrame(
            [
                event.to_series(index=zp.DIVIDEND_FIELDS)
                for event in dividend_events
            ],
        )
        perf_tracker.update_dividends(dividend_frame)

    # Raw trades
    trade_events = sorted(trade_events, key=lambda ev: (ev.dt, ev.source_id))

    # Add a benchmark event for each date.
    trades_plus_bm = date_sorted_sources(trade_events, host.benchmark_events)

    # Filter out benchmark events that are later than the last trade date.
    filtered_trades_plus_bm = (filt_event for filt_event in trades_plus_bm
                               if filt_event.dt <= trade_events[-1].dt)

    grouped_trades_plus_bm = itertools.groupby(filtered_trades_plus_bm,
                                               lambda x: x.dt)
    results = []

    bm_updated = False
    for date, group in grouped_trades_plus_bm:

        for txn in filter(lambda txn: txn.dt == date, txns):
            # Process txns for this date.
            perf_tracker.process_transaction(txn)

        for event in group:

            if event.type == zp.DATASOURCE_TYPE.TRADE:
                perf_tracker.process_trade(event)
            elif event.type == zp.DATASOURCE_TYPE.DIVIDEND:
                perf_tracker.process_dividend(event)
            elif event.type == zp.DATASOURCE_TYPE.BENCHMARK:
                perf_tracker.process_benchmark(event)
                bm_updated = True
            elif event.type == zp.DATASOURCE_TYPE.COMMISSION:
                perf_tracker.process_commission(event)

        for split in filter(lambda split: split.dt == date, splits):
            # Process splits for this date.
            perf_tracker.process_split(split)

        if bm_updated:
            msg = perf_tracker.handle_market_close_daily()
            msg['account'] = perf_tracker.get_account(True)
            results.append(msg)
            bm_updated = False
    return results


def check_perf_tracker_serialization(perf_tracker):
    scalar_keys = [
        'emission_rate',
        'txn_count',
        'market_open',
        'last_close',
        '_dividend_count',
        'period_start',
        'day_count',
        'capital_base',
        'market_close',
        'saved_dt',
        'period_end',
        'total_days',
    ]

    p_string = pickle.dumps(perf_tracker)

    test = pickle.loads(p_string)

    for k in scalar_keys:
        nt.assert_equal(getattr(test, k), getattr(perf_tracker, k), k)

    for period in test.perf_periods:
        nt.assert_true(hasattr(period, '_position_tracker'))


class TestSplitPerformance(unittest.TestCase):
    def setUp(self):
        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        # start with $10,000
        self.sim_params.capital_base = 10e3

        self.benchmark_events = benchmark_events_in_range(self.sim_params)

    def test_split_long_position(self):
        events = factory.create_trade_history(
            1,
            [20, 20],
            [100, 100],
            oneday,
            self.sim_params
        )

        # set up a long position in sid 1
        # 100 shares at $20 apiece = $2000 position
        txns = [create_txn(events[0], 20, 100)]

        # set up a split with ratio 3 occurring at the start of the second
        # day.
        splits = [
            factory.create_split(
                1,
                3,
                events[1].dt,
            ),
        ]

        results = calculate_results(self, events, txns=txns, splits=splits)

        # should have 33 shares (at $60 apiece) and $20 in cash
        self.assertEqual(2, len(results))

        latest_positions = results[1]['daily_perf']['positions']
        self.assertEqual(1, len(latest_positions))

        # check the last position to make sure it's been updated
        position = latest_positions[0]

        self.assertEqual(1, position['sid'])
        self.assertEqual(33, position['amount'])
        self.assertEqual(60, position['cost_basis'])
        self.assertEqual(60, position['last_sale_price'])

        # since we started with $10000, and we spent $2000 on the
        # position, but then got $20 back, we should have $8020
        # (or close to it) in cash.

        # we won't get exactly 8020 because sometimes a split is
        # denoted as a ratio like 0.3333, and we lose some digits
        # of precision.  thus, make sure we're pretty close.
        daily_perf = results[1]['daily_perf']

        self.assertTrue(
            zp_math.tolerant_equals(8020,
                                    daily_perf['ending_cash'], 1))

        # Validate that the account attributes were updated.
        account = results[1]['account']
        self.assertEqual(float('inf'), account['day_trades_remaining'])
        # this is a long only portfolio that is only partially invested
        # so net and gross leverage are equal.
        np.testing.assert_allclose(0.198, account['leverage'], rtol=1e-3)
        np.testing.assert_allclose(0.198, account['net_leverage'], rtol=1e-3)
        np.testing.assert_allclose(8020, account['regt_equity'], rtol=1e-3)
        self.assertEqual(float('inf'), account['regt_margin'])
        np.testing.assert_allclose(8020, account['available_funds'], rtol=1e-3)
        self.assertEqual(0, account['maintenance_margin_requirement'])
        np.testing.assert_allclose(10000,
                                   account['equity_with_loan'], rtol=1e-3)
        self.assertEqual(float('inf'), account['buying_power'])
        self.assertEqual(0, account['initial_margin_requirement'])
        np.testing.assert_allclose(8020, account['excess_liquidity'],
                                   rtol=1e-3)
        np.testing.assert_allclose(8020, account['settled_cash'], rtol=1e-3)
        np.testing.assert_allclose(10000, account['net_liquidation'],
                                   rtol=1e-3)
        np.testing.assert_allclose(0.802, account['cushion'], rtol=1e-3)
        np.testing.assert_allclose(1980, account['total_positions_value'],
                                   rtol=1e-3)
        self.assertEqual(0, account['accrued_interest'])

        for i, result in enumerate(results):
            for perf_kind in ('daily_perf', 'cumulative_perf'):
                perf_result = result[perf_kind]
                # prices aren't changing, so pnl and returns should be 0.0
                self.assertEqual(0.0, perf_result['pnl'],
                                 "day %s %s pnl %s instead of 0.0" %
                                 (i, perf_kind, perf_result['pnl']))
                self.assertEqual(0.0, perf_result['returns'],
                                 "day %s %s returns %s instead of 0.0" %
                                 (i, perf_kind, perf_result['returns']))


class TestCommissionEvents(unittest.TestCase):

    def setUp(self):
        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        logger.info("sim_params: %s, dt: %s, end_dt: %s" %
                    (self.sim_params, self.dt, self.end_dt))

        self.sim_params.capital_base = 10e3

        self.benchmark_events = benchmark_events_in_range(self.sim_params)

    def test_commission_event(self):
        with trading.TradingEnvironment():
            events = factory.create_trade_history(
                1,
                [10, 10, 10, 10, 10],
                [100, 100, 100, 100, 100],
                oneday,
                self.sim_params
            )

            # Test commission models and validate result
            # Expected commission amounts:
            # PerShare commission:  1.00, 1.00, 1.50 = $3.50
            # PerTrade commission:  5.00, 5.00, 5.00 = $15.00
            # PerDollar commission: 1.50, 3.00, 4.50 = $9.00
            # Total commission = $3.50 + $15.00 + $9.00 = $27.50

            # Create 3 transactions:  50, 100, 150 shares traded @ $20
            transactions = [create_txn(events[0], 20, i)
                            for i in [50, 100, 150]]

            # Create commission models and validate that produce expected
            # commissions.
            models = [PerShare(cost=0.01, min_trade_cost=1.00),
                      PerTrade(cost=5.00),
                      PerDollar(cost=0.0015)]
            expected_results = [3.50, 15.0, 9.0]

            for model, expected in zip(models, expected_results):
                total_commission = 0
                for trade in transactions:
                    total_commission += model.calculate(trade)[1]
                self.assertEqual(total_commission, expected)

            # Verify that commission events are handled correctly by
            # PerformanceTracker.
            cash_adj_dt = events[0].dt
            cash_adjustment = factory.create_commission(1, 300.0, cash_adj_dt)
            events.append(cash_adjustment)

            # Insert a purchase order.
            txns = [create_txn(events[0], 20, 1)]
            results = calculate_results(self, events, txns=txns)

            # Validate that we lost 320 dollars from our cash pool.
            self.assertEqual(results[-1]['cumulative_perf']['ending_cash'],
                             9680)
            # Validate that the cost basis of our position changed.
            self.assertEqual(results[-1]['daily_perf']['positions']
                             [0]['cost_basis'], 320.0)
            # Validate that the account attributes were updated.
            account = results[1]['account']
            self.assertEqual(float('inf'), account['day_trades_remaining'])
            np.testing.assert_allclose(0.001, account['leverage'], rtol=1e-3,
                                       atol=1e-4)
            np.testing.assert_allclose(9680, account['regt_equity'], rtol=1e-3)
            self.assertEqual(float('inf'), account['regt_margin'])
            np.testing.assert_allclose(9680, account['available_funds'],
                                       rtol=1e-3)
            self.assertEqual(0, account['maintenance_margin_requirement'])
            np.testing.assert_allclose(9690,
                                       account['equity_with_loan'], rtol=1e-3)
            self.assertEqual(float('inf'), account['buying_power'])
            self.assertEqual(0, account['initial_margin_requirement'])
            np.testing.assert_allclose(9680, account['excess_liquidity'],
                                       rtol=1e-3)
            np.testing.assert_allclose(9680, account['settled_cash'],
                                       rtol=1e-3)
            np.testing.assert_allclose(9690, account['net_liquidation'],
                                       rtol=1e-3)
            np.testing.assert_allclose(0.999, account['cushion'], rtol=1e-3)
            np.testing.assert_allclose(10, account['total_positions_value'],
                                       rtol=1e-3)
            self.assertEqual(0, account['accrued_interest'])

    def test_commission_zero_position(self):
        """
        Ensure no div-by-zero errors.
        """
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        # Buy and sell the same sid so that we have a zero position by the
        # time of events[3].
        txns = [
            create_txn(events[0], 20, 1),
            create_txn(events[1], 20, -1),
        ]

        # Add a cash adjustment at the time of event[3].
        cash_adj_dt = events[3].dt
        cash_adjustment = factory.create_commission(1, 300.0, cash_adj_dt)

        events.append(cash_adjustment)

        results = calculate_results(self, events, txns=txns)
        # Validate that we lost 300 dollars from our cash pool.
        self.assertEqual(results[-1]['cumulative_perf']['ending_cash'],
                         9700)

    def test_commission_no_position(self):
        """
        Ensure no position-not-found or sid-not-found errors.
        """
        with trading.TradingEnvironment():
            events = factory.create_trade_history(
                1,
                [10, 10, 10, 10, 10],
                [100, 100, 100, 100, 100],
                oneday,
                self.sim_params
            )

            # Add a cash adjustment at the time of event[3].
            cash_adj_dt = events[3].dt
            cash_adjustment = factory.create_commission(1, 300.0, cash_adj_dt)
            events.append(cash_adjustment)

            results = calculate_results(self, events)
            # Validate that we lost 300 dollars from our cash pool.
            self.assertEqual(results[-1]['cumulative_perf']['ending_cash'],
                             9700)


class TestDividendPerformance(unittest.TestCase):

    def setUp(self):

        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        self.sim_params.capital_base = 10e3

        self.benchmark_events = benchmark_events_in_range(self.sim_params)

    def test_market_hours_calculations(self):
        with trading.TradingEnvironment():
            # DST in US/Eastern began on Sunday March 14, 2010
            before = datetime(2010, 3, 12, 14, 31, tzinfo=pytz.utc)
            after = factory.get_next_trading_dt(
                before,
                timedelta(days=1)
            )
            self.assertEqual(after.hour, 13)

    def test_long_position_receives_dividend(self):
        with trading.TradingEnvironment():
            # post some trades in the market
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
                events[0].dt,
                # ex_date, the date before which the algorithm must hold stock
                # to receive the dividend
                events[1].dt,
                # pay date, when the algorithm receives the dividend.
                events[2].dt
            )

            # Simulate a transaction being filled prior to the ex_date.
            txns = [create_txn(events[0], 10.0, 100)]
            results = calculate_results(
                self,
                events,
                dividend_events=[dividend],
                txns=txns,
            )

            self.assertEqual(len(results), 5)
            cumulative_returns = \
                [event['cumulative_perf']['returns'] for event in results]
            self.assertEqual(cumulative_returns, [0.0, 0.0, 0.1, 0.1, 0.1])
            daily_returns = [event['daily_perf']['returns']
                             for event in results]
            self.assertEqual(daily_returns, [0.0, 0.0, 0.10, 0.0, 0.0])
            cash_flows = [event['daily_perf']['capital_used']
                          for event in results]
            self.assertEqual(cash_flows, [-1000, 0, 1000, 0, 0])
            cumulative_cash_flows = \
                [event['cumulative_perf']['capital_used'] for event in results]
            self.assertEqual(cumulative_cash_flows, [-1000, -1000, 0, 0, 0])
            cash_pos = \
                [event['cumulative_perf']['ending_cash'] for event in results]
            self.assertEqual(cash_pos, [9000, 9000, 10000, 10000, 10000])

    def test_long_position_receives_stock_dividend(self):
        with trading.TradingEnvironment():
            # post some trades in the market
            events = []
            for sid in (1, 2):
                events.extend(
                    factory.create_trade_history(
                        sid,
                        [10, 10, 10, 10, 10],
                        [100, 100, 100, 100, 100],
                        oneday,
                        self.sim_params)
                )

            dividend = factory.create_stock_dividend(
                1,
                payment_sid=2,
                ratio=2,
                # declared date, when the algorithm finds out about
                # the dividend
                declared_date=events[0].dt,
                # ex_date, the date before which the algorithm must hold stock
                # to receive the dividend
                ex_date=events[1].dt,
                # pay date, when the algorithm receives the dividend.
                pay_date=events[2].dt
            )

            txns = [create_txn(events[0], 10.0, 100)]

            results = calculate_results(
                self,
                events,
                dividend_events=[dividend],
                txns=txns,
            )

            self.assertEqual(len(results), 5)
            cumulative_returns = \
                [event['cumulative_perf']['returns'] for event in results]
            self.assertEqual(cumulative_returns, [0.0, 0.0, 0.2, 0.2, 0.2])
            daily_returns = [event['daily_perf']['returns']
                             for event in results]
            self.assertEqual(daily_returns, [0.0, 0.0, 0.2, 0.0, 0.0])
            cash_flows = [event['daily_perf']['capital_used']
                          for event in results]
            self.assertEqual(cash_flows, [-1000, 0, 0, 0, 0])
            cumulative_cash_flows = \
                [event['cumulative_perf']['capital_used'] for event in results]
            self.assertEqual(cumulative_cash_flows, [-1000] * 5)
            cash_pos = \
                [event['cumulative_perf']['ending_cash'] for event in results]
            self.assertEqual(cash_pos, [9000] * 5)

    def test_long_position_purchased_on_ex_date_receives_no_dividend(self):
        # post some trades in the market
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
            events[0].dt,  # Declared date
            events[1].dt,  # Exclusion date
            events[2].dt   # Pay date
        )

        # Simulate a transaction being filled on the ex_date.
        txns = [create_txn(events[1], 10.0, 100)]

        results = calculate_results(
            self,
            events,
            dividend_events=[dividend],
            txns=txns,
        )

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0, 0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, -1000, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows,
                         [0, -1000, -1000, -1000, -1000])

    def test_selling_before_dividend_payment_still_gets_paid(self):
        # post some trades in the market
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
            events[0].dt,  # Declared date
            events[1].dt,  # Exclusion date
            events[3].dt   # Pay date
        )

        buy_txn = create_txn(events[0], 10.0, 100)
        sell_txn = create_txn(events[2], 10.0, -100)
        txns = [buy_txn, sell_txn]

        results = calculate_results(
            self,
            events,
            dividend_events=[dividend],
            txns=txns,
        )

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
        # post some trades in the market
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

        buy_txn = create_txn(events[1], 10.0, 100)
        sell_txn = create_txn(events[2], 10.0, -100)
        txns = [buy_txn, sell_txn]

        results = calculate_results(
            self,
            events,
            dividend_events=[dividend],
            txns=txns,
        )

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
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params
        )

        pay_date = self.sim_params.first_open
        # find pay date that is much later.
        for i in range(30):
            pay_date = factory.get_next_trading_dt(pay_date, oneday)
        dividend = factory.create_dividend(
            1,
            10.00,
            events[0].dt,
            events[0].dt,
            pay_date
        )

        txns = [create_txn(events[1], 10.0, 100)]

        results = calculate_results(
            self,
            events,
            dividend_events=[dividend],
            txns=txns,
        )

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
        # post some trades in the market
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

        txns = [create_txn(events[1], 10.0, -100)]

        results = calculate_results(
            self,
            events,
            dividend_events=[dividend],
            txns=txns,
        )

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
        # post some trades in the market
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

        results = calculate_results(
            self,
            events,
            dividend_events=[dividend],
        )

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
        self.dt = datetime(2003, 11, 30, tzinfo=pytz.utc)
        self.end_dt = datetime(2004, 11, 25, tzinfo=pytz.utc)
        self.sim_params = SimulationParameters(
            self.dt,
            self.end_dt)
        self.benchmark_events = benchmark_events_in_range(self.sim_params)


class TestPositionPerformance(unittest.TestCase):

    def setUp(self):
        self.sim_params, self.dt, self.end_dt = \
            create_random_simulation_parameters()

        self.benchmark_events = benchmark_events_in_range(self.sim_params)

    def test_long_short_positions(self):
        """
        start with $1000
        buy 100 stock1 shares at $10
        sell short 100 stock2 shares at $10
        stock1 then goes down to $9
        stock2 goes to $11
        """

        trades_1 = factory.create_trade_history(
            1,
            [10, 10, 10, 9],
            [100, 100, 100, 100],
            onesec,
            self.sim_params
        )

        trades_2 = factory.create_trade_history(
            2,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            onesec,
            self.sim_params
        )

        txn1 = create_txn(trades_1[1], 10.0, 100)
        txn2 = create_txn(trades_2[1], 10.0, -100)
        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt
        pt.execute_transaction(txn1)
        pp.handle_execution(txn1)
        pt.execute_transaction(txn2)
        pp.handle_execution(txn2)

        for trade in itertools.chain(trades_1[:-2], trades_2[:-2]):
            pt.update_last_sale(trade)

        pp.calculate_performance()

        check_perf_period(
            pp,
            gross_leverage=2.0,
            net_leverage=0.0,
            long_exposure=1000.0,
            longs_count=1,
            short_exposure=-1000.0,
            shorts_count=1)
        # Validate that the account attributes were updated.
        account = pp.as_account()
        check_account(account,
                      settled_cash=1000.0,
                      equity_with_loan=1000.0,
                      total_positions_value=0.0,
                      regt_equity=1000.0,
                      available_funds=1000.0,
                      excess_liquidity=1000.0,
                      cushion=1.0,
                      leverage=2.0,
                      net_leverage=0.0,
                      net_liquidation=1000.0)

        # now simulate stock1 going to $9
        pt.update_last_sale(trades_1[-1])
        # and stock2 going to $11
        pt.update_last_sale(trades_2[-1])

        pp.calculate_performance()

        # Validate that the account attributes were updated.
        account = pp.as_account()

        check_perf_period(
            pp,
            gross_leverage=2.5,
            net_leverage=-0.25,
            long_exposure=900.0,
            longs_count=1,
            short_exposure=-1100.0,
            shorts_count=1)

        check_account(account,
                      settled_cash=1000.0,
                      equity_with_loan=800.0,
                      total_positions_value=-200.0,
                      regt_equity=1000.0,
                      available_funds=1000.0,
                      excess_liquidity=1000.0,
                      cushion=1.25,
                      leverage=2.5,
                      net_leverage=-0.25,
                      net_liquidation=800.0)

    def test_levered_long_position(self):
        """
            start with $1,000, then buy 1000 shares at $10.
            price goes to $11
        """
        # post some trades in the market
        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            onesec,
            self.sim_params
        )

        txn = create_txn(trades[1], 10.0, 1000)
        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        for trade in trades[:-2]:
            pt.update_last_sale(trade)

        pp.calculate_performance()

        check_perf_period(
            pp,
            gross_leverage=10.0,
            net_leverage=10.0,
            long_exposure=10000.0,
            longs_count=1,
            short_exposure=0.0,
            shorts_count=0)

        # Validate that the account attributes were updated.
        account = pp.as_account()
        check_account(account,
                      settled_cash=-9000.0,
                      equity_with_loan=1000.0,
                      total_positions_value=10000.0,
                      regt_equity=-9000.0,
                      available_funds=-9000.0,
                      excess_liquidity=-9000.0,
                      cushion=-9.0,
                      leverage=10.0,
                      net_leverage=10.0,
                      net_liquidation=1000.0)

        # now simulate a price jump to $11
        pt.update_last_sale(trades[-1])

        pp.calculate_performance()

        check_perf_period(
            pp,
            gross_leverage=5.5,
            net_leverage=5.5,
            long_exposure=11000.0,
            longs_count=1,
            short_exposure=0.0,
            shorts_count=0)

        # Validate that the account attributes were updated.
        account = pp.as_account()

        check_account(account,
                      settled_cash=-9000.0,
                      equity_with_loan=2000.0,
                      total_positions_value=11000.0,
                      regt_equity=-9000.0,
                      available_funds=-9000.0,
                      excess_liquidity=-9000.0,
                      cushion=-4.5,
                      leverage=5.5,
                      net_leverage=5.5,
                      net_liquidation=2000.0)

    def test_long_position(self):
        """
            verify that the performance period calculates properly for a
            single buy transaction
        """
        # post some trades in the market
        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            onesec,
            self.sim_params
        )

        txn = create_txn(trades[1], 10.0, 100)
        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        # This verifies that the last sale price is being correctly
        # set in the positions. If this is not the case then returns can
        # incorrectly show as sharply dipping if a transaction arrives
        # before a trade. This is caused by returns being based on holding
        # stocks with a last sale price of 0.
        self.assertEqual(pp.positions[1].last_sale_price, 10.0)

        for trade in trades:
            pt.update_last_sale(trade)

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

        check_perf_period(
            pp,
            gross_leverage=1.0,
            net_leverage=1.0,
            long_exposure=1100.0,
            longs_count=1,
            short_exposure=0.0,
            shorts_count=0)

        # Validate that the account attributes were updated.
        account = pp.as_account()
        check_account(account,
                      settled_cash=0.0,
                      equity_with_loan=1100.0,
                      total_positions_value=1100.0,
                      regt_equity=0.0,
                      available_funds=0.0,
                      excess_liquidity=0.0,
                      cushion=0.0,
                      leverage=1.0,
                      net_leverage=1.0,
                      net_liquidation=1100.0)

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

        txn = create_txn(trades[1], 10.0, -100)
        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)
        for trade in trades_1:
            pt.update_last_sale(trade)

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

        # simulate a rollover to a new period
        pp.rollover()

        for trade in trades_2:
            pt.update_last_sale(trade)

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

        # now run a performance period encompassing the entire trade sample.
        ptTotal = perf.PositionTracker()
        ppTotal = perf.PerformancePeriod(1000.0)
        ppTotal.position_tracker = pt

        for trade in trades_1:
            ptTotal.update_last_sale(trade)

        ptTotal.execute_transaction(txn)
        ppTotal.handle_execution(txn)

        for trade in trades_2:
            ptTotal.update_last_sale(trade)

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

        check_perf_period(
            pp,
            gross_leverage=0.8181,
            net_leverage=-0.8181,
            long_exposure=0.0,
            longs_count=0,
            short_exposure=-900.0,
            shorts_count=1)

        # Validate that the account attributes.
        account = ppTotal.as_account()
        check_account(account,
                      settled_cash=2000.0,
                      equity_with_loan=1100.0,
                      total_positions_value=-900.0,
                      regt_equity=2000.0,
                      available_funds=2000.0,
                      excess_liquidity=2000.0,
                      cushion=1.8181,
                      leverage=0.8181,
                      net_leverage=-0.8181,
                      net_liquidation=1100.0)

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
            trades[1],
            10.0,
            -100,
        )

        cover_txn = create_txn(trades[6], 7.0, 100)
        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt

        pt.execute_transaction(short_txn)
        pp.handle_execution(short_txn)
        pt.execute_transaction(cover_txn)
        pp.handle_execution(cover_txn)

        for trade in trades:
            pt.update_last_sale(trade)

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

        check_perf_period(
            pp,
            gross_leverage=0.0,
            net_leverage=0.0,
            long_exposure=0.0,
            longs_count=0,
            short_exposure=0.0,
            shorts_count=0)

        account = pp.as_account()
        check_account(account,
                      settled_cash=1300.0,
                      equity_with_loan=1300.0,
                      total_positions_value=0.0,
                      regt_equity=1300.0,
                      available_funds=1300.0,
                      excess_liquidity=1300.0,
                      cushion=1.0,
                      leverage=0.0,
                      net_leverage=0.0,
                      net_liquidation=1300.0)

    def test_cost_basis_calc(self):
        history_args = (
            1,
            [10, 11, 11, 12],
            [100, 100, 100, 100],
            onesec,
            self.sim_params
        )
        trades = factory.create_trade_history(*history_args)
        transactions = factory.create_txn_history(*history_args)

        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt

        average_cost = 0
        for i, txn in enumerate(transactions):
            pt.execute_transaction(txn)
            pp.handle_execution(txn)
            average_cost = (average_cost * i + txn.price) / (i + 1)
            self.assertEqual(pp.positions[1].cost_basis, average_cost)

        for trade in trades:
            pt.update_last_sale(trade)

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

        down_tick = factory.create_trade(
            1,
            10.0,
            100,
            trades[-1].dt + onesec)

        sale_txn = create_txn(
            down_tick,
            10.0,
            -100)

        pp.rollover()

        pt.execute_transaction(sale_txn)
        pp.handle_execution(sale_txn)
        pt.update_last_sale(down_tick)

        pp.calculate_performance()
        self.assertEqual(
            pp.positions[1].last_sale_price,
            10,
            "should have a last sale of 10, was {val}".format(
                val=pp.positions[1].last_sale_price)
        )

        self.assertEqual(
            pp.positions[1].cost_basis,
            11,
            "should have a cost basis of 11"
        )

        self.assertEqual(pp.pnl, -800, "this period goes from +400 to -400")

        pt3 = perf.PositionTracker()
        pp3 = perf.PerformancePeriod(1000.0)
        pp3.position_tracker = pt3

        average_cost = 0
        for i, txn in enumerate(transactions):
            pt3.execute_transaction(txn)
            pp3.handle_execution(txn)
            average_cost = (average_cost * i + txn.price) / (i + 1)
            self.assertEqual(pp3.positions[1].cost_basis, average_cost)

        pt3.execute_transaction(sale_txn)
        pp3.handle_execution(sale_txn)

        trades.append(down_tick)
        for trade in trades:
            pt3.update_last_sale(trade)

        pp3.calculate_performance()
        self.assertEqual(
            pp3.positions[1].last_sale_price,
            10,
            "should have a last sale of 10"
        )

        self.assertEqual(
            pp3.positions[1].cost_basis,
            11,
            "should have a cost basis of 11"
        )

        self.assertEqual(
            pp3.pnl,
            -400,
            "should be -400 for all trades and transactions in period"
        )

    def test_cost_basis_calc_close_pos(self):
        history_args = (
            1,
            [10, 9, 11, 8, 9, 12, 13, 14],
            [200, -100, -100, 100, -300, 100, 500, 400],
            onesec,
            self.sim_params
        )
        cost_bases = [10, 10, 0, 8, 9, 9, 13, 13.5]

        trades = factory.create_trade_history(*history_args)
        transactions = factory.create_txn_history(*history_args)

        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(1000.0)
        pp.position_tracker = pt

        for txn, cb in zip(transactions, cost_bases):
            pt.execute_transaction(txn)
            pp.handle_execution(txn)
            self.assertEqual(pp.positions[1].cost_basis, cb)

        for trade in trades:
            pt.update_last_sale(trade)

        pp.calculate_performance()

        self.assertEqual(pp.positions[1].cost_basis, cost_bases[-1])


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
        start_dt = datetime(year=2008,
                            month=10,
                            day=9,
                            tzinfo=pytz.utc)
        end_dt = datetime(year=2008,
                          month=10,
                          day=16,
                          tzinfo=pytz.utc)

        trade_count = 6
        sid = 133
        price = 10.1
        price_list = [price] * trade_count
        volume = [100] * trade_count
        trade_time_increment = timedelta(days=1)

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
                events if event.type == zp.DATASOURCE_TYPE.TRANSACTION]

        orders = [event for event in
                  events if event.type == zp.DATASOURCE_TYPE.ORDER]

        all_events = date_sorted_sources(events, benchmark_events)

        filtered_events = [filt_event for filt_event
                           in all_events if filt_event.dt <= end_dt]
        filtered_events.sort(key=lambda x: x.dt)
        grouped_events = itertools.groupby(filtered_events, lambda x: x.dt)
        perf_messages = []

        for date, group in grouped_events:
            for event in group:
                if event.type == zp.DATASOURCE_TYPE.TRADE:
                    perf_tracker.process_trade(event)
                elif event.type == zp.DATASOURCE_TYPE.ORDER:
                    perf_tracker.process_order(event)
                elif event.type == zp.DATASOURCE_TYPE.BENCHMARK:
                    perf_tracker.process_benchmark(event)
                elif event.type == zp.DATASOURCE_TYPE.TRANSACTION:
                    perf_tracker.process_transaction(event)
            msg = perf_tracker.handle_market_close_daily()
            perf_messages.append(msg)

        self.assertEqual(perf_tracker.txn_count, len(txns))
        self.assertEqual(perf_tracker.txn_count, len(orders))

        positions = perf_tracker.cumulative_performance.positions
        if len(txns) == 0:
            self.assertNotIn(sid, positions)
        else:
            expected_size = len(txns) / 2 * -25
            cumulative_pos = positions[sid]
            self.assertEqual(cumulative_pos.amount, expected_size)

            self.assertEqual(len(perf_messages),
                             sim_params.days_in_period)

        check_perf_tracker_serialization(perf_tracker)

    def trades_with_txns(self, events, no_txn_dt):
        for event in events:

            # create a transaction for all but
            # first trade in each sid, to simulate None transaction
            if event.dt != no_txn_dt:
                order = Order(
                    sid=event.sid,
                    amount=-25,
                    dt=event.dt
                )
                order.source_id = 'MockOrderSource'
                yield order
                yield event
                txn = Transaction(
                    sid=event.sid,
                    amount=-25,
                    dt=event.dt,
                    price=10.0,
                    commission=0.50,
                    order_id=order.id
                )
                txn.source_id = 'MockTransactionSource'
                yield txn
            else:
                yield event

    def test_minute_tracker(self):
        """ Tests minute performance tracking."""
        with trading.TradingEnvironment():
            start_dt = trading.environment.exchange_dt_in_utc(
                datetime(2013, 3, 1, 9, 31))
            end_dt = trading.environment.exchange_dt_in_utc(
                datetime(2013, 3, 1, 16, 0))

            sim_params = SimulationParameters(
                period_start=start_dt,
                period_end=end_dt,
                emission_rate='minute'
            )
            tracker = perf.PerformanceTracker(sim_params)

            foosid = 1
            barsid = 2

            trading.environment.update_asset_finder(
                identifiers=[foosid, barsid])

            foo_event_1 = factory.create_trade(foosid, 10.0, 20, start_dt)
            order_event_1 = Order(sid=foo_event_1.sid,
                                  amount=-25,
                                  dt=foo_event_1.dt)
            bar_event_1 = factory.create_trade(barsid, 100.0, 200, start_dt)
            txn_event_1 = Transaction(sid=foo_event_1.sid,
                                      amount=-25,
                                      dt=foo_event_1.dt,
                                      price=10.0,
                                      commission=0.50,
                                      order_id=order_event_1.id)
            benchmark_event_1 = Event({
                'dt': start_dt,
                'returns': 0.01,
                'type': zp.DATASOURCE_TYPE.BENCHMARK
            })

            foo_event_2 = factory.create_trade(
                foosid, 11.0, 20, start_dt + timedelta(minutes=1))
            bar_event_2 = factory.create_trade(
                barsid, 11.0, 20, start_dt + timedelta(minutes=1))
            benchmark_event_2 = Event({
                'dt': start_dt + timedelta(minutes=1),
                'returns': 0.02,
                'type': zp.DATASOURCE_TYPE.BENCHMARK
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
                    if event.type == zp.DATASOURCE_TYPE.TRADE:
                        tracker.process_trade(event)
                    elif event.type == zp.DATASOURCE_TYPE.BENCHMARK:
                        tracker.process_benchmark(event)
                    elif event.type == zp.DATASOURCE_TYPE.ORDER:
                        tracker.process_order(event)
                    elif event.type == zp.DATASOURCE_TYPE.TRANSACTION:
                        tracker.process_transaction(event)
                tracker.handle_minute_close(date)
                msg = tracker.to_dict()
                messages[date] = msg

            self.assertEquals(2, len(messages))

            msg_1 = messages[foo_event_1.dt]
            msg_2 = messages[foo_event_2.dt]

            self.assertEquals(1, len(msg_1['minute_perf']['transactions']),
                              "The first message should contain one "
                              "transaction.")
            # Check that transactions aren't emitted for previous events.
            self.assertEquals(0, len(msg_2['minute_perf']['transactions']),
                              "The second message should have no "
                              "transactions.")

            self.assertEquals(1, len(msg_1['minute_perf']['orders']),
                              "The first message should contain one orders.")
            # Check that orders aren't emitted for previous events.
            self.assertEquals(0, len(msg_2['minute_perf']['orders']),
                              "The second message should have no orders.")

            # Ensure that period_close moves through time.
            # Also, ensure that the period_closes are the expected dts.
            self.assertEquals(foo_event_1.dt,
                              msg_1['minute_perf']['period_close'])
            self.assertEquals(foo_event_2.dt,
                              msg_2['minute_perf']['period_close'])

            # In this test event1 transactions arrive on the first bar.
            # This leads to no returns as the price is constant.
            # Sharpe ratio cannot be computed and is None.
            # In the second bar we can start establishing a sharpe ratio.
            self.assertIsNone(msg_1['cumulative_risk_metrics']['sharpe'])
            self.assertIsNotNone(msg_2['cumulative_risk_metrics']['sharpe'])

            check_perf_tracker_serialization(tracker)

    def test_serialization(self):
        start_dt = datetime(year=2008,
                            month=10,
                            day=9,
                            tzinfo=pytz.utc)
        end_dt = datetime(year=2008,
                          month=10,
                          day=16,
                          tzinfo=pytz.utc)

        sim_params = SimulationParameters(
            period_start=start_dt,
            period_end=end_dt
        )

        perf_tracker = perf.PerformanceTracker(
            sim_params
        )
        check_perf_tracker_serialization(perf_tracker)


class TestPosition(unittest.TestCase):
    def setUp(self):
        pass

    def test_serialization(self):
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos = perf.Position(10, amount=np.float64(120.0), last_sale_date=dt,
                            last_sale_price=3.4)

        p_string = pickle.dumps(pos)

        test = pickle.loads(p_string)
        nt.assert_dict_equal(test.__dict__, pos.__dict__)


class TestPositionTracker(unittest.TestCase):
    def setUp(self):
        pass

    def test_empty_positions(self):
        """
        make sure all the empty position stats return a numeric 0

        Originally this bug was due to np.dot([], []) returning
        np.bool_(False)
        """
        pt = perf.PositionTracker()

        stats = [
            'calculate_positions_value',
            '_net_exposure',
            '_gross_value',
            '_gross_exposure',
            '_short_value',
            '_short_exposure',
            '_shorts_count',
            '_long_value',
            '_long_exposure',
            '_longs_count',
        ]
        for name in stats:
            meth = getattr(pt, name)
            val = meth()
            self.assertEquals(val, 0)
            self.assertNotIsInstance(val, (bool, np.bool_))

    def test_update_last_sale(self):
        metadata = {1: {'asset_type': EQUITY},
                    2: {'asset_type': FUTURE,
                        'contract_multiplier': 1000}}
        if trading.environment is None:
            trading.environment = trading.TradingEnvironment()
        trading.environment.update_asset_finder(asset_metadata=metadata)
        pt = perf.PositionTracker()
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos1 = perf.Position(1, amount=np.float64(100.0),
                             last_sale_date=dt, last_sale_price=10)
        pos2 = perf.Position(2, amount=np.float64(100.0),
                             last_sale_date=dt, last_sale_price=10)
        pt.update_positions({1: pos1, 2: pos2})

        event1 = Event({'sid': 1,
                        'price': 11,
                        'dt': dt})
        event2 = Event({'sid': 2,
                        'price': 11,
                        'dt': dt})

        # Check cash-adjustment return value
        self.assertEqual(0, pt.update_last_sale(event1))
        self.assertEqual(100000, pt.update_last_sale(event2))

    def test_position_values_and_exposures(self):
        metadata = {1: {'asset_type': EQUITY},
                    2: {'asset_type': EQUITY},
                    3: {'asset_type': FUTURE,
                        'contract_multiplier': 1000},
                    4: {'asset_type': FUTURE,
                        'contract_multiplier': 1000}}
        if trading.environment is None:
            trading.environment = trading.TradingEnvironment()
        trading.environment.update_asset_finder(asset_metadata=metadata)
        pt = perf.PositionTracker()
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos1 = perf.Position(1, amount=np.float64(10.0),
                             last_sale_date=dt, last_sale_price=10)
        pos2 = perf.Position(2, amount=np.float64(-20.0),
                             last_sale_date=dt, last_sale_price=10)
        pos3 = perf.Position(3, amount=np.float64(30.0),
                             last_sale_date=dt, last_sale_price=10)
        pos4 = perf.Position(4, amount=np.float64(-40.0),
                             last_sale_date=dt, last_sale_price=10)
        pt.update_positions({1: pos1, 2: pos2, 3: pos3, 4: pos4})

        # Test long-only methods
        self.assertEqual(100, pt._long_value())
        self.assertEqual(100 + 300000, pt._long_exposure())

        # Test short-only methods
        self.assertEqual(-200, pt._short_value())
        self.assertEqual(-200 - 400000, pt._short_exposure())

        # Test gross and net values
        self.assertEqual(100 + 200, pt._gross_value())
        self.assertEqual(100 - 200, pt._net_value())

        # Test gross and net exposures
        self.assertEqual(100 + 200 + 300000 + 400000, pt._gross_exposure())
        self.assertEqual(100 - 200 + 300000 - 400000, pt._net_exposure())


    def test_generate_end_sid_transaction(self):
        metadata = {1: {'asset_type': EQUITY},
                    2: {'asset_type': FUTURE,
                        'contract_multiplier': 1000}}
        if trading.environment is None:
            trading.environment = trading.TradingEnvironment()
        trading.environment.update_asset_finder(asset_metadata=metadata)
        pt = perf.PositionTracker()
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos1 = perf.Position(1, amount=np.float64(120.0),
                             last_sale_date=dt, last_sale_price=3.4)
        pos2 = perf.Position(2, amount=np.float64(-100.0),
                             last_sale_date=dt, last_sale_price=3.4)
        pt.update_positions({1: pos1, 2: pos2})

        # Test owned long
        txn = pt._generate_end_sid_transaction(1, dt)
        self.assertEqual(-120, txn.amount)

        # Test owned short
        txn = pt._generate_end_sid_transaction(2, dt)
        self.assertEqual(100, txn.amount)

        # Test not-owned SID
        self.assertIsNone(pt._generate_end_sid_transaction(3, dt))

    def test_serialization(self):
        metadata = {1: {'asset_type': EQUITY},
                    2: {'asset_type': FUTURE,
                        'contract_multiplier': 1000}}
        if trading.environment is None:
            trading.environment = trading.TradingEnvironment()
        trading.environment.update_asset_finder(asset_metadata=metadata)
        pt = perf.PositionTracker()
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos1 = perf.Position(1, amount=np.float64(120.0),
                             last_sale_date=dt, last_sale_price=3.4)
        pos2 = perf.Position(2, amount=np.float64(100.0),
                             last_sale_date=dt, last_sale_price=3.4)

        pt.update_positions({1: pos1, 2: pos2})
        p_string = pickle.dumps(pt)
        test = pickle.loads(p_string)
        nt.assert_dict_equal(test._position_amounts, pt._position_amounts)
        nt.assert_dict_equal(test._position_last_sale_prices,
                             pt._position_last_sale_prices)
        nt.assert_count_equal(test.positions.keys(), pt.positions.keys())
        for sid in pt.positions:
            nt.assert_dict_equal(test.positions[sid].__dict__,
                                 pt.positions[sid].__dict__)


class TestPerformancePeriod(unittest.TestCase):
    def setUp(self):
        pass

    def test_serialization(self):
        pt = perf.PositionTracker()
        pp = perf.PerformancePeriod(100)
        pp.position_tracker = pt

        p_string = pickle.dumps(pp)
        test = pickle.loads(p_string)

        correct = pp.__dict__.copy()
        del correct['_position_tracker']

        nt.assert_count_equal(test.__dict__.keys(), correct.keys())

        equal_keys = list(correct.keys())
        equal_keys.remove('_account_store')
        equal_keys.remove('_portfolio_store')

        for k in equal_keys:
            nt.assert_equal(test.__dict__[k], correct[k])
