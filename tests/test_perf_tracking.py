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

import collections
from datetime import (
    datetime,
    timedelta,
)
import logging

from testfixtures import TempDirectory
import unittest
from nose_parameterized import parameterized
import nose.tools as nt
import pytz
import itertools

import pandas as pd
import numpy as np
from six.moves import range, zip

from zipline.data.us_equity_pricing import (
    SQLiteAdjustmentWriter,
    SQLiteAdjustmentReader,
)
import zipline.utils.factory as factory
import zipline.finance.performance as perf
from zipline.finance.transaction import Transaction, create_transaction
import zipline.utils.math_utils as zp_math

from zipline.gens.composites import date_sorted_sources
from zipline.finance.trading import SimulationParameters
from zipline.finance.blotter import Order
from zipline.finance.commission import PerShare, PerTrade, PerDollar
from zipline.finance.trading import TradingEnvironment
from zipline.pipeline.loaders.synthetic import NullAdjustmentReader
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.serialization_utils import (
    loads_with_persistent_ids, dumps_with_persistent_ids
)
import zipline.protocol as zp
from zipline.protocol import Event
from zipline.utils.test_utils import create_data_portal_from_trade_history

logger = logging.getLogger('Test Perf Tracking')

onesec = timedelta(seconds=1)
oneday = timedelta(days=1)
tradingday = timedelta(hours=6, minutes=30)

# nose.tools changed name in python 3
if not hasattr(nt, 'assert_count_equal'):
    nt.assert_count_equal = nt.assert_items_equal


def check_perf_period(pp,
                      pt,
                      gross_leverage,
                      net_leverage,
                      long_exposure,
                      longs_count,
                      short_exposure,
                      shorts_count):

    pos_stats = pt.stats()
    pp_stats = pp.stats(pt.positions, pos_stats)
    perf_data = pp.to_dict(pos_stats, pp_stats, pt)
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


def create_txn(sid, dt, price, amount):
    """
    Create a fake transaction to be filled and processed prior to the execution
    of a given trade event.
    """
    mock_order = Order(dt, sid, amount, id=None)
    return create_transaction(sid, dt, mock_order, price, amount)


def benchmark_events_in_range(sim_params, env):
    return [
        Event({'dt': dt,
               'returns': ret,
               'type': zp.DATASOURCE_TYPE.BENCHMARK,
               # We explicitly rely on the behavior that benchmarks sort before
               # any other events.
               'source_id': '1Abenchmarks'})
        for dt, ret in env.benchmark_returns.iteritems()
        if dt.date() >= sim_params.period_start.date() and
        dt.date() <= sim_params.period_end.date()
    ]


def calculate_results(sim_params,
                      env,
                      tempdir,
                      benchmark_events,
                      trade_events,
                      adjustment_reader,
                      splits=None,
                      txns=None,
                      commissions=None):
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
    splits = splits or {}
    commissions = commissions or {}

    adjustment_reader = adjustment_reader or NullAdjustmentReader()

    data_portal = create_data_portal_from_trade_history(
        env,
        tempdir,
        sim_params,
        trade_events,
    )
    data_portal._adjustment_reader = adjustment_reader

    perf_tracker = perf.PerformanceTracker(sim_params, env, data_portal)

    results = []

    for date in sim_params.trading_days:

        for txn in filter(lambda txn: txn.dt == date, txns):
            # Process txns for this date.
            perf_tracker.process_transaction(txn)

        try:
            commissions_for_date = commissions[date]
            for comm in commissions_for_date:
                perf_tracker.process_commission(comm)
        except KeyError:
            pass

        try:
            splits_for_date = splits[date]
            perf_tracker.handle_splits(splits_for_date)
        except KeyError:
            pass

        msg = perf_tracker.handle_market_close_daily(date)
        msg['account'] = perf_tracker.get_account(date)
        results.append(msg)
    return results


def check_perf_tracker_serialization(perf_tracker):
    scalar_keys = [
        'emission_rate',
        'txn_count',
        'market_open',
        'last_close',
        'period_start',
        'day_count',
        'capital_base',
        'market_close',
        'saved_dt',
        'period_end',
        'total_days',
    ]
    p_string = dumps_with_persistent_ids(perf_tracker)

    test = loads_with_persistent_ids(p_string, env=perf_tracker.env)

    for k in scalar_keys:
        nt.assert_equal(getattr(test, k), getattr(perf_tracker, k), k)


def setup_env_data(env, sim_params, sids):
    data = {}
    for sid in sids:
        data[sid] = {
            "start_date": sim_params.trading_days[0],
            "end_date": sim_params.trading_days[-1]
        }

    env.write_data(equities_data=data)


class TestSplitPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.sim_params = create_simulation_parameters(num_days=2,
                                                      capital_base=10e3)

        setup_env_data(cls.env, cls.sim_params, [1])

        cls.benchmark_events = benchmark_events_in_range(cls.sim_params,
                                                         cls.env)
        cls.tempdir = TempDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_split_long_position(self):
        events = factory.create_trade_history(
            1,
            # TODO: Should we provide adjusted prices in the tests, or provide
            # raw prices and adjust via DataPortal?
            [20, 60],
            [100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        # set up a long position in sid 1
        # 100 shares at $20 apiece = $2000 position
        txns = [create_txn(events[0].sid, events[0].dt, 20, 100)]

        # set up a split with ratio 3 occurring at the start of the second
        # day.
        splits = {
            events[1].dt: [(1, 3)]
        }

        results = calculate_results(self.sim_params, self.env,
                                    self.tempdir,
                                    self.benchmark_events,
                                    {1: events},
                                    NullAdjustmentReader(),
                                    txns=txns, splits=splits)

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
                                    daily_perf['ending_cash'], 1),
            "ending_cash was {0}".format(daily_perf['ending_cash']))

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
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.sim_params = create_simulation_parameters(num_days=5,
                                                      capital_base=10e3)
        setup_env_data(cls.env, cls.sim_params, [0, 1, 133])

        cls.benchmark_events = benchmark_events_in_range(cls.sim_params,
                                                         cls.env)
        cls.tempdir = TempDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_commission_event(self):
        trade_events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        # Test commission models and validate result
        # Expected commission amounts:
        # PerShare commission:  1.00, 1.00, 1.50 = $3.50
        # PerTrade commission:  5.00, 5.00, 5.00 = $15.00
        # PerDollar commission: 1.50, 3.00, 4.50 = $9.00
        # Total commission = $3.50 + $15.00 + $9.00 = $27.50

        # Create 3 transactions:  50, 100, 150 shares traded @ $20
        first_trade = trade_events[0]
        transactions = [create_txn(first_trade.sid, first_trade.dt, 20, i)
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
        commissions = {}
        cash_adj_dt = trade_events[0].dt
        cash_adjustment = factory.create_commission(1, 300.0, cash_adj_dt)
        commissions[cash_adj_dt] = [cash_adjustment]

        # Insert a purchase order.
        txns = [create_txn(1, cash_adj_dt, 20, 1)]
        results = calculate_results(self.sim_params,
                                    self.env,
                                    self.tempdir,
                                    self.benchmark_events,
                                    {1: trade_events},
                                    NullAdjustmentReader(),
                                    txns=txns,
                                    commissions=commissions)

        # Validate that we lost 320 dollars from our cash pool.
        self.assertEqual(results[-1]['cumulative_perf']['ending_cash'],
                         9680, "Should have lost 320 from cash pool.")
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
            self.sim_params,
            env=self.env
        )

        # Buy and sell the same sid so that we have a zero position by the
        # time of events[3].
        txns = [
            create_txn(events[0].sid, events[0].dt, 20, 1),
            create_txn(events[1].sid, events[1].dt, 20, -1),
        ]

        # Add a cash adjustment at the time of event[3].
        cash_adj_dt = events[3].dt
        commissions = {}
        cash_adjustment = factory.create_commission(1, 300.0, cash_adj_dt)
        commissions[cash_adj_dt] = [cash_adjustment]

        results = calculate_results(self.sim_params,
                                    self.env,
                                    self.tempdir,
                                    self.benchmark_events,
                                    {1: events},
                                    NullAdjustmentReader(),
                                    txns=txns,
                                    commissions=commissions)
        # Validate that we lost 300 dollars from our cash pool.
        self.assertEqual(results[-1]['cumulative_perf']['ending_cash'],
                         9700)

    def test_commission_no_position(self):
        """
        Ensure no position-not-found or sid-not-found errors.
        """
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        # Add a cash adjustment at the time of event[3].
        cash_adj_dt = events[3].dt
        commissions = {}
        cash_adjustment = factory.create_commission(1, 300.0, cash_adj_dt)
        commissions[cash_adj_dt] = [cash_adjustment]

        results = calculate_results(self.sim_params,
                                    self.env,
                                    self.tempdir,
                                    self.benchmark_events,
                                    {1: events},
                                    NullAdjustmentReader(),
                                    commissions=commissions)
        # Validate that we lost 300 dollars from our cash pool.
        self.assertEqual(results[-1]['cumulative_perf']['ending_cash'],
                         9700)


class MockDailyBarSpotReader(object):

    def spot_price(self, sid, day, colname):
        return 100.0


class TestDividendPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.sim_params = create_simulation_parameters(num_days=6,
                                                      capital_base=10e3)

        setup_env_data(cls.env, cls.sim_params, [1, 2])

        cls.benchmark_events = benchmark_events_in_range(cls.sim_params,
                                                         cls.env)

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def setUp(self):
        self.tempdir = TempDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_market_hours_calculations(self):
        # DST in US/Eastern began on Sunday March 14, 2010
        before = datetime(2010, 3, 12, 14, 31, tzinfo=pytz.utc)
        after = factory.get_next_trading_dt(
            before,
            timedelta(days=1),
            self.env,
        )
        self.assertEqual(after.hour, 13)

    def test_long_position_receives_dividend(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([events[2].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        # Simulate a transaction being filled prior to the ex_date.
        txns = [create_txn(events[0].sid, events[0].dt, 10.0, 100)]
        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            adjustment_reader,
            txns=txns,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.1, 0.1, 0.1, 0.1])
        daily_returns = [event['daily_perf']['returns']
                         for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.10, 0.0, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used']
                      for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 1000, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [-1000, -1000, 0, 0, 0, 0])
        cash_pos = \
            [event['cumulative_perf']['ending_cash'] for event in results]
        self.assertEqual(cash_pos, [9000, 9000, 10000, 10000, 10000, 10000])

    def test_long_position_receives_stock_dividend(self):
        # post some trades in the market
        events = {}
        for sid in (1, 2):
            events[sid] = factory.create_trade_history(
                sid,
                [10, 10, 10, 10, 10, 10],
                [100, 100, 100, 100, 100, 100],
                oneday,
                self.sim_params,
                env=self.env
            )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([], dtype=np.uint32),
            'amount': np.array([], dtype=np.float64),
            'declared_date': np.array([], dtype='datetime64[ns]'),
            'ex_date': np.array([], dtype='datetime64[ns]'),
            'pay_date': np.array([], dtype='datetime64[ns]'),
            'record_date': np.array([], dtype='datetime64[ns]'),
        })
        sid_1 = events[1]
        stock_dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'payment_sid': np.array([2], dtype=np.uint32),
            'ratio': np.array([2], dtype=np.float64),
            'declared_date': np.array([sid_1[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([sid_1[1].dt], dtype='datetime64[ns]'),
            'record_date': np.array([sid_1[1].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([sid_1[2].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends, stock_dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        txns = [create_txn(events[1][0].sid, events[1][0].dt, 10.0, 100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            events,
            adjustment_reader,
            txns=txns,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.2, 0.2, 0.2, 0.2])
        daily_returns = [event['daily_perf']['returns']
                         for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used']
                      for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [-1000] * 6)
        cash_pos = \
            [event['cumulative_perf']['ending_cash'] for event in results]
        self.assertEqual(cash_pos, [9000] * 6)

    def test_long_position_purchased_on_ex_date_receives_no_dividend(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([events[2].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        # Simulate a transaction being filled on the ex_date.
        txns = [create_txn(events[1].sid, events[1].dt, 10.0, 100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            adjustment_reader,
            txns=txns,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0, 0, 0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, -1000, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows,
                         [0, -1000, -1000, -1000, -1000, -1000])

    def test_selling_before_dividend_payment_still_gets_paid(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([events[3].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        buy_txn = create_txn(events[0].sid, events[0].dt, 10.0, 100)
        sell_txn = create_txn(events[2].sid, events[2].dt, 10.0, -100)
        txns = [buy_txn, sell_txn]

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            adjustment_reader,
            txns=txns,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0.1, 0.1, 0.1])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0.1, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 1000, 1000, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows,
                         [-1000, -1000, 0, 1000, 1000, 1000])

    def test_buy_and_sell_before_ex(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )
        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )

        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.0], dtype=np.float64),
            'declared_date': np.array([events[3].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[4].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([events[5].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[4].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        buy_txn = create_txn(events[1].sid, events[1].dt, 10.0, 100)
        sell_txn = create_txn(events[2].sid, events[2].dt, 10.0, -100)
        txns = [buy_txn, sell_txn]

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            txns=txns,
            adjustment_reader=adjustment_reader,
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
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        pay_date = self.sim_params.first_open
        # find pay date that is much later.
        for i in range(30):
            pay_date = factory.get_next_trading_dt(pay_date, oneday, self.env)

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([pay_date], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        txns = [create_txn(events[1].sid, events[1].dt, 10.0, 100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            txns=txns,
            adjustment_reader=adjustment_reader,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0, 0, 0, 0.0, 0.0, 0.0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0, 0, 0, 0, 0, 0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, -1000, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(
            cumulative_cash_flows,
            [0, -1000, -1000, -1000, -1000, -1000]
        )

    def test_short_position_pays_dividend(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[2].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[2].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([events[3].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        txns = [create_txn(events[1].sid, events[1].dt, 10.0, -100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            adjustment_reader,
            txns=txns,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.0, -0.1, -0.1, -0.1])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.0, -0.1, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, 1000, 0, -1000, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [0, 1000, 1000, 0, 0, 0])

    def test_no_position_receives_no_dividend(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[1].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([events[2].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[2].dt], dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        results = calculate_results(
            self.sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            adjustment_reader,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [0, 0, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows, [0, 0, 0, 0, 0, 0])

    def test_no_dividend_at_simulation_end(self):
        # post some trades in the market
        events = factory.create_trade_history(
            1,
            [10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        dbpath = self.tempdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(dbpath, self.env.trading_days,
                                        MockDailyBarSpotReader())
        splits = mergers = pd.DataFrame(
            {
                # Hackery to make the dtypes correct on an empty frame.
                'effective_date': np.array([], dtype=int),
                'ratio': np.array([], dtype=float),
                'sid': np.array([], dtype=int),
            },
            index=pd.DatetimeIndex([], tz='UTC'),
            columns=['effective_date', 'ratio', 'sid'],
        )
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[-3].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[-2].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'pay_date': np.array([self.env.next_trading_day(events[-1].dt)],
                                 dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        # Set the last day to be the last event
        sim_params = create_simulation_parameters(
            num_days=6,
            capital_base=10e3,
            start=self.sim_params.period_start,
            end=self.sim_params.period_end
        )

        sim_params.period_end = events[-1].dt
        sim_params.update_internal_from_env(self.env)

        # Simulate a transaction being filled prior to the ex_date.
        txns = [create_txn(events[0].sid, events[0].dt, 10.0, 100)]
        results = calculate_results(
            sim_params,
            self.env,
            self.tempdir,
            self.benchmark_events,
            {1: events},
            adjustment_reader=adjustment_reader,
            txns=txns,
        )

        self.assertEqual(len(results), 5)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.0, 0.0, 0.0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.0, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows,
                         [-1000, -1000, -1000, -1000, -1000])


class TestDividendPerformanceHolidayStyle(TestDividendPerformance):

    # The holiday tests begins the simulation on the day
    # before Thanksgiving, so that the next trading day is
    # two days ahead. Any tests that hard code events
    # to be start + oneday will fail, since those events will
    # be skipped by the simulation.

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.sim_params = create_simulation_parameters(
            num_days=6,
            capital_base=10e3,
            start=pd.Timestamp("2003-11-30", tz='UTC'),
            end=pd.Timestamp("2003-12-08", tz='UTC')
        )

        setup_env_data(cls.env, cls.sim_params, [1, 2])

        cls.benchmark_events = benchmark_events_in_range(cls.sim_params,
                                                         cls.env)


class TestPositionPerformance(unittest.TestCase):

    def setUp(self):
        self.tempdir = TempDirectory()

    def create_environment_stuff(self, num_days=4, sids=[1, 2]):
        self.env = TradingEnvironment()
        self.sim_params = create_simulation_parameters(num_days=num_days)

        setup_env_data(self.env, self.sim_params, [1, 2])

        self.finder = self.env.asset_finder

        self.benchmark_events = benchmark_events_in_range(self.sim_params,
                                                          self.env)

    def tearDown(self):
        self.tempdir.cleanup()
        del self.env

    def test_long_short_positions(self):
        """
        start with $1000
        buy 100 stock1 shares at $10
        sell short 100 stock2 shares at $10
        stock1 then goes down to $9
        stock2 goes to $11
        """
        self.create_environment_stuff()

        trades_1 = factory.create_trade_history(
            1,
            [10, 10, 10, 9],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        trades_2 = factory.create_trade_history(
            2,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            onesec,
            self.sim_params,
            env=self.env
        )

        txn1 = create_txn(trades_1[1].sid, trades_1[1].dt, 10.0, 100)
        txn2 = create_txn(trades_2[1].sid, trades_1[1].dt, 10.0, -100)

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades_1, 2: trades_2}
        )

        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder, data_portal)
        pt.execute_transaction(txn1)
        pp.handle_execution(txn1)
        pt.execute_transaction(txn2)
        pp.handle_execution(txn2)

        check_perf_period(
            pp,
            pt,
            gross_leverage=2.0,
            net_leverage=0.0,
            long_exposure=1000.0,
            longs_count=1,
            short_exposure=-1000.0,
            shorts_count=1)

        dt = trades_1[-2].dt
        pt.sync_last_sale_prices(dt)

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)
        # Validate that the account attributes were updated.
        account = pp.as_account(pos_stats, pp_stats)
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

        # Validate that the account attributes were updated.
        dt = trades_1[-1].dt
        pt.sync_last_sale_prices(dt)
        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)
        account = pp.as_account(pos_stats, pp_stats)

        check_perf_period(
            pp,
            pt,
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

        self.create_environment_stuff()

        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades})

        txn = create_txn(trades[1].sid, trades[1].dt, 10.0, 1000)
        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder, data_portal)

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        check_perf_period(
            pp,
            pt,
            gross_leverage=10.0,
            net_leverage=10.0,
            long_exposure=10000.0,
            longs_count=1,
            short_exposure=0.0,
            shorts_count=0)

        # Validate that the account attributes were updated.
        pt.sync_last_sale_prices(trades[-2].dt)
        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)
        account = pp.as_account(pos_stats, pp_stats)
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
        pt.sync_last_sale_prices(trades[-1].dt)

        check_perf_period(
            pp,
            pt,
            gross_leverage=5.5,
            net_leverage=5.5,
            long_exposure=11000.0,
            longs_count=1,
            short_exposure=0.0,
            shorts_count=0)

        # Validate that the account attributes were updated.
        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)
        account = pp.as_account(pos_stats, pp_stats)

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
        self.create_environment_stuff()

        # post some trades in the market
        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            onesec,
            self.sim_params,
            env=self.env
        )

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades})

        txn = create_txn(trades[1].sid, trades[1].dt, 10.0, 100)
        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    data_portal,
                                    period_open=self.sim_params.period_start,
                                    period_close=self.sim_params.period_end)
        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        # This verifies that the last sale price is being correctly
        # set in the positions. If this is not the case then returns can
        # incorrectly show as sharply dipping if a transaction arrives
        # before a trade. This is caused by returns being based on holding
        # stocks with a last sale price of 0.
        self.assertEqual(pt.positions[1].last_sale_price, 10.0)

        for trade in trades:
            pt.update_last_sale(trade)

        self.assertEqual(
            pp.period_cash_flow,
            -1 * txn.price * txn.amount,
            "capital used should be equal to the opposite of the transaction \
            cost of sole txn in test"
        )

        self.assertEqual(
            len(pt.positions),
            1,
            "should be just one position")

        self.assertEqual(
            pt.positions[1].sid,
            txn.sid,
            "position should be in security with id 1")

        self.assertEqual(
            pt.positions[1].amount,
            txn.amount,
            "should have a position of {sharecount} shares".format(
                sharecount=txn.amount
            )
        )

        self.assertEqual(
            pt.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            pt.positions[1].last_sale_price,
            trades[-1]['price'],
            "last sale should be same as last trade. \
            expected {exp} actual {act}".format(
                exp=trades[-1]['price'],
                act=pt.positions[1].last_sale_price)
        )

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)

        self.assertEqual(
            pos_stats.net_value,
            1100,
            "ending value should be price of last trade times number of \
            shares in position"
        )

        self.assertEqual(pp_stats.pnl, 100,
                         "gain of 1 on 100 shares should be 100")

        check_perf_period(
            pp,
            pt,
            gross_leverage=1.0,
            net_leverage=1.0,
            long_exposure=1100.0,
            longs_count=1,
            short_exposure=0.0,
            shorts_count=0)

        # Validate that the account attributes were updated.
        account = pp.as_account(pos_stats, pp_stats)
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
        self.create_environment_stuff(num_days=6)

        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11, 10, 9],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            env=self.env
        )

        trades_1 = trades[:-2]

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades})

        txn = create_txn(trades[1].sid, trades[1].dt, 10.0, -100)
        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder, data_portal)

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        pt.sync_last_sale_prices(trades_1[-1].dt)

        self.assertEqual(
            pp.period_cash_flow,
            -1 * txn.price * txn.amount,
            "capital used should be equal to the opposite of the transaction\
             cost of sole txn in test"
        )

        self.assertEqual(
            len(pt.positions),
            1,
            "should be just one position")

        self.assertEqual(
            pt.positions[1].sid,
            txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            pt.positions[1].amount,
            -100,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            pt.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            pt.positions[1].last_sale_price,
            trades_1[-1]['price'],
            "last sale should be price of last trade"
        )

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)

        self.assertEqual(
            pos_stats.net_value,
            -1100,
            "ending value should be price of last trade times number of \
            shares in position"
        )

        self.assertEqual(pp_stats.pnl, -100,
                         "gain of 1 on 100 shares should be 100")

        # simulate additional trades, and ensure that the position value
        # reflects the new price
        trades_2 = trades[-2:]

        # simulate a rollover to a new period
        pp.rollover(pos_stats, pp_stats)

        pt.sync_last_sale_prices(trades[-1].dt)

        self.assertEqual(
            pp.period_cash_flow,
            0,
            "capital used should be zero, there were no transactions in \
            performance period"
        )

        self.assertEqual(
            len(pt.positions),
            1,
            "should be just one position"
        )

        self.assertEqual(
            pt.positions[1].sid,
            txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            pt.positions[1].amount,
            -100,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            pt.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            pt.positions[1].last_sale_price,
            trades_2[-1].price,
            "last sale should be price of last trade"
        )

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)

        self.assertEqual(
            pos_stats.net_value,
            -900,
            "ending value should be price of last trade times number of \
            shares in position")

        self.assertEqual(
            pp_stats.pnl,
            200,
            "drop of 2 on -100 shares should be 200"
        )

        # now run a performance period encompassing the entire trade sample.
        ptTotal = perf.PositionTracker(self.env.asset_finder, data_portal)
        ppTotal = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                         data_portal)

        ptTotal.execute_transaction(txn)
        ppTotal.handle_execution(txn)

        ptTotal.sync_last_sale_prices(trades[-1].dt)

        self.assertEqual(
            ppTotal.period_cash_flow,
            -1 * txn.price * txn.amount,
            "capital used should be equal to the opposite of the transaction \
cost of sole txn in test"
        )

        self.assertEqual(
            len(ptTotal.positions),
            1,
            "should be just one position"
        )
        self.assertEqual(
            ptTotal.positions[1].sid,
            txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            ptTotal.positions[1].amount,
            -100,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            ptTotal.positions[1].cost_basis,
            txn.price,
            "should have a cost basis of 10"
        )

        self.assertEqual(
            ptTotal.positions[1].last_sale_price,
            trades_2[-1].price,
            "last sale should be price of last trade"
        )

        pos_total_stats = ptTotal.stats()
        pp_total_stats = ppTotal.stats(ptTotal.positions, pos_total_stats)

        self.assertEqual(
            pos_total_stats.net_value,
            -900,
            "ending value should be price of last trade times number of \
            shares in position")

        self.assertEqual(
            pp_total_stats.pnl,
            100,
            "drop of 1 on -100 shares should be 100"
        )

        check_perf_period(
            pp,
            pt,
            gross_leverage=0.8181,
            net_leverage=-0.8181,
            long_exposure=0.0,
            longs_count=0,
            short_exposure=-900.0,
            shorts_count=1)

        # Validate that the account attributes.
        account = ppTotal.as_account(pos_stats, pp_stats)
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
        self.create_environment_stuff(num_days=10)

        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11, 9, 8, 7, 8, 9, 10],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            onesec,
            self.sim_params,
            env=self.env
        )

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades})

        short_txn = create_txn(
            trades[1].sid,
            trades[1].dt,
            10.0,
            -100,
        )
        cover_txn = create_txn(trades[6].sid, trades[6].dt, 7.0, 100)
        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    data_portal)

        pt.execute_transaction(short_txn)
        pp.handle_execution(short_txn)
        pt.execute_transaction(cover_txn)
        pp.handle_execution(cover_txn)

        for trade in trades:
            pt.update_last_sale(trade)

        short_txn_cost = short_txn.price * short_txn.amount
        cover_txn_cost = cover_txn.price * cover_txn.amount

        self.assertEqual(
            pp.period_cash_flow,
            -1 * short_txn_cost - cover_txn_cost,
            "capital used should be equal to the net transaction costs"
        )

        self.assertEqual(
            len(pt.positions),
            1,
            "should be just one position"
        )

        self.assertEqual(
            pt.positions[1].sid,
            short_txn.sid,
            "position should be in security from the transaction"
        )

        self.assertEqual(
            pt.positions[1].amount,
            0,
            "should have a position of -100 shares"
        )

        self.assertEqual(
            pt.positions[1].cost_basis,
            0,
            "a covered position should have a cost basis of 0"
        )

        self.assertEqual(
            pt.positions[1].last_sale_price,
            trades[-1].price,
            "last sale should be price of last trade"
        )

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)

        self.assertEqual(
            pos_stats.net_value,
            0,
            "ending value should be price of last trade times number of \
shares in position"
        )

        self.assertEqual(
            pp_stats.pnl,
            300,
            "gain of 1 on 100 shares should be 300"
        )

        check_perf_period(
            pp,
            pt,
            gross_leverage=0.0,
            net_leverage=0.0,
            long_exposure=0.0,
            longs_count=0,
            short_exposure=0.0,
            shorts_count=0)

        account = pp.as_account(pos_stats, pp_stats)
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
        self.create_environment_stuff(num_days=5)

        history_args = (
            1,
            [10, 11, 11, 12, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            self.env
        )
        trades = factory.create_trade_history(*history_args)
        transactions = factory.create_txn_history(*history_args)[:4]

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades})

        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(
            1000.0,
            self.env.asset_finder,
            data_portal,
            period_open=self.sim_params.period_start,
            period_close=self.sim_params.trading_days[-1]
        )
        average_cost = 0
        for i, txn in enumerate(transactions):
            pt.execute_transaction(txn)
            pp.handle_execution(txn)
            average_cost = (average_cost * i + txn.price) / (i + 1)
            self.assertEqual(pt.positions[1].cost_basis, average_cost)

        dt = trades[-2].dt
        self.assertEqual(
            pt.positions[1].last_sale_price,
            trades[-2].price,
            "should have a last sale of 12, got {val}".format(
                val=pt.positions[1].last_sale_price)
        )

        self.assertEqual(
            pt.positions[1].cost_basis,
            11,
            "should have a cost basis of 11"
        )

        pt.sync_last_sale_prices(dt)

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)

        self.assertEqual(
            pp_stats.pnl,
            400
        )

        down_tick = trades[-1]

        sale_txn = create_txn(
            down_tick.sid,
            down_tick.dt,
            10.0,
            -100)

        pp.rollover(pos_stats, pp_stats)

        pt.execute_transaction(sale_txn)
        pp.handle_execution(sale_txn)

        dt = down_tick.dt
        pt.sync_last_sale_prices(dt)

        self.assertEqual(
            pt.positions[1].last_sale_price,
            10,
            "should have a last sale of 10, was {val}".format(
                val=pt.positions[1].last_sale_price)
        )

        self.assertEqual(
            pt.positions[1].cost_basis,
            11,
            "should have a cost basis of 11"
        )

        pos_stats = pt.stats()
        pp_stats = pp.stats(pt.positions, pos_stats)

        self.assertEqual(pp_stats.pnl, -800,
                         "this period goes from +400 to -400")

        pt3 = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp3 = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                     data_portal)

        average_cost = 0
        for i, txn in enumerate(transactions):
            pt3.execute_transaction(txn)
            pp3.handle_execution(txn)
            average_cost = (average_cost * i + txn.price) / (i + 1)
            self.assertEqual(pt3.positions[1].cost_basis, average_cost)

        pt3.execute_transaction(sale_txn)
        pp3.handle_execution(sale_txn)

        trades.append(down_tick)
        for trade in trades:
            pt3.update_last_sale(trade)

        self.assertEqual(
            pt3.positions[1].last_sale_price,
            10,
            "should have a last sale of 10"
        )

        self.assertEqual(
            pt3.positions[1].cost_basis,
            11,
            "should have a cost basis of 11"
        )

        pt3.sync_last_sale_prices(dt)
        pt3_stats = pt3.stats()
        pp3_stats = pp3.stats(pt3.positions, pt3_stats)

        self.assertEqual(
            pp3_stats.pnl,
            -400,
            "should be -400 for all trades and transactions in period"
        )

    def test_cost_basis_calc_close_pos(self):
        self.create_environment_stuff(num_days=8)

        history_args = (
            1,
            [10, 9, 11, 8, 9, 12, 13, 14],
            [200, -100, -100, 100, -300, 100, 500, 400],
            onesec,
            self.sim_params,
            self.env
        )
        cost_bases = [10, 10, 0, 8, 9, 9, 13, 13.5]

        trades = factory.create_trade_history(*history_args)
        transactions = factory.create_txn_history(*history_args)

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            {1: trades})

        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder, data_portal)

        for txn, cb in zip(transactions, cost_bases):
            pt.execute_transaction(txn)
            pp.handle_execution(txn)
            self.assertEqual(pt.positions[1].cost_basis, cb)

        self.assertEqual(pt.positions[1].cost_basis, cost_bases[-1])


class TestPerformanceTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.env.write_data(equities_identifiers=[1, 2, 133, 134])

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def setUp(self):
        self.tempdir = TempDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

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
        price2 = 12.12
        price2_list = [price2] * trade_count
        volume = [100] * trade_count
        trade_time_increment = timedelta(days=1)

        price_list1 = np.array(price_list)
        price_list2 = np.array(price2_list)
        volume1 = np.array(volume)
        volume2 = np.array(volume)

        # 'middle' start of 3 depends on number of days == 7
        middle = 3

        # First delete from middle
        if days_to_delete.middle:
            volume1[middle:(middle + days_to_delete.middle)] = 0
            volume2[middle:(middle + days_to_delete.middle)] = 0
            price_list1[middle:(middle + days_to_delete.middle)] = 0
            price_list2[middle:(middle + days_to_delete.middle)] = 0

        # Delete start
        if days_to_delete.start:
            volume1[:days_to_delete.start]
            volume2[:days_to_delete.start] = 0
            price_list1[:days_to_delete.start] = 0
            price_list2[:days_to_delete.start] = 0

        # Delete from end
        if days_to_delete.end:
            volume1[-days_to_delete.end:] = 0
            volume2[-days_to_delete.end:] = 0
            price_list1[-days_to_delete.end:] = 0
            price_list2[-days_to_delete.end:] = 0

        env = TradingEnvironment()

        sim_params = SimulationParameters(
            period_start=start_dt,
            period_end=end_dt,
            env=env,
        )

        env.write_data(equities_data={
            133: {
                "start_date": sim_params.trading_days[0],
                "end_date": sim_params.trading_days[-1]
            },
            134: {
                "start_date": sim_params.trading_days[0],
                "end_date": sim_params.trading_days[-1]
            },
        })

        benchmark_events = benchmark_events_in_range(sim_params, env)

        trade_history = factory.create_trade_history(
            sid,
            price_list1,
            volume1,
            trade_time_increment,
            sim_params,
            source_id="factory1",
            env=env
        )

        sid2 = 134
        trade_history2 = factory.create_trade_history(
            sid2,
            price_list2,
            volume2,
            trade_time_increment,
            sim_params,
            source_id="factory2",
            env=env
        )

        sim_params.capital_base = 1000.0
        sim_params.frame_index = [
            'sid',
            'volume',
            'dt',
            'price',
            'changed']

        data_portal = create_data_portal_from_trade_history(
            env,
            self.tempdir,
            sim_params,
            {sid: trade_history,
             sid2: trade_history2},
        )
        data_portal._adjustment_reader = NullAdjustmentReader()

        perf_tracker = perf.PerformanceTracker(
            sim_params, env, data_portal
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
                elif event.type == zp.DATASOURCE_TYPE.TRANSACTION:
                    perf_tracker.process_transaction(event)
            msg = perf_tracker.handle_market_close_daily(date)
            perf_messages.append(msg)

        self.assertEqual(perf_tracker.txn_count, len(txns))
        self.assertEqual(perf_tracker.txn_count, len(orders))

        positions = perf_tracker.position_tracker.positions
        if len(txns) == 0:
            self.assertNotIn(sid, positions)
        else:
            expected_size = len(txns) / 2 * -25
            cumulative_pos = positions[sid]
            self.assertEqual(cumulative_pos.amount, expected_size)

            self.assertEqual(len(perf_messages),
                             sim_params.days_in_period)

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

    def write_equity_data(self, env, sim_params, sids):
        data = {}
        for sid in sids:
            data[sid] = {
                "start_date": sim_params.trading_days[0],
                "end_date": sim_params.trading_days[-1]
            }

        env.write_data(equities_data=data)


class TestPosition(unittest.TestCase):
    def setUp(self):
        pass

    def test_serialization(self):
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos = perf.Position(10, amount=np.float64(120.0), last_sale_date=dt,
                            last_sale_price=3.4)

        p_string = dumps_with_persistent_ids(pos)

        test = loads_with_persistent_ids(p_string, env=None)
        nt.assert_dict_equal(test.__dict__, pos.__dict__)


class TestPositionTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()

        equities_metadata = {1: {'asset_type': 'equity'},
                             2: {'asset_type': 'equity'}}
        futures_metadata = {3: {'asset_type': 'future',
                                'contract_multiplier': 1000},
                            4: {'asset_type': 'future',
                                'contract_multiplier': 1000}}
        cls.env.write_data(equities_data=equities_metadata,
                           futures_data=futures_metadata)

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def setUp(self):
        self.tempdir = TempDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_empty_positions(self):
        """
        make sure all the empty position stats return a numeric 0

        Originally this bug was due to np.dot([], []) returning
        np.bool_(False)
        """
        sim_params = factory.create_simulation_parameters(
            num_days=4, env=self.env
        )
        trades = factory.create_trade_history(
            1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            oneday,
            sim_params,
            env=self.env
        )

        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            sim_params,
            {1: trades})

        pt = perf.PositionTracker(self.env.asset_finder, data_portal)
        pos_stats = pt.stats()

        stats = [
            'net_value',
            'net_exposure',
            'gross_value',
            'gross_exposure',
            'short_value',
            'short_exposure',
            'shorts_count',
            'long_value',
            'long_exposure',
            'longs_count',
        ]
        for name in stats:
            val = getattr(pos_stats, name)
            self.assertEquals(val, 0)
            self.assertNotIsInstance(val, (bool, np.bool_))

    def test_position_values_and_exposures(self):
        pt = perf.PositionTracker(self.env.asset_finder, None)
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

        pos_stats = pt.stats()
        self.assertEqual(100, pos_stats.long_value)
        self.assertEqual(100 + 300000, pos_stats.long_exposure)
        self.assertEqual(2, pos_stats.longs_count)

        # Test short-only methods
        self.assertEqual(-200, pos_stats.short_value)
        self.assertEqual(-200 - 400000, pos_stats.short_exposure)
        self.assertEqual(2, pos_stats.shorts_count)

        # Test gross and net values
        self.assertEqual(100 + 200, pos_stats.gross_value)
        self.assertEqual(100 - 200, pos_stats.net_value)

        # Test gross and net exposures
        self.assertEqual(100 + 200 + 300000 + 400000, pos_stats.gross_exposure)
        self.assertEqual(100 - 200 + 300000 - 400000, pos_stats.net_exposure)

    def test_serialization(self):
        pt = perf.PositionTracker(self.env.asset_finder, None)
        dt = pd.Timestamp("1984/03/06 3:00PM")
        pos1 = perf.Position(1, amount=np.float64(120.0),
                             last_sale_date=dt, last_sale_price=3.4)
        pos3 = perf.Position(3, amount=np.float64(100.0),
                             last_sale_date=dt, last_sale_price=3.4)

        pt.update_positions({1: pos1, 3: pos3})
        p_string = dumps_with_persistent_ids(pt)
        test = loads_with_persistent_ids(p_string, env=self.env)
        nt.assert_count_equal(test.positions.keys(), pt.positions.keys())
        for sid in pt.positions:
            nt.assert_dict_equal(test.positions[sid].__dict__,
                                 pt.positions[sid].__dict__)


class TestPerformancePeriod(unittest.TestCase):

    def test_serialization(self):
        env = TradingEnvironment()
        pp = perf.PerformancePeriod(100, env.asset_finder, data_portal=None)

        p_string = dumps_with_persistent_ids(pp)
        test = loads_with_persistent_ids(p_string, env=env)

        correct = pp.__dict__.copy()

        nt.assert_count_equal(test.__dict__.keys(), correct.keys())

        equal_keys = list(correct.keys())
        equal_keys.remove('_account_store')
        equal_keys.remove('_portfolio_store')

        for k in equal_keys:
            nt.assert_equal(test.__dict__[k], correct[k])
