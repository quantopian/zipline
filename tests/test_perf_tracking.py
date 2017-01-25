#
# Copyright 2016 Quantopian, Inc.
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

import copy
from datetime import (
    datetime,
    timedelta,
)
import logging

import nose.tools as nt
import pytz

import pandas as pd
import numpy as np
from six.moves import range, zip

from zipline.assets import Asset
from zipline.assets.synthetic import make_simple_equity_info
from zipline.data.us_equity_pricing import (
    SQLiteAdjustmentWriter,
    SQLiteAdjustmentReader,
)
import zipline.utils.factory as factory
import zipline.finance.performance as perf
from zipline.finance.transaction import create_transaction
import zipline.utils.math_utils as zp_math

from zipline.finance.blotter import Order
from zipline.finance.performance.position import Position
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.serialization_utils import (
    loads_with_persistent_ids, dumps_with_persistent_ids
)
from zipline.testing import (
    MockDailyBarReader,
    create_data_portal_from_trade_history,
    create_empty_splits_mergers_frame,
    tmp_trading_env,
)
from zipline.testing.fixtures import (
    WithInstanceTmpDir,
    WithSimParams,
    WithTmpDir,
    WithTradingEnvironment,
    WithTradingCalendars,
    ZiplineTestCase,
)
from zipline.utils.calendars import get_calendar

logger = logging.getLogger('Test Perf Tracking')

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
                  total_positions_exposure,
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
                               account.settled_cash, rtol=1e-3)
    np.testing.assert_allclose(equity_with_loan,
                               account.equity_with_loan, rtol=1e-3)
    np.testing.assert_allclose(total_positions_value,
                               account.total_positions_value, rtol=1e-3)
    np.testing.assert_allclose(total_positions_exposure,
                               account.total_positions_exposure, rtol=1e-3)
    np.testing.assert_allclose(regt_equity,
                               account.regt_equity, rtol=1e-3)
    np.testing.assert_allclose(available_funds,
                               account.available_funds, rtol=1e-3)
    np.testing.assert_allclose(excess_liquidity,
                               account.excess_liquidity, rtol=1e-3)
    np.testing.assert_allclose(cushion,
                               account.cushion, rtol=1e-3)
    np.testing.assert_allclose(leverage, account.leverage, rtol=1e-3)
    np.testing.assert_allclose(net_leverage,
                               account.net_leverage, rtol=1e-3)
    np.testing.assert_allclose(net_liquidation,
                               account.net_liquidation, rtol=1e-3)


def create_txn(asset, dt, price, amount):
    """
    Create a fake transaction to be filled and processed prior to the execution
    of a given trade event.
    """
    if not isinstance(asset, Asset):
        raise ValueError("pass an asset to create_txn")

    mock_order = Order(dt, asset, amount, id=None)
    return create_transaction(mock_order, dt, price, amount)


def calculate_results(sim_params,
                      env,
                      data_portal,
                      splits=None,
                      txns=None,
                      commissions=None):
    """
    Run the given events through a stripped down version of the loop in
    AlgorithmSimulator.transform.

    IMPORTANT NOTE FOR TEST WRITERS/READERS:

    This loop has some wonky logic for the order of event processing for
    datasource types.  This exists mostly to accommodate legacy tests that were
    making assumptions about how events would be sorted.

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

    perf_tracker = perf.PerformanceTracker(
        sim_params, get_calendar("NYSE"), env
    )

    results = []

    for date in sim_params.sessions:
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

        msg = perf_tracker.handle_market_close(date, data_portal)
        perf_tracker.position_tracker.sync_last_sale_prices(
            date, False, data_portal,
        )
        msg['account'] = perf_tracker.get_account(True)
        results.append(copy.deepcopy(msg))
    return results


def check_perf_tracker_serialization(perf_tracker):
    scalar_keys = [
        'emission_rate',
        'txn_count',
        'market_open',
        'last_close',
        'start_session',
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

    perf_periods = (
        test.cumulative_performance,
        test.todays_performance
    )
    for period in perf_periods:
        nt.assert_true(hasattr(period, '_position_tracker'))


def setup_env_data(env, sim_params, sids, futures_sids=[]):
    data = {}
    for sid in sids:
        data[sid] = {
            "start_date": sim_params.sessions[0],
            "end_date": get_calendar("NYSE").next_session_label(
                sim_params.sessions[-1]
            )
        }

    env.write_data(equities_data=data)

    futures_data = {}
    for future_sid in futures_sids:
        futures_data[future_sid] = {
            "start_date": sim_params.sessions[0],
            # (obviously) FIXME once we have a future calendar
            "end_date": get_calendar("NYSE").next_session_label(
                sim_params.sessions[-1]
            ),
            "multiplier": 100
        }

    env.write_data(futures_data=futures_data)


class TestSplitPerformance(WithSimParams, WithTmpDir, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-04', tz='utc')
    SIM_PARAMS_CAPITAL_BASE = 10e3

    ASSET_FINDER_EQUITY_SIDS = 1, 2

    @classmethod
    def init_class_fixtures(cls):
        super(TestSplitPerformance, cls).init_class_fixtures()
        cls.asset1 = cls.env.asset_finder.retrieve_asset(1)

    def test_multiple_splits(self):
        # if multiple positions all have splits at the same time, verify that
        # the total leftover cash is correct
        perf_tracker = perf.PerformanceTracker(self.sim_params,
                                               self.trading_calendar,
                                               self.env)

        asset1 = self.asset_finder.retrieve_asset(1)
        asset2 = self.asset_finder.retrieve_asset(2)

        perf_tracker.position_tracker.positions[1] = \
            Position(asset1, amount=10, cost_basis=10, last_sale_price=11)

        perf_tracker.position_tracker.positions[2] = \
            Position(asset2, amount=10, cost_basis=10, last_sale_price=11)

        leftover_cash = perf_tracker.position_tracker.handle_splits(
            [(1, 0.333), (2, 0.333)]
        )

        # we used to have 10 shares that each cost us $10, total $100
        # now we have 33 shares that each cost us $3.33, total $99.9
        # each position returns $0.10 as leftover cash
        self.assertEqual(0.2, leftover_cash)

    def test_split_long_position(self):
        events = factory.create_trade_history(
            self.asset1,
            # TODO: Should we provide adjusted prices in the tests, or provide
            # raw prices and adjust via DataPortal?
            [20, 60],
            [100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        # set up a long position in sid 1
        # 100 shares at $20 apiece = $2000 position
        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.tmpdir,
            self.sim_params,
            {1: events},
        )

        txns = [create_txn(self.asset1, events[0].dt, 20, 100)]

        # set up a split with ratio 3 occurring at the start of the second
        # day.
        splits = {
            events[1].dt: [(1, 3)]
        }

        results = calculate_results(self.sim_params,
                                    self.env,
                                    data_portal,
                                    txns=txns,
                                    splits=splits)

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
        self.assertEqual(float('inf'), account.day_trades_remaining)
        # this is a long only portfolio that is only partially invested
        # so net and gross leverage are equal.
        np.testing.assert_allclose(0.198, account.leverage, rtol=1e-3)
        np.testing.assert_allclose(0.198, account.net_leverage, rtol=1e-3)
        np.testing.assert_allclose(8020, account.regt_equity, rtol=1e-3)
        self.assertEqual(float('inf'), account.regt_margin)
        np.testing.assert_allclose(8020, account.available_funds, rtol=1e-3)
        self.assertEqual(0, account.maintenance_margin_requirement)
        np.testing.assert_allclose(10000,
                                   account.equity_with_loan, rtol=1e-3)
        self.assertEqual(float('inf'), account.buying_power)
        self.assertEqual(0, account.initial_margin_requirement)
        np.testing.assert_allclose(8020, account.excess_liquidity,
                                   rtol=1e-3)
        np.testing.assert_allclose(8020, account.settled_cash, rtol=1e-3)
        np.testing.assert_allclose(10000, account.net_liquidation,
                                   rtol=1e-3)
        np.testing.assert_allclose(0.802, account.cushion, rtol=1e-3)
        np.testing.assert_allclose(1980, account.total_positions_value,
                                   rtol=1e-3)
        self.assertEqual(0, account.accrued_interest)

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


class TestDividendPerformance(WithSimParams,
                              WithInstanceTmpDir,
                              ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-10', tz='utc')
    ASSET_FINDER_EQUITY_SIDS = 1, 2
    SIM_PARAMS_CAPITAL_BASE = 10e3

    @classmethod
    def init_class_fixtures(cls):
        super(TestDividendPerformance, cls).init_class_fixtures()
        cls.asset1 = cls.asset_finder.retrieve_asset(1)
        cls.asset2 = cls.asset_finder.retrieve_asset(2)

    def test_market_hours_calculations(self):
        # DST in US/Eastern began on Sunday March 14, 2010
        before = datetime(2010, 3, 12, 14, 31, tzinfo=pytz.utc)
        after = factory.get_next_trading_dt(
            before,
            timedelta(days=1),
            self.trading_calendar,
        )
        self.assertEqual(after.hour, 13)

    def test_long_position_receives_dividend(self):
        # post some trades in the market
        events = factory.create_trade_history(
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()
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
        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader

        # Simulate a transaction being filled prior to the ex_date.
        txns = [create_txn(self.asset1, events[0].dt, 10.0, 100)]
        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
        for asset in [self.asset1, self.asset2]:
            events[asset.sid] = factory.create_trade_history(
                asset,
                [10, 10, 10, 10, 10, 10],
                [100, 100, 100, 100, 100, 100],
                oneday,
                self.sim_params,
                trading_calendar=self.trading_calendar,
            )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions
        )
        splits = mergers = create_empty_splits_mergers_frame()
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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            events,
        )

        data_portal._adjustment_reader = adjustment_reader
        txns = [create_txn(self.asset1, events[1][0].dt, 10.0, 100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar
        )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions
        )
        splits = mergers = create_empty_splits_mergers_frame()
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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader

        # Simulate a transaction being filled on the ex_date.
        txns = [create_txn(self.asset1, events[1].dt, 10.0, 100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()
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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader

        buy_txn = create_txn(self.asset1, events[0].dt, 10.0, 100)
        sell_txn = create_txn(self.asset1, events[2].dt, 10.0, -100)
        txns = [buy_txn, sell_txn]

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
        # need a six-day simparam

        # post some trades in the market
        events = factory.create_trade_history(
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )
        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()

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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader
        buy_txn = create_txn(self.asset1, events[1].dt, 10.0, 100)
        sell_txn = create_txn(self.asset1, events[2].dt, 10.0, -100)
        txns = [buy_txn, sell_txn]

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        pay_date = self.sim_params.first_open
        # find pay date that is much later.
        for i in range(30):
            pay_date = factory.get_next_trading_dt(pay_date, oneday,
                                                   self.trading_calendar)

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()
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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader
        txns = [create_txn(self.asset1, events[1].dt, 10.0, 100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
            txns=txns,
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
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()
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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader
        txns = [create_txn(self.asset1, events[1].dt, 10.0, -100)]

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()
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

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader

        results = calculate_results(
            self.sim_params,
            self.env,
            data_portal,
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
            self.asset1,
            [10, 10, 10, 10, 10, 10],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        dbpath = self.instance_tmpdir.getpath('adjustments.sqlite')

        writer = SQLiteAdjustmentWriter(
            dbpath,
            MockDailyBarReader(),
            self.trading_calendar.all_sessions,
        )
        splits = mergers = create_empty_splits_mergers_frame()
        dividends = pd.DataFrame({
            'sid': np.array([1], dtype=np.uint32),
            'amount': np.array([10.00], dtype=np.float64),
            'declared_date': np.array([events[-3].dt], dtype='datetime64[ns]'),
            'ex_date': np.array([events[-2].dt], dtype='datetime64[ns]'),
            'record_date': np.array([events[0].dt], dtype='datetime64[ns]'),
            'pay_date': np.array(
                [self.trading_calendar.next_session_label(
                    self.trading_calendar.minute_to_session_label(
                        events[-1].dt
                    )
                )],
                dtype='datetime64[ns]'),
        })
        writer.write(splits, mergers, dividends)
        adjustment_reader = SQLiteAdjustmentReader(dbpath)

        # Set the last day to be the last event
        sim_params = create_simulation_parameters(
            num_days=6,
            capital_base=10e3,
            start=self.sim_params.start_session,
            end=self.sim_params.end_session
        )

        sim_params = sim_params.create_new(
            sim_params.start_session,
            events[-1].dt
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            sim_params,
            {1: events},
        )
        data_portal._adjustment_reader = adjustment_reader
        # Simulate a transaction being filled prior to the ex_date.
        txns = [create_txn(self.asset1, events[0].dt, 10.0, 100)]
        results = calculate_results(
            sim_params,
            self.env,
            data_portal,
            txns=txns,
        )

        self.assertEqual(len(results), 6)
        cumulative_returns = \
            [event['cumulative_perf']['returns'] for event in results]
        self.assertEqual(cumulative_returns, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        daily_returns = [event['daily_perf']['returns'] for event in results]
        self.assertEqual(daily_returns, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cash_flows = [event['daily_perf']['capital_used'] for event in results]
        self.assertEqual(cash_flows, [-1000, 0, 0, 0, 0, 0])
        cumulative_cash_flows = \
            [event['cumulative_perf']['capital_used'] for event in results]
        self.assertEqual(cumulative_cash_flows,
                         [-1000, -1000, -1000, -1000, -1000, -1000])


class TestDividendPerformanceHolidayStyle(TestDividendPerformance):

    # The holiday tests begins the simulation on the day
    # before Thanksgiving, so that the next trading day is
    # two days ahead. Any tests that hard code events
    # to be start + oneday will fail, since those events will
    # be skipped by the simulation.
    START_DATE = pd.Timestamp('2003-11-30', tz='utc')
    END_DATE = pd.Timestamp('2003-12-08', tz='utc')


class TestPositionPerformance(WithInstanceTmpDir, WithTradingCalendars,
                              ZiplineTestCase):

    def create_environment_stuff(self,
                                 num_days=4,
                                 sids=[1, 2],
                                 futures_sids=[3]):
        start = pd.Timestamp('2006-01-01', tz='utc')
        end = start + timedelta(days=num_days * 2)
        equities = make_simple_equity_info(sids, start, end)
        futures = pd.DataFrame.from_dict(
            {
                sid: {
                    'start_date': start,
                    'end_date': end,
                    'multiplier': 100,
                    'exchange': "TEST",
                }
                for sid in futures_sids
            },
            orient='index',
        )
        self.env = self.enter_instance_context(tmp_trading_env(
            equities=equities,
            futures=futures,
        ))
        self.sim_params = create_simulation_parameters(
            start=start,
            num_days=num_days,
        )

        self.finder = self.env.asset_finder
        self.asset1 = self.env.asset_finder.retrieve_asset(1)
        self.asset2 = self.env.asset_finder.retrieve_asset(2)
        self.asset3 = self.env.asset_finder.retrieve_asset(3)

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
            self.asset1,
            [10, 10, 10, 9],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        trades_2 = factory.create_trade_history(
            self.asset2,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades_1, 2: trades_2}
        )

        txn1 = create_txn(self.asset1, trades_1[0].dt, 10.0, 100)
        txn2 = create_txn(self.asset2, trades_1[0].dt, 10.0, -100)

        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency)
        pp.position_tracker = pt
        pt.execute_transaction(txn1)
        pp.handle_execution(txn1)
        pt.execute_transaction(txn2)
        pp.handle_execution(txn2)

        dt = trades_1[-2].dt
        pt.sync_last_sale_prices(dt, False, data_portal)

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
                      total_positions_exposure=0.0,
                      regt_equity=1000.0,
                      available_funds=1000.0,
                      excess_liquidity=1000.0,
                      cushion=1.0,
                      leverage=2.0,
                      net_leverage=0.0,
                      net_liquidation=1000.0)

        dt = trades_1[-1].dt
        pt.sync_last_sale_prices(dt, False, data_portal)

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
                      total_positions_exposure=-200.0,
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
            self.asset1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})
        txn = create_txn(self.asset1, trades[1].dt, 10.0, 1000)
        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

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
        pt.sync_last_sale_prices(trades[-2].dt, False, data_portal)

        # Validate that the account attributes were updated.
        account = pp.as_account()
        check_account(account,
                      settled_cash=-9000.0,
                      equity_with_loan=1000.0,
                      total_positions_value=10000.0,
                      total_positions_exposure=10000.0,
                      regt_equity=-9000.0,
                      available_funds=-9000.0,
                      excess_liquidity=-9000.0,
                      cushion=-9.0,
                      leverage=10.0,
                      net_leverage=10.0,
                      net_liquidation=1000.0)

        # now simulate a price jump to $11
        pt.sync_last_sale_prices(trades[-1].dt, False, data_portal)

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
                      total_positions_exposure=11000.0,
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
            self.asset1,
            [10, 10, 10, 11],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})
        txn = create_txn(self.asset1, trades[1].dt, 10.0, 100)
        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency,
                                    period_open=self.sim_params.start_session,
                                    period_close=self.sim_params.end_session)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        # This verifies that the last sale price is being correctly
        # set in the positions. If this is not the case then returns can
        # incorrectly show as sharply dipping if a transaction arrives
        # before a trade. This is caused by returns being based on holding
        # stocks with a last sale price of 0.
        self.assertEqual(pp.positions[1].last_sale_price, 10.0)

        pt.sync_last_sale_prices(trades[-1].dt, False, data_portal)

        pp.calculate_performance()

        self.assertEqual(
            pp.cash_flow,
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
            trades[-1].price,
            "last sale should be same as last trade. \
            expected {exp} actual {act}".format(
                exp=trades[-1].price,
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
                      total_positions_exposure=1100.0,
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
            self.asset1,
            [10, 10, 10, 11, 10, 9],
            [100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        trades_1 = trades[:-2]

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})

        txn = create_txn(self.asset1, trades[1].dt, 10.0, -100)
        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(
            1000.0, self.env.asset_finder,
            self.sim_params.data_frequency)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        pt.sync_last_sale_prices(trades_1[-1].dt, False, data_portal)

        pp.calculate_performance()

        self.assertEqual(
            pp.cash_flow,
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
            trades_1[-1].price,
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

        pt.sync_last_sale_prices(trades[-1].dt, False, data_portal)

        pp.calculate_performance()

        self.assertEqual(
            pp.cash_flow,
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
        ptTotal = perf.PositionTracker(self.env.asset_finder,
                                       self.sim_params.data_frequency)
        ppTotal = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                         self.sim_params.data_frequency)
        ppTotal.position_tracker = pt

        ptTotal.execute_transaction(txn)
        ppTotal.handle_execution(txn)

        ptTotal.sync_last_sale_prices(trades[-1].dt, False, data_portal)

        ppTotal.calculate_performance()

        self.assertEqual(
            ppTotal.cash_flow,
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
                      total_positions_exposure=-900.0,
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
            self.asset1,
            [10, 10, 10, 11, 9, 8, 7, 8, 9, 10],
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})

        short_txn = create_txn(self.asset1, trades[1].dt, 10.0, -100)
        cover_txn = create_txn(self.asset1, trades[6].dt, 7.0, 100)
        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency)
        pp.position_tracker = pt

        pt.execute_transaction(short_txn)
        pp.handle_execution(short_txn)
        pt.execute_transaction(cover_txn)
        pp.handle_execution(cover_txn)

        pt.sync_last_sale_prices(trades[-1].dt, False, data_portal)

        pp.calculate_performance()

        short_txn_cost = short_txn.price * short_txn.amount
        cover_txn_cost = cover_txn.price * cover_txn.amount

        self.assertEqual(
            pp.cash_flow,
            -1 * short_txn_cost - cover_txn_cost,
            "capital used should be equal to the net transaction costs"
        )

        self.assertEqual(
            len(pp.positions),
            0,
            "should be zero positions"
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
                      total_positions_exposure=0.0,
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
            self.asset1,
            [10, 11, 11, 12, 10],
            [100, 100, 100, 100, 100],
            oneday,
            self.sim_params,
            self.trading_calendar,
        )
        trades = factory.create_trade_history(*history_args)
        transactions = factory.create_txn_history(*history_args)[:4]

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})

        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(
            1000.0,
            self.env.asset_finder,
            self.sim_params.data_frequency,
            period_open=self.sim_params.start_session,
            period_close=self.sim_params.sessions[-1]
        )
        pp.position_tracker = pt

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

        pt.sync_last_sale_prices(dt, False, data_portal)

        pp.calculate_performance()

        self.assertEqual(
            pp.pnl,
            400
        )

        down_tick = trades[-1]
        sale_txn = create_txn(self.asset1, down_tick.dt, 10.0, -100)
        pp.rollover()

        pt.execute_transaction(sale_txn)
        pp.handle_execution(sale_txn)

        dt = down_tick.dt
        pt.sync_last_sale_prices(dt, False, data_portal)

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

        pt3 = perf.PositionTracker(self.env.asset_finder,
                                   self.sim_params.data_frequency)
        pp3 = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                     self.sim_params.data_frequency)
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
        pt3.sync_last_sale_prices(trades[-1].dt, False, data_portal)

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
        self.create_environment_stuff(num_days=8)

        history_args = (
            1,
            [10, 9, 11, 8, 9, 12, 13, 14],
            [200, -100, -100, 100, -300, 100, 500, 400],
            oneday,
            self.sim_params,
            self.trading_calendar,
        )
        cost_bases = [10, 10, 0, 8, 9, 9, 13, 13.5]

        transactions = factory.create_txn_history(*history_args)

        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency)
        pp.position_tracker = pt

        for idx, (txn, cb) in enumerate(zip(transactions, cost_bases)):
            pt.execute_transaction(txn)
            pp.handle_execution(txn)

            if idx == 2:
                # buy 200, sell 100, sell 100 = 0 shares = no position
                self.assertNotIn(1, pp.positions)
            else:
                self.assertEqual(pp.positions[1].cost_basis, cb)

        pp.calculate_performance()

        self.assertEqual(pp.positions[1].cost_basis, cost_bases[-1])

    def test_capital_change_intra_period(self):
        self.create_environment_stuff()

        # post some trades in the market
        trades = factory.create_trade_history(
            self.asset1,
            [10.0, 11.0, 12.0, 13.0],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})
        txn = create_txn(self.asset1, trades[0].dt, 10.0, 100)
        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency,
                                    period_open=self.sim_params.start_session,
                                    period_close=self.sim_params.end_session)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)

        # sync prices before we introduce a capital change
        pt.sync_last_sale_prices(trades[2].dt, False, data_portal)

        pp.initialize_subperiod_divider()
        pp.set_current_subperiod_starting_values(1000.0)

        pt.sync_last_sale_prices(trades[-1].dt, False, data_portal)
        pp.calculate_performance()

        self.assertAlmostEqual(pp.returns, 1200/1000 * 2300/2200 - 1)
        self.assertAlmostEqual(pp.pnl, 300)
        self.assertAlmostEqual(pp.cash_flow, -1000)

    def test_capital_change_inter_period(self):
        self.create_environment_stuff()

        # post some trades in the market
        trades = factory.create_trade_history(
            self.asset1,
            [10.0, 11.0, 12.0, 13.0],
            [100, 100, 100, 100],
            oneday,
            self.sim_params,
            trading_calendar=self.trading_calendar,
        )

        data_portal = create_data_portal_from_trade_history(
            self.env.asset_finder,
            self.trading_calendar,
            self.instance_tmpdir,
            self.sim_params,
            {1: trades})
        txn = create_txn(self.asset1, trades[0].dt, 10.0, 100)
        pt = perf.PositionTracker(self.env.asset_finder,
                                  self.sim_params.data_frequency)
        pp = perf.PerformancePeriod(1000.0, self.env.asset_finder,
                                    self.sim_params.data_frequency,
                                    period_open=self.sim_params.start_session,
                                    period_close=self.sim_params.end_session)
        pp.position_tracker = pt

        pt.execute_transaction(txn)
        pp.handle_execution(txn)
        pt.sync_last_sale_prices(trades[0].dt, False, data_portal)
        pp.calculate_performance()
        self.assertAlmostEqual(pp.returns, 0)
        self.assertAlmostEqual(pp.pnl, 0)
        self.assertAlmostEqual(pp.cash_flow, -1000)
        pp.rollover()

        pt.sync_last_sale_prices(trades[1].dt, False, data_portal)
        pp.calculate_performance()
        self.assertAlmostEqual(pp.returns, 1100.0/1000.0 - 1)
        self.assertAlmostEqual(pp.pnl, 100)
        self.assertAlmostEqual(pp.cash_flow, 0)
        pp.rollover()

        pp.adjust_period_starting_capital(1000)
        pt.sync_last_sale_prices(trades[2].dt, False, data_portal)
        pp.calculate_performance()
        self.assertAlmostEqual(pp.returns, 2200.0/2100.0 - 1)
        self.assertAlmostEqual(pp.pnl, 100)
        self.assertAlmostEqual(pp.cash_flow, 0)
        pp.rollover()

        pt.sync_last_sale_prices(trades[3].dt, False, data_portal)
        pp.calculate_performance()
        self.assertAlmostEqual(pp.returns, 2300.0/2200.0 - 1)
        self.assertAlmostEqual(pp.pnl, 100)
        self.assertAlmostEqual(pp.cash_flow, 0)


class TestPositionTracker(WithTradingEnvironment,
                          WithInstanceTmpDir,
                          ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = 1, 2

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                3: {'multiplier': 1000, 'exchange': 'TEST'},
                4: {'multiplier': 1000, 'exchange': 'TEST'},
                1032201401: {'multiplier': 50, 'exchange': 'TEST'},
            },
            orient='index',
        )

    def test_empty_positions(self):
        """
        make sure all the empty position stats return a numeric 0

        Originally this bug was due to np.dot([], []) returning
        np.bool_(False)
        """
        sim_params = factory.create_simulation_parameters(num_days=4)

        pt = perf.PositionTracker(self.env.asset_finder,
                                  sim_params.data_frequency)
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

    def test_update_positions(self):
        pt = perf.PositionTracker(self.env.asset_finder, None)
        dt = pd.Timestamp("2014/01/01 3:00PM")
        pos1 = perf.Position(1, amount=np.float64(10.0),
                             last_sale_date=dt, last_sale_price=10)
        pos2 = perf.Position(2, amount=np.float64(-20.0),
                             last_sale_date=dt, last_sale_price=10)
        pos3 = perf.Position(1032201401, amount=np.float64(30.0),
                             last_sale_date=dt, last_sale_price=100)

        # Call update_positions twice. When the second call is made,
        # self.positions will already contain data. The order of this data
        # needs to be preserved so that it is consistent with the order of the
        # data stored in the multipliers OrderedDict()'s. If self.positions
        # were to be stored as a dict, then its order could change in arbitrary
        # ways when the second update_positions call is made. Hence we also
        # store it as an OrderedDict.
        pt.update_positions({1: pos1, 1032201401: pos3})
        pt.update_positions({2: pos2})

        pos_stats = pt.stats()
        # Test long-only methods
        self.assertEqual(100, pos_stats.long_value)
        # 150,000 = 30 * 100 * 50 (amount * last_sale_price * multiplier)
        self.assertEqual(100 + 150000, pos_stats.long_exposure)
        self.assertEqual(2, pos_stats.longs_count)

        # Test short-only methods
        self.assertEqual(-200, pos_stats.short_value)
        self.assertEqual(-200, pos_stats.short_exposure)
        self.assertEqual(1, pos_stats.shorts_count)

        # Test gross and net values
        self.assertEqual(100 + 200, pos_stats.gross_value)
        self.assertEqual(100 - 200, pos_stats.net_value)

        # Test gross and net exposures
        self.assertEqual(100 + 150000 + 200, pos_stats.gross_exposure)
        self.assertEqual(100 + 150000 - 200, pos_stats.net_exposure)

    def test_close_position(self):
        future_sid = 1032201401
        equity_sid = 1
        pt = perf.PositionTracker(self.env.asset_finder, None)
        dt = pd.Timestamp('2017/01/04 3:00PM')

        pos1 = perf.Position(
            sid=future_sid,
            amount=np.float64(30.0),
            last_sale_date=dt,
            last_sale_price=100,
        )
        pos2 = perf.Position(
            sid=equity_sid,
            amount=np.float64(10.0),
            last_sale_date=dt,
            last_sale_price=10,
        )

        # Update the positions dictionary with `future_sid` first. The order
        # matters because it affects the multipliers dictionaries, which are
        # OrderedDicts. If `future_sid` is not removed from the multipliers
        # dictionaries, equities will hit the incorrect multiplier when
        # computing `pt.stats()`.
        pt.update_positions({future_sid: pos1, equity_sid: pos2})

        asset_to_close = self.env.asset_finder.retrieve_asset(future_sid)
        txn = create_txn(asset_to_close, dt, 100, -30)
        pt.execute_transaction(txn)

        pos_stats = pt.stats()

        # Test long-only methods.
        self.assertEqual(100, pos_stats.long_value)
        self.assertEqual(100, pos_stats.long_exposure)
        self.assertEqual(1, pos_stats.longs_count)

        # Test short-only methods.
        self.assertEqual(0, pos_stats.short_value)
        self.assertEqual(0, pos_stats.short_exposure)
        self.assertEqual(0, pos_stats.shorts_count)

        # Test gross and net values.
        self.assertEqual(100, pos_stats.gross_value)
        self.assertEqual(100, pos_stats.net_value)

        # Test gross and net exposures.
        self.assertEqual(100, pos_stats.gross_exposure)
        self.assertEqual(100, pos_stats.net_exposure)
