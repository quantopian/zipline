#
# Copyright 2014 Quantopian, Inc.
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
import warnings
from collections import namedtuple
import datetime
from datetime import timedelta
from textwrap import dedent
from unittest import skip
from copy import deepcopy

import logbook
import toolz
from logbook import TestHandler, WARNING
from mock import MagicMock
from nose_parameterized import parameterized
from six import iteritems, itervalues, string_types
from six.moves import range
from testfixtures import TempDirectory

import numpy as np
import pandas as pd
import pytz
from pandas.core.common import PerformanceWarning

from zipline import run_algorithm
from zipline import TradingAlgorithm
from zipline.api import FixedSlippage
from zipline.assets import Equity, Future, Asset
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.assets.synthetic import (
    make_jagged_equity_info,
    make_simple_equity_info,
)
from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import (
    BcolzMinuteBarReader,
    BcolzMinuteBarWriter,
    US_EQUITIES_MINUTES_PER_DAY,
)
from zipline.data.us_equity_pricing import (
    BcolzDailyBarReader,
    BcolzDailyBarWriter,
)
from zipline.errors import (
    OrderDuringInitialize,
    RegisterTradingControlPostInit,
    TradingControlViolation,
    AccountControlViolation,
    SymbolNotFound,
    UnsupportedDatetimeFormat,
    CannotOrderDelistedAsset,
    SetCancelPolicyPostInit,
    UnsupportedCancelPolicy,
    OrderInBeforeTradingStart)
from zipline.api import (
    order,
    order_value,
    order_percent,
    order_target,
    order_target_value,
    order_target_percent
)

from zipline.finance.commission import PerShare
from zipline.finance.execution import LimitOrder
from zipline.finance.order import ORDER_STATUS
from zipline.finance.trading import SimulationParameters
from zipline.finance.asset_restrictions import (
    Restriction,
    HistoricalRestrictions,
    StaticRestrictions,
    RESTRICTION_STATES,
)
from zipline.testing import (
    FakeDataPortal,
    create_daily_df_for_asset,
    create_data_portal,
    create_data_portal_from_trade_history,
    create_minute_df_for_asset,
    empty_trading_env,
    make_test_handler,
    make_trade_data_for_asset_info,
    parameter_space,
    str_to_seconds,
    tmp_trading_env,
    to_utc,
    trades_by_sid_to_dfs,
)
from zipline.testing import RecordBatchBlotter
from zipline.testing.fixtures import (
    WithDataPortal,
    WithLogger,
    WithSimParams,
    WithTradingEnvironment,
    WithTmpDir,
    WithTradingCalendars,
    ZiplineTestCase,
)
from zipline.test_algorithms import (
    access_account_in_init,
    access_portfolio_in_init,
    AmbitiousStopLimitAlgorithm,
    EmptyPositionsAlgorithm,
    InvalidOrderAlgorithm,
    RecordAlgorithm,
    FutureFlipAlgo,
    TestOrderAlgorithm,
    TestOrderPercentAlgorithm,
    TestOrderStyleForwardingAlgorithm,
    TestOrderValueAlgorithm,
    TestRegisterTransformAlgorithm,
    TestTargetAlgorithm,
    TestTargetPercentAlgorithm,
    TestTargetValueAlgorithm,
    TestBatchTargetPercentAlgorithm,
    SetLongOnlyAlgorithm,
    SetAssetDateBoundsAlgorithm,
    SetMaxPositionSizeAlgorithm,
    SetMaxOrderCountAlgorithm,
    SetMaxOrderSizeAlgorithm,
    SetDoNotOrderListAlgorithm,
    SetAssetRestrictionsAlgorithm,
    SetMultipleAssetRestrictionsAlgorithm,
    SetMaxLeverageAlgorithm,
    api_algo,
    api_get_environment_algo,
    api_symbol_algo,
    call_all_order_methods,
    call_order_in_init,
    handle_data_api,
    handle_data_noop,
    initialize_api,
    initialize_noop,
    noop_algo,
    record_float_magic,
    record_variables,
    call_with_kwargs,
    call_without_kwargs,
    call_with_bad_kwargs_current,
    call_with_bad_kwargs_history,
    bad_type_history_assets,
    bad_type_history_fields,
    bad_type_history_bar_count,
    bad_type_history_frequency,
    bad_type_history_assets_kwarg_list,
    bad_type_current_assets,
    bad_type_current_fields,
    bad_type_can_trade_assets,
    bad_type_is_stale_assets,
    bad_type_history_assets_kwarg,
    bad_type_history_fields_kwarg,
    bad_type_history_bar_count_kwarg,
    bad_type_history_frequency_kwarg,
    bad_type_current_assets_kwarg,
    bad_type_current_fields_kwarg,
    call_with_bad_kwargs_get_open_orders,
    call_with_good_kwargs_get_open_orders,
    call_with_no_kwargs_get_open_orders,
    empty_positions,
    set_benchmark_algo,
    no_handle_data,
)
from zipline.testing.predicates import assert_equal
from zipline.utils.api_support import ZiplineAPI, set_algo_instance
from zipline.utils.calendars import get_calendar, register_calendar
from zipline.utils.context_tricks import CallbackManager
from zipline.utils.control_flow import nullctx
import zipline.utils.events
from zipline.utils.events import date_rules, time_rules, Always
import zipline.utils.factory as factory

# Because test cases appear to reuse some resources.


_multiprocess_can_split_ = False


class TestRecordAlgorithm(WithSimParams, WithDataPortal, ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = 133,

    def test_record_incr(self):
        algo = RecordAlgorithm(sim_params=self.sim_params, env=self.env)
        output = algo.run(self.data_portal)

        np.testing.assert_array_equal(output['incr'].values,
                                      range(1, len(output) + 1))
        np.testing.assert_array_equal(output['name'].values,
                                      range(1, len(output) + 1))
        np.testing.assert_array_equal(output['name2'].values,
                                      [2] * len(output))
        np.testing.assert_array_equal(output['name3'].values,
                                      range(1, len(output) + 1))


class TestMiscellaneousAPI(WithLogger,
                           WithSimParams,
                           WithDataPortal,
                           ZiplineTestCase):

    START_DATE = pd.Timestamp('2006-01-03', tz='UTC')
    END_DATE = pd.Timestamp('2006-01-04', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    sids = 1, 2

    @classmethod
    def make_equity_info(cls):
        return pd.concat((
            make_simple_equity_info(cls.sids, '2002-02-1', '2007-01-01'),
            pd.DataFrame.from_dict(
                {3: {'symbol': 'PLAY',
                     'start_date': '2002-01-01',
                     'end_date': '2004-01-01',
                     'exchange': 'TEST'},
                 4: {'symbol': 'PLAY',
                     'start_date': '2005-01-01',
                     'end_date': '2006-01-01',
                     'exchange': 'TEST'}},
                orient='index',
            ),
        ))

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                5: {
                    'symbol': 'CLG06',
                    'root_symbol': 'CL',
                    'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                    'notice_date': pd.Timestamp('2005-12-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2006-01-20', tz='UTC'),
                    'exchange': 'TEST'
                },
                6: {
                    'root_symbol': 'CL',
                    'symbol': 'CLK06',
                    'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                    'notice_date': pd.Timestamp('2006-03-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2006-04-20', tz='UTC'),
                    'exchange': 'TEST',
                },
                7: {
                    'symbol': 'CLQ06',
                    'root_symbol': 'CL',
                    'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                    'notice_date': pd.Timestamp('2006-06-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2006-07-20', tz='UTC'),
                    'exchange': 'TEST',
                },
                8: {
                    'symbol': 'CLX06',
                    'root_symbol': 'CL',
                    'start_date': pd.Timestamp('2006-02-01', tz='UTC'),
                    'notice_date': pd.Timestamp('2006-09-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2006-10-20', tz='UTC'),
                    'exchange': 'TEST',
                }
            },
            orient='index',
        )

    def test_cancel_policy_outside_init(self):
        code = """
from zipline.api import cancel_policy, set_cancel_policy

def initialize(algo):
    pass

def handle_data(algo, data):
    set_cancel_policy(cancel_policy.NeverCancel())
"""

        algo = TradingAlgorithm(script=code,
                                sim_params=self.sim_params,
                                env=self.env)

        with self.assertRaises(SetCancelPolicyPostInit):
            algo.run(self.data_portal)

    def test_cancel_policy_invalid_param(self):
        code = """
from zipline.api import set_cancel_policy

def initialize(algo):
    set_cancel_policy("foo")

def handle_data(algo, data):
    pass
"""
        algo = TradingAlgorithm(script=code,
                                sim_params=self.sim_params,
                                env=self.env)

        with self.assertRaises(UnsupportedCancelPolicy):
            algo.run(self.data_portal)

    def test_zipline_api_resolves_dynamically(self):
        # Make a dummy algo.
        algo = TradingAlgorithm(
            initialize=lambda context: None,
            handle_data=lambda context, data: None,
            sim_params=self.sim_params,
        )

        # Verify that api methods get resolved dynamically by patching them out
        # and then calling them
        for method in algo.all_api_methods():
            name = method.__name__
            sentinel = object()

            def fake_method(*args, **kwargs):
                return sentinel
            setattr(algo, name, fake_method)
            with ZiplineAPI(algo):
                self.assertIs(sentinel, getattr(zipline.api, name)())

    def test_sid_datetime(self):
        algo_text = """
from zipline.api import sid, get_datetime

def initialize(context):
    pass

def handle_data(context, data):
    aapl_dt = data.current(sid(1), "last_traded")
    assert_equal(aapl_dt, get_datetime())
"""
        algo = TradingAlgorithm(script=algo_text,
                                sim_params=self.sim_params,
                                env=self.env)
        algo.namespace['assert_equal'] = self.assertEqual
        algo.run(self.data_portal)

    def test_datetime_bad_params(self):
        algo_text = """
from zipline.api import get_datetime
from pytz import timezone

def initialize(context):
    pass

def handle_data(context, data):
    get_datetime(timezone)
"""
        with self.assertRaises(TypeError):
            algo = TradingAlgorithm(script=algo_text,
                                    sim_params=self.sim_params,
                                    env=self.env)
            algo.run(self.data_portal)

    def test_get_environment(self):
        expected_env = {
            'arena': 'backtest',
            'data_frequency': 'minute',
            'start': pd.Timestamp('2006-01-03 14:31:00+0000', tz='utc'),
            'end': pd.Timestamp('2006-01-04 21:00:00+0000', tz='utc'),
            'capital_base': 100000.0,
            'platform': 'zipline'
        }

        def initialize(algo):
            self.assertEqual('zipline', algo.get_environment())
            self.assertEqual(expected_env, algo.get_environment('*'))

        def handle_data(algo, data):
            pass

        algo = TradingAlgorithm(initialize=initialize,
                                handle_data=handle_data,
                                sim_params=self.sim_params,
                                env=self.env)
        algo.run(self.data_portal)

    def test_get_open_orders(self):
        def initialize(algo):
            algo.minute = 0

        def handle_data(algo, data):
            if algo.minute == 0:

                # Should be filled by the next minute
                algo.order(algo.sid(1), 1)

                # Won't be filled because the price is too low.
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01))
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01))
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01))

                all_orders = algo.get_open_orders()
                self.assertEqual(list(all_orders.keys()), [1, 2])

                self.assertEqual(all_orders[1], algo.get_open_orders(1))
                self.assertEqual(len(all_orders[1]), 1)

                self.assertEqual(all_orders[2], algo.get_open_orders(2))
                self.assertEqual(len(all_orders[2]), 3)

            if algo.minute == 1:
                # First order should have filled.
                # Second order should still be open.
                all_orders = algo.get_open_orders()
                self.assertEqual(list(all_orders.keys()), [2])

                self.assertEqual([], algo.get_open_orders(1))

                orders_2 = algo.get_open_orders(2)
                self.assertEqual(all_orders[2], orders_2)
                self.assertEqual(len(all_orders[2]), 3)

                for order_ in orders_2:
                    algo.cancel_order(order_)

                all_orders = algo.get_open_orders()
                self.assertEqual(all_orders, {})

            algo.minute += 1

        algo = TradingAlgorithm(initialize=initialize,
                                handle_data=handle_data,
                                sim_params=self.sim_params,
                                env=self.env)
        algo.run(self.data_portal)

    def test_schedule_function_custom_cal(self):
        # run a simulation on the CME cal, and schedule a function
        # using the NYSE cal
        algotext = """
from zipline.api import schedule_function, get_datetime, time_rules, date_rules
from zipline.utils.calendars import get_calendar

def initialize(context):
    schedule_function(
        func=log_nyse_open,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_open(),
        calendar=get_calendar("NYSE")
    )

    schedule_function(
        func=log_nyse_close,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_close(),
        calendar=get_calendar("NYSE")
    )

    context.nyse_opens = []
    context.nyse_closes = []

def log_nyse_open(context, data):
    context.nyse_opens.append(get_datetime())

def log_nyse_close(context, data):
    context.nyse_closes.append(get_datetime())
        """

        algo = TradingAlgorithm(
            script=algotext,
            sim_params=self.sim_params,
            env=self.env,
            trading_calendar=get_calendar("CME")
        )

        algo.run(self.data_portal)

        nyse = get_calendar("NYSE")

        for minute in algo.nyse_opens:
            # each minute should be a nyse session open
            session_label = nyse.minute_to_session_label(minute)
            session_open = nyse.open_and_close_for_session(session_label)[0]
            self.assertEqual(session_open, minute)

        for minute in algo.nyse_closes:
            # each minute should be a minute before a nyse session close
            session_label = nyse.minute_to_session_label(minute)
            session_close = nyse.open_and_close_for_session(session_label)[1]
            self.assertEqual(session_close - timedelta(minutes=1), minute)

    def test_schedule_function(self):
        us_eastern = pytz.timezone('US/Eastern')

        def incrementer(algo, data):
            algo.func_called += 1
            curdt = algo.get_datetime().tz_convert(pytz.utc)
            self.assertEqual(
                curdt,
                us_eastern.localize(
                    datetime.datetime.combine(
                        curdt.date(),
                        datetime.time(9, 31)
                    ),
                ),
            )

        def initialize(algo):
            algo.func_called = 0
            algo.days = 1
            algo.date = None
            algo.schedule_function(
                func=incrementer,
                date_rule=date_rules.every_day(),
                time_rule=time_rules.market_open(),
            )

        def handle_data(algo, data):
            if not algo.date:
                algo.date = algo.get_datetime().date()

            if algo.date < algo.get_datetime().date():
                algo.days += 1
                algo.date = algo.get_datetime().date()

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            sim_params=self.sim_params,
            env=self.env,
        )
        algo.run(self.data_portal)

        self.assertEqual(algo.func_called, algo.days)

    def test_event_context(self):
        expected_data = []
        collected_data_pre = []
        collected_data_post = []
        function_stack = []

        def pre(data):
            function_stack.append(pre)
            collected_data_pre.append(data)

        def post(data):
            function_stack.append(post)
            collected_data_post.append(data)

        def initialize(context):
            context.add_event(Always(), f)
            context.add_event(Always(), g)

        def handle_data(context, data):
            function_stack.append(handle_data)
            expected_data.append(data)

        def f(context, data):
            function_stack.append(f)

        def g(context, data):
            function_stack.append(g)

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            sim_params=self.sim_params,
            create_event_context=CallbackManager(pre, post),
            env=self.env,
        )
        algo.run(self.data_portal)

        self.assertEqual(len(expected_data), 780)
        self.assertEqual(collected_data_pre, expected_data)
        self.assertEqual(collected_data_post, expected_data)

        self.assertEqual(
            len(function_stack),
            3900,
            'Incorrect number of functions called: %s != 3900' %
            len(function_stack),
        )
        expected_functions = [pre, handle_data, f, g, post] * 97530
        for n, (f, g) in enumerate(zip(function_stack, expected_functions)):
            self.assertEqual(
                f,
                g,
                'function at position %d was incorrect, expected %s but got %s'
                % (n, g.__name__, f.__name__),
            )

    @parameterized.expand([
        ('daily',),
        ('minute'),
    ])
    def test_schedule_function_rule_creation(self, mode):
        def nop(*args, **kwargs):
            return None

        self.sim_params.data_frequency = mode
        algo = TradingAlgorithm(
            initialize=nop,
            handle_data=nop,
            sim_params=self.sim_params,
            env=self.env,
        )

        # Schedule something for NOT Always.
        algo.schedule_function(nop, time_rule=zipline.utils.events.Never())

        event_rule = algo.event_manager._events[1].rule

        self.assertIsInstance(event_rule, zipline.utils.events.OncePerDay)

        inner_rule = event_rule.rule
        self.assertIsInstance(inner_rule, zipline.utils.events.ComposedRule)

        first = inner_rule.first
        second = inner_rule.second
        composer = inner_rule.composer

        self.assertIsInstance(first, zipline.utils.events.Always)

        if mode == 'daily':
            self.assertIsInstance(second, zipline.utils.events.Always)
        else:
            self.assertIsInstance(second, zipline.utils.events.Never)

        self.assertIs(composer, zipline.utils.events.ComposedRule.lazy_and)

    def test_asset_lookup(self):
        algo = TradingAlgorithm(env=self.env)

        # this date doesn't matter
        start_session = pd.Timestamp("2000-01-01", tz="UTC")

        # Test before either PLAY existed
        algo.sim_params = algo.sim_params.create_new(
            start_session,
            pd.Timestamp('2001-12-01', tz='UTC')
        )
        with self.assertRaises(SymbolNotFound):
            algo.symbol('PLAY')
        with self.assertRaises(SymbolNotFound):
            algo.symbols('PLAY')

        # Test when first PLAY exists
        algo.sim_params = algo.sim_params.create_new(
            start_session,
            pd.Timestamp('2002-12-01', tz='UTC')
        )
        list_result = algo.symbols('PLAY')
        self.assertEqual(3, list_result[0])

        # Test after first PLAY ends
        algo.sim_params = algo.sim_params.create_new(
            start_session,
            pd.Timestamp('2004-12-01', tz='UTC')
        )
        self.assertEqual(3, algo.symbol('PLAY'))

        # Test after second PLAY begins
        algo.sim_params = algo.sim_params.create_new(
            start_session,
            pd.Timestamp('2005-12-01', tz='UTC')
        )
        self.assertEqual(4, algo.symbol('PLAY'))

        # Test after second PLAY ends
        algo.sim_params = algo.sim_params.create_new(
            start_session,
            pd.Timestamp('2006-12-01', tz='UTC')
        )
        self.assertEqual(4, algo.symbol('PLAY'))
        list_result = algo.symbols('PLAY')
        self.assertEqual(4, list_result[0])

        # Test lookup SID
        self.assertIsInstance(algo.sid(3), Equity)
        self.assertIsInstance(algo.sid(4), Equity)

        # Supplying a non-string argument to symbol()
        # should result in a TypeError.
        with self.assertRaises(TypeError):
            algo.symbol(1)

        with self.assertRaises(TypeError):
            algo.symbol((1,))

        with self.assertRaises(TypeError):
            algo.symbol({1})

        with self.assertRaises(TypeError):
            algo.symbol([1])

        with self.assertRaises(TypeError):
            algo.symbol({'foo': 'bar'})

    def test_future_symbol(self):
        """ Tests the future_symbol API function.
        """
        algo = TradingAlgorithm(env=self.env)
        algo.datetime = pd.Timestamp('2006-12-01', tz='UTC')

        # Check that we get the correct fields for the CLG06 symbol
        cl = algo.future_symbol('CLG06')
        self.assertEqual(cl.sid, 5)
        self.assertEqual(cl.symbol, 'CLG06')
        self.assertEqual(cl.root_symbol, 'CL')
        self.assertEqual(cl.start_date, pd.Timestamp('2005-12-01', tz='UTC'))
        self.assertEqual(cl.notice_date, pd.Timestamp('2005-12-20', tz='UTC'))
        self.assertEqual(cl.expiration_date,
                         pd.Timestamp('2006-01-20', tz='UTC'))

        with self.assertRaises(SymbolNotFound):
            algo.future_symbol('')

        with self.assertRaises(SymbolNotFound):
            algo.future_symbol('PLAY')

        with self.assertRaises(SymbolNotFound):
            algo.future_symbol('FOOBAR')

        # Supplying a non-string argument to future_symbol()
        # should result in a TypeError.
        with self.assertRaises(TypeError):
            algo.future_symbol(1)

        with self.assertRaises(TypeError):
            algo.future_symbol((1,))

        with self.assertRaises(TypeError):
            algo.future_symbol({1})

        with self.assertRaises(TypeError):
            algo.future_symbol([1])

        with self.assertRaises(TypeError):
            algo.future_symbol({'foo': 'bar'})

    def test_set_symbol_lookup_date(self):
        """
        Test the set_symbol_lookup_date API method.
        """
        # Note we start sid enumeration at i+3 so as not to
        # collide with sids [1, 2] added in the setUp() method.
        dates = pd.date_range('2013-01-01', freq='2D', periods=2, tz='UTC')
        # Create two assets with the same symbol but different
        # non-overlapping date ranges.
        metadata = pd.DataFrame.from_records(
            [
                {
                    'sid': i + 3,
                    'symbol': 'DUP',
                    'start_date': date.value,
                    'end_date': (date + timedelta(days=1)).value,
                    'exchange': 'TEST',
                }
                for i, date in enumerate(dates)
            ]
        )
        with tmp_trading_env(equities=metadata) as env:
            algo = TradingAlgorithm(env=env)

            # Set the period end to a date after the period end
            # dates for our assets.
            algo.sim_params = algo.sim_params.create_new(
                algo.sim_params.start_session,
                pd.Timestamp('2015-01-01', tz='UTC')
            )

            # With no symbol lookup date set, we will use the period end date
            # for the as_of_date, resulting here in the asset with the earlier
            # start date being returned.
            result = algo.symbol('DUP')
            self.assertEqual(result.symbol, 'DUP')

            # By first calling set_symbol_lookup_date, the relevant asset
            # should be returned by lookup_symbol
            for i, date in enumerate(dates):
                algo.set_symbol_lookup_date(date)
                result = algo.symbol('DUP')
                self.assertEqual(result.symbol, 'DUP')
                self.assertEqual(result.sid, i + 3)

            with self.assertRaises(UnsupportedDatetimeFormat):
                algo.set_symbol_lookup_date('foobar')


class TestTransformAlgorithm(WithLogger,
                             WithDataPortal,
                             WithSimParams,
                             ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')

    sids = ASSET_FINDER_EQUITY_SIDS = [0, 1, 133]

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict({
            3: {
                'multiplier': 10,
                'symbol': 'F',
                'exchange': 'TEST'
            }
        }, orient='index')

    @classmethod
    def make_equity_daily_bar_data(cls):
        return trades_by_sid_to_dfs(
            {
                sid: factory.create_trade_history(
                    sid,
                    [10.0, 10.0, 11.0, 11.0],
                    [100, 100, 100, 300],
                    timedelta(days=1),
                    cls.sim_params,
                    cls.trading_calendar,
                ) for sid in cls.sids
            },
            index=cls.sim_params.sessions,
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestTransformAlgorithm, cls).init_class_fixtures()
        cls.futures_env = cls.enter_class_context(
            tmp_trading_env(futures=cls.make_futures_info()),
        )

    def test_invalid_order_parameters(self):
        algo = InvalidOrderAlgorithm(
            sids=[133],
            sim_params=self.sim_params,
            env=self.env,
        )
        algo.run(self.data_portal)

    @parameterized.expand([
        (order, 1),
        (order_value, 1000),
        (order_target, 1),
        (order_target_value, 1000),
        (order_percent, 1),
        (order_target_percent, 1),
    ])
    def test_cannot_order_in_before_trading_start(self, order_method, amount):
        algotext = """
from zipline.api import sid
from zipline.api import {order_func}

def initialize(context):
     context.asset = sid(133)

def before_trading_start(context, data):
     {order_func}(context.asset, {arg})
     """.format(order_func=order_method.__name__, arg=amount)

        algo = TradingAlgorithm(script=algotext, sim_params=self.sim_params,
                                data_frequency='daily', env=self.env)

        with self.assertRaises(OrderInBeforeTradingStart):
            algo.run(self.data_portal)

    def test_run_twice(self):
        algo1 = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[0, 1]
        )

        res1 = algo1.run(self.data_portal)

        # Create a new trading algorithm, which will
        # use the newly instantiated environment.
        algo2 = TestRegisterTransformAlgorithm(
            sim_params=self.sim_params,
            sids=[0, 1]
        )

        res2 = algo2.run(self.data_portal)

        # There are some np.NaN values in the first row because there is not
        # enough data to calculate the metric, e.g. beta.
        res1 = res1.fillna(value=0)
        res2 = res2.fillna(value=0)

        np.testing.assert_array_equal(res1, res2)

    def test_data_frequency_setting(self):
        self.sim_params.data_frequency = 'daily'

        sim_params = factory.create_simulation_parameters(
            num_days=4, data_frequency='daily')

        algo = TestRegisterTransformAlgorithm(
            sim_params=sim_params,
            env=self.env,
        )
        self.assertEqual(algo.sim_params.data_frequency, 'daily')

        sim_params = factory.create_simulation_parameters(
            num_days=4, data_frequency='minute')

        algo = TestRegisterTransformAlgorithm(
            sim_params=sim_params,
            env=self.env,
        )
        self.assertEqual(algo.sim_params.data_frequency, 'minute')

    @parameterized.expand([
        ('order', TestOrderAlgorithm,),
        ('order_value', TestOrderValueAlgorithm,),
        ('order_target', TestTargetAlgorithm,),
        ('order_percent', TestOrderPercentAlgorithm,),
        ('order_target_percent', TestTargetPercentAlgorithm,),
        ('order_target_value', TestTargetValueAlgorithm,),
        ('batch_order_target_percent', TestBatchTargetPercentAlgorithm,),
    ])
    def test_order_methods(self, test_name, algo_class):
        algo = algo_class(
            sim_params=self.sim_params,
            env=self.env,
        )
        # Ensure that the environment's asset 0 is an Equity
        asset_to_test = algo.sid(0)
        self.assertIsInstance(asset_to_test, Equity)

        algo.run(self.data_portal)

    @parameterized.expand([
        (TestOrderAlgorithm,),
        (TestOrderValueAlgorithm,),
        (TestTargetAlgorithm,),
        (TestOrderPercentAlgorithm,),
        (TestTargetValueAlgorithm,),
    ])
    def test_order_methods_for_future(self, algo_class):
        algo = algo_class(
            sim_params=self.sim_params,
            env=self.env,
        )
        # Ensure that the environment's asset 3 is a Future
        asset_to_test = algo.sid(3)
        self.assertIsInstance(asset_to_test, Future)

        algo.run(self.data_portal)

    @parameterized.expand([
        ("order",),
        ("order_value",),
        ("order_percent",),
        ("order_target",),
        ("order_target_percent",),
        ("order_target_value",),
    ])
    def test_order_method_style_forwarding(self, order_style):
        algo = TestOrderStyleForwardingAlgorithm(
            sim_params=self.sim_params,
            method_name=order_style,
            env=self.env
        )
        algo.run(self.data_portal)

    def test_order_on_each_day_of_asset_lifetime(self):
        algo_code = dedent("""
        from zipline.api import sid, schedule_function, date_rules, order
        def initialize(context):
            schedule_function(order_it, date_rule=date_rules.every_day())

        def order_it(context, data):
            order(sid(133), 1)

        def handle_data(context, data):
            pass
        """)

        asset133 = self.env.asset_finder.retrieve_asset(133)

        sim_params = SimulationParameters(
            start_session=asset133.start_date,
            end_session=asset133.end_date,
            data_frequency="minute",
            trading_calendar=self.trading_calendar
        )

        algo = TradingAlgorithm(
            script=algo_code,
            sim_params=sim_params,
            env=self.env
        )

        results = algo.run(FakeDataPortal(self.env))

        for orders_for_day in results.orders:
            self.assertEqual(1, len(orders_for_day))
            self.assertEqual(orders_for_day[0]["status"], ORDER_STATUS.FILLED)

        for txns_for_day in results.transactions:
            self.assertEqual(1, len(txns_for_day))
            self.assertEqual(1, txns_for_day[0]["amount"])

    @parameterized.expand([
        (TestOrderAlgorithm,),
        (TestOrderValueAlgorithm,),
        (TestTargetAlgorithm,),
        (TestOrderPercentAlgorithm,)
    ])
    def test_minute_data(self, algo_class):
        start_session = pd.Timestamp('2002-1-2', tz='UTC')
        period_end = pd.Timestamp('2002-1-4', tz='UTC')
        equities = pd.DataFrame([{
            'start_date': start_session,
            'end_date': period_end + timedelta(days=1),
            'exchange': "TEST",
        }] * 2)
        equities['symbol'] = ['A', 'B']
        with TempDirectory() as tempdir, \
                tmp_trading_env(equities=equities) as env:
            sim_params = SimulationParameters(
                start_session=start_session,
                end_session=period_end,
                capital_base=1.0e5,
                data_frequency='minute',
                trading_calendar=self.trading_calendar,
            )

            data_portal = create_data_portal(
                env.asset_finder,
                tempdir,
                sim_params,
                equities.index,
                self.trading_calendar,
            )
            algo = algo_class(sim_params=sim_params, env=env)
            algo.run(data_portal)


class TestPositions(WithLogger,
                    WithDataPortal,
                    WithSimParams,
                    ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')

    sids = ASSET_FINDER_EQUITY_SIDS = [1, 133]

    def test_empty_portfolio(self):
        algo = EmptyPositionsAlgorithm(self.sids,
                                       sim_params=self.sim_params,
                                       env=self.env)
        daily_stats = algo.run(self.data_portal)

        expected_position_count = [
            0,  # Before entering the first position
            2,  # After entering, exiting on this date
            0,  # After exiting
            0,
        ]

        for i, expected in enumerate(expected_position_count):
            self.assertEqual(daily_stats.ix[i]['num_positions'],
                             expected)

    def test_noop_orders(self):
        algo = AmbitiousStopLimitAlgorithm(sid=1,
                                           sim_params=self.sim_params,
                                           env=self.env)
        daily_stats = algo.run(self.data_portal)

        # Verify that positions are empty for all dates.
        empty_positions = daily_stats.positions.map(lambda x: len(x) == 0)
        self.assertTrue(empty_positions.all())


class TestBeforeTradingStart(WithDataPortal,
                             WithSimParams,
                             ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-06', tz='utc')
    END_DATE = pd.Timestamp('2016-01-07', tz='utc')
    SIM_PARAMS_CAPITAL_BASE = 10000
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = EQUITY_MINUTE_BAR_LOOKBACK_DAYS = 1

    DATA_PORTAL_FIRST_TRADING_DAY = pd.Timestamp("2016-01-05", tz='UTC')
    EQUITY_MINUTE_BAR_START_DATE = pd.Timestamp("2016-01-05", tz='UTC')
    FUTURE_MINUTE_BAR_START_DATE = pd.Timestamp("2016-01-05", tz='UTC')

    data_start = ASSET_FINDER_EQUITY_START_DATE = pd.Timestamp(
        '2016-01-05',
        tz='utc',
    )

    SPLIT_ASSET_SID = 3
    ASSET_FINDER_EQUITY_SIDS = 1, 2, SPLIT_ASSET_SID

    @classmethod
    def make_equity_minute_bar_data(cls):
        asset_minutes = \
            cls.trading_calendar.minutes_in_range(
                cls.data_start,
                cls.END_DATE,
            )
        minutes_count = len(asset_minutes)
        minutes_arr = np.arange(minutes_count) + 1
        split_data = pd.DataFrame(
            {
                'open': minutes_arr + 1,
                'high': minutes_arr + 2,
                'low': minutes_arr - 1,
                'close': minutes_arr,
                'volume': 100 * minutes_arr,
            },
            index=asset_minutes,
        )
        split_data.iloc[780:] = split_data.iloc[780:] / 2.0
        for sid in (1, 8554):
            yield sid, create_minute_df_for_asset(
                cls.trading_calendar,
                cls.data_start,
                cls.sim_params.end_session,
            )

        yield 2, create_minute_df_for_asset(
            cls.trading_calendar,
            cls.data_start,
            cls.sim_params.end_session,
            50,
        )
        yield cls.SPLIT_ASSET_SID, split_data

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame.from_records([
            {
                'effective_date': str_to_seconds('2016-01-07'),
                'ratio': 0.5,
                'sid': cls.SPLIT_ASSET_SID,
            }
        ])

    @classmethod
    def make_equity_daily_bar_data(cls):
        for sid in cls.ASSET_FINDER_EQUITY_SIDS:
            yield sid, create_daily_df_for_asset(
                cls.trading_calendar,
                cls.data_start,
                cls.sim_params.end_session,
            )

    def test_data_in_bts_minute(self):
        algo_code = dedent("""
        from zipline.api import record, sid
        def initialize(context):
            context.history_values = []

        def before_trading_start(context, data):
            record(the_price1=data.current(sid(1), "price"))
            record(the_high1=data.current(sid(1), "high"))
            record(the_price2=data.current(sid(2), "price"))
            record(the_high2=data.current(sid(2), "high"))

            context.history_values.append(data.history(
                [sid(1), sid(2)],
                ["price", "high"],
                60,
                "1m"
            ))

        def handle_data(context, data):
            pass
        """)

        algo = TradingAlgorithm(
            script=algo_code,
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)

        # fetching data at midnight gets us the previous market minute's data
        self.assertEqual(390, results.iloc[0].the_price1)
        self.assertEqual(392, results.iloc[0].the_high1)

        # make sure that price is ffilled, but not other fields
        self.assertEqual(350, results.iloc[0].the_price2)
        self.assertTrue(np.isnan(results.iloc[0].the_high2))

        # 10-minute history

        # asset1 day1 price should be 331-390
        np.testing.assert_array_equal(
            range(331, 391), algo.history_values[0]["price"][1]
        )

        # asset1 day1 high should be 333-392
        np.testing.assert_array_equal(
            range(333, 393), algo.history_values[0]["high"][1]
        )

        # asset2 day1 price should be 19 300s, then 40 350s
        np.testing.assert_array_equal(
            [300] * 19, algo.history_values[0]["price"][2][0:19]
        )

        np.testing.assert_array_equal(
            [350] * 40, algo.history_values[0]["price"][2][20:]
        )

        # asset2 day1 high should be all NaNs except for the 19th item
        # = 2016-01-05 20:20:00+00:00
        np.testing.assert_array_equal(
            np.full(19, np.nan), algo.history_values[0]["high"][2][0:19]
        )

        self.assertEqual(352, algo.history_values[0]["high"][2][19])

        np.testing.assert_array_equal(
            np.full(40, np.nan), algo.history_values[0]["high"][2][20:]
        )

    def test_data_in_bts_daily(self):
        algo_code = dedent("""
        from zipline.api import record, sid
        def initialize(context):
            context.history_values = []

        def before_trading_start(context, data):
            record(the_price1=data.current(sid(1), "price"))
            record(the_high1=data.current(sid(1), "high"))
            record(the_price2=data.current(sid(2), "price"))
            record(the_high2=data.current(sid(2), "high"))

            context.history_values.append(data.history(
                [sid(1), sid(2)],
                ["price", "high"],
                1,
                "1m"
            ))

        def handle_data(context, data):
            pass
        """)

        algo = TradingAlgorithm(
            script=algo_code,
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)

        self.assertEqual(392, results.the_high1[0])
        self.assertEqual(390, results.the_price1[0])

        # nan because asset2 only trades every 50 minutes
        self.assertTrue(np.isnan(results.the_high2[0]))

        self.assertTrue(350, results.the_price2[0])

        self.assertEqual(392, algo.history_values[0]["high"][1][0])
        self.assertEqual(390, algo.history_values[0]["price"][1][0])

        self.assertTrue(np.isnan(algo.history_values[0]["high"][2][0]))
        self.assertEqual(350, algo.history_values[0]["price"][2][0])

    def test_portfolio_bts(self):
        algo_code = dedent("""
        from zipline.api import order, sid, record

        def initialize(context):
            context.ordered = False
            context.hd_portfolio = context.portfolio

        def before_trading_start(context, data):
            bts_portfolio = context.portfolio

            # Assert that the portfolio in BTS is the same as the last
            # portfolio in handle_data
            assert (context.hd_portfolio == bts_portfolio)
            record(pos_value=bts_portfolio.positions_value)

        def handle_data(context, data):
            if not context.ordered:
                order(sid(1), 1)
                context.ordered = True
            context.hd_portfolio = context.portfolio
        """)

        algo = TradingAlgorithm(
            script=algo_code,
            data_frequency="minute",
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)

        # Asset starts with price 1 on 1/05 and increases by 1 every minute.
        # Simulation starts on 1/06, where the price in bts is 390, and
        # positions_value is 0. On 1/07, price is 780, and after buying one
        # share on the first bar of 1/06, positions_value is 780
        self.assertEqual(results.pos_value.iloc[0], 0)
        self.assertEqual(results.pos_value.iloc[1], 780)

    def test_account_bts(self):
        algo_code = dedent("""
        from zipline.api import order, sid, record

        def initialize(context):
            context.ordered = False
            context.hd_account = context.account

        def before_trading_start(context, data):
            bts_account = context.account

            # Assert that the account in BTS is the same as the last account
            # in handle_data
            assert (context.hd_account == bts_account)
            record(port_value=context.account.equity_with_loan)

        def handle_data(context, data):
            if not context.ordered:
                order(sid(1), 1)
                context.ordered = True
            context.hd_account = context.account
        """)

        algo = TradingAlgorithm(
            script=algo_code,
            data_frequency="minute",
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)

        # Starting portfolio value is 10000. Order for the asset fills on the
        # second bar of 1/06, where the price is 391, and costs the default
        # commission of 1. On 1/07, the price is 780, and the increase in
        # portfolio value is 780-392-1
        self.assertEqual(results.port_value.iloc[0], 10000)
        self.assertAlmostEqual(results.port_value.iloc[1],
                               10000 + 780 - 392 - 1)

    def test_portfolio_bts_with_overnight_split(self):
        algo_code = dedent("""
        from zipline.api import order, sid, record
        def initialize(context):
            context.ordered = False
            context.hd_portfolio = context.portfolio
        def before_trading_start(context, data):
            bts_portfolio = context.portfolio
            # Assert that the portfolio in BTS is the same as the last
            # portfolio in handle_data, except for the positions
            for k in bts_portfolio.__dict__:
                if k != 'positions':
                    assert (context.hd_portfolio.__dict__[k]
                            == bts_portfolio.__dict__[k])
            record(pos_value=bts_portfolio.positions_value)
            record(pos_amount=bts_portfolio.positions[sid(3)].amount)
            record(
                last_sale_price=bts_portfolio.positions[sid(3)].last_sale_price
            )
        def handle_data(context, data):
            if not context.ordered:
                order(sid(3), 1)
                context.ordered = True
            context.hd_portfolio = context.portfolio
        """)

        algo = TradingAlgorithm(
            script=algo_code,
            data_frequency="minute",
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)

        # On 1/07, positions value should by 780, same as without split
        self.assertEqual(results.pos_value.iloc[0], 0)
        self.assertEqual(results.pos_value.iloc[1], 780)

        # On 1/07, after applying the split, 1 share becomes 2
        self.assertEqual(results.pos_amount.iloc[0], 0)
        self.assertEqual(results.pos_amount.iloc[1], 2)

        # On 1/07, after applying the split, last sale price is halved
        self.assertEqual(results.last_sale_price.iloc[0], 0)
        self.assertEqual(results.last_sale_price.iloc[1], 390)

    def test_account_bts_with_overnight_split(self):
        algo_code = dedent("""
        from zipline.api import order, sid, record
        def initialize(context):
            context.ordered = False
            context.hd_account = context.account
        def before_trading_start(context, data):
            bts_account = context.account
            # Assert that the account in BTS is the same as the last account
            # in handle_data
            assert (context.hd_account == bts_account)
            record(port_value=bts_account.equity_with_loan)
        def handle_data(context, data):
            if not context.ordered:
                order(sid(1), 1)
                context.ordered = True
            context.hd_account = context.account
        """)

        algo = TradingAlgorithm(
            script=algo_code,
            data_frequency="minute",
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)

        # On 1/07, portfolio value is the same as without split
        self.assertEqual(results.port_value.iloc[0], 10000)
        self.assertAlmostEqual(results.port_value.iloc[1],
                               10000 + 780 - 392 - 1)


class TestAlgoScript(WithLogger,
                     WithDataPortal,
                     WithSimParams,
                     ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-12-31', tz='utc')
    DATA_PORTAL_USE_MINUTE_DATA = False
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = 5  # max history window length

    STRING_TYPE_NAMES = [s.__name__ for s in string_types]
    STRING_TYPE_NAMES_STRING = ', '.join(STRING_TYPE_NAMES)
    ASSET_TYPE_NAME = Asset.__name__
    CONTINUOUS_FUTURE_NAME = ContinuousFuture.__name__
    ASSET_OR_STRING_TYPE_NAMES = ', '.join([ASSET_TYPE_NAME] +
                                           STRING_TYPE_NAMES)
    ASSET_OR_STRING_OR_CF_TYPE_NAMES = ', '.join([ASSET_TYPE_NAME,
                                                  CONTINUOUS_FUTURE_NAME] +
                                                 STRING_TYPE_NAMES)
    ARG_TYPE_TEST_CASES = (
        ('history__assets', (bad_type_history_assets,
                             ASSET_OR_STRING_OR_CF_TYPE_NAMES,
                             True)),
        ('history__fields', (bad_type_history_fields,
                             STRING_TYPE_NAMES_STRING,
                             True)),
        ('history__bar_count', (bad_type_history_bar_count, 'int', False)),
        ('history__frequency', (bad_type_history_frequency,
                                STRING_TYPE_NAMES_STRING,
                                False)),
        ('current__assets', (bad_type_current_assets,
                             ASSET_OR_STRING_OR_CF_TYPE_NAMES,
                             True)),
        ('current__fields', (bad_type_current_fields,
                             STRING_TYPE_NAMES_STRING,
                             True)),
        ('is_stale__assets', (bad_type_is_stale_assets, 'Asset', True)),
        ('can_trade__assets', (bad_type_can_trade_assets, 'Asset', True)),
        ('history_kwarg__assets',
         (bad_type_history_assets_kwarg,
          ASSET_OR_STRING_OR_CF_TYPE_NAMES,
          True)),
        ('history_kwarg_bad_list__assets',
         (bad_type_history_assets_kwarg_list,
          ASSET_OR_STRING_OR_CF_TYPE_NAMES,
          True)),
        ('history_kwarg__fields',
         (bad_type_history_fields_kwarg, STRING_TYPE_NAMES_STRING, True)),
        ('history_kwarg__bar_count',
         (bad_type_history_bar_count_kwarg, 'int', False)),
        ('history_kwarg__frequency',
         (bad_type_history_frequency_kwarg, STRING_TYPE_NAMES_STRING, False)),
        ('current_kwarg__assets',
         (bad_type_current_assets_kwarg,
          ASSET_OR_STRING_OR_CF_TYPE_NAMES,
          True)),
        ('current_kwarg__fields',
         (bad_type_current_fields_kwarg, STRING_TYPE_NAMES_STRING, True)),
    )

    sids = 0, 1, 3, 133

    @classmethod
    def make_equity_info(cls):
        register_calendar("TEST", get_calendar("NYSE"), force=True)

        data = make_simple_equity_info(
            cls.sids,
            cls.START_DATE,
            cls.END_DATE,
        )
        data.loc[3, 'symbol'] = 'TEST'
        return data

    @classmethod
    def make_equity_daily_bar_data(cls):
        days = len(cls.equity_daily_bar_days)
        return trades_by_sid_to_dfs(
            {
                0: factory.create_trade_history(
                    0,
                    [10.0] * days,
                    [100] * days,
                    timedelta(days=1),
                    cls.sim_params,
                    cls.trading_calendar),
                3: factory.create_trade_history(
                    3,
                    [10.0] * days,
                    [100] * days,
                    timedelta(days=1),
                    cls.sim_params,
                    cls.trading_calendar)
            },
            index=cls.equity_daily_bar_days,
        )

    def test_noop(self):
        algo = TradingAlgorithm(initialize=initialize_noop,
                                handle_data=handle_data_noop)
        algo.run(self.data_portal)

    def test_noop_string(self):
        algo = TradingAlgorithm(script=noop_algo)
        algo.run(self.data_portal)

    def test_no_handle_data(self):
        algo = TradingAlgorithm(script=no_handle_data)
        algo.run(self.data_portal)

    def test_api_calls(self):
        algo = TradingAlgorithm(initialize=initialize_api,
                                handle_data=handle_data_api,
                                env=self.env)
        algo.run(self.data_portal)

    def test_api_calls_string(self):
        algo = TradingAlgorithm(script=api_algo, env=self.env)
        algo.run(self.data_portal)

    def test_api_get_environment(self):
        platform = 'zipline'
        algo = TradingAlgorithm(script=api_get_environment_algo,
                                platform=platform)
        algo.run(self.data_portal)
        self.assertEqual(algo.environment, platform)

    def test_api_symbol(self):
        algo = TradingAlgorithm(script=api_symbol_algo,
                                env=self.env,
                                sim_params=self.sim_params)
        algo.run(self.data_portal)

    def test_fixed_slippage(self):
        # verify order -> transaction -> portfolio position.
        # --------------
        test_algo = TradingAlgorithm(
            script="""
from zipline.api import (slippage,
                         commission,
                         set_slippage,
                         set_commission,
                         order,
                         record,
                         sid)

def initialize(context):
    model = slippage.FixedSlippage(spread=0.10)
    set_slippage(model)
    set_commission(commission.PerTrade(100.00))
    context.count = 1
    context.incr = 0

def handle_data(context, data):
    if context.incr < context.count:
        order(sid(0), -1000)
    record(price=data.current(sid(0), "price"))

    context.incr += 1""",
            sim_params=self.sim_params,
            env=self.env,
        )
        results = test_algo.run(self.data_portal)

        # flatten the list of txns
        all_txns = [val for sublist in results["transactions"].tolist()
                    for val in sublist]

        self.assertEqual(len(all_txns), 1)
        txn = all_txns[0]

        expected_spread = 0.05
        expected_price = test_algo.recorded_vars["price"] - expected_spread

        self.assertEqual(expected_price, txn['price'])

        # make sure that the $100 commission was applied to our cash
        # the txn was for -1000 shares at 9.95, means -9.95k.  our capital_used
        # for that day was therefore 9.95k, but after the $100 commission,
        # it should be 9.85k.
        self.assertEqual(9850, results.capital_used[1])
        self.assertEqual(100, results["orders"][1][0]["commission"])

    @parameterized.expand(
        [
            ('no_minimum_commission', 0,),
            ('default_minimum_commission', 1,),
            ('alternate_minimum_commission', 2,),
        ]
    )
    def test_volshare_slippage(self, name, minimum_commission):
        tempdir = TempDirectory()
        try:
            if name == "default_minimum_commission":
                commission_line = "set_commission(commission.PerShare(0.02))"
            else:
                commission_line = \
                    "set_commission(commission.PerShare(0.02, " \
                    "min_trade_cost={0}))".format(minimum_commission)

            # verify order -> transaction -> portfolio position.
            # --------------
            test_algo = TradingAlgorithm(
                script="""
from zipline.api import *

def initialize(context):
    model = slippage.VolumeShareSlippage(
                            volume_limit=.3,
                            price_impact=0.05
                       )
    set_slippage(model)
    {0}

    context.count = 2
    context.incr = 0

def handle_data(context, data):
    if context.incr < context.count:
        # order small lots to be sure the
        # order will fill in a single transaction
        order(sid(0), 5000)
    record(price=data.current(sid(0), "price"))
    record(volume=data.current(sid(0), "volume"))
    record(incr=context.incr)
    context.incr += 1
    """.format(commission_line),
                sim_params=self.sim_params,
                env=self.env,
            )
            trades = factory.create_daily_trade_source(
                [0], self.sim_params, self.env, self.trading_calendar)
            data_portal = create_data_portal_from_trade_history(
                self.env.asset_finder, self.trading_calendar, tempdir,
                self.sim_params, {0: trades})
            results = test_algo.run(data_portal)

            all_txns = [
                val for sublist in results["transactions"].tolist()
                for val in sublist]

            self.assertEqual(len(all_txns), 67)
            # all_orders are all the incremental versions of the
            # orders as each new fill comes in.
            all_orders = list(toolz.concat(results['orders']))

            if minimum_commission == 0:
                # for each incremental version of each order, the commission
                # should be its filled amount * 0.02
                for order_ in all_orders:
                    self.assertAlmostEqual(
                        order_["filled"] * 0.02,
                        order_["commission"]
                    )
            else:
                # the commission should be at least the min_trade_cost
                for order_ in all_orders:
                    if order_["filled"] > 0:
                        self.assertAlmostEqual(
                            max(order_["filled"] * 0.02, minimum_commission),
                            order_["commission"]
                        )
                    else:
                        self.assertEqual(0, order_["commission"])
        finally:
            tempdir.cleanup()

    def test_algo_record_vars(self):
        test_algo = TradingAlgorithm(
            script=record_variables,
            sim_params=self.sim_params,
            env=self.env,
        )
        results = test_algo.run(self.data_portal)

        for i in range(1, 252):
            self.assertEqual(results.iloc[i-1]["incr"], i)

    def test_algo_record_allow_mock(self):
        """
        Test that values from "MagicMock"ed methods can be passed to record.

        Relevant for our basic/validation and methods like history, which
        will end up returning a MagicMock instead of a DataFrame.
        """
        test_algo = TradingAlgorithm(
            script=record_variables,
            sim_params=self.sim_params,
        )
        set_algo_instance(test_algo)

        test_algo.record(foo=MagicMock())

    def test_algo_record_nan(self):
        test_algo = TradingAlgorithm(
            script=record_float_magic % 'nan',
            sim_params=self.sim_params,
            env=self.env,
        )
        results = test_algo.run(self.data_portal)

        for i in range(1, 252):
            self.assertTrue(np.isnan(results.iloc[i-1]["data"]))

    def test_order_methods(self):
        """
        Only test that order methods can be called without error.
        Correct filling of orders is tested in zipline.
        """
        test_algo = TradingAlgorithm(
            script=call_all_order_methods,
            sim_params=self.sim_params,
            env=self.env,
        )
        test_algo.run(self.data_portal)

    def test_batch_order_target_percent_matches_multi_order(self):
        weights = pd.Series([.3, .7])

        multi_blotter = RecordBatchBlotter(self.SIM_PARAMS_DATA_FREQUENCY,
                                           self.asset_finder)
        multi_test_algo = TradingAlgorithm(
            script=dedent("""\
                from collections import OrderedDict
                from six import iteritems

                from zipline.api import sid, order_target_percent


                def initialize(context):
                    context.assets = [sid(0), sid(3)]
                    context.placed = False

                def handle_data(context, data):
                    if not context.placed:
                        for asset, weight in iteritems(OrderedDict(zip(
                            context.assets, {weights}
                        ))):
                            order_target_percent(asset, weight)

                        context.placed = True

            """).format(weights=list(weights)),
            blotter=multi_blotter,
            env=self.env,
        )
        multi_stats = multi_test_algo.run(self.data_portal)
        self.assertFalse(multi_blotter.order_batch_called)

        batch_blotter = RecordBatchBlotter(self.SIM_PARAMS_DATA_FREQUENCY,
                                           self.asset_finder)
        batch_test_algo = TradingAlgorithm(
            script=dedent("""\
                from collections import OrderedDict

                from zipline.api import sid, batch_order_target_percent


                def initialize(context):
                    context.assets = [sid(0), sid(3)]
                    context.placed = False

                def handle_data(context, data):
                    if not context.placed:
                        orders = batch_order_target_percent(OrderedDict(zip(
                            context.assets, {weights}
                        )))
                        assert len(orders) == 2, \
                            "len(orders) was %s but expected 2" % len(orders)
                        for o in orders:
                            assert o is not None, "An order is None"

                        context.placed = True

            """).format(weights=list(weights)),
            blotter=batch_blotter,
            env=self.env,
        )
        batch_stats = batch_test_algo.run(self.data_portal)
        self.assertTrue(batch_blotter.order_batch_called)

        for stats in (multi_stats, batch_stats):
            stats.orders = stats.orders.apply(
                lambda orders: [toolz.dissoc(o, 'id') for o in orders]
            )
            stats.transactions = stats.transactions.apply(
                lambda txns: [toolz.dissoc(txn, 'order_id') for txn in txns]
            )
        assert_equal(multi_stats, batch_stats)

    def test_batch_order_target_percent_filters_null_orders(self):
        weights = pd.Series([1, 0])

        batch_blotter = RecordBatchBlotter(self.SIM_PARAMS_DATA_FREQUENCY,
                                           self.asset_finder)
        batch_test_algo = TradingAlgorithm(
            script=dedent("""\
                from collections import OrderedDict

                from zipline.api import sid, batch_order_target_percent


                def initialize(context):
                    context.assets = [sid(0), sid(3)]
                    context.placed = False

                def handle_data(context, data):
                    if not context.placed:
                        orders = batch_order_target_percent(OrderedDict(zip(
                            context.assets, {weights}
                        )))
                        assert len(orders) == 1, \
                            "len(orders) was %s but expected 1" % len(orders)
                        for o in orders:
                            assert o is not None, "An order is None"

                        context.placed = True

            """).format(weights=list(weights)),
            blotter=batch_blotter,
            env=self.env,
        )
        batch_test_algo.run(self.data_portal)
        self.assertTrue(batch_blotter.order_batch_called)

    def test_order_dead_asset(self):
        # after asset 0 is dead
        params = SimulationParameters(
            start_session=pd.Timestamp("2007-01-03", tz='UTC'),
            end_session=pd.Timestamp("2007-01-05", tz='UTC'),
            trading_calendar=self.trading_calendar,
        )

        # order method shouldn't blow up
        test_algo = TradingAlgorithm(
            script="""
from zipline.api import order, sid

def initialize(context):
    pass

def handle_data(context, data):
    order(sid(0), 10)
        """,
            sim_params=params,
            env=self.env
        )

        test_algo.run(self.data_portal)

        # order_value and order_percent should blow up
        for order_str in ["order_value", "order_percent"]:
            test_algo = TradingAlgorithm(
                script="""
from zipline.api import order_percent, order_value, sid

def initialize(context):
    pass

def handle_data(context, data):
    {0}(sid(0), 10)
        """.format(order_str),
                sim_params=params,
                env=self.env
            )

        with self.assertRaises(CannotOrderDelistedAsset):
            test_algo.run(self.data_portal)

    def test_order_in_init(self):
        """
        Test that calling order in initialize
        will raise an error.
        """
        with self.assertRaises(OrderDuringInitialize):
            test_algo = TradingAlgorithm(
                script=call_order_in_init,
                sim_params=self.sim_params,
                env=self.env,
            )
            test_algo.run(self.data_portal)

    def test_portfolio_in_init(self):
        """
        Test that accessing portfolio in init doesn't break.
        """
        test_algo = TradingAlgorithm(
            script=access_portfolio_in_init,
            sim_params=self.sim_params,
            env=self.env,
        )
        test_algo.run(self.data_portal)

    def test_account_in_init(self):
        """
        Test that accessing account in init doesn't break.
        """
        test_algo = TradingAlgorithm(
            script=access_account_in_init,
            sim_params=self.sim_params,
            env=self.env,
        )
        test_algo.run(self.data_portal)

    def test_without_kwargs(self):
        """
        Test that api methods on the data object can be called with positional
        arguments.
        """

        params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-10", tz='UTC'),
            end_session=pd.Timestamp("2006-01-11", tz='UTC'),
            trading_calendar=self.trading_calendar,
        )

        test_algo = TradingAlgorithm(
            script=call_without_kwargs,
            sim_params=params,
            env=self.env,
        )
        test_algo.run(self.data_portal)

    def test_good_kwargs(self):
        """
        Test that api methods on the data object can be called with keyword
        arguments.
        """
        params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-10", tz='UTC'),
            end_session=pd.Timestamp("2006-01-11", tz='UTC'),
            trading_calendar=self.trading_calendar,
        )

        test_algo = TradingAlgorithm(
            script=call_with_kwargs,
            sim_params=params,
            env=self.env,
        )
        test_algo.run(self.data_portal)

    @parameterized.expand([('history', call_with_bad_kwargs_history),
                           ('current', call_with_bad_kwargs_current)])
    def test_bad_kwargs(self, name, algo_text):
        """
        Test that api methods on the data object called with bad kwargs return
        a meaningful TypeError that we create, rather than an unhelpful cython
        error
        """
        with self.assertRaises(TypeError) as cm:
            test_algo = TradingAlgorithm(
                script=algo_text,
                sim_params=self.sim_params,
                env=self.env,
            )
            test_algo.run(self.data_portal)

        self.assertEqual("%s() got an unexpected keyword argument 'blahblah'"
                         % name, cm.exception.args[0])

    @parameterized.expand(ARG_TYPE_TEST_CASES)
    def test_arg_types(self, name, inputs):

        keyword = name.split('__')[1]

        with self.assertRaises(TypeError) as cm:
            algo = TradingAlgorithm(
                script=inputs[0],
                sim_params=self.sim_params,
                env=self.env
            )
            algo.run(self.data_portal)

        expected = "Expected %s argument to be of type %s%s" % (
            keyword,
            'or iterable of type ' if inputs[2] else '',
            inputs[1]
        )

        self.assertEqual(expected, cm.exception.args[0])

    def test_empty_asset_list_to_history(self):
        params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-10", tz='UTC'),
            end_session=pd.Timestamp("2006-01-11", tz='UTC'),
            trading_calendar=self.trading_calendar,
        )

        algo = TradingAlgorithm(
            script=dedent("""
                def initialize(context):
                    pass

                def handle_data(context, data):
                    data.history([], "price", 5, '1d')
                """),
            sim_params=params,
            env=self.env
        )

        algo.run(self.data_portal)

    @parameterized.expand(
        [('bad_kwargs', call_with_bad_kwargs_get_open_orders),
         ('good_kwargs', call_with_good_kwargs_get_open_orders),
         ('no_kwargs', call_with_no_kwargs_get_open_orders)]
    )
    def test_get_open_orders_kwargs(self, name, script):
        algo = TradingAlgorithm(
            script=script,
            sim_params=self.sim_params,
            env=self.env
        )

        if name == 'bad_kwargs':
            with self.assertRaises(TypeError) as cm:
                algo.run(self.data_portal)
                self.assertEqual('Keyword argument `sid` is no longer '
                                 'supported for get_open_orders. Use `asset` '
                                 'instead.', cm.exception.args[0])
        else:
            algo.run(self.data_portal)

    def test_empty_positions(self):
        """
        Test that when we try context.portfolio.positions[stock] on a stock
        for which we have no positions, we return a Position with values 0
        (but more importantly, we don't crash) and don't save this Position
        to the user-facing dictionary PositionTracker._positions_store
        """
        algo = TradingAlgorithm(
            script=empty_positions,
            sim_params=self.sim_params,
            env=self.env
        )

        results = algo.run(self.data_portal)
        num_positions = results.num_positions
        amounts = results.amounts
        self.assertTrue(all(num_positions == 0))
        self.assertTrue(all(amounts == 0))

    @parameterized.expand([
        ('noop_algo', noop_algo),
        ('with_benchmark_set', set_benchmark_algo)]
    )
    def test_zero_trading_days(self, name, algocode):
        """
        Test that when we run a simulation with no trading days (e.g. beginning
        and ending the same weekend), we don't crash on calculating the
        benchmark
        """
        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp('2006-01-14', tz='UTC'),
            end=pd.Timestamp('2006-01-15', tz='UTC')
        )

        algo = TradingAlgorithm(
            script=algocode,
            sim_params=sim_params,
            env=self.env
        )
        algo.run(self.data_portal)

    def test_schedule_function_time_rule_positionally_misplaced(self):
        """
        Test that when a user specifies a time rule for the date_rule argument,
        but no rule in the time_rule argument
        (e.g. schedule_function(func, <time_rule>)), we assume that means
        assign a time rule but no date rule
        """

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp('2006-01-12', tz='UTC'),
            end=pd.Timestamp('2006-01-13', tz='UTC'),
            data_frequency='minute'
        )

        algocode = dedent("""
        from zipline.api import time_rules, schedule_function

        def do_at_open(context, data):
            context.done_at_open.append(context.get_datetime())

        def do_at_close(context, data):
            context.done_at_close.append(context.get_datetime())

        def initialize(context):
            context.done_at_open = []
            context.done_at_close = []
            schedule_function(do_at_open, time_rules.market_open())
            schedule_function(do_at_close, time_rules.market_close())

        def handle_data(algo, data):
            pass
        """)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)

            algo = TradingAlgorithm(
                script=algocode,
                sim_params=sim_params,
                env=self.env
            )
            algo.run(self.data_portal)

            self.assertEqual(len(w), 2)

            for i, warning in enumerate(w):
                self.assertIsInstance(warning.message, UserWarning)
                self.assertEqual(
                    warning.message.args[0],
                    'Got a time rule for the second positional argument '
                    'date_rule. You should use keyword argument '
                    'time_rule= when calling schedule_function without '
                    'specifying a date_rule'
                )
                # The warnings come from line 13 and 14 in the algocode
                self.assertEqual(warning.lineno, 13 + i)

        self.assertEqual(
            algo.done_at_open,
            [pd.Timestamp('2006-01-12 14:31:00', tz='UTC'),
             pd.Timestamp('2006-01-13 14:31:00', tz='UTC')]
        )

        self.assertEqual(
            algo.done_at_close,
            [pd.Timestamp('2006-01-12 20:59:00', tz='UTC'),
             pd.Timestamp('2006-01-13 20:59:00', tz='UTC')]
        )


class TestCapitalChanges(WithLogger,
                         WithDataPortal,
                         WithSimParams,
                         ZiplineTestCase):

    sids = 0, 1

    @classmethod
    def make_equity_info(cls):
        data = make_simple_equity_info(
            cls.sids,
            pd.Timestamp('2006-01-03', tz='UTC'),
            pd.Timestamp('2006-01-09', tz='UTC'),
        )
        return data

    @classmethod
    def make_equity_minute_bar_data(cls):
        minutes = cls.trading_calendar.minutes_in_range(
            pd.Timestamp('2006-01-03', tz='UTC'),
            pd.Timestamp('2006-01-09', tz='UTC')
        )
        return trades_by_sid_to_dfs(
            {
                1: factory.create_trade_history(
                    1,
                    np.arange(100.0, 100.0 + len(minutes), 1),
                    [10000] * len(minutes),
                    timedelta(minutes=1),
                    cls.sim_params,
                    cls.trading_calendar),
            },
            index=pd.DatetimeIndex(minutes),
        )

    @classmethod
    def make_equity_daily_bar_data(cls):
        days = cls.trading_calendar.minutes_in_range(
            pd.Timestamp('2006-01-03', tz='UTC'),
            pd.Timestamp('2006-01-09', tz='UTC')
        )
        return trades_by_sid_to_dfs(
            {
                0: factory.create_trade_history(
                    0,
                    np.arange(10.0, 10.0 + len(days), 1.0),
                    [10000] * len(days),
                    timedelta(days=1),
                    cls.sim_params,
                    cls.trading_calendar),
            },
            index=pd.DatetimeIndex(days),
        )

    @parameterized.expand([
        ('target', 153000.0), ('delta', 50000.0)
    ])
    def test_capital_changes_daily_mode(self, change_type, value):
        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp('2006-01-03', tz='UTC'),
            end=pd.Timestamp('2006-01-09', tz='UTC')
        )

        capital_changes = {
            pd.Timestamp('2006-01-06', tz='UTC'):
                {'type': change_type, 'value': value}
        }

        algocode = """
from zipline.api import set_slippage, set_commission, slippage, commission, \
    schedule_function, time_rules, order, sid

def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(0, 0))
    schedule_function(order_stuff, time_rule=time_rules.market_open())

def order_stuff(context, data):
    order(sid(0), 1000)
"""

        algo = TradingAlgorithm(
            script=algocode,
            sim_params=sim_params,
            env=self.env,
            data_portal=self.data_portal,
            capital_changes=capital_changes
        )

        gen = algo.get_generator()
        results = list(gen)

        cumulative_perf = \
            [r['cumulative_perf'] for r in results if 'cumulative_perf' in r]
        daily_perf = [r['daily_perf'] for r in results if 'daily_perf' in r]
        capital_change_packets = \
            [r['capital_change'] for r in results if 'capital_change' in r]

        self.assertEqual(len(capital_change_packets), 1)
        self.assertEqual(
            capital_change_packets[0],
            {'date': pd.Timestamp('2006-01-06', tz='UTC'),
             'type': 'cash',
             'target': 153000.0 if change_type == 'target' else None,
             'delta': 50000.0})

        # 1/03: price = 10, place orders
        # 1/04: orders execute at price = 11, place orders
        # 1/05: orders execute at price = 12, place orders
        # 1/06: +50000 capital change,
        #       orders execute at price = 13, place orders
        # 1/09: orders execute at price = 14, place orders

        expected_daily = {}

        expected_capital_changes = np.array([
            0.0, 0.0, 0.0, 50000.0, 0.0
        ])

        # Day 1, no transaction. Day 2, we transact, but the price of our stock
        # does not change. Day 3, we start getting returns
        expected_daily['returns'] = np.array([
            0.0,
            0.0,
            # 1000 shares * gain of 1
            (100000.0 + 1000.0)/100000.0 - 1.0,
            # 2000 shares * gain of 1, capital change of +5000
            (151000.0 + 2000.0)/151000.0 - 1.0,
            # 3000 shares * gain of 1
            (153000.0 + 3000.0)/153000.0 - 1.0,
        ])

        expected_daily['pnl'] = np.array([
            0.0,
            0.0,
            1000.00,  # 1000 shares * gain of 1
            2000.00,  # 2000 shares * gain of 1
            3000.00,  # 3000 shares * gain of 1
        ])

        expected_daily['capital_used'] = np.array([
            0.0,
            -11000.0,  # 1000 shares at price = 11
            -12000.0,  # 1000 shares at price = 12
            -13000.0,  # 1000 shares at price = 13
            -14000.0,  # 1000 shares at price = 14
        ])

        expected_daily['ending_cash'] = \
            np.array([100000.0] * 5) + \
            np.cumsum(expected_capital_changes) + \
            np.cumsum(expected_daily['capital_used'])

        expected_daily['starting_cash'] = \
            expected_daily['ending_cash'] - \
            expected_daily['capital_used']

        expected_daily['starting_value'] = [
            0.0,
            0.0,
            11000.0,  # 1000 shares at price = 11
            24000.0,  # 2000 shares at price = 12
            39000.0,  # 3000 shares at price = 13
        ]

        expected_daily['ending_value'] = \
            expected_daily['starting_value'] + \
            expected_daily['pnl'] - \
            expected_daily['capital_used']

        expected_daily['portfolio_value'] = \
            expected_daily['ending_value'] + \
            expected_daily['ending_cash']

        stats = [
            'returns', 'pnl', 'capital_used', 'starting_cash', 'ending_cash',
            'starting_value', 'ending_value', 'portfolio_value'
        ]

        expected_cumulative = {
            'returns': np.cumprod(expected_daily['returns'] + 1) - 1,
            'pnl': np.cumsum(expected_daily['pnl']),
            'capital_used': np.cumsum(expected_daily['capital_used']),
            'starting_cash':
                np.repeat(expected_daily['starting_cash'][0:1], 5),
            'ending_cash': expected_daily['ending_cash'],
            'starting_value':
                np.repeat(expected_daily['starting_value'][0:1], 5),
            'ending_value': expected_daily['ending_value'],
            'portfolio_value': expected_daily['portfolio_value'],
        }

        for stat in stats:
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in daily_perf]),
                expected_daily[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in cumulative_perf]),
                expected_cumulative[stat]
            )

        self.assertEqual(
            algo.capital_change_deltas,
            {pd.Timestamp('2006-01-06', tz='UTC'): 50000.0}
        )

    @parameterized.expand([
        ('interday_target', [('2006-01-04', 2388.0)]),
        ('interday_delta', [('2006-01-04', 1000.0)]),
        ('intraday_target', [('2006-01-04 17:00', 2186.0),
                             ('2006-01-04 18:00', 2806.0)]),
        ('intraday_delta', [('2006-01-04 17:00', 500.0),
                            ('2006-01-04 18:00', 500.0)]),
    ])
    def test_capital_changes_minute_mode_daily_emission(self, change, values):
        change_loc, change_type = change.split('_')

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp('2006-01-03', tz='UTC'),
            end=pd.Timestamp('2006-01-05', tz='UTC'),
            data_frequency='minute',
            capital_base=1000.0
        )

        capital_changes = {pd.Timestamp(val[0], tz='UTC'): {
            'type': change_type, 'value': val[1]} for val in values}

        algocode = """
from zipline.api import set_slippage, set_commission, slippage, commission, \
    schedule_function, time_rules, order, sid

def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(0, 0))
    schedule_function(order_stuff, time_rule=time_rules.market_open())

def order_stuff(context, data):
    order(sid(1), 1)
"""

        algo = TradingAlgorithm(
            script=algocode,
            sim_params=sim_params,
            env=self.env,
            data_portal=self.data_portal,
            capital_changes=capital_changes
        )

        gen = algo.get_generator()
        results = list(gen)

        cumulative_perf = \
            [r['cumulative_perf'] for r in results if 'cumulative_perf' in r]
        daily_perf = [r['daily_perf'] for r in results if 'daily_perf' in r]
        capital_change_packets = \
            [r['capital_change'] for r in results if 'capital_change' in r]

        self.assertEqual(len(capital_change_packets), len(capital_changes))
        expected = [
            {'date': pd.Timestamp(val[0], tz='UTC'),
             'type': 'cash',
             'target': val[1] if change_type == 'target' else None,
             'delta': 1000.0 if len(values) == 1 else 500.0}
            for val in values]
        self.assertEqual(capital_change_packets, expected)

        # 1/03: place orders at price = 100, execute at 101
        # 1/04: place orders at price = 490, execute at 491,
        #       +500 capital change at 17:00 and 18:00 (intraday)
        #       or +1000 at 00:00 (interday),
        # 1/05: place orders at price = 880, execute at 881

        expected_daily = {}

        expected_capital_changes = np.array([
            0.0, 1000.0, 0.0
        ])

        if change_loc == 'intraday':
            # Fills at 491, +500 capital change comes at 638 (17:00) and
            # 698 (18:00), ends day at 879
            day2_return = (1388.0 + 149.0 + 147.0)/1388.0 * \
                          (2184.0 + 60.0 + 60.0)/2184.0 * \
                          (2804.0 + 181.0 + 181.0)/2804.0 - 1.0
        else:
            # Fills at 491, ends day at 879, capital change +1000
            day2_return = (2388.0 + 390.0 + 388.0)/2388.0 - 1

        expected_daily['returns'] = np.array([
            # Fills at 101, ends day at 489
            (1000.0 + 388.0)/1000.0 - 1.0,
            day2_return,
            # Fills at 881, ends day at 1269
            (3166.0 + 390.0 + 390.0 + 388.0)/3166.0 - 1.0,
        ])

        expected_daily['pnl'] = np.array([
            388.0,
            390.0 + 388.0,
            390.0 + 390.0 + 388.0,
        ])

        expected_daily['capital_used'] = np.array([
            -101.0, -491.0, -881.0
        ])

        expected_daily['ending_cash'] = \
            np.array([1000.0] * 3) + \
            np.cumsum(expected_capital_changes) + \
            np.cumsum(expected_daily['capital_used'])

        expected_daily['starting_cash'] = \
            expected_daily['ending_cash'] - \
            expected_daily['capital_used']

        if change_loc == 'intraday':
            # Capital changes come after day start
            expected_daily['starting_cash'] -= expected_capital_changes

        expected_daily['starting_value'] = np.array([
            0.0, 489.0, 879.0 * 2
        ])

        expected_daily['ending_value'] = \
            expected_daily['starting_value'] + \
            expected_daily['pnl'] - \
            expected_daily['capital_used']

        expected_daily['portfolio_value'] = \
            expected_daily['ending_value'] + \
            expected_daily['ending_cash']

        stats = [
            'returns', 'pnl', 'capital_used', 'starting_cash', 'ending_cash',
            'starting_value', 'ending_value', 'portfolio_value'
        ]

        expected_cumulative = {
            'returns': np.cumprod(expected_daily['returns'] + 1) - 1,
            'pnl': np.cumsum(expected_daily['pnl']),
            'capital_used': np.cumsum(expected_daily['capital_used']),
            'starting_cash':
                np.repeat(expected_daily['starting_cash'][0:1], 3),
            'ending_cash': expected_daily['ending_cash'],
            'starting_value':
                np.repeat(expected_daily['starting_value'][0:1], 3),
            'ending_value': expected_daily['ending_value'],
            'portfolio_value': expected_daily['portfolio_value'],
        }

        for stat in stats:
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in daily_perf]),
                expected_daily[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in cumulative_perf]),
                expected_cumulative[stat]
            )

        if change_loc == 'interday':
            self.assertEqual(
                algo.capital_change_deltas,
                {pd.Timestamp('2006-01-04', tz='UTC'): 1000.0}
            )
        else:
            self.assertEqual(
                algo.capital_change_deltas,
                {pd.Timestamp('2006-01-04 17:00', tz='UTC'): 500.0,
                 pd.Timestamp('2006-01-04 18:00', tz='UTC'): 500.0}
            )

    @parameterized.expand([
        ('interday_target', [('2006-01-04', 2388.0)]),
        ('interday_delta', [('2006-01-04', 1000.0)]),
        ('intraday_target', [('2006-01-04 17:00', 2186.0),
                             ('2006-01-04 18:00', 2806.0)]),
        ('intraday_delta', [('2006-01-04 17:00', 500.0),
                            ('2006-01-04 18:00', 500.0)]),
    ])
    def test_capital_changes_minute_mode_minute_emission(self, change, values):
        change_loc, change_type = change.split('_')

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp('2006-01-03', tz='UTC'),
            end=pd.Timestamp('2006-01-05', tz='UTC'),
            data_frequency='minute',
            emission_rate='minute',
            capital_base=1000.0
        )

        capital_changes = {pd.Timestamp(val[0], tz='UTC'): {
            'type': change_type, 'value': val[1]} for val in values}

        algocode = """
from zipline.api import set_slippage, set_commission, slippage, commission, \
    schedule_function, time_rules, order, sid

def initialize(context):
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(0, 0))
    schedule_function(order_stuff, time_rule=time_rules.market_open())

def order_stuff(context, data):
    order(sid(1), 1)
"""

        algo = TradingAlgorithm(
            script=algocode,
            sim_params=sim_params,
            env=self.env,
            data_portal=self.data_portal,
            capital_changes=capital_changes
        )

        gen = algo.get_generator()
        results = list(gen)

        cumulative_perf = \
            [r['cumulative_perf'] for r in results if 'cumulative_perf' in r]
        minute_perf = [r['minute_perf'] for r in results if 'minute_perf' in r]
        daily_perf = [r['daily_perf'] for r in results if 'daily_perf' in r]
        capital_change_packets = \
            [r['capital_change'] for r in results if 'capital_change' in r]

        self.assertEqual(len(capital_change_packets), len(capital_changes))
        expected = [
            {'date': pd.Timestamp(val[0], tz='UTC'),
             'type': 'cash',
             'target': val[1] if change_type == 'target' else None,
             'delta': 1000.0 if len(values) == 1 else 500.0}
            for val in values]
        self.assertEqual(capital_change_packets, expected)

        # 1/03: place orders at price = 100, execute at 101
        # 1/04: place orders at price = 490, execute at 491,
        #       +500 capital change at 17:00 and 18:00 (intraday)
        #       or +1000 at 00:00 (interday),
        # 1/05: place orders at price = 880, execute at 881

        # Minute perfs are cumulative for the day
        expected_minute = {}

        capital_changes_after_start = np.array([0.0] * 1170)
        if change_loc == 'intraday':
            capital_changes_after_start[539:599] = 500.0
            capital_changes_after_start[599:780] = 1000.0

        expected_minute['pnl'] = np.array([0.0] * 1170)
        expected_minute['pnl'][:2] = 0.0
        expected_minute['pnl'][2:392] = 1.0
        expected_minute['pnl'][392:782] = 2.0
        expected_minute['pnl'][782:] = 3.0
        for start, end in ((0, 390), (390, 780), (780, 1170)):
            expected_minute['pnl'][start:end] = \
                np.cumsum(expected_minute['pnl'][start:end])

        expected_minute['capital_used'] = np.concatenate((
            [0.0] * 1, [-101.0] * 389,
            [0.0] * 1, [-491.0] * 389,
            [0.0] * 1, [-881.0] * 389,
        ))

        # +1000 capital changes comes before the day start if interday
        day2adj = 0.0 if change_loc == 'intraday' else 1000.0

        expected_minute['starting_cash'] = np.concatenate((
            [1000.0] * 390,
            # 101 spent on 1/03
            [1000.0 - 101.0 + day2adj] * 390,
            # 101 spent on 1/03, 491 on 1/04, +1000 capital change on 1/04
            [1000.0 - 101.0 - 491.0 + 1000] * 390
        ))

        expected_minute['ending_cash'] = \
            expected_minute['starting_cash'] + \
            expected_minute['capital_used'] + \
            capital_changes_after_start

        expected_minute['starting_value'] = np.concatenate((
            [0.0] * 390,
            [489.0] * 390,
            [879.0 * 2] * 390
        ))

        expected_minute['ending_value'] = \
            expected_minute['starting_value'] + \
            expected_minute['pnl'] - \
            expected_minute['capital_used']

        expected_minute['portfolio_value'] = \
            expected_minute['ending_value'] + \
            expected_minute['ending_cash']

        expected_minute['returns'] = \
            expected_minute['pnl'] / \
            (expected_minute['starting_value'] +
             expected_minute['starting_cash'])

        # If the change is interday, we can just calculate the returns from
        # the pnl, starting_value and starting_cash. If the change is intraday,
        # the returns after the change have to be calculated from two
        # subperiods
        if change_loc == 'intraday':
            # The last packet (at 1/04 16:59) before the first capital change
            prev_subperiod_return = expected_minute['returns'][538]

            # From 1/04 17:00 to 17:59
            cur_subperiod_pnl = \
                expected_minute['pnl'][539:599] - expected_minute['pnl'][538]
            cur_subperiod_starting_value = \
                np.array([expected_minute['ending_value'][538]] * 60)
            cur_subperiod_starting_cash = \
                np.array([expected_minute['ending_cash'][538] + 500] * 60)

            cur_subperiod_returns = cur_subperiod_pnl / \
                (cur_subperiod_starting_value + cur_subperiod_starting_cash)
            expected_minute['returns'][539:599] = \
                (cur_subperiod_returns + 1.0) * \
                (prev_subperiod_return + 1.0) - \
                1.0

            # The last packet (at 1/04 17:59) before the second capital change
            prev_subperiod_return = expected_minute['returns'][598]

            # From 1/04 18:00 to 21:00
            cur_subperiod_pnl = \
                expected_minute['pnl'][599:780] - expected_minute['pnl'][598]
            cur_subperiod_starting_value = \
                np.array([expected_minute['ending_value'][598]] * 181)
            cur_subperiod_starting_cash = \
                np.array([expected_minute['ending_cash'][598] + 500] * 181)

            cur_subperiod_returns = cur_subperiod_pnl / \
                (cur_subperiod_starting_value + cur_subperiod_starting_cash)
            expected_minute['returns'][599:780] = \
                (cur_subperiod_returns + 1.0) * \
                (prev_subperiod_return + 1.0) - \
                1.0

        # The last minute packet of each day
        expected_daily = {
            k: np.array([v[389], v[779], v[1169]])
            for k, v in iteritems(expected_minute)
        }

        stats = [
            'pnl', 'capital_used', 'starting_cash', 'ending_cash',
            'starting_value', 'ending_value', 'portfolio_value', 'returns'
        ]

        expected_cumulative = deepcopy(expected_minute)

        # "Add" daily return from 1/03 to minute returns on 1/04 and 1/05
        # "Add" daily return from 1/04 to minute returns on 1/05
        expected_cumulative['returns'][390:] = \
            (expected_cumulative['returns'][390:] + 1) * \
            (expected_daily['returns'][0] + 1) - 1
        expected_cumulative['returns'][780:] = \
            (expected_cumulative['returns'][780:] + 1) * \
            (expected_daily['returns'][1] + 1) - 1

        # Add daily pnl/capital_used from 1/03 to 1/04 and 1/05
        # Add daily pnl/capital_used from 1/04 to 1/05
        expected_cumulative['pnl'][390:] += expected_daily['pnl'][0]
        expected_cumulative['pnl'][780:] += expected_daily['pnl'][1]
        expected_cumulative['capital_used'][390:] += \
            expected_daily['capital_used'][0]
        expected_cumulative['capital_used'][780:] += \
            expected_daily['capital_used'][1]

        # starting_cash, starting_value are same as those of the first daily
        # packet
        expected_cumulative['starting_cash'] = \
            np.repeat(expected_daily['starting_cash'][0:1], 1170)
        expected_cumulative['starting_value'] = \
            np.repeat(expected_daily['starting_value'][0:1], 1170)

        # extra cumulative packet per day from the daily packet
        for stat in stats:
            for i in (390, 781, 1172):
                expected_cumulative[stat] = np.insert(
                    expected_cumulative[stat],
                    i,
                    expected_cumulative[stat][i-1]
                )

        for stat in stats:
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in minute_perf]),
                expected_minute[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in daily_perf]),
                expected_daily[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in cumulative_perf]),
                expected_cumulative[stat]
            )

        if change_loc == 'interday':
            self.assertEqual(
                algo.capital_change_deltas,
                {pd.Timestamp('2006-01-04', tz='UTC'): 1000.0}
            )
        else:
            self.assertEqual(
                algo.capital_change_deltas,
                {pd.Timestamp('2006-01-04 17:00', tz='UTC'): 500.0,
                 pd.Timestamp('2006-01-04 18:00', tz='UTC'): 500.0}
            )


class TestGetDatetime(WithLogger,
                      WithSimParams,
                      WithDataPortal,
                      ZiplineTestCase):
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    START_DATE = to_utc('2014-01-02 9:31')
    END_DATE = to_utc('2014-01-03 9:31')

    ASSET_FINDER_EQUITY_SIDS = 0, 1

    @parameterized.expand(
        [
            ('default', None,),
            ('utc', 'UTC',),
            ('us_east', 'US/Eastern',),
        ]
    )
    def test_get_datetime(self, name, tz):
        algo = dedent(
            """
            import pandas as pd
            from zipline.api import get_datetime

            def initialize(context):
                context.tz = {tz} or 'UTC'
                context.first_bar = True

            def handle_data(context, data):
                dt = get_datetime({tz})
                if dt.tz.zone != context.tz:
                    raise ValueError("Mismatched Zone")

                if context.first_bar:
                    if dt.tz_convert("US/Eastern").hour != 9:
                        raise ValueError("Mismatched Hour")
                    elif dt.tz_convert("US/Eastern").minute != 31:
                        raise ValueError("Mismatched Minute")

                    context.first_bar = False
            """.format(tz=repr(tz))
        )

        algo = TradingAlgorithm(
            script=algo,
            sim_params=self.sim_params,
            env=self.env,
        )
        algo.run(self.data_portal)
        self.assertFalse(algo.first_bar)


class TestTradingControls(WithSimParams, WithDataPortal, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')

    sid = 133
    sids = ASSET_FINDER_EQUITY_SIDS = 133, 134

    @classmethod
    def init_class_fixtures(cls):
        super(TestTradingControls, cls).init_class_fixtures()
        cls.asset = cls.asset_finder.retrieve_asset(cls.sid)
        cls.another_asset = cls.asset_finder.retrieve_asset(134)

    def _check_algo(self,
                    algo,
                    handle_data,
                    expected_order_count,
                    expected_exc):

        algo._handle_data = handle_data
        with self.assertRaises(expected_exc) if expected_exc else nullctx():
            algo.run(self.data_portal)
        self.assertEqual(algo.order_count, expected_order_count)

    def check_algo_succeeds(self, algo, handle_data, order_count=4):
        # Default for order_count assumes one order per handle_data call.
        self._check_algo(algo, handle_data, order_count, None)

    def check_algo_fails(self, algo, handle_data, order_count):
        self._check_algo(algo,
                         handle_data,
                         order_count,
                         TradingControlViolation)

    def test_set_max_position_size(self):

        # Buy one share four times.  Should be fine.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1
        algo = SetMaxPositionSizeAlgorithm(asset=self.asset,
                                           max_shares=10,
                                           max_notional=500.0,
                                           sim_params=self.sim_params,
                                           env=self.env)
        self.check_algo_succeeds(algo, handle_data)

        # Buy three shares four times.  Should bail on the fourth before it's
        # placed.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 3)
            algo.order_count += 1

        algo = SetMaxPositionSizeAlgorithm(asset=self.asset,
                                           max_shares=10,
                                           max_notional=500.0,
                                           sim_params=self.sim_params,
                                           env=self.env)
        self.check_algo_fails(algo, handle_data, 3)

        # Buy three shares four times. Should bail due to max_notional on the
        # third attempt.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 3)
            algo.order_count += 1

        algo = SetMaxPositionSizeAlgorithm(asset=self.asset,
                                           max_shares=10,
                                           max_notional=67.0,
                                           sim_params=self.sim_params,
                                           env=self.env)
        self.check_algo_fails(algo, handle_data, 2)

        # Set the trading control to a different sid, then BUY ALL THE THINGS!.
        # Should continue normally.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = SetMaxPositionSizeAlgorithm(asset=self.another_asset,
                                           max_shares=10,
                                           max_notional=67.0,
                                           sim_params=self.sim_params,
                                           env=self.env)
        self.check_algo_succeeds(algo, handle_data)

        # Set the trading control sid to None, then BUY ALL THE THINGS!. Should
        # fail because setting sid to None makes the control apply to all sids.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = SetMaxPositionSizeAlgorithm(max_shares=10, max_notional=61.0,
                                           sim_params=self.sim_params,
                                           env=self.env)
        self.check_algo_fails(algo, handle_data, 0)

    def test_set_asset_restrictions(self):

        def handle_data(algo, data):
            algo.could_trade = data.can_trade(algo.sid(self.sid))
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1

        # Set HistoricalRestrictions for one sid for the entire simulation,
        # and fail.
        rlm = HistoricalRestrictions([
            Restriction(
                self.sid,
                self.sim_params.start_session,
                RESTRICTION_STATES.FROZEN)
        ])
        algo = SetAssetRestrictionsAlgorithm(
            sid=self.sid,
            restrictions=rlm,
            sim_params=self.sim_params,
            env=self.env,
        )
        self.check_algo_fails(algo, handle_data, 0)
        self.assertFalse(algo.could_trade)

        # Set StaticRestrictions for one sid and fail.
        rlm = StaticRestrictions([self.sid])
        algo = SetAssetRestrictionsAlgorithm(
            sid=self.sid,
            restrictions=rlm,
            sim_params=self.sim_params,
            env=self.env,
        )
        self.check_algo_fails(algo, handle_data, 0)
        self.assertFalse(algo.could_trade)

        # just log an error on the violation if we choose not to fail.
        algo = SetAssetRestrictionsAlgorithm(
            sid=self.sid,
            restrictions=rlm,
            sim_params=self.sim_params,
            env=self.env,
            on_error='log'
        )
        with make_test_handler(self) as log_catcher:
            self.check_algo_succeeds(algo, handle_data)
        logs = [r.message for r in log_catcher.records]
        self.assertIn("Order for 100 shares of Equity(133 [A]) at "
                      "2006-01-03 21:00:00+00:00 violates trading constraint "
                      "RestrictedListOrder({})", logs)
        self.assertFalse(algo.could_trade)

        # set the restricted list to exclude the sid, and succeed
        rlm = HistoricalRestrictions([
            Restriction(
                sid,
                self.sim_params.start_session,
                RESTRICTION_STATES.FROZEN) for sid in [134, 135, 136]
        ])
        algo = SetAssetRestrictionsAlgorithm(
            sid=self.sid,
            restrictions=rlm,
            sim_params=self.sim_params,
            env=self.env,
        )
        self.check_algo_succeeds(algo, handle_data)
        self.assertTrue(algo.could_trade)

    @parameterized.expand([
        ('order_first_restricted_sid', 0),
        ('order_second_restricted_sid', 1)
    ])
    def test_set_multiple_asset_restrictions(self, name, to_order_idx):

        def handle_data(algo, data):
            algo.could_trade1 = data.can_trade(algo.sid(self.sids[0]))
            algo.could_trade2 = data.can_trade(algo.sid(self.sids[1]))
            algo.order(algo.sid(self.sids[to_order_idx]), 100)
            algo.order_count += 1

        rl1 = StaticRestrictions([self.sids[0]])
        rl2 = StaticRestrictions([self.sids[1]])
        algo = SetMultipleAssetRestrictionsAlgorithm(
            restrictions1=rl1,
            restrictions2=rl2,
            sim_params=self.sim_params,
            env=self.env,
        )
        self.check_algo_fails(algo, handle_data, 0)
        self.assertFalse(algo.could_trade1)
        self.assertFalse(algo.could_trade2)

    def test_set_do_not_order_list(self):

        def handle_data(algo, data):
            algo.could_trade = data.can_trade(algo.sid(self.sid))
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1

        rlm = [self.sid]
        algo = SetDoNotOrderListAlgorithm(
            sid=self.sid,
            restricted_list=rlm,
            sim_params=self.sim_params,
            env=self.env,
        )

        self.check_algo_fails(algo, handle_data, 0)
        self.assertFalse(algo.could_trade)

    def test_set_max_order_size(self):

        # Buy one share.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1

        algo = SetMaxOrderSizeAlgorithm(asset=self.asset,
                                        max_shares=10,
                                        max_notional=500.0,
                                        sim_params=self.sim_params,
                                        env=self.env)
        self.check_algo_succeeds(algo, handle_data)

        # Buy 1, then 2, then 3, then 4 shares.  Bail on the last attempt
        # because we exceed shares.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), algo.order_count + 1)
            algo.order_count += 1

        algo = SetMaxOrderSizeAlgorithm(asset=self.asset,
                                        max_shares=3,
                                        max_notional=500.0,
                                        sim_params=self.sim_params,
                                        env=self.env)
        self.check_algo_fails(algo, handle_data, 3)

        # Buy 1, then 2, then 3, then 4 shares.  Bail on the last attempt
        # because we exceed notional.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), algo.order_count + 1)
            algo.order_count += 1

        algo = SetMaxOrderSizeAlgorithm(asset=self.asset,
                                        max_shares=10,
                                        max_notional=40.0,
                                        sim_params=self.sim_params,
                                        env=self.env)
        self.check_algo_fails(algo, handle_data, 3)

        # Set the trading control to a different sid, then BUY ALL THE THINGS!.
        # Should continue normally.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = SetMaxOrderSizeAlgorithm(asset=self.another_asset,
                                        max_shares=1,
                                        max_notional=1.0,
                                        sim_params=self.sim_params,
                                        env=self.env)
        self.check_algo_succeeds(algo, handle_data)

        # Set the trading control sid to None, then BUY ALL THE THINGS!.
        # Should fail because not specifying a sid makes the trading control
        # apply to all sids.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = SetMaxOrderSizeAlgorithm(max_shares=1,
                                        max_notional=1.0,
                                        sim_params=self.sim_params,
                                        env=self.env)
        self.check_algo_fails(algo, handle_data, 0)

    def test_set_max_order_count(self):
        start = pd.Timestamp('2006-01-05', tz='utc')
        metadata = pd.DataFrame.from_dict(
            {
                1: {
                    'symbol': 'SYM',
                    'start_date': start,
                    'end_date': start + timedelta(days=6),
                    'exchange': "TEST",
                },
            },
            orient='index',
        )
        with TempDirectory() as tempdir, \
                tmp_trading_env(equities=metadata) as env:
            sim_params = factory.create_simulation_parameters(
                start=start,
                num_days=4,
                data_frequency='minute',
            )

            data_portal = create_data_portal(
                env.asset_finder,
                tempdir,
                sim_params,
                [1],
                self.trading_calendar,
            )

            def handle_data(algo, data):
                for i in range(5):
                    algo.order(algo.sid(1), 1)
                    algo.order_count += 1

            algo = SetMaxOrderCountAlgorithm(3, sim_params=sim_params,
                                             env=env)
            with self.assertRaises(TradingControlViolation):
                algo._handle_data = handle_data
                algo.run(data_portal)

            self.assertEqual(algo.order_count, 3)

            # This time, order 5 times twice in a single day. The last order
            # of the second batch should fail.
            def handle_data2(algo, data):
                if algo.minute_count == 0 or algo.minute_count == 100:
                    for i in range(5):
                        algo.order(algo.sid(1), 1)
                        algo.order_count += 1

                algo.minute_count += 1

            algo = SetMaxOrderCountAlgorithm(9, sim_params=sim_params,
                                             env=env)
            with self.assertRaises(TradingControlViolation):
                algo._handle_data = handle_data2
                algo.run(data_portal)

            self.assertEqual(algo.order_count, 9)

            def handle_data3(algo, data):
                if (algo.minute_count % 390) == 0:
                    for i in range(5):
                        algo.order(algo.sid(1), 1)
                        algo.order_count += 1

                algo.minute_count += 1

            # Only 5 orders are placed per day, so this should pass even
            # though in total more than 20 orders are placed.
            algo = SetMaxOrderCountAlgorithm(5, sim_params=sim_params,
                                             env=env)
            algo._handle_data = handle_data3
            algo.run(data_portal)

    def test_long_only(self):
        # Sell immediately -> fail immediately.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), -1)
            algo.order_count += 1
        algo = SetLongOnlyAlgorithm(sim_params=self.sim_params, env=self.env)
        self.check_algo_fails(algo, handle_data, 0)

        # Buy on even days, sell on odd days.  Never takes a short position, so
        # should succeed.
        def handle_data(algo, data):
            if (algo.order_count % 2) == 0:
                algo.order(algo.sid(self.sid), 1)
            else:
                algo.order(algo.sid(self.sid), -1)
            algo.order_count += 1
        algo = SetLongOnlyAlgorithm(sim_params=self.sim_params, env=self.env)
        self.check_algo_succeeds(algo, handle_data)

        # Buy on first three days, then sell off holdings.  Should succeed.
        def handle_data(algo, data):
            amounts = [1, 1, 1, -3]
            algo.order(algo.sid(self.sid), amounts[algo.order_count])
            algo.order_count += 1
        algo = SetLongOnlyAlgorithm(sim_params=self.sim_params, env=self.env)
        self.check_algo_succeeds(algo, handle_data)

        # Buy on first three days, then sell off holdings plus an extra share.
        # Should fail on the last sale.
        def handle_data(algo, data):
            amounts = [1, 1, 1, -4]
            algo.order(algo.sid(self.sid), amounts[algo.order_count])
            algo.order_count += 1
        algo = SetLongOnlyAlgorithm(sim_params=self.sim_params, env=self.env)
        self.check_algo_fails(algo, handle_data, 3)

    def test_register_post_init(self):

        def initialize(algo):
            algo.initialized = True

        def handle_data(algo, data):
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_max_position_size(self.sid, 1, 1)
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_max_order_size(self.sid, 1, 1)
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_max_order_count(1)
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_long_only()

        algo = TradingAlgorithm(initialize=initialize,
                                handle_data=handle_data,
                                sim_params=self.sim_params,
                                env=self.env)
        algo.run(self.data_portal)

    def test_asset_date_bounds(self):
        metadata = pd.DataFrame([{
            'symbol': 'SYM',
            'start_date': self.sim_params.start_session,
            'end_date': '2020-01-01',
            'exchange': "TEST",
            'sid': 999,
        }])
        with TempDirectory() as tempdir, \
                tmp_trading_env(equities=metadata) as env:
            algo = SetAssetDateBoundsAlgorithm(
                sim_params=self.sim_params,
                env=env,
            )
            data_portal = create_data_portal(
                env.asset_finder,
                tempdir,
                self.sim_params,
                [999],
                self.trading_calendar,
            )
            algo.run(data_portal)

        metadata = pd.DataFrame([{
            'symbol': 'SYM',
            'start_date': '1989-01-01',
            'end_date': '1990-01-01',
            'exchange': "TEST",
            'sid': 999,
        }])
        with TempDirectory() as tempdir, \
                tmp_trading_env(equities=metadata) as env:
            data_portal = create_data_portal(
                env.asset_finder,
                tempdir,
                self.sim_params,
                [999],
                self.trading_calendar,
            )
            algo = SetAssetDateBoundsAlgorithm(
                sim_params=self.sim_params,
                env=env,
            )
            with self.assertRaises(TradingControlViolation):
                algo.run(data_portal)

        metadata = pd.DataFrame([{
            'symbol': 'SYM',
            'start_date': '2020-01-01',
            'end_date': '2021-01-01',
            'exchange': "TEST",
            'sid': 999,
        }])
        with TempDirectory() as tempdir, \
                tmp_trading_env(equities=metadata) as env:
            data_portal = create_data_portal(
                env.asset_finder,
                tempdir,
                self.sim_params,
                [999],
                self.trading_calendar,
            )
            algo = SetAssetDateBoundsAlgorithm(
                sim_params=self.sim_params,
                env=env,
            )
            with self.assertRaises(TradingControlViolation):
                algo.run(data_portal)


class TestAccountControls(WithDataPortal, WithSimParams, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')

    sidint, = ASSET_FINDER_EQUITY_SIDS = (133,)

    @classmethod
    def make_equity_daily_bar_data(cls):
        return trades_by_sid_to_dfs(
            {
                cls.sidint: factory.create_trade_history(
                    cls.sidint,
                    [10.0, 10.0, 11.0, 11.0],
                    [100, 100, 100, 300],
                    timedelta(days=1),
                    cls.sim_params,
                    cls.trading_calendar,
                ),
            },
            index=cls.sim_params.sessions,
        )

    def _check_algo(self,
                    algo,
                    handle_data,
                    expected_exc):

        algo._handle_data = handle_data
        with self.assertRaises(expected_exc) if expected_exc else nullctx():
            algo.run(self.data_portal)

    def check_algo_succeeds(self, algo, handle_data):
        # Default for order_count assumes one order per handle_data call.
        self._check_algo(algo, handle_data, None)

    def check_algo_fails(self, algo, handle_data):
        self._check_algo(algo,
                         handle_data,
                         AccountControlViolation)

    def test_set_max_leverage(self):

        # Set max leverage to 0 so buying one share fails.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sidint), 1)

        algo = SetMaxLeverageAlgorithm(0, sim_params=self.sim_params,
                                       env=self.env)
        self.check_algo_fails(algo, handle_data)

        # Set max leverage to 1 so buying one share passes
        def handle_data(algo, data):
            algo.order(algo.sid(self.sidint), 1)

        algo = SetMaxLeverageAlgorithm(1,  sim_params=self.sim_params,
                                       env=self.env)
        self.check_algo_succeeds(algo, handle_data)


# FIXME re-implement this testcase in q2
# class TestClosePosAlgo(TestCase):
#     def setUp(self):
#         self.env = TradingEnvironment()
#         self.days = self.env.trading_days[:5]
#         self.panel = pd.Panel({1: pd.DataFrame({
#             'price': [1, 1, 2, 4, 8], 'volume': [1e9, 1e9, 1e9, 1e9, 0],
#             'type': [DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.CLOSE_POSITION]},
#             index=self.days)
#         })
#         self.no_close_panel = pd.Panel({1: pd.DataFrame({
#             'price': [1, 1, 2, 4, 8], 'volume': [1e9, 1e9, 1e9, 1e9, 1e9],
#             'type': [DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE,
#                      DATASOURCE_TYPE.TRADE]},
#             index=self.days)
#         })
#
#     def test_close_position_equity(self):
#         metadata = {1: {'symbol': 'TEST',
#                         'end_date': self.days[4]}}
#         self.env.write_data(equities_data=metadata)
#         algo = TestAlgorithm(sid=1, amount=1, order_count=1,
#                              commission=PerShare(0),
#                              env=self.env)
#         data = DataPanelSource(self.panel)
#
#         # Check results
#         expected_positions = [0, 1, 1, 1, 0]
#         expected_pnl = [0, 0, 1, 2, 4]
#         results = algo.run(data)
#         self.check_algo_positions(results, expected_positions)
#         self.check_algo_pnl(results, expected_pnl)
#
#     def test_close_position_future(self):
#         metadata = {1: {'symbol': 'TEST'}}
#         self.env.write_data(futures_data=metadata)
#         algo = TestAlgorithm(sid=1, amount=1, order_count=1,
#                              commission=PerShare(0),
#                              env=self.env)
#         data = DataPanelSource(self.panel)
#
#         # Check results
#         expected_positions = [0, 1, 1, 1, 0]
#         expected_pnl = [0, 0, 1, 2, 4]
#         results = algo.run(data)
#         self.check_algo_pnl(results, expected_pnl)
#         self.check_algo_positions(results, expected_positions)
#
#     def test_auto_close_future(self):
#         metadata = {1: {'symbol': 'TEST',
#                         'auto_close_date': self.env.trading_days[4]}}
#         self.env.write_data(futures_data=metadata)
#         algo = TestAlgorithm(sid=1, amount=1, order_count=1,
#                              commission=PerShare(0),
#                              env=self.env)
#         data = DataPanelSource(self.no_close_panel)
#
#         # Check results
#         results = algo.run(data)
#
#         expected_positions = [0, 1, 1, 1, 0]
#         self.check_algo_positions(results, expected_positions)
#
#         expected_pnl = [0, 0, 1, 2, 0]
#         self.check_algo_pnl(results, expected_pnl)
#
#     def check_algo_pnl(self, results, expected_pnl):
#         np.testing.assert_array_almost_equal(results.pnl, expected_pnl)
#
#     def check_algo_positions(self, results, expected_positions):
#         for i, amount in enumerate(results.positions):
#             if amount:
#                 actual_position = amount[0]['amount']
#             else:
#                 actual_position = 0
#
#             self.assertEqual(
#                 actual_position, expected_positions[i],
#                 "position for day={0} not equal, actual={1}, expected={2}".
#                 format(i, actual_position, expected_positions[i]))


class TestFutureFlip(WithDataPortal, WithSimParams, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-09', tz='utc')
    END_DATE = pd.Timestamp('2006-01-10', tz='utc')
    sid, = ASSET_FINDER_EQUITY_SIDS = (1,)

    @classmethod
    def make_equity_daily_bar_data(cls):
        return trades_by_sid_to_dfs(
            {
                cls.sid: factory.create_trade_history(
                    cls.sid,
                    [1, 2],
                    [1e9, 1e9],
                    timedelta(days=1),
                    cls.sim_params,
                    cls.trading_calendar,
                ),
            },
            index=cls.sim_params.sessions,
        )

    @skip('broken in zipline 1.0.0')
    def test_flip_algo(self):
        metadata = {1: {'symbol': 'TEST',
                        'start_date': self.sim_params.trading_days[0],
                        'end_date': self.trading_calendar.next_session_label(
                            self.sim_params.sessions[-1]
                        ),
                        'multiplier': 5}}

        self.env.write_data(futures_data=metadata)

        algo = FutureFlipAlgo(sid=1, amount=1, env=self.env,
                              commission=PerShare(0),
                              order_count=0,  # not applicable but required
                              sim_params=self.sim_params)

        results = algo.run(self.data_portal)

        expected_positions = [0, 1, -1]
        self.check_algo_positions(results, expected_positions)

        expected_pnl = [0, 5, -10]
        self.check_algo_pnl(results, expected_pnl)

    def check_algo_pnl(self, results, expected_pnl):
        np.testing.assert_array_almost_equal(results.pnl, expected_pnl)

    def check_algo_positions(self, results, expected_positions):
        for i, amount in enumerate(results.positions):
            if amount:
                actual_position = amount[0]['amount']
            else:
                actual_position = 0

            self.assertEqual(
                actual_position, expected_positions[i],
                "position for day={0} not equal, actual={1}, expected={2}".
                format(i, actual_position, expected_positions[i]))


class TestFuturesAlgo(WithDataPortal, WithSimParams, ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-06', tz='utc')
    END_DATE = pd.Timestamp('2016-01-07', tz='utc')
    FUTURE_MINUTE_BAR_START_DATE = pd.Timestamp('2016-01-05', tz='UTC')

    SIM_PARAMS_DATA_FREQUENCY = 'minute'

    TRADING_CALENDAR_STRS = ('us_futures',)
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                1: {
                    'symbol': 'CLG16',
                    'root_symbol': 'CL',
                    'start_date': pd.Timestamp('2015-12-01', tz='UTC'),
                    'notice_date': pd.Timestamp('2016-01-20', tz='UTC'),
                    'expiration_date': pd.Timestamp('2016-02-19', tz='UTC'),
                    'auto_close_date': pd.Timestamp('2016-01-18', tz='UTC'),
                    'exchange': 'TEST',
                },
            },
            orient='index',
        )

    def test_futures_history(self):
        algo_code = dedent(
            """
            from datetime import time
            from zipline.api import (
                date_rules,
                get_datetime,
                schedule_function,
                sid,
                time_rules,
            )

            def initialize(context):
                context.history_values = []

                schedule_function(
                    make_history_call,
                    date_rules.every_day(),
                    time_rules.market_open(),
                )

                schedule_function(
                    check_market_close_time,
                    date_rules.every_day(),
                    time_rules.market_close(),
                )

            def make_history_call(context, data):
                # Ensure that the market open is 6:31am US/Eastern.
                open_time = get_datetime().tz_convert('US/Eastern').time()
                assert open_time == time(6, 31)
                context.history_values.append(
                    data.history(sid(1), 'close', 5, '1m'),
                )

            def check_market_close_time(context, data):
                # Ensure that this function is called at 4:59pm US/Eastern.
                # By default, `market_close()` uses an offset of 1 minute.
                close_time = get_datetime().tz_convert('US/Eastern').time()
                assert close_time == time(16, 59)
            """
        )

        algo = TradingAlgorithm(
            script=algo_code,
            sim_params=self.sim_params,
            env=self.env,
            trading_calendar=get_calendar('us_futures'),
        )
        algo.run(self.data_portal)

        # Assert that we were able to retrieve history data for minutes outside
        # of the 6:31am US/Eastern to 5:00pm US/Eastern futures open times.
        np.testing.assert_array_equal(
            algo.history_values[0].index,
            pd.date_range(
                '2016-01-06 6:27',
                '2016-01-06 6:31',
                freq='min',
                tz='US/Eastern',
            ),
        )
        np.testing.assert_array_equal(
            algo.history_values[1].index,
            pd.date_range(
                '2016-01-07 6:27',
                '2016-01-07 6:31',
                freq='min',
                tz='US/Eastern',
            ),
        )

        # Expected prices here are given by the range values created by the
        # default `make_future_minute_bar_data` method.
        np.testing.assert_array_equal(
            algo.history_values[0].values, list(map(float, range(2196, 2201))),
        )
        np.testing.assert_array_equal(
            algo.history_values[1].values, list(map(float, range(3636, 3641))),
        )


class TestTradingAlgorithm(ZiplineTestCase):
    def test_analyze_called(self):
        self.perf_ref = None

        def initialize(context):
            pass

        def handle_data(context, data):
            pass

        def analyze(context, perf):
            self.perf_ref = perf

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
        )

        with empty_trading_env() as env:
            data_portal = FakeDataPortal(env)
            results = algo.run(data_portal)

        self.assertIs(results, self.perf_ref)


class TestOrderCancelation(WithDataPortal,
                           WithSimParams,
                           ZiplineTestCase):

    START_DATE = pd.Timestamp('2016-01-05', tz='utc')
    END_DATE = pd.Timestamp('2016-01-07', tz='utc')

    ASSET_FINDER_EQUITY_SIDS = (1,)
    ASSET_FINDER_EQUITY_SYMBOLS = ('ASSET1',)

    code = dedent(
        """
        from zipline.api import (
            sid, order, set_slippage, slippage, VolumeShareSlippage,
            set_cancel_policy, cancel_policy, EODCancel
        )


        def initialize(context):
            set_slippage(
                slippage.VolumeShareSlippage(
                    volume_limit=1,
                    price_impact=0
                )
            )

            {0}
            context.ordered = False


        def handle_data(context, data):
            if not context.ordered:
                order(sid(1), {1})
                context.ordered = True
        """,
    )

    @classmethod
    def make_equity_minute_bar_data(cls):
        asset_minutes = \
            cls.trading_calendar.minutes_for_sessions_in_range(
                cls.sim_params.start_session,
                cls.sim_params.end_session,
            )

        minutes_count = len(asset_minutes)
        minutes_arr = np.arange(1, 1 + minutes_count)

        # normal test data, but volume is pinned at 1 share per minute
        yield 1, pd.DataFrame(
            {
                'open': minutes_arr + 1,
                'high': minutes_arr + 2,
                'low': minutes_arr - 1,
                'close': minutes_arr,
                'volume': np.full(minutes_count, 1.0),
            },
            index=asset_minutes,
        )

    @classmethod
    def make_equity_daily_bar_data(cls):
        yield 1, pd.DataFrame(
            {
                'open': np.full(3, 1, dtype=np.float64),
                'high': np.full(3, 1, dtype=np.float64),
                'low': np.full(3, 1, dtype=np.float64),
                'close': np.full(3, 1, dtype=np.float64),
                'volume': np.full(3, 1, dtype=np.float64),
            },
            index=cls.sim_params.sessions,
        )

    def prep_algo(self, cancelation_string, data_frequency="minute",
                  amount=1000, minute_emission=False):
        code = self.code.format(cancelation_string, amount)
        algo = TradingAlgorithm(
            script=code,
            env=self.env,
            sim_params=SimulationParameters(
                start_session=self.sim_params.start_session,
                end_session=self.sim_params.end_session,
                trading_calendar=self.trading_calendar,
                data_frequency=data_frequency,
                emission_rate='minute' if minute_emission else 'daily'
            )
        )

        return algo

    @parameter_space(
        direction=[1, -1],
        minute_emission=[True, False]
    )
    def test_eod_order_cancel_minute(self, direction, minute_emission):
        """
        Test that EOD order cancel works in minute mode for both shorts and
        longs, and both daily emission and minute emission
        """
        # order 1000 shares of asset1.  the volume is only 1 share per bar,
        # so the order should be cancelled at the end of the day.
        algo = self.prep_algo(
            "set_cancel_policy(cancel_policy.EODCancel())",
            amount=np.copysign(1000, direction),
            minute_emission=minute_emission
        )

        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run(self.data_portal)

            for daily_positions in results.positions:
                self.assertEqual(1, len(daily_positions))
                self.assertEqual(
                    np.copysign(389, direction),
                    daily_positions[0]["amount"],
                )
                self.assertEqual(1, results.positions[0][0]["sid"].sid)

            # should be an order on day1, but no more orders afterwards
            np.testing.assert_array_equal([1, 0, 0],
                                          list(map(len, results.orders)))

            # should be 389 txns on day 1, but no more afterwards
            np.testing.assert_array_equal([389, 0, 0],
                                          list(map(len, results.transactions)))

            the_order = results.orders[0][0]

            self.assertEqual(ORDER_STATUS.CANCELLED, the_order["status"])
            self.assertEqual(np.copysign(389, direction), the_order["filled"])

            warnings = [record for record in log_catcher.records if
                        record.level == WARNING]

            self.assertEqual(1, len(warnings))

            if direction == 1:
                self.assertEqual(
                    "Your order for 1000 shares of ASSET1 has been partially "
                    "filled. 389 shares were successfully purchased. "
                    "611 shares were not filled by the end of day and "
                    "were canceled.",
                    str(warnings[0].message)
                )
            elif direction == -1:
                self.assertEqual(
                    "Your order for -1000 shares of ASSET1 has been partially "
                    "filled. 389 shares were successfully sold. "
                    "611 shares were not filled by the end of day and "
                    "were canceled.",
                    str(warnings[0].message)
                )

    def test_default_cancelation_policy(self):
        algo = self.prep_algo("")

        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run(self.data_portal)

            # order stays open throughout simulation
            np.testing.assert_array_equal([1, 1, 1],
                                          list(map(len, results.orders)))

            # one txn per minute.  389 the first day (since no order until the
            # end of the first minute).  390 on the second day.  221 on the
            # the last day, sum = 1000.
            np.testing.assert_array_equal([389, 390, 221],
                                          list(map(len, results.transactions)))

            self.assertFalse(log_catcher.has_warnings)

    def test_eod_order_cancel_daily(self):
        # in daily mode, EODCancel does nothing.
        algo = self.prep_algo(
            "set_cancel_policy(cancel_policy.EODCancel())",
            "daily"
        )

        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run(self.data_portal)

            # order stays open throughout simulation
            np.testing.assert_array_equal([1, 1, 1],
                                          list(map(len, results.orders)))

            # one txn per day
            np.testing.assert_array_equal([0, 1, 1],
                                          list(map(len, results.transactions)))

            self.assertFalse(log_catcher.has_warnings)


class TestEquityAutoClose(WithTmpDir, WithTradingCalendars, ZiplineTestCase):
    """
    Tests if delisted equities are properly removed from a portfolio holding
    positions in said equities.
    """
    @classmethod
    def init_class_fixtures(cls):
        super(TestEquityAutoClose, cls).init_class_fixtures()
        trading_sessions = cls.trading_calendar.all_sessions
        start_date = pd.Timestamp('2015-01-05', tz='UTC')
        start_date_loc = trading_sessions.get_loc(start_date)
        test_duration = 7
        cls.test_days = trading_sessions[
            start_date_loc:start_date_loc + test_duration
        ]
        cls.first_asset_expiration = cls.test_days[2]

    def make_data(self, auto_close_delta, frequency,
                  capital_base=1.0e5):

        asset_info = make_jagged_equity_info(
            num_assets=3,
            start_date=self.test_days[0],
            first_end=self.first_asset_expiration,
            frequency=self.trading_calendar.day,
            periods_between_ends=2,
            auto_close_delta=auto_close_delta,
        )

        sids = asset_info.index

        env = self.enter_instance_context(tmp_trading_env(equities=asset_info))

        if frequency == 'daily':
            dates = self.test_days
            trade_data_by_sid = make_trade_data_for_asset_info(
                dates=dates,
                asset_info=asset_info,
                price_start=10,
                price_step_by_sid=10,
                price_step_by_date=1,
                volume_start=100,
                volume_step_by_sid=100,
                volume_step_by_date=10,
                frequency=frequency
            )
            path = self.tmpdir.getpath("testdaily.bcolz")
            writer = BcolzDailyBarWriter(
                path, self.trading_calendar, dates[0], dates[-1]
            )
            writer.write(iteritems(trade_data_by_sid))
            reader = BcolzDailyBarReader(path)
            data_portal = DataPortal(
                env.asset_finder, self.trading_calendar,
                first_trading_day=reader.first_trading_day,
                equity_daily_reader=reader,
            )
        elif frequency == 'minute':
            dates = self.trading_calendar.minutes_for_sessions_in_range(
                self.test_days[0],
                self.test_days[-1],
            )
            writer = BcolzMinuteBarWriter(
                self.tmpdir.path,
                self.trading_calendar,
                self.test_days[0],
                self.test_days[-1],
                US_EQUITIES_MINUTES_PER_DAY
            )
            trade_data_by_sid = make_trade_data_for_asset_info(
                writer=writer,
                dates=dates,
                asset_info=asset_info,
                price_start=10,
                price_step_by_sid=10,
                price_step_by_date=1,
                volume_start=100,
                volume_step_by_sid=100,
                volume_step_by_date=10,
                frequency=frequency
            )
            reader = BcolzMinuteBarReader(self.tmpdir.path)
            data_portal = DataPortal(
                env.asset_finder, self.trading_calendar,
                first_trading_day=reader.first_trading_day,
                equity_minute_reader=reader,
            )
        else:
            self.fail("Unknown frequency in make_data: %r" % frequency)

        assets = env.asset_finder.retrieve_all(sids)

        sim_params = factory.create_simulation_parameters(
            start=self.test_days[0],
            end=self.test_days[-1],
            data_frequency=frequency,
            emission_rate=frequency,
            capital_base=capital_base,
        )

        if frequency == 'daily':
            final_prices = {
                asset.sid: trade_data_by_sid[asset.sid].
                loc[asset.end_date].close
                for asset in assets
            }
        else:
            final_prices = {
                asset.sid: trade_data_by_sid[asset.sid].loc[
                    self.trading_calendar.open_and_close_for_session(
                        asset.end_date
                    )[1]
                ].close
                for asset in assets
            }

        TestData = namedtuple(
            'TestData',
            [
                'asset_info',
                'assets',
                'env',
                'data_portal',
                'final_prices',
                'trade_data_by_sid',
                'sim_params'
            ],
        )
        return TestData(
            asset_info=asset_info,
            assets=assets,
            env=env,
            data_portal=data_portal,
            final_prices=final_prices,
            trade_data_by_sid=trade_data_by_sid,
            sim_params=sim_params
        )

    def prices_on_tick(self, trades_by_sid, row):
        return [trades.iloc[row].close
                for trades in itervalues(trades_by_sid)]

    def default_initialize(self):
        """
        Initialize function shared between test algos.
        """
        def initialize(context):
            context.ordered = False
            context.set_commission(PerShare(0, 0))
            context.set_slippage(FixedSlippage(spread=0))
            context.num_positions = []
            context.cash = []

        return initialize

    def default_handle_data(self, assets, order_size):
        """
        Handle data function shared between test algos.
        """
        def handle_data(context, data):
            if not context.ordered:
                for asset in assets:
                    context.order(asset, order_size)
                context.ordered = True

            context.cash.append(context.portfolio.cash)
            context.num_positions.append(len(context.portfolio.positions))

        return handle_data

    @parameter_space(
        order_size=[10, -10],
        capital_base=[0, 100000],
        auto_close_lag=[1, 2],
    )
    def test_daily_delisted_equities(self,
                                     order_size,
                                     capital_base,
                                     auto_close_lag):
        """
        Make sure that after an equity gets delisted, our portfolio holds the
        correct number of equities and correct amount of cash.
        """
        auto_close_delta = self.trading_calendar.day * auto_close_lag
        resources = self.make_data(auto_close_delta, 'daily', capital_base)

        assets = resources.assets
        sids = [asset.sid for asset in assets]
        final_prices = resources.final_prices

        # Prices at which we expect our orders to be filled.
        initial_fill_prices = \
            self.prices_on_tick(resources.trade_data_by_sid, 1)
        cost_basis = sum(initial_fill_prices) * order_size

        # Last known prices of assets that will be auto-closed.
        fp0 = final_prices[0]
        fp1 = final_prices[1]

        algo = TradingAlgorithm(
            initialize=self.default_initialize(),
            handle_data=self.default_handle_data(assets, order_size),
            env=resources.env,
            sim_params=resources.sim_params
        )
        output = algo.run(resources.data_portal)

        initial_cash = capital_base
        after_fills = initial_cash - cost_basis
        after_first_auto_close = after_fills + fp0 * (order_size)
        after_second_auto_close = after_first_auto_close + fp1 * (order_size)

        if auto_close_lag == 1:
            # Day 1: Order 10 shares of each equity; there are 3 equities.
            # Day 2: Order goes through at the day 2 price of each equity.
            # Day 3: End date of Equity 0.
            # Day 4: Auto close date of Equity 0. Add cash == (fp0 * size).
            # Day 5: End date of Equity 1.
            # Day 6: Auto close date of Equity 1. Add cash == (fp1 * size).
            # Day 7: End date of Equity 2 and last day of backtest; no changes.
            expected_cash = [
                initial_cash,
                after_fills,
                after_fills,
                after_first_auto_close,
                after_first_auto_close,
                after_second_auto_close,
                after_second_auto_close,
            ]
            expected_num_positions = [0, 3, 3, 2, 2, 1, 1]
        elif auto_close_lag == 2:
            # Day 1: Order 10 shares of each equity; there are 3 equities.
            # Day 2: Order goes through at the day 2 price of each equity.
            # Day 3: End date of Equity 0.
            # Day 4: Nothing happens.
            # Day 5: End date of Equity 1. Auto close of equity 0.
            #        Add cash == (fp0 * size).
            # Day 6: Nothing happens.
            # Day 7: End date of Equity 2 and auto-close date of Equity 1.
            #        Add cash equal to (fp1 * size).
            expected_cash = [
                initial_cash,
                after_fills,
                after_fills,
                after_fills,
                after_first_auto_close,
                after_first_auto_close,
                after_second_auto_close,
            ]
            expected_num_positions = [0, 3, 3, 3, 2, 2, 1]
        else:
            self.fail(
                "Don't know about auto_close lags other than 1 or 2. "
                "Add test answers please!"
            )

        # Check expected cash.
        self.assertEqual(algo.cash, expected_cash)
        self.assertEqual(expected_cash, list(output['ending_cash']))

        # Check expected long/short counts.
        # We have longs if order_size > 0.
        # We have shrots if order_size > 0.
        self.assertEqual(algo.num_positions, expected_num_positions)
        if order_size > 0:
            self.assertEqual(
                expected_num_positions,
                list(output['longs_count']),
            )
            self.assertEqual(
                [0] * len(self.test_days),
                list(output['shorts_count']),
            )
        else:
            self.assertEqual(
                expected_num_positions,
                list(output['shorts_count']),
            )
            self.assertEqual(
                [0] * len(self.test_days),
                list(output['longs_count']),
            )

        # Check expected transactions.
        # We should have a transaction of order_size shares per sid.
        transactions = output['transactions']
        initial_fills = transactions.iloc[1]
        self.assertEqual(len(initial_fills), len(assets))

        last_minute_of_session = \
            self.trading_calendar.open_and_close_for_session(
                self.test_days[1]
            )[1]

        for sid, txn in zip(sids, initial_fills):
            self.assertDictContainsSubset(
                {
                    'amount': order_size,
                    'commission': None,
                    'dt': last_minute_of_session,
                    'price': initial_fill_prices[sid],
                    'sid': sid,
                },
                txn,
            )
            # This will be a UUID.
            self.assertIsInstance(txn['order_id'], str)

        def transactions_for_date(date):
            return transactions.iloc[self.test_days.get_loc(date)]

        # We should have exactly one auto-close transaction on the close date
        # of asset 0.
        (first_auto_close_transaction,) = transactions_for_date(
            assets[0].auto_close_date
        )
        self.assertEqual(
            first_auto_close_transaction,
            {
                'amount': -order_size,
                'commission': 0.0,
                'dt': assets[0].auto_close_date,
                'price': fp0,
                'sid': sids[0],
                'order_id': None,  # Auto-close txns emit Nones for order_id.
            },
        )

        (second_auto_close_transaction,) = transactions_for_date(
            assets[1].auto_close_date
        )
        self.assertEqual(
            second_auto_close_transaction,
            {
                'amount': -order_size,
                'commission': 0.0,
                'dt': assets[1].auto_close_date,
                'price': fp1,
                'sid': sids[1],
                'order_id': None,  # Auto-close txns emit Nones for order_id.
            },
        )

    def test_cancel_open_orders(self):
        """
        Test that any open orders for an equity that gets delisted are
        canceled.  Unless an equity is auto closed, any open orders for that
        equity will persist indefinitely.
        """
        auto_close_delta = self.trading_calendar.day
        resources = self.make_data(auto_close_delta, 'daily')
        env = resources.env
        assets = resources.assets

        first_asset_end_date = assets[0].end_date
        first_asset_auto_close_date = assets[0].auto_close_date

        def initialize(context):
            pass

        def handle_data(context, data):
            # The only order we place in this test should never be filled.
            assert (
                context.portfolio.cash == context.portfolio.starting_cash
            )

            today_session = self.trading_calendar.minute_to_session_label(
                context.get_datetime()
            )

            if today_session == first_asset_end_date:
                # Equity 0 will no longer exist tomorrow, so this order will
                # never be filled.
                assert len(context.get_open_orders()) == 0
                context.order(context.sid(0), 10)
                assert len(context.get_open_orders()) == 1
            elif today_session == first_asset_auto_close_date:
                assert len(context.get_open_orders()) == 0

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            env=env,
            sim_params=resources.sim_params
        )
        results = algo.run(resources.data_portal)

        orders = results['orders']

        def orders_for_date(date):
            return orders.iloc[self.test_days.get_loc(date)]

        original_open_orders = orders_for_date(first_asset_end_date)
        assert len(original_open_orders) == 1

        last_close_for_asset = \
            algo.trading_calendar.open_and_close_for_session(
                first_asset_end_date
            )[1]

        self.assertDictContainsSubset(
            {
                'amount': 10,
                'commission': 0,
                'created': last_close_for_asset,
                'dt': last_close_for_asset,
                'sid': assets[0],
                'status': ORDER_STATUS.OPEN,
                'filled': 0,
            },
            original_open_orders[0],
        )

        orders_after_auto_close = orders_for_date(first_asset_auto_close_date)
        assert len(orders_after_auto_close) == 1
        self.assertDictContainsSubset(
            {
                'amount': 10,
                'commission': 0,
                'created': last_close_for_asset,
                'dt': first_asset_auto_close_date,
                'sid': assets[0],
                'status': ORDER_STATUS.CANCELLED,
                'filled': 0,
            },
            orders_after_auto_close[0],
        )

    def test_minutely_delisted_equities(self):
        resources = self.make_data(self.trading_calendar.day, 'minute')

        env = resources.env
        assets = resources.assets
        sids = [a.sid for a in assets]
        final_prices = resources.final_prices
        backtest_minutes = resources.trade_data_by_sid[0].index.tolist()

        order_size = 10

        capital_base = 100000
        algo = TradingAlgorithm(
            initialize=self.default_initialize(),
            handle_data=self.default_handle_data(assets, order_size),
            env=env,
            sim_params=resources.sim_params,
            data_frequency='minute',
        )

        output = algo.run(resources.data_portal)
        initial_fill_prices = \
            self.prices_on_tick(resources.trade_data_by_sid, 1)
        cost_basis = sum(initial_fill_prices) * order_size

        # Last known prices of assets that will be auto-closed.
        fp0 = final_prices[0]
        fp1 = final_prices[1]

        initial_cash = capital_base
        after_fills = initial_cash - cost_basis
        after_first_auto_close = after_fills + fp0 * (order_size)
        after_second_auto_close = after_first_auto_close + fp1 * (order_size)

        expected_cash = [initial_cash]
        expected_position_counts = [0]

        # We have the rest of the first sim day, plus the second and third
        # days' worth of minutes with cash spent.
        expected_cash.extend([after_fills] * (389 + 390 + 390))
        expected_position_counts.extend([3] * (389 + 390 + 390))

        # We then have two days with the cash refunded from asset 0.
        expected_cash.extend([after_first_auto_close] * (390 + 390))
        expected_position_counts.extend([2] * (390 + 390))

        # We then have two days with cash refunded from asset 1
        expected_cash.extend([after_second_auto_close] * (390 + 390))
        expected_position_counts.extend([1] * (390 + 390))

        # Check list lengths first to avoid expensive comparison
        self.assertEqual(len(algo.cash), len(expected_cash))
        # TODO find more efficient way to compare these lists
        self.assertEqual(algo.cash, expected_cash)
        self.assertEqual(
            list(output['ending_cash']),
            [
                after_fills,
                after_fills,
                after_fills,
                after_first_auto_close,
                after_first_auto_close,
                after_second_auto_close,
                after_second_auto_close,
            ],
        )

        self.assertEqual(algo.num_positions, expected_position_counts)
        self.assertEqual(
            list(output['longs_count']),
            [3, 3, 3, 2, 2, 1, 1],
        )

        # Check expected transactions.
        # We should have a transaction of order_size shares per sid.
        transactions = output['transactions']

        # Note that the transactions appear on the first day rather than the
        # second in minute mode, because the fills happen on the second tick of
        # the backtest, which is still on the first day in minute mode.
        initial_fills = transactions.iloc[0]
        self.assertEqual(len(initial_fills), len(assets))
        for sid, txn in zip(sids, initial_fills):
            self.assertDictContainsSubset(
                {
                    'amount': order_size,
                    'commission': None,
                    'dt': backtest_minutes[1],
                    'price': initial_fill_prices[sid],
                    'sid': sid,
                },
                txn,
            )
            # This will be a UUID.
            self.assertIsInstance(txn['order_id'], str)

        def transactions_for_date(date):
            return transactions.iloc[self.test_days.get_loc(date)]

        # We should have exactly one auto-close transaction on the close date
        # of asset 0.
        (first_auto_close_transaction,) = transactions_for_date(
            assets[0].auto_close_date
        )
        self.assertEqual(
            first_auto_close_transaction,
            {
                'amount': -order_size,
                'commission': 0.0,
                'dt': assets[0].auto_close_date,
                'price': fp0,
                'sid': sids[0],
                'order_id': None,  # Auto-close txns emit Nones for order_id.
            },
        )

        (second_auto_close_transaction,) = transactions_for_date(
            assets[1].auto_close_date
        )
        self.assertEqual(
            second_auto_close_transaction,
            {
                'amount': -order_size,
                'commission': 0.0,
                'dt': assets[1].auto_close_date,
                'price': fp1,
                'sid': sids[1],
                'order_id': None,  # Auto-close txns emit Nones for order_id.
            },
        )


class TestOrderAfterDelist(WithTradingEnvironment, ZiplineTestCase):
    start = pd.Timestamp('2016-01-05', tz='utc')
    day_1 = pd.Timestamp('2016-01-06', tz='utc')
    day_4 = pd.Timestamp('2016-01-11', tz='utc')
    end = pd.Timestamp('2016-01-15', tz='utc')

    @classmethod
    def make_equity_info(cls):
        return pd.DataFrame.from_dict(
            {
                # Asset whose auto close date is after its end date.
                1: {
                    'start_date': cls.start,
                    'end_date': cls.day_1,
                    'auto_close_date': cls.day_4,
                    'symbol': "ASSET1",
                    'exchange': "TEST",
                },
                # Asset whose auto close date is before its end date.
                2: {
                    'start_date': cls.start,
                    'end_date': cls.day_4,
                    'auto_close_date': cls.day_1,
                    'symbol': 'ASSET2',
                    'exchange': 'TEST',
                },
            },
            orient='index',
        )

    @classmethod
    def init_class_fixtures(cls):
        super(TestOrderAfterDelist, cls).init_class_fixtures()
        cls.data_portal = FakeDataPortal(cls.env)

    @parameterized.expand([
        ('auto_close_after_end_date', 1),
        ('auto_close_before_end_date', 2),
    ])
    def test_order_in_quiet_period(self, name, sid):
        asset = self.asset_finder.retrieve_asset(sid)

        algo_code = dedent("""
        from zipline.api import (
            sid,
            order,
            order_value,
            order_percent,
            order_target,
            order_target_percent,
            order_target_value
        )

        def initialize(context):
            pass

        def handle_data(context, data):
            order(sid({sid}), 1)
            order_value(sid({sid}), 100)
            order_percent(sid({sid}), 0.5)
            order_target(sid({sid}), 50)
            order_target_percent(sid({sid}), 0.5)
            order_target_value(sid({sid}), 50)
        """).format(sid=sid)

        # run algo from 1/6 to 1/7
        algo = TradingAlgorithm(
            script=algo_code,
            env=self.env,
            sim_params=SimulationParameters(
                start_session=pd.Timestamp("2016-01-06", tz='UTC'),
                end_session=pd.Timestamp("2016-01-07", tz='UTC'),
                trading_calendar=self.trading_calendar,
                data_frequency="minute"
            )
        )

        with make_test_handler(self) as log_catcher:
            algo.run(self.data_portal)

            warnings = [r for r in log_catcher.records
                        if r.level == logbook.WARNING]

            # one warning per order on the second day
            self.assertEqual(6 * 390, len(warnings))

            for w in warnings:
                expected_message = (
                    'Cannot place order for ASSET{sid}, as it has de-listed. '
                    'Any existing positions for this asset will be liquidated '
                    'on {date}.'.format(sid=sid, date=asset.auto_close_date)
                )
                self.assertEqual(expected_message, w.message)


class AlgoInputValidationTestCase(ZiplineTestCase):

    def test_reject_passing_both_api_methods_and_script(self):
        script = dedent(
            """
            def initialize(context):
                pass

            def handle_data(context, data):
                pass

            def before_trading_start(context, data):
                pass

            def analyze(context, results):
                pass
            """
        )
        for method in ('initialize',
                       'handle_data',
                       'before_trading_start',
                       'analyze'):

            with self.assertRaises(ValueError):
                TradingAlgorithm(
                    script=script,
                    **{method: lambda *args, **kwargs: None}
                )


class TestPanelData(ZiplineTestCase):

    @parameterized.expand([
        ('daily',
         pd.Timestamp('2015-12-23', tz='UTC'),
         pd.Timestamp('2016-01-05', tz='UTC'),),
        ('minute',
         pd.Timestamp('2015-12-23', tz='UTC'),
         pd.Timestamp('2015-12-24', tz='UTC'),),
    ])
    def test_panel_data(self, data_frequency, start_dt, end_dt):
        trading_calendar = get_calendar('NYSE')
        if data_frequency == 'daily':
            history_freq = '1d'
            create_df_for_asset = create_daily_df_for_asset
            dt_transform = trading_calendar.minute_to_session_label
        elif data_frequency == 'minute':
            history_freq = '1m'
            create_df_for_asset = create_minute_df_for_asset

            def dt_transform(dt):
                return dt

        sids = range(1, 3)
        dfs = {}
        for sid in sids:
            dfs[sid] = create_df_for_asset(trading_calendar,
                                           start_dt, end_dt, interval=sid)
            dfs[sid]['prev_close'] = dfs[sid]['close'].shift(1)
        panel = pd.Panel(dfs)

        price_record = pd.Panel(items=sids,
                                major_axis=panel.major_axis,
                                minor_axis=['current', 'previous'])

        def initialize(algo):
            algo.first_bar = True
            algo.equities = []
            for sid in sids:
                algo.equities.append(algo.sid(sid))

        def handle_data(algo, data):
            price_record.loc[:, dt_transform(algo.get_datetime()),
                             'current'] = (
                data.current(algo.equities, 'price')
            )
            if algo.first_bar:
                algo.first_bar = False
            else:
                price_record.loc[:, dt_transform(algo.get_datetime()),
                                 'previous'] = (
                    data.history(algo.equities, 'price',
                                 2, history_freq).iloc[0]
                )

        def check_panels():
            np.testing.assert_array_equal(
                price_record.values.astype('float64'),
                panel.loc[:, :, ['close',
                                 'prev_close']].values.astype('float64')
            )

        trading_algo = TradingAlgorithm(initialize=initialize,
                                        handle_data=handle_data)
        trading_algo.run(data=panel)
        check_panels()
        price_record.loc[:] = np.nan

        run_algorithm(
            start=start_dt,
            end=end_dt,
            capital_base=1,
            initialize=initialize,
            handle_data=handle_data,
            data_frequency=data_frequency,
            data=panel
        )
        check_panels()
