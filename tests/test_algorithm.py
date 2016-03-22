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
from collections import namedtuple
import datetime
from datetime import timedelta

from logbook import TestHandler, WARNING
from mock import MagicMock
from nose_parameterized import parameterized
from six import iteritems, itervalues
from six.moves import range
from testfixtures import TempDirectory
from textwrap import dedent
from unittest import TestCase, skip

import numpy as np
import pandas as pd
from contextlib2 import ExitStack

from zipline import TradingAlgorithm
from zipline.api import FixedSlippage
from zipline.data.data_portal import DataPortal
from zipline.data.minute_bars import BcolzMinuteBarWriter, \
    US_EQUITIES_MINUTES_PER_DAY, BcolzMinuteBarReader
from zipline.data.us_equity_pricing import BcolzDailyBarReader
from zipline.finance.commission import PerShare
from zipline.finance.execution import LimitOrder
from zipline.finance.order import ORDER_STATUS
from zipline.finance.trading import TradingEnvironment, SimulationParameters
from zipline.sources import DataPanelSource
from zipline.testing.core import (
    FakeDataPortal,
    make_trade_data_for_asset_info,
    create_data_portal,
    create_data_portal_from_trade_history,
    DailyBarWriterFromDataFrames,
    create_daily_df_for_asset, write_minute_data_for_asset
)
from zipline.errors import (
    OrderDuringInitialize,
    RegisterTradingControlPostInit,
    TradingControlViolation,
    AccountControlViolation,
    SymbolNotFound,
    RootSymbolNotFound,
    UnsupportedDatetimeFormat,
    CannotOrderDelistedAsset,
    SetCancelPolicyPostInit,
    UnsupportedCancelPolicy
)
from zipline.assets import Equity, Future
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
    SetLongOnlyAlgorithm,
    SetAssetDateBoundsAlgorithm,
    SetMaxPositionSizeAlgorithm,
    SetMaxOrderCountAlgorithm,
    SetMaxOrderSizeAlgorithm,
    SetDoNotOrderListAlgorithm,
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
    call_with_bad_kwargs_history
)
from zipline.testing import (
    make_jagged_equity_info,
    to_utc,
    setup_logger,
    teardown_logger,
    parameter_space,
)
from zipline.utils.api_support import ZiplineAPI, set_algo_instance
from zipline.utils.context_tricks import CallbackManager
from zipline.utils.control_flow import nullctx
import zipline.utils.events
from zipline.utils.events import DateRuleFactory, TimeRuleFactory, Always
import zipline.utils.factory as factory
from zipline.utils.tradingcalendar import trading_day, trading_days

# Because test cases appear to reuse some resources.


_multiprocess_can_split_ = False


class TestRecordAlgorithm(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.sids = [133]
        cls.env.write_data(equities_identifiers=cls.sids)

        cls.sim_params = factory.create_simulation_parameters(
            num_days=4,
            env=cls.env
        )

        cls.tempdir = TempDirectory()

        cls.data_portal = create_data_portal(
            cls.env,
            cls.tempdir,
            cls.sim_params,
            cls.sids
        )

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()

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


class TestMiscellaneousAPI(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sids = [1, 2]
        cls.env = TradingEnvironment()

        metadata = {3: {'symbol': 'PLAY',
                        'start_date': '2002-01-01',
                        'end_date': '2004-01-01'},
                    4: {'symbol': 'PLAY',
                        'start_date': '2005-01-01',
                        'end_date': '2006-01-01'}}

        futures_metadata = {
            5: {
                'symbol': 'CLG06',
                'root_symbol': 'CL',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2005-12-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-01-20', tz='UTC')},
            6: {
                'root_symbol': 'CL',
                'symbol': 'CLK06',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-03-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-04-20', tz='UTC')},
            7: {
                'symbol': 'CLQ06',
                'root_symbol': 'CL',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-06-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-07-20', tz='UTC')},
            8: {
                'symbol': 'CLX06',
                'root_symbol': 'CL',
                'start_date': pd.Timestamp('2006-02-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-09-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-10-20', tz='UTC')}
        }
        cls.env.write_data(equities_identifiers=cls.sids,
                           equities_data=metadata,
                           futures_data=futures_metadata)

        setup_logger(cls)

        cls.sim_params = factory.create_simulation_parameters(
            num_days=2,
            data_frequency='minute',
            emission_rate='daily',
            env=cls.env,
        )

        cls.temp_dir = TempDirectory()

        cls.data_portal = create_data_portal(
            cls.env,
            cls.temp_dir,
            cls.sim_params,
            cls.sids
        )

    @classmethod
    def tearDownClass(cls):
        del cls.env
        teardown_logger(cls)
        cls.temp_dir.cleanup()

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
    assert aapl_dt == get_datetime()
"""

        algo = TradingAlgorithm(script=algo_text,
                                sim_params=self.sim_params,
                                env=self.env)

        algo.run(self.data_portal)

    def test_get_environment(self):
        expected_env = {
            'arena': 'backtest',
            'data_frequency': 'minute',
            'start': pd.Timestamp('2006-01-03 14:31:00+0000', tz='UTC'),
            'end': pd.Timestamp('2006-01-04 21:00:00+0000', tz='UTC'),
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

                for order in orders_2:
                    algo.cancel_order(order)

                all_orders = algo.get_open_orders()
                self.assertEqual(all_orders, {})

            algo.minute += 1

        algo = TradingAlgorithm(initialize=initialize,
                                handle_data=handle_data,
                                sim_params=self.sim_params,
                                env=self.env)
        algo.run(self.data_portal)

    def test_schedule_function(self):
        date_rules = DateRuleFactory
        time_rules = TimeRuleFactory

        def incrementer(algo, data):
            algo.func_called += 1
            self.assertEqual(
                algo.get_datetime().time(),
                datetime.time(hour=14, minute=31),
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
            780 * 5,
            'Incorrect number of functions called: %s != 780' %
            len(function_stack),
        )
        expected_functions = [pre, handle_data, f, g, post] * 780
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

        # Test before either PLAY existed
        algo.sim_params.period_end = pd.Timestamp('2001-12-01', tz='UTC')
        with self.assertRaises(SymbolNotFound):
            algo.symbol('PLAY')
        with self.assertRaises(SymbolNotFound):
            algo.symbols('PLAY')

        # Test when first PLAY exists
        algo.sim_params.period_end = pd.Timestamp('2002-12-01', tz='UTC')
        list_result = algo.symbols('PLAY')
        self.assertEqual(3, list_result[0])

        # Test after first PLAY ends
        algo.sim_params.period_end = pd.Timestamp('2004-12-01', tz='UTC')
        self.assertEqual(3, algo.symbol('PLAY'))

        # Test after second PLAY begins
        algo.sim_params.period_end = pd.Timestamp('2005-12-01', tz='UTC')
        self.assertEqual(4, algo.symbol('PLAY'))

        # Test after second PLAY ends
        algo.sim_params.period_end = pd.Timestamp('2006-12-01', tz='UTC')
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

    def test_future_chain(self):
        """ Tests the future_chain API function.
        """
        algo = TradingAlgorithm(env=self.env)
        algo.datetime = pd.Timestamp('2006-12-01', tz='UTC')

        # Check that the fields of the FutureChain object are set correctly
        cl = algo.future_chain('CL')
        self.assertEqual(cl.root_symbol, 'CL')
        self.assertEqual(cl.as_of_date, algo.datetime)

        # Check the fields are set correctly if an as_of_date is supplied
        as_of_date = pd.Timestamp('1952-08-11', tz='UTC')

        cl = algo.future_chain('CL', as_of_date=as_of_date)
        self.assertEqual(cl.root_symbol, 'CL')
        self.assertEqual(cl.as_of_date, as_of_date)

        cl = algo.future_chain('CL', as_of_date='1952-08-11')
        self.assertEqual(cl.root_symbol, 'CL')
        self.assertEqual(cl.as_of_date, as_of_date)

        # Check that weird capitalization is corrected
        cl = algo.future_chain('cL')
        self.assertEqual(cl.root_symbol, 'CL')

        cl = algo.future_chain('cl')
        self.assertEqual(cl.root_symbol, 'CL')

        # Check that invalid root symbols raise RootSymbolNotFound
        with self.assertRaises(RootSymbolNotFound):
            algo.future_chain('CLZ')

        with self.assertRaises(RootSymbolNotFound):
            algo.future_chain('')

        # Check that invalid dates raise UnsupportedDatetimeFormat
        with self.assertRaises(UnsupportedDatetimeFormat):
            algo.future_chain('CL', 'my_finger_slipped')

        with self.assertRaises(UnsupportedDatetimeFormat):
            algo.future_chain('CL', '2015-09-')

        # Supplying a non-string argument to future_chain()
        # should result in a TypeError.
        with self.assertRaises(TypeError):
            algo.future_chain(1)

        with self.assertRaises(TypeError):
            algo.future_chain((1,))

        with self.assertRaises(TypeError):
            algo.future_chain({1})

        with self.assertRaises(TypeError):
            algo.future_chain([1])

        with self.assertRaises(TypeError):
            algo.future_chain({'foo': 'bar'})

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
                }
                for i, date in enumerate(dates)
            ]
        )
        env = TradingEnvironment()
        env.write_data(equities_df=metadata)
        algo = TradingAlgorithm(env=env)

        # Set the period end to a date after the period end
        # dates for our assets.
        algo.sim_params.period_end = pd.Timestamp('2015-01-01', tz='UTC')

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


class TestTransformAlgorithm(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logger(cls)
        cls.env = TradingEnvironment()
        cls.sim_params = factory.create_simulation_parameters(num_days=4,
                                                              env=cls.env)
        cls.sids = [0, 1, 133]
        cls.tempdir = TempDirectory()

        futures_metadata = {3: {'multiplier': 10}}
        equities_metadata = {}

        cls.futures_env = TradingEnvironment()
        cls.futures_env.write_data(futures_data=futures_metadata)

        for sid in cls.sids:
            equities_metadata[sid] = {
                'start_date': cls.sim_params.period_start,
                'end_date': cls.sim_params.period_end
            }

        cls.env.write_data(equities_data=equities_metadata,
                           futures_data=futures_metadata)

        trades_by_sid = {}
        for sid in cls.sids:
            trades_by_sid[sid] = factory.create_trade_history(
                sid,
                [10.0, 10.0, 11.0, 11.0],
                [100, 100, 100, 300],
                timedelta(days=1),
                cls.sim_params,
                cls.env
            )

        cls.data_portal = create_data_portal_from_trade_history(cls.env,
                                                                cls.tempdir,
                                                                cls.sim_params,
                                                                trades_by_sid)

    @classmethod
    def tearDownClass(cls):
        teardown_logger(cls)
        del cls.env
        cls.tempdir.cleanup()

    def test_invalid_order_parameters(self):
        algo = InvalidOrderAlgorithm(
            sids=[133],
            sim_params=self.sim_params,
            env=self.env,
        )
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

        # FIXME I think we are getting Nans due to fixed benchmark,
        # so dropping them for now.
        res1 = res1.fillna(method='ffill')
        res2 = res2.fillna(method='ffill')

        np.testing.assert_array_equal(res1, res2)

    def test_data_frequency_setting(self):
        self.sim_params.data_frequency = 'daily'

        sim_params = factory.create_simulation_parameters(
            num_days=4, env=self.env, data_frequency='daily')

        algo = TestRegisterTransformAlgorithm(
            sim_params=sim_params,
            env=self.env,
        )
        self.assertEqual(algo.sim_params.data_frequency, 'daily')

        sim_params = factory.create_simulation_parameters(
            num_days=4, env=self.env, data_frequency='minute')

        algo = TestRegisterTransformAlgorithm(
            sim_params=sim_params,
            env=self.env,
        )
        self.assertEqual(algo.sim_params.data_frequency, 'minute')

    @parameterized.expand([
        (TestOrderAlgorithm,),
        (TestOrderValueAlgorithm,),
        (TestTargetAlgorithm,),
        (TestOrderPercentAlgorithm,),
        (TestTargetPercentAlgorithm,),
        (TestTargetValueAlgorithm,),
    ])
    def test_order_methods(self, algo_class):
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
        # Ensure that the environment's asset 0 is a Future
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
            period_start=asset133.start_date,
            period_end=asset133.end_date,
            data_frequency="minute"
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
        tempdir = TempDirectory()

        try:
            env = TradingEnvironment()

            sim_params = SimulationParameters(
                period_start=pd.Timestamp('2002-1-2', tz='UTC'),
                period_end=pd.Timestamp('2002-1-4', tz='UTC'),
                capital_base=float("1.0e5"),
                data_frequency='minute',
                env=env
            )

            equities_metadata = {}

            for sid in [0, 1]:
                equities_metadata[sid] = {
                    'start_date': sim_params.period_start,
                    'end_date': sim_params.period_end + timedelta(days=1)
                }

            env.write_data(equities_data=equities_metadata)

            data_portal = create_data_portal(
                env,
                tempdir,
                sim_params,
                [0, 1]
            )

            algo = algo_class(sim_params=sim_params, env=env)
            algo.run(data_portal)
        finally:
            tempdir.cleanup()


class TestPositions(TestCase):
    @classmethod
    def setUpClass(cls):
        setup_logger(cls)
        cls.env = TradingEnvironment()
        cls.sim_params = factory.create_simulation_parameters(num_days=4,
                                                              env=cls.env)

        cls.sids = [1, 133]
        cls.tempdir = TempDirectory()

        equities_metadata = {}

        for sid in cls.sids:
            equities_metadata[sid] = {
                'start_date': cls.sim_params.period_start,
                'end_date': cls.sim_params.period_end
            }

        cls.env.write_data(equities_data=equities_metadata)

        cls.data_portal = create_data_portal(
            cls.env,
            cls.tempdir,
            cls.sim_params,
            cls.sids
        )

    @classmethod
    def tearDownClass(cls):
        teardown_logger(cls)
        cls.tempdir.cleanup()

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


class TestBeforeTradingStart(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.tempdir = TempDirectory()

        cls.trading_days = cls.env.days_in_range(
            start=pd.Timestamp("2016-01-05", tz='UTC'),
            end=pd.Timestamp("2016-01-07", tz='UTC')
        )

        equities_data = {}
        for sid in [1, 2]:
            equities_data[sid] = {
                "start_date": cls.trading_days[0],
                "end_date": cls.trading_days[-1],
                "symbol": "ASSET{0}".format(sid),
            }

        cls.env.write_data(equities_data=equities_data)

        cls.asset1 = cls.env.asset_finder.retrieve_asset(1)
        cls.asset2 = cls.env.asset_finder.retrieve_asset(2)

        market_opens = cls.env.open_and_closes.market_open.loc[
            cls.trading_days]

        minute_writer = BcolzMinuteBarWriter(
            cls.trading_days[0],
            cls.tempdir.path,
            market_opens,
            US_EQUITIES_MINUTES_PER_DAY
        )

        for sid in [1, 8554]:
            write_minute_data_for_asset(
                cls.env, minute_writer, cls.trading_days[0],
                cls.trading_days[-1], sid
            )

        # asset2 only trades every 50 minutes
        write_minute_data_for_asset(
            cls.env, minute_writer, cls.trading_days[0],
            cls.trading_days[-1], 2, 50
        )

        cls.minute_reader = BcolzMinuteBarReader(cls.tempdir.path)

        cls.daily_path = cls.tempdir.getpath("testdaily.bcolz")
        dfs = {
            1: create_daily_df_for_asset(cls.env, cls.trading_days[0],
                                         cls.trading_days[-1]),
            2: create_daily_df_for_asset(cls.env, cls.trading_days[0],
                                         cls.trading_days[-1])
        }
        daily_writer = DailyBarWriterFromDataFrames(dfs)
        daily_writer.write(cls.daily_path, cls.trading_days, dfs)

        cls.sim_params = SimulationParameters(
            period_start=cls.trading_days[1],
            period_end=cls.trading_days[-1],
            data_frequency="minute",
            env=cls.env
        )

        cls.data_portal = DataPortal(
            env=cls.env,
            equity_daily_reader=BcolzDailyBarReader(cls.daily_path),
            equity_minute_reader=cls.minute_reader
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

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
            data_frequency="minute",
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

    def data_in_bts_daily(self):
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
            data_frequency="minute",
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


class TestAlgoScript(TestCase):

    @classmethod
    def setUpClass(cls):
        setup_logger(cls)
        cls.env = TradingEnvironment()
        cls.sim_params = factory.create_simulation_parameters(num_days=251,
                                                              env=cls.env)

        cls.sids = [0, 1, 3, 133]
        cls.tempdir = TempDirectory()

        equities_metadata = {}

        for sid in cls.sids:
            equities_metadata[sid] = {
                'start_date': cls.sim_params.period_start,
                'end_date': cls.env.next_trading_day(cls.sim_params.period_end)
            }

            if sid == 3:
                equities_metadata[sid]["symbol"] = "TEST"
                equities_metadata[sid]["asset_type"] = "equity"

        cls.env.write_data(equities_data=equities_metadata)

        days = 251

        cls.trades_by_sid = {
            0: factory.create_trade_history(
                0,
                [10.0] * days,
                [100] * days,
                timedelta(days=1),
                cls.sim_params,
                cls.env),
            3: factory.create_trade_history(
                3,
                [10.0] * days,
                [100] * days,
                timedelta(days=1),
                cls.sim_params,
                cls.env)
        }

        cls.data_portal = create_data_portal_from_trade_history(
            cls.env, cls.tempdir, cls.sim_params, cls.trades_by_sid
        )

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()
        teardown_logger(cls)

    def test_noop(self):
        algo = TradingAlgorithm(initialize=initialize_noop,
                                handle_data=handle_data_noop)
        algo.run(self.data_portal)

    def test_noop_string(self):
        algo = TradingAlgorithm(script=noop_algo)
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

        self.assertEqual(100.0, txn["commission"])
        expected_spread = 0.05
        expected_commish = 0.10
        expected_price = test_algo.recorded_vars["price"] - expected_spread \
            - expected_commish

        self.assertEqual(expected_price, txn['price'])

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
            set_algo_instance(test_algo)
            trades = factory.create_daily_trade_source(
                [0], self.sim_params, self.env)
            data_portal = create_data_portal_from_trade_history(
                self.env, tempdir, self.sim_params, {0: trades})
            results = test_algo.run(data_portal)

            all_txns = [
                val for sublist in results["transactions"].tolist()
                for val in sublist]

            self.assertEqual(len(all_txns), 67)
            first_txn = all_txns[0]

            if minimum_commission == 0:
                commish = first_txn["amount"] * 0.02
                self.assertEqual(commish, first_txn["commission"])
            else:
                self.assertEqual(minimum_commission, first_txn["commission"])
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

    def test_order_dead_asset(self):
        # after asset 0 is dead
        params = SimulationParameters(
            period_start=pd.Timestamp("2007-01-03", tz='UTC'),
            period_end=pd.Timestamp("2007-01-05", tz='UTC'),
            env=self.env
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
        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            self.trades_by_sid
        )

        test_algo = TradingAlgorithm(
            script=call_without_kwargs,
            sim_params=self.sim_params,
            env=self.env,
        )
        test_algo.run(data_portal)

    def test_good_kwargs(self):
        """
        Test that api methods on the data object can be called with keyword
        arguments.
        """
        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            self.trades_by_sid
        )

        test_algo = TradingAlgorithm(
            script=call_with_kwargs,
            sim_params=self.sim_params,
            env=self.env,
        )
        test_algo.run(data_portal)

    @parameterized.expand([('history', call_with_bad_kwargs_history),
                           ('current', call_with_bad_kwargs_current)])
    def test_bad_kwargs(self, name, algo_text):
        """
        Test that api methods on the data object called with bad kwargs return
        a meaningful TypeError that we create, rather than an unhelpful cython
        error
        """
        data_portal = create_data_portal_from_trade_history(
            self.env,
            self.tempdir,
            self.sim_params,
            self.trades_by_sid
        )

        with self.assertRaises(TypeError) as cm:
            test_algo = TradingAlgorithm(
                script=algo_text,
                sim_params=self.sim_params,
                env=self.env,
            )
            test_algo.run(data_portal)

        self.assertEqual("%s() got an unexpected keyword argument 'blahblah'"
                         % name, cm.exception.args[0])


class TestGetDatetime(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.env.write_data(equities_identifiers=[0, 1])

        setup_logger(cls)

        cls.sim_params = factory.create_simulation_parameters(
            data_frequency='minute',
            env=cls.env,
            start=to_utc('2014-01-02 9:31'),
            end=to_utc('2014-01-03 9:31')
        )

        cls.tempdir = TempDirectory()

        cls.data_portal = create_data_portal(
            cls.env,
            cls.tempdir,
            cls.sim_params,
            [1]
        )

    @classmethod
    def tearDownClass(cls):
        del cls.env
        teardown_logger(cls)
        cls.tempdir.cleanup()

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


class TestTradingControls(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sid = 133
        cls.env = TradingEnvironment()
        cls.sim_params = factory.create_simulation_parameters(num_days=4,
                                                              env=cls.env)

        cls.env.write_data(equities_data={
            133: {
                'start_date': cls.sim_params.period_start,
                'end_date': cls.env.next_trading_day(cls.sim_params.period_end)
            }
        })

        cls.tempdir = TempDirectory()

        cls.data_portal = create_data_portal(
            cls.env,
            cls.tempdir,
            cls.sim_params,
            [cls.sid]
        )

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()

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
        algo = SetMaxPositionSizeAlgorithm(sid=self.sid,
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

        algo = SetMaxPositionSizeAlgorithm(sid=self.sid,
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

        algo = SetMaxPositionSizeAlgorithm(sid=self.sid,
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
        algo = SetMaxPositionSizeAlgorithm(sid=self.sid + 1,
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

    def test_set_do_not_order_list(self):
        # set the restricted list to be the sid, and fail.
        algo = SetDoNotOrderListAlgorithm(
            sid=self.sid,
            restricted_list=[self.sid],
            sim_params=self.sim_params,
            env=self.env,
        )

        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1

        self.check_algo_fails(algo, handle_data, 0)

        # set the restricted list to exclude the sid, and succeed
        algo = SetDoNotOrderListAlgorithm(
            sid=self.sid,
            restricted_list=[134, 135, 136],
            sim_params=self.sim_params,
            env=self.env,
        )

        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1

        self.check_algo_succeeds(algo, handle_data)

    def test_set_max_order_size(self):

        # Buy one share.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1
        algo = SetMaxOrderSizeAlgorithm(sid=self.sid,
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

        algo = SetMaxOrderSizeAlgorithm(sid=self.sid,
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

        algo = SetMaxOrderSizeAlgorithm(sid=self.sid,
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
        algo = SetMaxOrderSizeAlgorithm(sid=self.sid + 1,
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
        tempdir = TempDirectory()
        try:
            env = TradingEnvironment()
            sim_params = factory.create_simulation_parameters(
                num_days=4, env=env, data_frequency="minute")

            env.write_data(equities_data={
                1: {
                    'start_date': sim_params.period_start,
                    'end_date': sim_params.period_end + timedelta(days=1)
                }
            })

            data_portal = create_data_portal(
                env,
                tempdir,
                sim_params,
                [1]
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
        finally:
            tempdir.cleanup()

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
        tempdir = TempDirectory()
        try:
            # Run the algorithm with a sid that ends far in the future
            temp_env = TradingEnvironment()

            data_portal = create_data_portal(
                temp_env,
                tempdir,
                self.sim_params,
                [0]
            )

            metadata = {0: {'start_date': self.sim_params.period_start,
                            'end_date': '2020-01-01'}}

            algo = SetAssetDateBoundsAlgorithm(
                equities_metadata=metadata,
                sim_params=self.sim_params,
                env=temp_env,
            )
            algo.run(data_portal)
        finally:
            tempdir.cleanup()

        # Run the algorithm with a sid that has already ended
        tempdir = TempDirectory()
        try:
            temp_env = TradingEnvironment()

            data_portal = create_data_portal(
                temp_env,
                tempdir,
                self.sim_params,
                [0]
            )
            metadata = {0: {'start_date': '1989-01-01',
                            'end_date': '1990-01-01'}}

            algo = SetAssetDateBoundsAlgorithm(
                equities_metadata=metadata,
                sim_params=self.sim_params,
                env=temp_env,
            )
            with self.assertRaises(TradingControlViolation):
                algo.run(data_portal)
        finally:
            tempdir.cleanup()

        # Run the algorithm with a sid that has not started
        tempdir = TempDirectory()
        try:
            temp_env = TradingEnvironment()
            data_portal = create_data_portal(
                temp_env,
                tempdir,
                self.sim_params,
                [0]
            )

            metadata = {0: {'start_date': '2020-01-01',
                            'end_date': '2021-01-01'}}

            algo = SetAssetDateBoundsAlgorithm(
                equities_metadata=metadata,
                sim_params=self.sim_params,
                env=temp_env,
            )

            with self.assertRaises(TradingControlViolation):
                algo.run(data_portal)

        finally:
            tempdir.cleanup()


class TestAccountControls(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sidint = 133
        cls.env = TradingEnvironment()
        cls.sim_params = factory.create_simulation_parameters(
            num_days=4, env=cls.env
        )

        cls.env.write_data(equities_data={
            133: {
                'start_date': cls.sim_params.period_start,
                'end_date': cls.sim_params.period_end + timedelta(days=1)
            }
        })

        cls.tempdir = TempDirectory()

        trades_by_sid = {
            cls.sidint: factory.create_trade_history(
                cls.sidint,
                [10.0, 10.0, 11.0, 11.0],
                [100, 100, 100, 300],
                timedelta(days=1),
                cls.sim_params,
                cls.env,
            )
        }

        cls.data_portal = create_data_portal_from_trade_history(cls.env,
                                                                cls.tempdir,
                                                                cls.sim_params,
                                                                trades_by_sid)

    @classmethod
    def tearDownClass(cls):
        del cls.env
        cls.tempdir.cleanup()

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


class TestFutureFlip(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = TempDirectory()

        cls.env = TradingEnvironment()
        cls.days = pd.date_range(start=pd.Timestamp("2006-01-09", tz='UTC'),
                                 end=pd.Timestamp("2006-01-12", tz='UTC'))

        cls.sid = 1

        cls.sim_params = factory.create_simulation_parameters(
            start=cls.days[0],
            end=cls.days[-2]
        )

        trades = factory.create_trade_history(
            cls.sid,
            [1, 2, 4],
            [1e9, 1e9, 1e9],
            timedelta(days=1),
            cls.sim_params,
            cls.env
        )

        trades_by_sid = {
            cls.sid: trades
        }

        cls.data_portal = create_data_portal_from_trade_history(
            cls.env,
            cls.tempdir,
            cls.sim_params,
            trades_by_sid
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @skip
    def test_flip_algo(self):
        metadata = {1: {'symbol': 'TEST',
                        'start_date': self.sim_params.trading_days[0],
                        'end_date': self.env.next_trading_day(
                            self.sim_params.trading_days[-1]),
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


class TestTradingAlgorithm(TestCase):
    def setUp(self):
        self.env = TradingEnvironment()
        self.days = self.env.trading_days[:4]

    def test_analyze_called(self):
        self.perf_ref = None

        def initialize(context):
            pass

        def handle_data(context, data):
            pass

        def analyze(context, perf):
            self.perf_ref = perf

        algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
                                analyze=analyze)

        data_portal = FakeDataPortal(self.env)

        results = algo.run(data_portal)
        self.assertIs(results, self.perf_ref)


class TestOrderCancelation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnvironment()
        cls.tempdir = TempDirectory()

        cls.days = cls.env.days_in_range(
            start=pd.Timestamp("2016-01-05", tz='UTC'),
            end=pd.Timestamp("2016-01-07", tz='UTC')
        )

        cls.env.write_data(equities_data={
            1: {
                'start_date': cls.days[0],
                'end_date': cls.days[-1],
                'symbol': "ASSET1"
            }
        })

        cls.data_portal = DataPortal(
            cls.env,
            equity_minute_reader=cls.build_minute_data(),
            equity_daily_reader=cls.build_daily_data()
        )

        cls.code = dedent(
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
                    order(sid(1), 1000)
                    context.ordered = True
            """
        )

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    @classmethod
    def build_minute_data(cls):
        market_opens = cls.env.open_and_closes.market_open.loc[cls.days]

        writer = BcolzMinuteBarWriter(
            cls.days[0],
            cls.tempdir.path,
            market_opens,
            US_EQUITIES_MINUTES_PER_DAY
        )

        asset_minutes = cls.env.minutes_for_days_in_range(
            cls.days[0], cls.days[-1]
        )

        minutes_count = len(asset_minutes)
        minutes_arr = np.array(range(1, 1 + minutes_count))

        # normal test data, but volume is pinned at 1 share per minute
        df = pd.DataFrame({
            "open": minutes_arr + 1,
            "high": minutes_arr + 2,
            "low": minutes_arr - 1,
            "close": minutes_arr,
            "volume": np.full(minutes_count, 1),
            "dt": asset_minutes
        }).set_index("dt")

        writer.write(1, df)

        return BcolzMinuteBarReader(cls.tempdir.path)

    @classmethod
    def build_daily_data(cls):
        path = cls.tempdir.getpath("testdaily.bcolz")

        dfs = {
            1: pd.DataFrame({
                "open": np.full(3, 1),
                "high": np.full(3, 1),
                "low": np.full(3, 1),
                "close": np.full(3, 1),
                "volume": np.full(3, 1),
                "day": [day.value for day in cls.days]
            })
        }

        daily_writer = DailyBarWriterFromDataFrames(dfs)
        daily_writer.write(path, cls.days, dfs)

        return BcolzDailyBarReader(path)

    def prep_algo(self, cancelation_string, data_frequency="minute"):
        code = self.code.format(cancelation_string)
        algo = TradingAlgorithm(
            script=code,
            env=self.env,
            sim_params=SimulationParameters(
                period_start=self.days[0],
                period_end=self.days[-1],
                env=self.env,
                data_frequency=data_frequency
            )
        )

        return algo

    def test_eod_order_cancel_minute(self):
        # order 1000 shares of asset1.  the volume is only 1 share per bar,
        # so the order should be cancelled at the end of the day.
        algo = self.prep_algo(
            "set_cancel_policy(cancel_policy.EODCancel())"
        )

        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run(self.data_portal)

            for daily_positions in results.positions:
                self.assertEqual(1, len(daily_positions))
                self.assertEqual(389, daily_positions[0]["amount"])
                self.assertEqual(1, results.positions[0][0]["sid"].sid)

            # should be an order on day1, but no more orders afterwards
            np.testing.assert_array_equal([1, 0, 0],
                                          list(map(len, results.orders)))

            # should be 389 txns on day 1, but no more afterwards
            np.testing.assert_array_equal([389, 0, 0],
                                          list(map(len, results.transactions)))

            the_order = results.orders[0][0]

            self.assertEqual(ORDER_STATUS.CANCELLED, the_order["status"])
            self.assertEqual(389, the_order["filled"])

            warnings = [record for record in log_catcher.records if
                        record.level == WARNING]

            self.assertEqual(1, len(warnings))

            self.assertEqual(
                "Your order for 1000 shares of ASSET1 has been partially "
                "filled. 389 shares were successfully purchased. "
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


@skip("fix in Q2")
class TestRemoveData(TestCase):
    """
    tests if futures data is removed after max(expiration_date, end_date)
    """
    def setUp(self):
        self.env = env = TradingEnvironment()
        start_date = pd.Timestamp('2015-01-02', tz='UTC')
        start_ix = env.trading_days.get_loc(start_date)
        days = env.trading_days

        metadata = {
            0: {
                'symbol': 'X',
                'start_date': env.trading_days[start_ix + 2],
                'expiration_date': env.trading_days[start_ix + 5],
                'end_date': env.trading_days[start_ix + 6],
            },
            1: {
                'symbol': 'Y',
                'start_date': env.trading_days[start_ix + 4],
                'expiration_date': env.trading_days[start_ix + 7],
                'end_date': env.trading_days[start_ix + 8],
            }
        }

        env.write_data(futures_data=metadata)
        assetX, assetY = env.asset_finder.retrieve_all([0, 1])

        index_x = days[days.slice_indexer(assetX.start_date, assetX.end_date)]
        data_x = pd.DataFrame([[1, 100], [2, 100], [3, 100], [4, 100],
                               [5, 100]],
                              index=index_x, columns=['price', 'volume'])

        index_y = days[days.slice_indexer(assetY.start_date, assetY.end_date)]
        data_y = pd.DataFrame([[6, 100], [7, 100], [8, 100], [9, 100],
                               [10, 100]],
                              index=index_y, columns=['price', 'volume'])

        self.trade_data = pd.Panel({0: data_x, 1: data_y})
        self.live_asset_counts = []
        assets = env.asset_finder.retrieve_all([0, 1])
        for day in self.trade_data.major_axis:
            count = 0
            for asset in assets:
                # We shouldn't see assets on their expiration dates.
                if asset.start_date <= day <= asset.end_date:
                    count += 1
            self.live_asset_counts.append(count)

    def test_remove_data(self):
        source = DataPanelSource(self.trade_data)

        def initialize(context):
            context.data_lengths = []

        def handle_data(context, data):
            context.data_lengths.append(len(data))

        algo = TradingAlgorithm(
            initialize=initialize,
            handle_data=handle_data,
            env=self.env,
        )

        algo.run(source)
        self.assertEqual(algo.data_lengths, self.live_asset_counts)


class TestEquityAutoClose(TestCase):
    """
    Tests if delisted equities are properly removed from a portfolio holding
    positions in said equities.
    """
    @classmethod
    def setUpClass(cls):
        start_date = pd.Timestamp('2015-01-05', tz='UTC')
        start_date_loc = trading_days.get_loc(start_date)
        test_duration = 7
        cls.test_days = trading_days[
            start_date_loc:start_date_loc + test_duration
        ]
        cls.first_asset_expiration = cls.test_days[2]
        cls.tempdir = TempDirectory()

    def setUp(self):
        self._teardown_stack = ExitStack()

    def tearDown(self):
        self._teardown_stack.close()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def make_temp_resource(self, resource_context):
        return self._teardown_stack.enter_context(resource_context)

    def make_data(self, auto_close_delta, frequency,
                  capital_base=float("1.0e5")):

        asset_info = make_jagged_equity_info(
            num_assets=3,
            start_date=self.test_days[0],
            first_end=self.first_asset_expiration,
            frequency=trading_day,
            periods_between_ends=2,
            auto_close_delta=auto_close_delta,
        )

        sids = asset_info.keys()

        env = TradingEnvironment()
        env.write_data(equities_data=asset_info)
        market_opens = env.open_and_closes.market_open.loc[self.test_days]

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
            path = self.tempdir.getpath("testdaily.bcolz")
            writer = DailyBarWriterFromDataFrames(trade_data_by_sid)
            writer.write(path, dates, trade_data_by_sid)
            data_portal = DataPortal(
                env,
                equity_daily_reader=BcolzDailyBarReader(path)
            )
        elif frequency == 'minute':
            dates = env.minutes_for_days_in_range(
                self.test_days[0],
                self.test_days[-1],
            )
            writer = BcolzMinuteBarWriter(
                self.test_days[0],
                self.tempdir.path,
                market_opens,
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
            data_portal = DataPortal(
                env,
                equity_minute_reader=BcolzMinuteBarReader(self.tempdir.path)
            )
        else:
            self.fail("Unknown frequency in make_data: %r" % frequency)

        assets = env.asset_finder.retrieve_all(sids)

        sim_params = factory.create_simulation_parameters(
            start=self.test_days[0],
            end=self.test_days[-1],
            data_frequency=frequency,
            emission_rate=frequency,
            env=env,
            capital_base=capital_base,
        )

        if frequency == 'daily':
            trade_data_by_sid = {
                sid: df.set_index('day') for sid, df in
                iteritems(trade_data_by_sid)
            }
            final_prices = {
                asset.sid: trade_data_by_sid[asset.sid].
                loc[asset.end_date.value].close
                for asset in assets
            }
        else:
            final_prices = {
                asset.sid: trade_data_by_sid[asset.sid].loc[
                    env.get_open_and_close(asset.end_date)[1]
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
        auto_close_delta = trading_day * auto_close_lag
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
        # We have shrots if order_size < 0.
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
        for sid, txn in zip(sids, initial_fills):
            self.assertDictContainsSubset(
                {
                    'amount': order_size,
                    'commission': 0.0,
                    'dt': self.test_days[1],
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
        auto_close_delta = trading_day
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

            now = context.get_datetime()

            if now == first_asset_end_date:
                # Equity 0 will no longer exist tomorrow, so this order will
                # never be filled.
                assert len(context.get_open_orders()) == 0
                context.order(context.sid(0), 10)
                assert len(context.get_open_orders()) == 1
            elif now == first_asset_auto_close_date:
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
        self.assertDictContainsSubset(
            {
                'amount': 10,
                'commission': None,
                'created': first_asset_end_date,
                'dt': first_asset_end_date,
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
                'commission': None,
                'created': first_asset_end_date,
                'dt': first_asset_auto_close_date,
                'sid': assets[0],
                'status': ORDER_STATUS.CANCELLED,
                'filled': 0,
            },
            orders_after_auto_close[0],
        )

    def test_minutely_delisted_equities(self):
        resources = self.make_data(trading_day, 'minute')

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
            capital_base=capital_base,
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
                    'commission': 0.0,
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
