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
import datetime
from datetime import timedelta
from mock import MagicMock
from nose_parameterized import parameterized
from six.moves import range
from testfixtures import TempDirectory
from textwrap import dedent
from unittest import TestCase

import numpy as np
import pandas as pd

from zipline.utils.api_support import ZiplineAPI
from zipline.utils.control_flow import nullctx
from zipline.utils.test_utils import (
    setup_logger,
    teardown_logger
)
import zipline.utils.factory as factory

from zipline.errors import (
    OrderDuringInitialize,
    RegisterTradingControlPostInit,
    TradingControlViolation,
    AccountControlViolation,
    SymbolNotFound,
    RootSymbolNotFound,
    UnsupportedDatetimeFormat,
)
from zipline.test_algorithms import (
    access_account_in_init,
    access_portfolio_in_init,
    AmbitiousStopLimitAlgorithm,
    EmptyPositionsAlgorithm,
    InvalidOrderAlgorithm,
    RecordAlgorithm,
    TestAlgorithm,
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
)

import zipline.utils.events
from zipline.utils.test_utils import to_utc
from zipline.assets import Equity

from zipline.finance.execution import LimitOrder
from zipline.finance.trading import SimulationParameters
from zipline.utils.api_support import set_algo_instance
from zipline.utils.events import DateRuleFactory, TimeRuleFactory
from zipline.algorithm import TradingAlgorithm
from zipline.finance.trading import TradingEnvironment
from zipline.finance.commission import PerShare

from zipline.utils.test_utils import (
    create_data_portal,
    create_data_portal_from_trade_history
)

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
                        'asset_type': 'equity',
                        'start_date': '2002-01-01',
                        'end_date': '2004-01-01'},
                    4: {'symbol': 'PLAY',
                        'asset_type': 'equity',
                        'start_date': '2005-01-01',
                        'end_date': '2006-01-01'}}

        futures_metadata = {
            5: {
                'symbol': 'CLG06',
                'root_symbol': 'CL',
                'asset_type': 'future',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2005-12-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-01-20', tz='UTC')},
            6: {
                'root_symbol': 'CL',
                'symbol': 'CLK06',
                'asset_type': 'future',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-03-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-04-20', tz='UTC')},
            7: {
                'symbol': 'CLQ06',
                'root_symbol': 'CL',
                'asset_type': 'future',
                'start_date': pd.Timestamp('2005-12-01', tz='UTC'),
                'notice_date': pd.Timestamp('2006-06-20', tz='UTC'),
                'expiration_date': pd.Timestamp('2006-07-20', tz='UTC')},
            8: {
                'symbol': 'CLX06',
                'root_symbol': 'CL',
                'asset_type': 'future',
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
        futures_metadata = {3: {'asset_type': 'future',
                                'contract_multiplier': 10}}
        cls.env = TradingEnvironment()
        cls.sim_params = factory.create_simulation_parameters(num_days=4,
                                                              env=cls.env)
        cls.sids = [0, 1, 133]
        cls.tempdir = TempDirectory()

        equities_metadata = {}

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

        cls.data_portal.current_day = cls.sim_params.trading_days[0]

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
            instant_fill=False,
            method_name=order_style,
            env=self.env
        )
        algo.run(self.data_portal)

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
                'end_date': cls.sim_params.period_end
            }

            if sid == 3:
                equities_metadata[sid]["symbol"] = "TEST"
                equities_metadata[sid]["asset_type"] = "equity"

        cls.env.write_data(equities_data=equities_metadata)

        days = 251

        trades_by_sid = {
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

        cls.data_portal = create_data_portal_from_trade_history(cls.env,
                                                                cls.tempdir,
                                                                cls.sim_params,
                                                                trades_by_sid)

        cls.zipline_test_config = {
            'sid': 0,
        }

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
    record(price=data[0].price)

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

    def test_volshare_slippage(self):
        tempdir = TempDirectory()
        try:
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
    set_commission(commission.PerShare(0.02))
    context.count = 2
    context.incr = 0

def handle_data(context, data):
    if context.incr < context.count:
        # order small lots to be sure the
        # order will fill in a single transaction
        order(sid(0), 5000)
    record(price=data[0].price)
    record(volume=data[0].volume)
    record(incr=context.incr)
    context.incr += 1
    """,
                sim_params=self.sim_params,
                env=self.env,
            )
            set_algo_instance(test_algo)
            trades = factory.create_daily_trade_source(
                [0], self.sim_params, self.env)
            data_portal = create_data_portal_from_trade_history(
                self.env, tempdir, self.sim_params, {0: trades})
            results = test_algo.run(data_portal=data_portal)

            all_txns = [
                val for sublist in results["transactions"].tolist()
                for val in sublist]

            self.assertEqual(len(all_txns), 67)

            per_share_commish = 0.02
            first_txn = all_txns[0]
            commish = first_txn["amount"] * per_share_commish
            self.assertEqual(commish, first_txn["commission"])
            self.assertEqual(2.029, first_txn["price"])
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
                if context.first_bar:
                    dt = get_datetime({tz})
                    if dt.tz.zone != context.tz:
                        raise ValueError("Mismatched Zone")
                    elif dt.tz_convert("US/Eastern").hour != 9:
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
                'end_date': cls.sim_params.period_end
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


class TestClosePosAlgo(TestCase):

    def setUp(self):
        self.tempdir = TempDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_auto_close_future(self):
        sidint = 1
        env = TradingEnvironment()

        sim_params = factory.create_simulation_parameters(
            num_days=4, env=env
        )

        trades_by_sid = {}
        trades_by_sid[sidint] = factory.create_trade_history(
            sidint,
            [1, 1, 2, 4],
            [1e9, 1e9, 1e9, 1e9],
            timedelta(days=1),
            sim_params,
            env
        )

        day_after_ix = env.trading_days.searchsorted(
            sim_params.trading_days[-1]) + 1
        auto_close_date = env.trading_days[day_after_ix]

        env.write_data(futures_data={
            1: {
                "start_date": sim_params.trading_days[0],
                "end_date": sim_params.trading_days[-1] + timedelta(days=1),
                'symbol': 'TEST',
                'asset_type': 'future',
                'auto_close_date': auto_close_date
            }
        })

        algo = TestAlgorithm(sid=1, amount=1, order_count=1,
                             commission=PerShare(0),
                             sim_params=sim_params,
                             env=env)
        data_portal = create_data_portal_from_trade_history(env,
                                                            self.tempdir,
                                                            sim_params,
                                                            trades_by_sid)

        # Check results
        results = algo.run(data_portal)

        expected_pnl = [0, 0, 1, 2]
        self.check_algo_pnl(results, expected_pnl)

        expected_positions = [0, 1, 1, 0]
        self.check_algo_positions(results, expected_positions)

        expected_pnl = [0, 0, 1, 2]
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
