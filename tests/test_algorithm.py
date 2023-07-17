#
# Copyright 2018 Quantopian, Inc.
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
import logging
import warnings
from copy import deepcopy
from datetime import timedelta
from functools import partial
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import pytz
import toolz
from parameterized import parameterized
from testfixtures import TempDirectory

import zipline.api
import zipline.testing.fixtures as zf
from zipline.api import FixedSlippage
from zipline.assets import Asset, Equity, Future
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.assets.synthetic import make_jagged_equity_info, make_simple_equity_info
from zipline.errors import (
    AccountControlViolation,
    CannotOrderDelistedAsset,
    IncompatibleSlippageModel,
    RegisterTradingControlPostInit,
    ScheduleFunctionInvalidCalendar,
    SetCancelPolicyPostInit,
    SymbolNotFound,
    TradingControlViolation,
    UnsupportedCancelPolicy,
    UnsupportedDatetimeFormat,
    ZeroCapitalError,
)
from zipline.finance.asset_restrictions import (
    RESTRICTION_STATES,
    HistoricalRestrictions,
    Restriction,
    StaticRestrictions,
)
from zipline.finance.commission import PerShare, PerTrade
from zipline.finance.controls import AssetDateBounds
from zipline.finance.execution import LimitOrder
from zipline.finance.order import ORDER_STATUS
from zipline.finance.trading import SimulationParameters
from zipline.test_algorithms import (
    access_account_in_init,
    access_portfolio_in_init,
    api_algo,
    api_get_environment_algo,
    api_symbol_algo,
    bad_type_can_trade_assets,
    bad_type_current_assets,
    bad_type_current_assets_kwarg,
    bad_type_current_fields,
    bad_type_current_fields_kwarg,
    bad_type_history_assets,
    bad_type_history_assets_kwarg,
    bad_type_history_assets_kwarg_list,
    bad_type_history_bar_count,
    bad_type_history_bar_count_kwarg,
    bad_type_history_fields,
    bad_type_history_fields_kwarg,
    bad_type_history_frequency,
    bad_type_history_frequency_kwarg,
    bad_type_is_stale_assets,
    call_with_bad_kwargs_current,
    call_with_bad_kwargs_get_open_orders,
    call_with_bad_kwargs_history,
    call_with_good_kwargs_get_open_orders,
    call_with_kwargs,
    call_with_no_kwargs_get_open_orders,
    call_without_kwargs,
    empty_positions,
    handle_data_api,
    handle_data_noop,
    initialize_api,
    initialize_noop,
    no_handle_data,
    noop_algo,
    record_float_magic,
    record_variables,
)
from zipline.testing import (
    FakeDataPortal,
    RecordBatchBlotter,
    create_daily_df_for_asset,
    create_data_portal_from_trade_history,
    create_minute_df_for_asset,
    # make_test_handler,
    make_trade_data_for_asset_info,
    parameter_space,
    str_to_seconds,
    to_utc,
)
from zipline.testing.predicates import assert_equal
from zipline.utils import factory
from zipline.utils.api_support import ZiplineAPI
from zipline.utils.calendar_utils import get_calendar, register_calendar
from zipline.utils.context_tricks import CallbackManager, nop_context
from zipline.utils.events import (
    Always,
    ComposedRule,
    Never,
    OncePerDay,
    date_rules,
    time_rules,
)
from zipline.utils.pandas_utils import PerformanceWarning

# Because test cases appear to reuse some resources.


_multiprocess_can_split_ = False


class TestRecord(zf.WithMakeAlgo, zf.ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (133,)
    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_record_incr(self):
        def initialize(self):
            self.incr = 0

        def handle_data(self, data):
            self.incr += 1
            self.record(incr=self.incr)
            name = "name"
            self.record(name, self.incr)
            zipline.api.record(name, self.incr, "name2", 2, name3=self.incr)

        output = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
        )

        np.testing.assert_array_equal(output["incr"].values, range(1, len(output) + 1))
        np.testing.assert_array_equal(output["name"].values, range(1, len(output) + 1))
        np.testing.assert_array_equal(output["name2"].values, [2] * len(output))
        np.testing.assert_array_equal(output["name3"].values, range(1, len(output) + 1))


class TestMiscellaneousAPI(zf.WithMakeAlgo, zf.ZiplineTestCase):

    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-04")
    SIM_PARAMS_DATA_FREQUENCY = "minute"
    sids = 1, 2

    # FIXME: Pass a benchmark source instead of this.
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        return pd.concat(
            (
                make_simple_equity_info(cls.sids, "2002-02-1", "2007-01-01"),
                pd.DataFrame.from_dict(
                    {
                        3: {
                            "symbol": "PLAY",
                            "start_date": "2002-01-01",
                            "end_date": "2004-01-01",
                            "exchange": "TEST",
                        },
                        4: {
                            "symbol": "PLAY",
                            "start_date": "2005-01-01",
                            "end_date": "2006-01-01",
                            "exchange": "TEST",
                        },
                    },
                    orient="index",
                ),
            )
        )

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                5: {
                    "symbol": "CLG06",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2005-12-01"),
                    "notice_date": pd.Timestamp("2005-12-20"),
                    "expiration_date": pd.Timestamp("2006-01-20"),
                    "exchange": "TEST",
                },
                6: {
                    "root_symbol": "CL",
                    "symbol": "CLK06",
                    "start_date": pd.Timestamp("2005-12-01"),
                    "notice_date": pd.Timestamp("2006-03-20"),
                    "expiration_date": pd.Timestamp("2006-04-20"),
                    "exchange": "TEST",
                },
                7: {
                    "symbol": "CLQ06",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2005-12-01"),
                    "notice_date": pd.Timestamp("2006-06-20"),
                    "expiration_date": pd.Timestamp("2006-07-20"),
                    "exchange": "TEST",
                },
                8: {
                    "symbol": "CLX06",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2006-02-01"),
                    "notice_date": pd.Timestamp("2006-09-20"),
                    "expiration_date": pd.Timestamp("2006-10-20"),
                    "exchange": "TEST",
                },
            },
            orient="index",
        )

    def test_cancel_policy_outside_init(self):
        code = dedent(
            """
            from zipline.api import cancel_policy, set_cancel_policy

            def initialize(algo):
                pass

            def handle_data(algo, data):
                set_cancel_policy(cancel_policy.NeverCancel())
            """
        )
        algo = self.make_algo(script=code)
        with pytest.raises(SetCancelPolicyPostInit):
            algo.run()

    def test_cancel_policy_invalid_param(self):
        code = dedent(
            """
            from zipline.api import set_cancel_policy

            def initialize(algo):
                set_cancel_policy("foo")

            def handle_data(algo, data):
                pass
            """
        )
        algo = self.make_algo(script=code)
        with pytest.raises(UnsupportedCancelPolicy):
            algo.run()

    def test_zipline_api_resolves_dynamically(self):
        # Make a dummy algo.
        algo = self.make_algo(
            initialize=lambda context: None,
            handle_data=lambda context, data: None,
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
                assert sentinel is getattr(zipline.api, name)()

    def test_sid_datetime(self):
        algo_text = dedent(
            """
            from zipline.api import sid, get_datetime

            def initialize(context):
                pass

            def handle_data(context, data):
                aapl_dt = data.current(sid(1), "last_traded")
                assert_equal(aapl_dt, get_datetime())
            """
        )
        self.run_algorithm(
            script=algo_text,
            namespace={"assert_equal": self.assertEqual},
        )

    def test_datetime_bad_params(self):
        algo_text = dedent(
            """
            from zipline.api import get_datetime
            from pytz import timezone

            def initialize(context):
                pass

            def handle_data(context, data):
                get_datetime(timezone)
            """
        )
        algo = self.make_algo(script=algo_text)
        with pytest.raises(TypeError):
            algo.run()

    @parameterized.expand([(-1000, "invalid_base"), (0, "invalid_base")])
    def test_invalid_capital_base(self, cap_base, name):
        """Test that the appropriate error is being raised and orders aren't
        filled for algos with capital base <= 0
        """

        algo_text = dedent(
            """
            def initialize(context):
                pass

            def handle_data(context, data):
                order(sid(24), 1000)
             """
        )
        sim_params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-03"),
            end_session=pd.Timestamp("2006-01-06"),
            capital_base=cap_base,
            data_frequency="minute",
            trading_calendar=self.trading_calendar,
        )

        expected_msg = "initial capital base must be greater than zero"
        with pytest.raises(ZeroCapitalError, match=expected_msg):
            # make_algo will trace to TradingAlgorithm,
            # where the exception will be raised
            self.make_algo(script=algo_text, sim_params=sim_params)
            # Make sure the correct error was raised

    def test_get_environment(self):
        expected_env = {
            "arena": "backtest",
            "data_frequency": "minute",
            "start": pd.Timestamp("2006-01-03 14:31:00+0000", tz="utc"),
            "end": pd.Timestamp("2006-01-04 21:00:00+0000", tz="utc"),
            "capital_base": 100000.0,
            "platform": "zipline",
        }

        def initialize(algo):
            assert "zipline" == algo.get_environment()
            assert expected_env == algo.get_environment("*")

        def handle_data(algo, data):
            pass

        self.run_algorithm(initialize=initialize, handle_data=handle_data)

    def test_get_open_orders(self):
        def initialize(algo):
            algo.minute = 0

        def handle_data(algo, data):
            if algo.minute == 0:

                # Should be filled by the next minute
                algo.order(algo.sid(1), 1)

                # Won't be filled because the price is too low.
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01, asset=algo.sid(2)))
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01, asset=algo.sid(2)))
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01, asset=algo.sid(2)))

                all_orders = algo.get_open_orders()
                assert list(all_orders.keys()) == [1, 2]

                assert all_orders[1] == algo.get_open_orders(1)
                assert len(all_orders[1]) == 1

                assert all_orders[2] == algo.get_open_orders(2)
                assert len(all_orders[2]) == 3

            if algo.minute == 1:
                # First order should have filled.
                # Second order should still be open.
                all_orders = algo.get_open_orders()
                assert list(all_orders.keys()) == [2]

                assert [] == algo.get_open_orders(1)

                orders_2 = algo.get_open_orders(2)
                assert all_orders[2] == orders_2
                assert len(all_orders[2]) == 3

                for order_ in orders_2:
                    algo.cancel_order(order_)

                all_orders = algo.get_open_orders()
                assert all_orders == {}

            algo.minute += 1

        self.run_algorithm(initialize=initialize, handle_data=handle_data)

    def test_schedule_function_custom_cal(self):
        # run a simulation on the CMES cal, and schedule a function
        # using the NYSE cal
        algotext = dedent(
            """
            from zipline.api import (
                schedule_function,
                get_datetime,
                time_rules,
                date_rules,
                calendars,
            )

            def initialize(context):
                schedule_function(
                    func=log_nyse_open,
                    date_rule=date_rules.every_day(),
                    time_rule=time_rules.market_open(),
                    calendar=calendars.US_EQUITIES,
                )

                schedule_function(
                    func=log_nyse_close,
                    date_rule=date_rules.every_day(),
                    time_rule=time_rules.market_close(),
                    calendar=calendars.US_EQUITIES,
                )

                context.nyse_opens = []
                context.nyse_closes = []

            def log_nyse_open(context, data):
                context.nyse_opens.append(get_datetime())

            def log_nyse_close(context, data):
                context.nyse_closes.append(get_datetime())
            """
        )

        algo = self.make_algo(
            script=algotext,
            sim_params=self.make_simparams(
                trading_calendar=get_calendar("CMES"),
            ),
        )
        algo.run()

        nyse = get_calendar("NYSE")

        for minute in algo.nyse_opens:
            # each minute should be a nyse session open
            session_label = nyse.minute_to_session(minute)
            session_open = nyse.session_first_minute(session_label)
            assert session_open == minute

        for minute in algo.nyse_closes:
            # each minute should be a minute before a nyse session close
            session_label = nyse.minute_to_session(minute)
            session_close = nyse.session_last_minute(session_label)
            assert session_close - timedelta(minutes=1) == minute

        # Test that passing an invalid calendar parameter raises an error.
        erroring_algotext = dedent(
            """
            from zipline.api import schedule_function
            from zipline.utils.calendar_utils import get_calendar

            def initialize(context):
                schedule_function(func=my_func, calendar=get_calendar('XNYS'))

            def my_func(context, data):
                pass
            """
        )

        algo = self.make_algo(
            script=erroring_algotext,
            sim_params=self.make_simparams(
                trading_calendar=get_calendar("CMES"),
            ),
        )

        with pytest.raises(ScheduleFunctionInvalidCalendar):
            algo.run()

    def test_schedule_function(self):
        us_eastern = pytz.timezone("US/Eastern")

        def incrementer(algo, data):
            algo.func_called += 1
            curdt = algo.get_datetime().tz_convert(pytz.utc)
            assert curdt == us_eastern.localize(
                datetime.datetime.combine(curdt.date(), datetime.time(9, 31))
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

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
        )
        algo.run()

        assert algo.func_called == algo.days

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

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            create_event_context=CallbackManager(pre, post),
        )
        algo.run()

        assert len(expected_data) == 780
        assert collected_data_pre == expected_data
        assert collected_data_post == expected_data

        assert (
            len(function_stack) == 3900
        ), "Incorrect number of functions called: %s != 3900" % len(function_stack)
        expected_functions = [pre, handle_data, f, g, post] * 97530
        for n, (f, g) in enumerate(zip(function_stack, expected_functions)):
            assert (
                f == g
            ), "function at position %d was incorrect, expected %s but got %s" % (
                n,
                g.__name__,
                f.__name__,
            )

    @parameterized.expand(
        [
            ("daily",),
            ("minute"),
        ]
    )
    def test_schedule_function_rule_creation(self, mode):
        def nop(*args, **kwargs):
            return None

        self.sim_params.data_frequency = mode
        algo = self.make_algo(
            initialize=nop,
            handle_data=nop,
            sim_params=self.sim_params,
        )

        # Schedule something for NOT Always.
        # Compose two rules to ensure calendar is set properly.
        algo.schedule_function(nop, time_rule=Never() & Always())

        event_rule = algo.event_manager._events[1].rule
        assert isinstance(event_rule, OncePerDay)
        assert event_rule.cal == algo.trading_calendar

        inner_rule = event_rule.rule
        assert isinstance(inner_rule, ComposedRule)
        assert inner_rule.cal == algo.trading_calendar

        first = inner_rule.first
        second = inner_rule.second
        composer = inner_rule.composer

        assert isinstance(first, Always)
        assert first.cal == algo.trading_calendar
        assert second.cal == algo.trading_calendar

        if mode == "daily":
            assert isinstance(second, Always)
        else:
            assert isinstance(second, ComposedRule)
            assert isinstance(second.first, Never)
            assert second.first.cal == algo.trading_calendar

            assert isinstance(second.second, Always)
            assert second.second.cal == algo.trading_calendar

        assert composer is ComposedRule.lazy_and

    def test_asset_lookup(self):
        algo = self.make_algo()

        # this date doesn't matter
        start_session = pd.Timestamp("2000-01-01")

        # Test before either PLAY existed
        algo.sim_params = algo.sim_params.create_new(
            start_session, pd.Timestamp("2001-12-01")
        )

        with pytest.raises(SymbolNotFound):
            algo.symbol("PLAY")
        with pytest.raises(SymbolNotFound):
            algo.symbols("PLAY")

        # Test when first PLAY exists
        algo.sim_params = algo.sim_params.create_new(
            start_session, pd.Timestamp("2002-12-01")
        )
        list_result = algo.symbols("PLAY")
        assert 3 == list_result[0]

        # Test after first PLAY ends
        algo.sim_params = algo.sim_params.create_new(
            start_session, pd.Timestamp("2004-12-01")
        )
        assert 3 == algo.symbol("PLAY")

        # Test after second PLAY begins
        algo.sim_params = algo.sim_params.create_new(
            start_session, pd.Timestamp("2005-12-01")
        )
        assert 4 == algo.symbol("PLAY")

        # Test after second PLAY ends
        algo.sim_params = algo.sim_params.create_new(
            start_session, pd.Timestamp("2006-12-01")
        )
        assert 4 == algo.symbol("PLAY")
        list_result = algo.symbols("PLAY")
        assert 4 == list_result[0]

        # Test lookup SID
        assert isinstance(algo.sid(3), Equity)
        assert isinstance(algo.sid(4), Equity)

        # Supplying a non-string argument to symbol()
        # should result in a TypeError.
        with pytest.raises(TypeError):
            algo.symbol(1)

        with pytest.raises(TypeError):
            algo.symbol((1,))

        with pytest.raises(TypeError):
            algo.symbol({1})

        with pytest.raises(TypeError):
            algo.symbol([1])

        with pytest.raises(TypeError):
            algo.symbol({"foo": "bar"})

    def test_future_symbol(self):
        """Tests the future_symbol API function."""

        algo = self.make_algo()
        algo.datetime = pd.Timestamp("2006-12-01")

        # Check that we get the correct fields for the CLG06 symbol
        cl = algo.future_symbol("CLG06")
        assert cl.sid == 5
        assert cl.symbol == "CLG06"
        assert cl.root_symbol == "CL"
        assert cl.start_date == pd.Timestamp("2005-12-01")
        assert cl.notice_date == pd.Timestamp("2005-12-20")
        assert cl.expiration_date == pd.Timestamp("2006-01-20")

        with pytest.raises(SymbolNotFound):
            algo.future_symbol("")

        with pytest.raises(SymbolNotFound):
            algo.future_symbol("PLAY")

        with pytest.raises(SymbolNotFound):
            algo.future_symbol("FOOBAR")

        # Supplying a non-string argument to future_symbol()
        # should result in a TypeError.
        with pytest.raises(TypeError):
            algo.future_symbol(1)

        with pytest.raises(TypeError):
            algo.future_symbol((1,))

        with pytest.raises(TypeError):
            algo.future_symbol({1})

        with pytest.raises(TypeError):
            algo.future_symbol([1])

        with pytest.raises(TypeError):
            algo.future_symbol({"foo": "bar"})


class TestSetSymbolLookupDate(zf.WithMakeAlgo, zf.ZiplineTestCase):
    #     January 2006
    # Su Mo Tu We Th Fr Sa
    #  1  2  3  4  5  6  7
    #  8  9 10 11 12 13 14
    # 15 16 17 18 19 20 21
    # 22 23 24 25 26 27 28
    # 29 30 31
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-06")
    SIM_PARAMS_START_DATE = pd.Timestamp("2006-01-04")
    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = 3

    @classmethod
    def make_equity_info(cls):
        dates = pd.date_range(cls.START_DATE, cls.END_DATE)
        assert len(dates) == 4, "Expected four dates."

        # Two assets with the same ticker, ending on days[1] and days[3], plus
        # a benchmark that spans the whole period.
        cls.sids = [1, 2, 3]
        cls.asset_starts = [dates[0], dates[2]]
        cls.asset_ends = [dates[1], dates[3]]
        return pd.DataFrame.from_records(
            [
                {
                    "symbol": "DUP",
                    "start_date": cls.asset_starts[0],
                    "end_date": cls.asset_ends[0],
                    "exchange": "TEST",
                    "asset_name": "FIRST",
                },
                {
                    "symbol": "DUP",
                    "start_date": cls.asset_starts[1],
                    "end_date": cls.asset_ends[1],
                    "exchange": "TEST",
                    "asset_name": "SECOND",
                },
                {
                    "symbol": "BENCH",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE,
                    "exchange": "TEST",
                    "asset_name": "BENCHMARK",
                },
            ],
            index=cls.sids,
        )

    # TODO FIXME IMPORTANT pytest crashes with internal error if test below is uncommented
    # def test_set_symbol_lookup_date(self):
    #     """Test the set_symbol_lookup_date API method."""

    #     set_symbol_lookup_date = zipline.api.set_symbol_lookup_date

    #     def initialize(context):
    #         set_symbol_lookup_date(self.asset_ends[0])
    #         assert zipline.api.symbol("DUP").sid == self.sids[0]

    #         set_symbol_lookup_date(self.asset_ends[1])
    #         assert zipline.api.symbol("DUP").sid == self.sids[1]

    #         with pytest.raises(UnsupportedDatetimeFormat):
    #             set_symbol_lookup_date("foobar")

    #     self.run_algorithm(initialize=initialize)


class TestPositions(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-06")
    SIM_PARAMS_CAPITAL_BASE = 1000

    ASSET_FINDER_EQUITY_SIDS = (1, 133)

    SIM_PARAMS_DATA_FREQUENCY = "daily"

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        frame = pd.DataFrame(
            {
                "open": [90, 95, 100, 105],
                "high": [90, 95, 100, 105],
                "low": [90, 95, 100, 105],
                "close": [90, 95, 100, 105],
                "volume": 100,
            },
            index=cls.equity_daily_bar_days,
        )
        return ((sid, frame) for sid in sids)

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                1000: {
                    "symbol": "CLF06",
                    "root_symbol": "CL",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE,
                    "auto_close_date": cls.END_DATE + cls.trading_calendar.day,
                    "exchange": "CMES",
                    "multiplier": 100,
                },
            },
            orient="index",
        )

    @classmethod
    def make_future_minute_bar_data(cls):
        trading_calendar = cls.trading_calendars[Future]

        sids = cls.asset_finder.futures_sids
        minutes = trading_calendar.sessions_minutes(
            cls.future_minute_bar_days[0],
            cls.future_minute_bar_days[-1],
        )
        frame = pd.DataFrame(
            {
                "open": 2.0,
                "high": 2.0,
                "low": 2.0,
                "close": 2.0,
                "volume": 100,
            },
            index=minutes,
        )
        return ((sid, frame) for sid in sids)

    def test_portfolio_exited_position(self):
        # This test ensures ensures that 'phantom' positions do not appear in
        # context.portfolio.positions in the case that a position has been
        # entered and fully exited.

        def initialize(context, sids):
            context.ordered = False
            context.exited = False
            context.sids = sids

        def handle_data(context, data):
            if not context.ordered:
                for s in context.sids:
                    context.order(context.sid(s), 1)
                context.ordered = True

            if not context.exited:
                amounts = [pos.amount for pos in context.portfolio.positions.values()]

                if len(amounts) > 0 and all([(amount == 1) for amount in amounts]):
                    for stock in context.portfolio.positions:
                        context.order(context.sid(stock), -1)
                    context.exited = True

            # Should be 0 when all positions are exited.
            context.record(num_positions=len(context.portfolio.positions))

        result = self.run_algorithm(
            initialize=initialize,
            handle_data=handle_data,
            sids=self.ASSET_FINDER_EQUITY_SIDS,
        )

        expected_position_count = [
            0,  # Before entering the first position
            2,  # After entering, exiting on this date
            0,  # After exiting
            0,
        ]
        for i, expected in enumerate(expected_position_count):
            assert result.iloc[i]["num_positions"] == expected

    def test_noop_orders(self):
        asset = self.asset_finder.retrieve_asset(1)

        # Algorithm that tries to buy with extremely low stops/limits and tries
        # to sell with extremely high versions of same. Should not end up with
        # any positions for reasonable data.
        def handle_data(algo, data):

            ########
            # Buys #
            ########

            # Buy with low limit, shouldn't trigger.
            algo.order(asset, 100, limit_price=1)

            # But with high stop, shouldn't trigger
            algo.order(asset, 100, stop_price=10000000)

            # Buy with high limit (should trigger) but also high stop (should
            # prevent trigger).
            algo.order(asset, 100, limit_price=10000000, stop_price=10000000)

            # Buy with low stop (should trigger), but also low limit (should
            # prevent trigger).
            algo.order(asset, 100, limit_price=1, stop_price=1)

            #########
            # Sells #
            #########

            # Sell with high limit, shouldn't trigger.
            algo.order(asset, -100, limit_price=1000000)

            # Sell with low stop, shouldn't trigger.
            algo.order(asset, -100, stop_price=1)

            # Sell with low limit (should trigger), but also high stop (should
            # prevent trigger).
            algo.order(asset, -100, limit_price=1000000, stop_price=1000000)

            # Sell with low limit (should trigger), but also low stop (should
            # prevent trigger).
            algo.order(asset, -100, limit_price=1, stop_price=1)

            ###################
            # Rounding Checks #
            ###################
            algo.order(asset, 100, limit_price=0.00000001)
            algo.order(asset, -100, stop_price=0.00000001)

        daily_stats = self.run_algorithm(handle_data=handle_data)

        # Verify that positions are empty for all dates.
        empty_positions = daily_stats.positions.map(lambda x: len(x) == 0)
        assert empty_positions.all()

    def test_position_weights(self):
        sids = (1, 133, 1000)
        equity_1, equity_133, future_1000 = self.asset_finder.retrieve_all(sids)

        def initialize(algo, sids_and_amounts, *args, **kwargs):
            algo.ordered = False
            algo.sids_and_amounts = sids_and_amounts
            algo.set_commission(
                us_equities=PerTrade(0),
                us_futures=PerTrade(0),
            )
            algo.set_slippage(
                us_equities=FixedSlippage(0),
                us_futures=FixedSlippage(0),
            )

        def handle_data(algo, data):
            if not algo.ordered:
                for s, amount in algo.sids_and_amounts:
                    algo.order(algo.sid(s), amount)
                algo.ordered = True

            algo.record(
                position_weights=algo.portfolio.current_portfolio_weights,
            )

        daily_stats = self.run_algorithm(
            sids_and_amounts=zip(sids, [2, -1, 1]),
            initialize=initialize,
            handle_data=handle_data,
        )

        expected_position_weights = [
            # No positions held on the first day.
            pd.Series({}, dtype=float),
            # Each equity's position value is its price times the number of
            # shares held. In this example, we hold a long position in 2 shares
            # of equity_1 so its weight is (95.0 * 2) = 190.0 divided by the
            # total portfolio value. The total portfolio value is the sum of
            # cash ($905.00) plus the value of all equity positions.
            #
            # For a futures contract, its weight is the unit price times number
            # of shares held times the multiplier. For future_1000, this is
            # (2.0 * 1 * 100) = 200.0 divided by total portfolio value.
            pd.Series(
                {
                    equity_1: 190.0 / (190.0 - 95.0 + 905.0),
                    equity_133: -95.0 / (190.0 - 95.0 + 905.0),
                    future_1000: 200.0 / (190.0 - 95.0 + 905.0),
                }
            ),
            pd.Series(
                {
                    equity_1: 200.0 / (200.0 - 100.0 + 905.0),
                    equity_133: -100.0 / (200.0 - 100.0 + 905.0),
                    future_1000: 200.0 / (200.0 - 100.0 + 905.0),
                }
            ),
            pd.Series(
                {
                    equity_1: 210.0 / (210.0 - 105.0 + 905.0),
                    equity_133: -105.0 / (210.0 - 105.0 + 905.0),
                    future_1000: 200.0 / (210.0 - 105.0 + 905.0),
                }
            ),
        ]

        for i, expected in enumerate(expected_position_weights):
            assert_equal(daily_stats.iloc[i]["position_weights"], expected)


class TestBeforeTradingStart(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2016-01-06")
    END_DATE = pd.Timestamp("2016-01-07")
    SIM_PARAMS_CAPITAL_BASE = 10000
    SIM_PARAMS_DATA_FREQUENCY = "minute"
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = EQUITY_MINUTE_BAR_LOOKBACK_DAYS = 1

    DATA_PORTAL_FIRST_TRADING_DAY = pd.Timestamp("2016-01-05")
    EQUITY_MINUTE_BAR_START_DATE = pd.Timestamp("2016-01-05")
    FUTURE_MINUTE_BAR_START_DATE = pd.Timestamp("2016-01-05")

    data_start = ASSET_FINDER_EQUITY_START_DATE = pd.Timestamp("2016-01-05")

    SPLIT_ASSET_SID = 3
    ASSET_FINDER_EQUITY_SIDS = 1, 2, SPLIT_ASSET_SID

    @classmethod
    def make_equity_minute_bar_data(cls):
        asset_minutes = cls.trading_calendar.minutes_in_range(
            cls.data_start,
            cls.END_DATE,
        )
        minutes_count = len(asset_minutes)
        minutes_arr = np.arange(minutes_count) + 1
        split_data = pd.DataFrame(
            {
                "open": minutes_arr + 1,
                "high": minutes_arr + 2,
                "low": minutes_arr - 1,
                "close": minutes_arr,
                "volume": 100 * minutes_arr,
            },
            index=asset_minutes,
        )
        split_data.iloc[780:] = split_data.iloc[780:] / 2.0
        for sid in (1, 8554):
            yield sid, create_minute_df_for_asset(
                cls.trading_calendar,
                cls.data_start,
                cls.END_DATE,
            )

        yield 2, create_minute_df_for_asset(
            cls.trading_calendar,
            cls.data_start,
            cls.END_DATE,
            50,
        )
        yield cls.SPLIT_ASSET_SID, split_data

    @classmethod
    def make_splits_data(cls):
        return pd.DataFrame.from_records(
            [
                {
                    "effective_date": str_to_seconds("2016-01-07"),
                    "ratio": 0.5,
                    "sid": cls.SPLIT_ASSET_SID,
                }
            ]
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        for sid in sids:
            yield sid, create_daily_df_for_asset(
                cls.trading_calendar,
                cls.data_start,
                cls.END_DATE,
            )

    def test_data_in_bts_minute(self):
        algo_code = dedent(
            """
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
            """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        # fetching data at midnight gets us the previous market minute's data
        assert 390 == results.iloc[0].the_price1
        assert 392 == results.iloc[0].the_high1

        # make sure that price is ffilled, but not other fields
        assert 350 == results.iloc[0].the_price2
        assert np.isnan(results.iloc[0].the_high2)

        # 10-minute history

        # asset1 day1 price should be 331-390
        np.testing.assert_array_equal(
            range(331, 391), algo.history_values[0].loc[pd.IndexSlice[:, 1], "price"]
        )

        # asset1 day1 high should be 333-392
        np.testing.assert_array_equal(
            range(333, 393), algo.history_values[0].loc[pd.IndexSlice[:, 1], "high"]
        )

        # asset2 day1 price should be 19 300s, then 40 350s
        np.testing.assert_array_equal(
            [300] * 19,
            algo.history_values[0].loc[pd.IndexSlice[:, 2], "price"].iloc[:19],
        )

        np.testing.assert_array_equal(
            [350] * 40,
            algo.history_values[0].loc[pd.IndexSlice[:, 2], "price"].iloc[20:],
        )

        # asset2 day1 high should be all NaNs except for the 19th item
        # = 2016-01-05 20:20:00+00:00
        np.testing.assert_array_equal(
            np.full(19, np.nan),
            algo.history_values[0].loc[pd.IndexSlice[:, 2], "high"].iloc[:19],
        )

        assert 352 == algo.history_values[0].loc[pd.IndexSlice[:, 2], "high"].iloc[19]

        np.testing.assert_array_equal(
            np.full(40, np.nan),
            algo.history_values[0].loc[pd.IndexSlice[:, 2], "high"].iloc[20:],
        )

    def test_data_in_bts_daily(self):
        algo_code = dedent(
            """
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
                    "1d",
                ))

            def handle_data(context, data):
                pass
            """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        assert 392 == results.the_high1[0]
        assert 390 == results.the_price1[0]

        # nan because asset2 only trades every 50 minutes
        assert np.isnan(results.the_high2[0])

        assert 350, results.the_price2[0]

        assert 392 == algo.history_values[0]["high"][0]
        assert 390 == algo.history_values[0]["price"][0]

        assert 352 == algo.history_values[0]["high"][1]
        assert 350 == algo.history_values[0]["price"][1]

    def test_portfolio_bts(self):
        algo_code = dedent(
            """
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
            """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        # Asset starts with price 1 on 1/05 and increases by 1 every minute.
        # Simulation starts on 1/06, where the price in bts is 390, and
        # positions_value is 0. On 1/07, price is 780, and after buying one
        # share on the first bar of 1/06, positions_value is 780
        assert results.pos_value.iloc[0] == 0
        assert results.pos_value.iloc[1] == 780

    def test_account_bts(self):
        algo_code = dedent(
            """
            from zipline.api import order, sid, record, set_slippage, slippage

            def initialize(context):
                context.ordered = False
                context.hd_account = context.account
                set_slippage(slippage.VolumeShareSlippage())

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
            """
        )

        algo = self.make_algo(script=algo_code)
        results = algo.run()

        # Starting portfolio value is 10000. Order for the asset fills on the
        # second bar of 1/06, where the price is 391, and costs the default
        # commission of 0. On 1/07, the price is 780, and the increase in
        # portfolio value is 780-392-0
        assert results.port_value.iloc[0] == 10000
        self.assertAlmostEqual(
            results.port_value.iloc[1], 10000 + 780 - 392 - 0, places=2
        )

    def test_portfolio_bts_with_overnight_split(self):
        algo_code = dedent(
            """
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
            """
        )

        results = self.run_algorithm(script=algo_code)

        # On 1/07, positions value should by 780, same as without split
        assert results.pos_value.iloc[0] == 0
        assert results.pos_value.iloc[1] == 780

        # On 1/07, after applying the split, 1 share becomes 2
        assert results.pos_amount.iloc[0] == 0
        assert results.pos_amount.iloc[1] == 2

        # On 1/07, after applying the split, last sale price is halved
        assert results.last_sale_price.iloc[0] == 0
        assert results.last_sale_price.iloc[1] == 390

    def test_account_bts_with_overnight_split(self):
        algo_code = dedent(
            """
            from zipline.api import order, sid, record, set_slippage, slippage

            def initialize(context):
                context.ordered = False
                context.hd_account = context.account
                set_slippage(slippage.VolumeShareSlippage())


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
            """
        )

        results = self.run_algorithm(script=algo_code)

        # On 1/07, portfolio value is the same as without split
        assert results.port_value.iloc[0] == 10000
        self.assertAlmostEqual(
            results.port_value.iloc[1], 10000 + 780 - 392 - 0, places=2
        )


class TestAlgoScript(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-12-31")
    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = 5  # max history window length

    STRING_TYPE_NAMES = [str.__name__]
    STRING_TYPE_NAMES_STRING = ", ".join(STRING_TYPE_NAMES)
    ASSET_TYPE_NAME = Asset.__name__
    CONTINUOUS_FUTURE_NAME = ContinuousFuture.__name__
    ASSET_OR_STRING_TYPE_NAMES = ", ".join([ASSET_TYPE_NAME] + STRING_TYPE_NAMES)
    ASSET_OR_STRING_OR_CF_TYPE_NAMES = ", ".join(
        [ASSET_TYPE_NAME, CONTINUOUS_FUTURE_NAME] + STRING_TYPE_NAMES
    )
    ARG_TYPE_TEST_CASES = (
        (
            "history__assets",
            (bad_type_history_assets, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True),
        ),
        ("history__fields", (bad_type_history_fields, STRING_TYPE_NAMES_STRING, True)),
        ("history__bar_count", (bad_type_history_bar_count, "int", False)),
        (
            "history__frequency",
            (bad_type_history_frequency, STRING_TYPE_NAMES_STRING, False),
        ),
        (
            "current__assets",
            (bad_type_current_assets, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True),
        ),
        ("current__fields", (bad_type_current_fields, STRING_TYPE_NAMES_STRING, True)),
        ("is_stale__assets", (bad_type_is_stale_assets, "Asset", True)),
        ("can_trade__assets", (bad_type_can_trade_assets, "Asset", True)),
        (
            "history_kwarg__assets",
            (bad_type_history_assets_kwarg, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True),
        ),
        (
            "history_kwarg_bad_list__assets",
            (
                bad_type_history_assets_kwarg_list,
                ASSET_OR_STRING_OR_CF_TYPE_NAMES,
                True,
            ),
        ),
        (
            "history_kwarg__fields",
            (bad_type_history_fields_kwarg, STRING_TYPE_NAMES_STRING, True),
        ),
        ("history_kwarg__bar_count", (bad_type_history_bar_count_kwarg, "int", False)),
        (
            "history_kwarg__frequency",
            (bad_type_history_frequency_kwarg, STRING_TYPE_NAMES_STRING, False),
        ),
        (
            "current_kwarg__assets",
            (bad_type_current_assets_kwarg, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True),
        ),
        (
            "current_kwarg__fields",
            (bad_type_current_fields_kwarg, STRING_TYPE_NAMES_STRING, True),
        ),
    )

    sids = 0, 1, 3, 133

    # FIXME: Pass a benchmark explicitly here.
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        register_calendar("TEST", get_calendar("NYSE"), force=True)

        data = make_simple_equity_info(
            cls.sids,
            cls.START_DATE,
            cls.END_DATE,
        )
        data.loc[3, "symbol"] = "TEST"
        return data

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        cal = cls.trading_calendars[Equity]
        sessions = cal.sessions_in_range(cls.START_DATE, cls.END_DATE)
        frame = pd.DataFrame(
            {
                "close": 10.0,
                "high": 10.5,
                "low": 9.5,
                "open": 10.0,
                "volume": 100,
            },
            index=sessions,
        )

        for sid in sids:
            yield sid, frame

    def test_noop(self):
        self.run_algorithm(
            initialize=initialize_noop,
            handle_data=handle_data_noop,
        )

    def test_noop_string(self):
        self.run_algorithm(script=noop_algo)

    def test_no_handle_data(self):
        self.run_algorithm(script=no_handle_data)

    def test_api_calls(self):
        self.run_algorithm(
            initialize=initialize_api,
            handle_data=handle_data_api,
        )

    def test_api_calls_string(self):
        self.run_algorithm(script=api_algo)

    def test_api_get_environment(self):
        platform = "zipline"
        algo = self.make_algo(
            script=api_get_environment_algo,
            platform=platform,
        )
        algo.run()
        assert algo.environment == platform

    def test_api_symbol(self):
        self.run_algorithm(script=api_symbol_algo)

    def test_fixed_slippage(self):
        # verify order -> transaction -> portfolio position.
        # --------------
        test_algo = self.make_algo(
            script=dedent(
                """
                from zipline.api import (
                    slippage,
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

                    context.incr += 1
                """
            ),
        )
        results = test_algo.run()

        # flatten the list of txns
        all_txns = [
            val for sublist in results["transactions"].tolist() for val in sublist
        ]

        assert len(all_txns) == 1
        txn = all_txns[0]

        expected_spread = 0.05
        expected_price = test_algo.recorded_vars["price"] - expected_spread

        assert expected_price == txn["price"]

        # make sure that the $100 commission was applied to our cash
        # the txn was for -1000 shares at 9.95, means -9.95k.  our capital_used
        # for that day was therefore 9.95k, but after the $100 commission,
        # it should be 9.85k.
        assert 9850 == results.capital_used[1]
        assert 100 == results["orders"].iloc[1][0]["commission"]

    @parameterized.expand(
        [
            ("no_minimum_commission", 0),
            ("default_minimum_commission", 0),
            ("alternate_minimum_commission", 2),
        ]
    )
    def test_volshare_slippage(self, name, minimum_commission):
        tempdir = TempDirectory()
        try:
            if name == "default_minimum_commission":
                commission_line = "set_commission(commission.PerShare(0.02))"
            else:
                commission_line = (
                    "set_commission(commission.PerShare(0.02, "
                    "min_trade_cost={0}))".format(minimum_commission)
                )

            # verify order -> transaction -> portfolio position.
            # --------------
            # XXX: This is the last remaining consumer of
            #      create_daily_trade_source.
            trades = factory.create_daily_trade_source(
                [0], self.sim_params, self.asset_finder, self.trading_calendar
            )
            data_portal = create_data_portal_from_trade_history(
                self.asset_finder,
                self.trading_calendar,
                tempdir,
                self.sim_params,
                {0: trades},
            )
            test_algo = self.make_algo(
                data_portal=data_portal,
                script=dedent(
                    f"""
                    from zipline.api import *

                    def initialize(context):
                        model = slippage.VolumeShareSlippage(
                                                volume_limit=.3,
                                                price_impact=0.05
                                        )
                        set_slippage(model)
                        {commission_line}

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
                        """
                ),
            )
            results = test_algo.run()

            all_txns = [
                val for sublist in results["transactions"].tolist() for val in sublist
            ]

            assert len(all_txns) == 67
            # all_orders are all the incremental versions of the
            # orders as each new fill comes in.
            all_orders = list(toolz.concat(results["orders"]))

            if minimum_commission == 0:
                # for each incremental version of each order, the commission
                # should be its filled amount * 0.02
                for order_ in all_orders:
                    assert (
                        round(abs(order_["filled"] * 0.02 - order_["commission"]), 7)
                        == 0
                    )
            else:
                # the commission should be at least the min_trade_cost
                for order_ in all_orders:
                    if order_["filled"] > 0:
                        assert (
                            round(
                                abs(
                                    max(order_["filled"] * 0.02, minimum_commission)
                                    - order_["commission"]
                                ),
                                7,
                            )
                            == 0
                        )
                    else:
                        assert 0 == order_["commission"]
        finally:
            tempdir.cleanup()

    def test_incorrectly_set_futures_slippage_model(self):
        code = dedent(
            """
            from zipline.api import set_slippage, slippage

            class MySlippage(slippage.FutureSlippageModel):
                def process_order(self, data, order):
                    return data.current(order.asset, 'price'), order.amount

            def initialize(context):
                set_slippage(MySlippage())
            """
        )
        test_algo = self.make_algo(script=code)
        with pytest.raises(IncompatibleSlippageModel):
            # Passing a futures slippage model as the first argument, which is
            # for setting equity models, should fail.
            test_algo.run()

    def test_algo_record_vars(self):
        test_algo = self.make_algo(script=record_variables)
        results = test_algo.run()

        for i in range(1, 252):
            assert results.iloc[i - 1]["incr"] == i

    def test_algo_record_nan(self):
        test_algo = self.make_algo(script=record_float_magic % "nan")
        results = test_algo.run()
        for i in range(1, 252):
            assert np.isnan(results.iloc[i - 1]["data"])

    def test_batch_market_order_matches_multiple_manual_orders(self):
        share_counts = pd.Series([50, 100])

        multi_blotter = RecordBatchBlotter()
        multi_test_algo = self.make_algo(
            script=dedent(
                """
                from collections import OrderedDict
                from zipline.api import sid, order


                def initialize(context):
                    context.assets = [sid(0), sid(3)]
                    context.placed = False

                def handle_data(context, data):
                    if not context.placed:
                        it = zip(context.assets, {share_counts})
                        for asset, shares in it:
                            order(asset, shares)

                        context.placed = True

                """
            ).format(share_counts=list(share_counts)),
            blotter=multi_blotter,
        )
        multi_stats = multi_test_algo.run()
        assert not multi_blotter.order_batch_called

        batch_blotter = RecordBatchBlotter()
        batch_test_algo = self.make_algo(
            script=dedent(
                """
                import pandas as pd

                from zipline.api import sid, batch_market_order


                def initialize(context):
                    context.assets = [sid(0), sid(3)]
                    context.placed = False

                def handle_data(context, data):
                    if not context.placed:
                        orders = batch_market_order(pd.Series(
                            index=context.assets, data={share_counts}
                        ))
                        assert len(orders) == 2, \
                            "len(orders) was %s but expected 2" % len(orders)
                        for o in orders:
                            assert o is not None, "An order is None"

                        context.placed = True

                """
            ).format(share_counts=list(share_counts)),
            blotter=batch_blotter,
        )
        batch_stats = batch_test_algo.run()
        assert batch_blotter.order_batch_called

        for stats in (multi_stats, batch_stats):
            stats.orders = stats.orders.apply(
                lambda orders: [toolz.dissoc(o, "id") for o in orders]
            )
            stats.transactions = stats.transactions.apply(
                lambda txns: [toolz.dissoc(txn, "order_id") for txn in txns]
            )
        assert_equal(multi_stats.sort_index(axis=1), batch_stats.sort_index(axis=1))

    def test_batch_market_order_filters_null_orders(self):
        share_counts = [50, 0]

        batch_blotter = RecordBatchBlotter()
        batch_test_algo = self.make_algo(
            script=dedent(
                """
                import pandas as pd

                from zipline.api import sid, batch_market_order

                def initialize(context):
                    context.assets = [sid(0), sid(3)]
                    context.placed = False

                def handle_data(context, data):
                    if not context.placed:
                        orders = batch_market_order(pd.Series(
                            index=context.assets, data={share_counts}
                        ))
                        assert len(orders) == 1, \
                            "len(orders) was %s but expected 1" % len(orders)
                        for o in orders:
                            assert o is not None, "An order is None"

                        context.placed = True

                """
            ).format(share_counts=share_counts),
            blotter=batch_blotter,
        )
        batch_test_algo.run()
        assert batch_blotter.order_batch_called

    def test_order_dead_asset(self):
        # after asset 0 is dead
        params = SimulationParameters(
            start_session=pd.Timestamp("2007-01-03"),
            end_session=pd.Timestamp("2007-01-05"),
            trading_calendar=self.trading_calendar,
        )

        # order method shouldn't blow up
        self.run_algorithm(
            script=dedent(
                """
                from zipline.api import order, sid

                def initialize(context):
                    pass

                def handle_data(context, data):
                    order(sid(0), 10)
                """
            )
        )

        # order_value and order_percent should blow up
        for order_str in ["order_value", "order_percent"]:
            test_algo = self.make_algo(
                script=dedent(
                    f"""
                from zipline.api import order_percent, order_value, sid

                def initialize(context):
                    pass

                def handle_data(context, data):
                    {order_str}(sid(0), 10)"""
                ),
                sim_params=params,
            )

        with pytest.raises(CannotOrderDelistedAsset):
            test_algo.run()

    def test_portfolio_in_init(self):
        """Test that accessing portfolio in init doesn't break."""
        self.run_algorithm(script=access_portfolio_in_init)

    def test_account_in_init(self):
        """Test that accessing account in init doesn't break."""
        self.run_algorithm(script=access_account_in_init)

    def test_without_kwargs(self):
        """Test that api methods on the data object can be called with positional
        arguments.
        """
        params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-10"),
            end_session=pd.Timestamp("2006-01-11"),
            trading_calendar=self.trading_calendar,
        )
        self.run_algorithm(sim_params=params, script=call_without_kwargs)

    def test_good_kwargs(self):
        """Test that api methods on the data object can be called with keyword
        arguments.
        """
        params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-10"),
            end_session=pd.Timestamp("2006-01-11"),
            trading_calendar=self.trading_calendar,
        )
        self.run_algorithm(script=call_with_kwargs, sim_params=params)

    @parameterized.expand(
        [
            ("history", call_with_bad_kwargs_history),
            ("current", call_with_bad_kwargs_current),
        ]
    )
    def test_bad_kwargs(self, name, algo_text):
        """Test that api methods on the data object called with bad kwargs return
        a meaningful TypeError that we create, rather than an unhelpful cython
        error
        """
        algo = self.make_algo(script=algo_text)
        with pytest.raises(TypeError) as cm:
            algo.run()

        assert (
            "%s() got an unexpected keyword argument 'blahblah'" % name
            == cm.value.args[0]
        )

    @parameterized.expand(ARG_TYPE_TEST_CASES)
    def test_arg_types(self, name, inputs):

        keyword = name.split("__")[1]

        algo = self.make_algo(script=inputs[0])
        with pytest.raises(TypeError) as cm:
            algo.run()

        expected = "Expected %s argument to be of type %s%s" % (
            keyword,
            "or iterable of type " if inputs[2] else "",
            inputs[1],
        )

        assert expected == cm.value.args[0]

    def test_empty_asset_list_to_history(self):
        params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-10"),
            end_session=pd.Timestamp("2006-01-11"),
            trading_calendar=self.trading_calendar,
        )

        self.run_algorithm(
            script=dedent(
                """
                def initialize(context):
                    pass

                def handle_data(context, data):
                    data.history([], "price", 5, '1d')
                """
            ),
            sim_params=params,
        )

    @parameterized.expand(
        [
            ("bad_kwargs", call_with_bad_kwargs_get_open_orders),
            ("good_kwargs", call_with_good_kwargs_get_open_orders),
            ("no_kwargs", call_with_no_kwargs_get_open_orders),
        ]
    )
    def test_get_open_orders_kwargs(self, name, script):
        algo = self.make_algo(script=script)
        if name == "bad_kwargs":
            with pytest.raises(TypeError) as cm:
                algo.run()
                assert (
                    "Keyword argument `sid` is no longer "
                    "supported for get_open_orders. Use `asset` "
                    "instead." == cm.value.args[0]
                )
        else:
            algo.run()

    def test_empty_positions(self):
        """Test that when we try context.portfolio.positions[stock] on a stock
        for which we have no positions, we return a Position with values 0
        (but more importantly, we don't crash) and don't save this Position
        to the user-facing dictionary PositionTracker._positions_store
        """
        results = self.run_algorithm(script=empty_positions)
        num_positions = results.num_positions
        amounts = results.amounts
        assert all(num_positions == 0)
        assert all(amounts == 0)

    def test_schedule_function_time_rule_positionally_misplaced(self):
        """Test that when a user specifies a time rule for the date_rule argument,
        but no rule in the time_rule argument
        (e.g. schedule_function(func, <time_rule>)), we assume that means
        assign a time rule but no date rule
        """

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-12"),
            end=pd.Timestamp("2006-01-13"),
            data_frequency="minute",
        )

        algocode = dedent(
            """
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
        """
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", PerformanceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

            algo = self.make_algo(script=algocode, sim_params=sim_params)
            algo.run()

            assert len(w) == 2

            for i, warning in enumerate(w):
                assert isinstance(warning.message, UserWarning)
                assert (
                    warning.message.args[0]
                    == "Got a time rule for the second positional argument "
                    "date_rule. You should use keyword argument "
                    "time_rule= when calling schedule_function without "
                    "specifying a date_rule"
                )

                # The warnings come from line 13 and 14 in the algocode
                assert warning.lineno == 13 + i

        assert algo.done_at_open == [
            pd.Timestamp("2006-01-12 14:31:00", tz="UTC"),
            pd.Timestamp("2006-01-13 14:31:00", tz="UTC"),
        ]

        assert algo.done_at_close == [
            pd.Timestamp("2006-01-12 20:59:00", tz="UTC"),
            pd.Timestamp("2006-01-13 20:59:00", tz="UTC"),
        ]


class TestCapitalChanges(zf.WithMakeAlgo, zf.ZiplineTestCase):

    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-09")

    # XXX: This suite only has daily data for sid 0 and only has minutely data
    #      for sid 1.
    sids = ASSET_FINDER_EQUITY_SIDS = (0, 1)
    DAILY_SID = 0
    MINUTELY_SID = 1

    # FIXME: Pass a benchmark source explicitly here.
    BENCHMARK_SID = None

    @classmethod
    def make_equity_minute_bar_data(cls):
        minutes = cls.trading_calendar.minutes_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )
        closes = np.arange(100, 100 + len(minutes), 1)
        opens = closes
        highs = closes + 5
        lows = closes - 5

        frame = pd.DataFrame(
            index=minutes,
            data={
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": 10000,
            },
        )

        yield cls.MINUTELY_SID, frame

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        days = cls.trading_calendar.sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )

        closes = np.arange(10.0, 10.0 + len(days), 1.0)
        opens = closes
        highs = closes + 0.5
        lows = closes - 0.5

        frame = pd.DataFrame(
            index=days,
            data={
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": 10000,
            },
        )

        yield cls.DAILY_SID, frame

    @parameterized.expand([("target", 151000.0), ("delta", 50000.0)])
    def test_capital_changes_daily_mode(self, change_type, value):
        capital_changes = {
            pd.Timestamp("2006-01-06", tz="UTC"): {"type": change_type, "value": value}
        }

        algocode = dedent(
            """
            from zipline.api import (
                set_slippage,
                set_commission,
                slippage,
                commission,
                schedule_function,
                time_rules,
                order,
                sid)

            def initialize(context):
                set_slippage(slippage.FixedSlippage(spread=0))
                set_commission(commission.PerShare(0, 0))
                schedule_function(order_stuff, time_rule=time_rules.market_open())

            def order_stuff(context, data):
                order(sid(0), 1000)
            """
        )
        algo = self.make_algo(
            script=algocode,
            capital_changes=capital_changes,
            sim_params=SimulationParameters(
                start_session=self.START_DATE,
                end_session=self.END_DATE,
                trading_calendar=self.nyse_calendar,
            ),
        )

        # We call get_generator rather than `run()` here because we care about
        # the raw capital change packets.
        gen = algo.get_generator()
        results = list(gen)

        cumulative_perf = [
            r["cumulative_perf"] for r in results if "cumulative_perf" in r
        ]
        daily_perf = [r["daily_perf"] for r in results if "daily_perf" in r]
        capital_change_packets = [
            r["capital_change"] for r in results if "capital_change" in r
        ]

        assert len(capital_change_packets) == 1
        assert capital_change_packets[0] == {
            "date": pd.Timestamp("2006-01-06", tz="UTC"),
            "type": "cash",
            "target": 151000.0 if change_type == "target" else None,
            "delta": 50000.0,
        }

        # 1/03: price = 10, place orders
        # 1/04: orders execute at price = 11, place orders
        # 1/05: orders execute at price = 12, place orders
        # 1/06: +50000 capital change,
        #       orders execute at price = 13, place orders
        # 1/09: orders execute at price = 14, place orders

        expected_capital_changes = np.array([0.0, 0.0, 0.0, 50000.0, 0.0])

        expected_daily = {}
        # Day 1, no transaction. Day 2, we transact, but the price of our stock
        # does not change. Day 3, we start getting returns
        expected_daily["returns"] = np.array(
            [
                0.0,
                0.0,
                # 1000 shares * gain of 1
                (100000.0 + 1000.0) / 100000.0 - 1.0,
                # 2000 shares * gain of 1, capital change of +50000
                (151000.0 + 2000.0) / 151000.0 - 1.0,
                # 3000 shares * gain of 1
                (153000.0 + 3000.0) / 153000.0 - 1.0,
            ]
        )

        expected_daily["pnl"] = np.array(
            [
                0.0,
                0.0,
                1000.00,  # 1000 shares * gain of 1
                2000.00,  # 2000 shares * gain of 1
                3000.00,  # 3000 shares * gain of 1
            ]
        )

        expected_daily["capital_used"] = np.array(
            [
                0.0,
                -11000.0,  # 1000 shares at price = 11
                -12000.0,  # 1000 shares at price = 12
                -13000.0,  # 1000 shares at price = 13
                -14000.0,  # 1000 shares at price = 14
            ]
        )

        expected_daily["ending_cash"] = (
            np.array([100000.0] * 5)
            + np.cumsum(expected_capital_changes)
            + np.cumsum(expected_daily["capital_used"])
        )

        expected_daily["starting_cash"] = (
            expected_daily["ending_cash"] - expected_daily["capital_used"]
        )

        expected_daily["starting_value"] = np.array(
            [
                0.0,
                0.0,
                11000.0,  # 1000 shares at price = 11
                24000.0,  # 2000 shares at price = 12
                39000.0,  # 3000 shares at price = 13
            ]
        )

        expected_daily["ending_value"] = (
            expected_daily["starting_value"]
            + expected_daily["pnl"]
            - expected_daily["capital_used"]
        )

        expected_daily["portfolio_value"] = (
            expected_daily["ending_value"] + expected_daily["ending_cash"]
        )

        stats = [
            "returns",
            "pnl",
            "capital_used",
            "starting_cash",
            "ending_cash",
            "starting_value",
            "ending_value",
            "portfolio_value",
        ]

        expected_cumulative = {
            "returns": np.cumprod(expected_daily["returns"] + 1) - 1,
            "pnl": np.cumsum(expected_daily["pnl"]),
            "capital_used": np.cumsum(expected_daily["capital_used"]),
            "starting_cash": np.repeat(expected_daily["starting_cash"][0:1], 5),
            "ending_cash": expected_daily["ending_cash"],
            "starting_value": np.repeat(expected_daily["starting_value"][0:1], 5),
            "ending_value": expected_daily["ending_value"],
            "portfolio_value": expected_daily["portfolio_value"],
        }

        for stat in stats:
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in daily_perf]),
                expected_daily[stat],
                err_msg="daily " + stat,
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in cumulative_perf]),
                expected_cumulative[stat],
                err_msg="cumulative " + stat,
            )

        assert algo.capital_change_deltas == {
            pd.Timestamp("2006-01-06", tz="UTC"): 50000.0
        }

    @parameterized.expand(
        [
            ("interday_target", [("2006-01-04", 2388.0)]),
            ("interday_delta", [("2006-01-04", 1000.0)]),
            (
                "intraday_target",
                [("2006-01-04 17:00", 2184.0), ("2006-01-04 18:00", 2804.0)],
            ),
            (
                "intraday_delta",
                [("2006-01-04 17:00", 500.0), ("2006-01-04 18:00", 500.0)],
            ),
        ]
    )
    def test_capital_changes_minute_mode_daily_emission(self, change, values):
        change_loc, change_type = change.split("_")

        sim_params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-03"),
            end_session=pd.Timestamp("2006-01-05"),
            data_frequency="minute",
            capital_base=1000.0,
            trading_calendar=self.nyse_calendar,
        )

        capital_changes = {
            pd.Timestamp(datestr, tz="UTC"): {"type": change_type, "value": value}
            for datestr, value in values
        }

        algocode = dedent(
            """
            from zipline.api import (
                set_slippage,
                set_commission,
                slippage,
                commission,
                schedule_function,
                time_rules,
                order,
                sid,
                )

            def initialize(context):
                set_slippage(slippage.FixedSlippage(spread=0))
                set_commission(commission.PerShare(0, 0))
                schedule_function(order_stuff, time_rule=time_rules.market_open())

            def order_stuff(context, data):
                order(sid(1), 1)
            """
        )

        algo = self.make_algo(
            script=algocode, sim_params=sim_params, capital_changes=capital_changes
        )

        gen = algo.get_generator()
        results = list(gen)

        cumulative_perf = [
            r["cumulative_perf"] for r in results if "cumulative_perf" in r
        ]
        daily_perf = [r["daily_perf"] for r in results if "daily_perf" in r]
        capital_change_packets = [
            r["capital_change"] for r in results if "capital_change" in r
        ]

        assert len(capital_change_packets) == len(capital_changes)
        expected = [
            {
                "date": pd.Timestamp(val[0], tz="UTC"),
                "type": "cash",
                "target": val[1] if change_type == "target" else None,
                "delta": 1000.0 if len(values) == 1 else 500.0,
            }
            for val in values
        ]
        assert capital_change_packets == expected

        # 1/03: place orders at price = 100, execute at 101
        # 1/04: place orders at price = 490, execute at 491,
        #       +500 capital change at 17:00 and 18:00 (intraday)
        #       or +1000 at 00:00 (interday),
        # 1/05: place orders at price = 880, execute at 881

        expected_daily = {}

        expected_capital_changes = np.array([0.0, 1000.0, 0.0])

        if change_loc == "intraday":
            # Fills at 491, +500 capital change comes at 638 (17:00) and
            # 698 (18:00), ends day at 879
            day2_return = (1388.0 + 149.0 + 147.0) / 1388.0 * (
                2184.0 + 60.0 + 60.0
            ) / 2184.0 * (2804.0 + 181.0 + 181.0) / 2804.0 - 1.0
        else:
            # Fills at 491, ends day at 879, capital change +1000
            day2_return = (2388.0 + 390.0 + 388.0) / 2388.0 - 1

        expected_daily["returns"] = np.array(
            [
                # Fills at 101, ends day at 489
                (1000.0 + 489 - 101) / 1000.0 - 1.0,
                day2_return,
                # Fills at 881, ends day at 1269
                (3166.0 + 390.0 + 390.0 + 388.0) / 3166.0 - 1.0,
            ]
        )

        expected_daily["pnl"] = np.array(
            [
                388.0,
                390.0 + 388.0,
                390.0 + 390.0 + 388.0,
            ]
        )

        expected_daily["capital_used"] = np.array([-101.0, -491.0, -881.0])

        expected_daily["ending_cash"] = (
            np.array([1000.0] * 3)
            + np.cumsum(expected_capital_changes)
            + np.cumsum(expected_daily["capital_used"])
        )

        expected_daily["starting_cash"] = (
            expected_daily["ending_cash"] - expected_daily["capital_used"]
        )

        if change_loc == "intraday":
            # Capital changes come after day start
            expected_daily["starting_cash"] -= expected_capital_changes

        expected_daily["starting_value"] = np.array([0.0, 489.0, 879.0 * 2])

        expected_daily["ending_value"] = (
            expected_daily["starting_value"]
            + expected_daily["pnl"]
            - expected_daily["capital_used"]
        )

        expected_daily["portfolio_value"] = (
            expected_daily["ending_value"] + expected_daily["ending_cash"]
        )

        stats = [
            "returns",
            "pnl",
            "capital_used",
            "starting_cash",
            "ending_cash",
            "starting_value",
            "ending_value",
            "portfolio_value",
        ]

        expected_cumulative = {
            "returns": np.cumprod(expected_daily["returns"] + 1) - 1,
            "pnl": np.cumsum(expected_daily["pnl"]),
            "capital_used": np.cumsum(expected_daily["capital_used"]),
            "starting_cash": np.repeat(expected_daily["starting_cash"][0:1], 3),
            "ending_cash": expected_daily["ending_cash"],
            "starting_value": np.repeat(expected_daily["starting_value"][0:1], 3),
            "ending_value": expected_daily["ending_value"],
            "portfolio_value": expected_daily["portfolio_value"],
        }

        for stat in stats:
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in daily_perf]), expected_daily[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in cumulative_perf]),
                expected_cumulative[stat],
            )

        if change_loc == "interday":
            assert algo.capital_change_deltas == {
                pd.Timestamp("2006-01-04", tz="UTC"): 1000.0
            }
        else:
            assert algo.capital_change_deltas == {
                pd.Timestamp("2006-01-04 17:00", tz="UTC"): 500.0,
                pd.Timestamp("2006-01-04 18:00", tz="UTC"): 500.0,
            }

    @parameterized.expand(
        [
            ("interday_target", [("2006-01-04", 2388.0)]),
            ("interday_delta", [("2006-01-04", 1000.0)]),
            (
                "intraday_target",
                [("2006-01-04 17:00", 2184.0), ("2006-01-04 18:00", 2804.0)],
            ),
            (
                "intraday_delta",
                [("2006-01-04 17:00", 500.0), ("2006-01-04 18:00", 500.0)],
            ),
        ]
    )
    def test_capital_changes_minute_mode_minute_emission(self, change, values):
        change_loc, change_type = change.split("_")

        sim_params = SimulationParameters(
            start_session=pd.Timestamp("2006-01-03"),
            end_session=pd.Timestamp("2006-01-05"),
            data_frequency="minute",
            emission_rate="minute",
            capital_base=1000.0,
            trading_calendar=self.nyse_calendar,
        )

        capital_changes = {
            pd.Timestamp(val[0], tz="UTC"): {"type": change_type, "value": val[1]}
            for val in values
        }

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

        algo = self.make_algo(
            script=algocode, sim_params=sim_params, capital_changes=capital_changes
        )

        gen = algo.get_generator()
        results = list(gen)

        cumulative_perf = [
            r["cumulative_perf"] for r in results if "cumulative_perf" in r
        ]
        minute_perf = [r["minute_perf"] for r in results if "minute_perf" in r]
        daily_perf = [r["daily_perf"] for r in results if "daily_perf" in r]
        capital_change_packets = [
            r["capital_change"] for r in results if "capital_change" in r
        ]

        assert len(capital_change_packets) == len(capital_changes)
        expected = [
            {
                "date": pd.Timestamp(val[0], tz="UTC"),
                "type": "cash",
                "target": val[1] if change_type == "target" else None,
                "delta": 1000.0 if len(values) == 1 else 500.0,
            }
            for val in values
        ]
        assert capital_change_packets == expected

        # 1/03: place orders at price = 100, execute at 101
        # 1/04: place orders at price = 490, execute at 491,
        #       +500 capital change at 17:00 and 18:00 (intraday)
        #       or +1000 at 00:00 (interday),
        # 1/05: place orders at price = 880, execute at 881

        # Minute perfs are cumulative for the day
        expected_minute = {}

        capital_changes_after_start = np.array([0.0] * 1170)
        if change_loc == "intraday":
            capital_changes_after_start[539:599] = 500.0
            capital_changes_after_start[599:780] = 1000.0

        expected_minute["pnl"] = np.array([0.0] * 1170)
        expected_minute["pnl"][:2] = 0.0
        expected_minute["pnl"][2:392] = 1.0
        expected_minute["pnl"][392:782] = 2.0
        expected_minute["pnl"][782:] = 3.0
        for start, end in ((0, 390), (390, 780), (780, 1170)):
            expected_minute["pnl"][start:end] = np.cumsum(
                expected_minute["pnl"][start:end]
            )

        expected_minute["capital_used"] = np.concatenate(
            (
                [0.0] * 1,
                [-101.0] * 389,
                [0.0] * 1,
                [-491.0] * 389,
                [0.0] * 1,
                [-881.0] * 389,
            )
        )

        # +1000 capital changes comes before the day start if interday
        day2adj = 0.0 if change_loc == "intraday" else 1000.0

        expected_minute["starting_cash"] = np.concatenate(
            (
                [1000.0] * 390,
                # 101 spent on 1/03
                [1000.0 - 101.0 + day2adj] * 390,
                # 101 spent on 1/03, 491 on 1/04, +1000 capital change on 1/04
                [1000.0 - 101.0 - 491.0 + 1000] * 390,
            )
        )

        expected_minute["ending_cash"] = (
            expected_minute["starting_cash"]
            + expected_minute["capital_used"]
            + capital_changes_after_start
        )

        expected_minute["starting_value"] = np.concatenate(
            ([0.0] * 390, [489.0] * 390, [879.0 * 2] * 390)
        )

        expected_minute["ending_value"] = (
            expected_minute["starting_value"]
            + expected_minute["pnl"]
            - expected_minute["capital_used"]
        )

        expected_minute["portfolio_value"] = (
            expected_minute["ending_value"] + expected_minute["ending_cash"]
        )

        expected_minute["returns"] = expected_minute["pnl"] / (
            expected_minute["starting_value"] + expected_minute["starting_cash"]
        )

        # If the change is interday, we can just calculate the returns from
        # the pnl, starting_value and starting_cash. If the change is intraday,
        # the returns after the change have to be calculated from two
        # subperiods
        if change_loc == "intraday":
            # The last packet (at 1/04 16:59) before the first capital change
            prev_subperiod_return = expected_minute["returns"][538]

            # From 1/04 17:00 to 17:59
            cur_subperiod_pnl = (
                expected_minute["pnl"][539:599] - expected_minute["pnl"][538]
            )
            cur_subperiod_starting_value = np.array(
                [expected_minute["ending_value"][538]] * 60
            )
            cur_subperiod_starting_cash = np.array(
                [expected_minute["ending_cash"][538] + 500] * 60
            )

            cur_subperiod_returns = cur_subperiod_pnl / (
                cur_subperiod_starting_value + cur_subperiod_starting_cash
            )
            expected_minute["returns"][539:599] = (cur_subperiod_returns + 1.0) * (
                prev_subperiod_return + 1.0
            ) - 1.0

            # The last packet (at 1/04 17:59) before the second capital change
            prev_subperiod_return = expected_minute["returns"][598]

            # From 1/04 18:00 to 21:00
            cur_subperiod_pnl = (
                expected_minute["pnl"][599:780] - expected_minute["pnl"][598]
            )
            cur_subperiod_starting_value = np.array(
                [expected_minute["ending_value"][598]] * 181
            )
            cur_subperiod_starting_cash = np.array(
                [expected_minute["ending_cash"][598] + 500] * 181
            )

            cur_subperiod_returns = cur_subperiod_pnl / (
                cur_subperiod_starting_value + cur_subperiod_starting_cash
            )
            expected_minute["returns"][599:780] = (cur_subperiod_returns + 1.0) * (
                prev_subperiod_return + 1.0
            ) - 1.0

        # The last minute packet of each day
        expected_daily = {
            k: np.array([v[389], v[779], v[1169]]) for k, v in expected_minute.items()
        }

        stats = [
            "pnl",
            "capital_used",
            "starting_cash",
            "ending_cash",
            "starting_value",
            "ending_value",
            "portfolio_value",
            "returns",
        ]

        expected_cumulative = deepcopy(expected_minute)

        # "Add" daily return from 1/03 to minute returns on 1/04 and 1/05
        # "Add" daily return from 1/04 to minute returns on 1/05
        expected_cumulative["returns"][390:] = (
            expected_cumulative["returns"][390:] + 1
        ) * (expected_daily["returns"][0] + 1) - 1
        expected_cumulative["returns"][780:] = (
            expected_cumulative["returns"][780:] + 1
        ) * (expected_daily["returns"][1] + 1) - 1

        # Add daily pnl/capital_used from 1/03 to 1/04 and 1/05
        # Add daily pnl/capital_used from 1/04 to 1/05
        expected_cumulative["pnl"][390:] += expected_daily["pnl"][0]
        expected_cumulative["pnl"][780:] += expected_daily["pnl"][1]
        expected_cumulative["capital_used"][390:] += expected_daily["capital_used"][0]
        expected_cumulative["capital_used"][780:] += expected_daily["capital_used"][1]

        # starting_cash, starting_value are same as those of the first daily
        # packet
        expected_cumulative["starting_cash"] = np.repeat(
            expected_daily["starting_cash"][0:1], 1170
        )
        expected_cumulative["starting_value"] = np.repeat(
            expected_daily["starting_value"][0:1], 1170
        )

        # extra cumulative packet per day from the daily packet
        for stat in stats:
            for i in (390, 781, 1172):
                expected_cumulative[stat] = np.insert(
                    expected_cumulative[stat], i, expected_cumulative[stat][i - 1]
                )

        for stat in stats:
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in minute_perf]), expected_minute[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in daily_perf]), expected_daily[stat]
            )
            np.testing.assert_array_almost_equal(
                np.array([perf[stat] for perf in cumulative_perf]),
                expected_cumulative[stat],
            )

        if change_loc == "interday":
            assert algo.capital_change_deltas == {
                pd.Timestamp("2006-01-04", tz="UTC"): 1000.0
            }
        else:
            assert algo.capital_change_deltas == {
                pd.Timestamp("2006-01-04 17:00", tz="UTC"): 500.0,
                pd.Timestamp("2006-01-04 18:00", tz="UTC"): 500.0,
            }


class TestGetDatetime(zf.WithMakeAlgo, zf.ZiplineTestCase):
    SIM_PARAMS_DATA_FREQUENCY = "minute"
    START_DATE = pd.Timestamp("2014-01-02 9:31")
    END_DATE = pd.Timestamp("2014-01-03 9:31")

    ASSET_FINDER_EQUITY_SIDS = 0, 1

    # FIXME: Pass a benchmark source explicitly here.
    BENCHMARK_SID = None

    @parameterized.expand(
        [
            (
                "default",
                None,
            ),
            (
                "utc",
                "UTC",
            ),
            (
                "us_east",
                "US/Eastern",
            ),
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
                if str(dt.tz) != context.tz:
                    raise ValueError("Mismatched Zone")

                if context.first_bar:
                    if dt.tz_convert("US/Eastern").hour != 9:
                        raise ValueError("Mismatched Hour")
                    elif dt.tz_convert("US/Eastern").minute != 31:
                        raise ValueError("Mismatched Minute")

                    context.first_bar = False
            """.format(
                tz=repr(tz)
            )
        )

        algo = self.make_algo(script=algo)
        algo.run()
        assert not algo.first_bar


class TestTradingControls(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-06")

    sid = 133
    sids = ASSET_FINDER_EQUITY_SIDS = 133, 134

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = True

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    @classmethod
    def init_class_fixtures(cls):
        super(TestTradingControls, cls).init_class_fixtures()
        cls.asset = cls.asset_finder.retrieve_asset(cls.sid)
        cls.another_asset = cls.asset_finder.retrieve_asset(134)

    def _check_algo(self, algo, expected_order_count, expected_exc):

        with pytest.raises(expected_exc) if expected_exc else nop_context:
            algo.run()
        assert algo.order_count == expected_order_count

    def check_algo_succeeds(self, algo, order_count=4):
        # Default for order_count assumes one order per handle_data call.
        self._check_algo(algo, order_count, None)

    def check_algo_fails(self, algo, order_count):
        self._check_algo(algo, order_count, TradingControlViolation)

    def test_set_max_position_size(self):
        def initialize(self, asset, max_shares, max_notional):
            self.set_slippage(FixedSlippage())
            self.order_count = 0
            self.set_max_position_size(
                asset=asset, max_shares=max_shares, max_notional=max_notional
            )

        # Buy one share four times.  Should be fine.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1

        algo = self.make_algo(
            asset=self.asset,
            max_shares=10,
            max_notional=500.0,
            initialize=initialize,
            handle_data=handle_data,
        )
        self.check_algo_succeeds(algo)

        # Buy three shares four times.  Should bail on the fourth before it's
        # placed.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 3)
            algo.order_count += 1

        algo = self.make_algo(
            asset=self.asset,
            max_shares=10,
            max_notional=500.0,
            initialize=initialize,
            handle_data=handle_data,
        )
        self.check_algo_fails(algo, 3)

        # Buy three shares four times. Should bail due to max_notional on the
        # third attempt.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 3)
            algo.order_count += 1

        algo = self.make_algo(
            asset=self.asset,
            max_shares=10,
            max_notional=67.0,
            initialize=initialize,
            handle_data=handle_data,
        )
        self.check_algo_fails(algo, 2)

        # Set the trading control to a different sid, then BUY ALL THE THINGS!.
        # Should continue normally.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1

        algo = self.make_algo(
            asset=self.another_asset,
            max_shares=10,
            max_notional=67.0,
            initialize=initialize,
            handle_data=handle_data,
        )
        self.check_algo_succeeds(algo)

        # Set the trading control sid to None, then BUY ALL THE THINGS!. Should
        # fail because setting sid to None makes the control apply to all sids.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1

        algo = self.make_algo(
            max_shares=10,
            max_notional=61.0,
            asset=None,
            initialize=initialize,
            handle_data=handle_data,
        )

        self.check_algo_fails(algo, 0)

    def test_set_asset_restrictions(self):
        def initialize(algo, sid, restrictions, on_error):
            algo.order_count = 0
            algo.set_asset_restrictions(restrictions, on_error)

        def handle_data(algo, data):
            algo.could_trade = data.can_trade(algo.sid(self.sid))
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1

        # Set HistoricalRestrictions for one sid for the entire simulation,
        # and fail.
        rlm = HistoricalRestrictions(
            [
                Restriction(
                    self.sid, self.sim_params.start_session, RESTRICTION_STATES.FROZEN
                )
            ]
        )
        algo = self.make_algo(
            sid=self.sid,
            restrictions=rlm,
            on_error="fail",
            initialize=initialize,
            handle_data=handle_data,
        )
        self.check_algo_fails(algo, 0)
        assert not algo.could_trade

        # Set StaticRestrictions for one sid and fail.
        rlm = StaticRestrictions([self.sid])
        algo = self.make_algo(
            sid=self.sid,
            restrictions=rlm,
            on_error="fail",
            initialize=initialize,
            handle_data=handle_data,
        )

        self.check_algo_fails(algo, 0)
        assert not algo.could_trade

        # just log an error on the violation if we choose not to fail.
        algo = self.make_algo(
            sid=self.sid,
            restrictions=rlm,
            on_error="log",
            initialize=initialize,
            handle_data=handle_data,
        )

        self.check_algo_succeeds(algo)

        assert (
            "Order for 100 shares of Equity(133 [A]) at "
            "2006-01-03 21:00:00+00:00 violates trading constraint "
            "RestrictedListOrder({})" in self._caplog.messages
        )
        assert not algo.could_trade

        # set the restricted list to exclude the sid, and succeed
        rlm = HistoricalRestrictions(
            [
                Restriction(
                    sid, self.sim_params.start_session, RESTRICTION_STATES.FROZEN
                )
                for sid in [134, 135, 136]
            ]
        )
        algo = self.make_algo(
            sid=self.sid,
            restrictions=rlm,
            on_error="fail",
            initialize=initialize,
            handle_data=handle_data,
        )
        self.check_algo_succeeds(algo)
        assert algo.could_trade

    @parameterized.expand(
        [("order_first_restricted_sid", 0), ("order_second_restricted_sid", 1)]
    )
    def test_set_multiple_asset_restrictions(self, name, to_order_idx):
        def initialize(algo, restrictions1, restrictions2, on_error):
            algo.order_count = 0
            algo.set_asset_restrictions(restrictions1, on_error)
            algo.set_asset_restrictions(restrictions2, on_error)

        def handle_data(algo, data):
            algo.could_trade1 = data.can_trade(algo.sid(self.sids[0]))
            algo.could_trade2 = data.can_trade(algo.sid(self.sids[1]))
            algo.order(algo.sid(self.sids[to_order_idx]), 100)
            algo.order_count += 1

        rl1 = StaticRestrictions([self.sids[0]])
        rl2 = StaticRestrictions([self.sids[1]])
        algo = self.make_algo(
            restrictions1=rl1,
            restrictions2=rl2,
            initialize=initialize,
            handle_data=handle_data,
            on_error="fail",
        )
        self.check_algo_fails(algo, 0)
        assert not algo.could_trade1
        assert not algo.could_trade2

    def test_set_do_not_order_list(self):
        def initialize(self, restricted_list):
            self.order_count = 0
            # self.set_do_not_order_list(restricted_list, on_error="fail")
            self.set_asset_restrictions(
                StaticRestrictions(restricted_list), on_error="fail"
            )

        def handle_data(algo, data):
            algo.could_trade = data.can_trade(algo.sid(self.sid))
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1

        rlm = [self.sid]
        algo = self.make_algo(
            restricted_list=rlm,
            initialize=initialize,
            handle_data=handle_data,
        )

        self.check_algo_fails(algo, 0)
        assert not algo.could_trade

    def test_set_max_order_size(self):
        def initialize(algo, asset, max_shares, max_notional):
            algo.order_count = 0
            algo.set_max_order_size(
                asset=asset, max_shares=max_shares, max_notional=max_notional
            )

        # Buy one share.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            asset=self.asset,
            max_shares=10,
            max_notional=500.0,
        )
        self.check_algo_succeeds(algo)

        # Buy 1, then 2, then 3, then 4 shares.  Bail on the last attempt
        # because we exceed shares.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), algo.order_count + 1)
            algo.order_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            asset=self.asset,
            max_shares=3,
            max_notional=500.0,
        )
        self.check_algo_fails(algo, 3)

        # Buy 1, then 2, then 3, then 4 shares.  Bail on the last attempt
        # because we exceed notional.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), algo.order_count + 1)
            algo.order_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            asset=self.asset,
            max_shares=10,
            max_notional=40.0,
        )
        self.check_algo_fails(algo, 3)

        # Set the trading control to a different sid, then BUY ALL THE THINGS!.
        # Should continue normally.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            asset=self.another_asset,
            max_shares=1,
            max_notional=1.0,
        )
        self.check_algo_succeeds(algo)

        # Set the trading control sid to None, then BUY ALL THE THINGS!.
        # Should fail because not specifying a sid makes the trading control
        # apply to all sids.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            asset=None,
            max_shares=1,
            max_notional=1.0,
        )
        self.check_algo_fails(algo, 0)

    def test_set_max_order_count(self):
        def initialize(algo, count):
            algo.order_count = 0
            algo.set_max_order_count(count)

        def handle_data(algo, data):
            for _ in range(5):
                algo.order(self.asset, 1)
                algo.order_count += 1

        algo = self.make_algo(
            count=3,
            initialize=initialize,
            handle_data=handle_data,
        )
        with pytest.raises(TradingControlViolation):
            algo.run()

        assert algo.order_count == 3

    def test_set_max_order_count_minutely(self):
        sim_params = self.make_simparams(data_frequency="minute")

        def initialize(algo, max_orders_per_day):
            algo.minute_count = 0
            algo.order_count = 0
            algo.set_max_order_count(max_orders_per_day)

        # Order 5 times twice in a single day, and set a max order count of
        # 9. The last order of the second batch should fail.
        def handle_data(algo, data):
            if algo.minute_count == 0 or algo.minute_count == 100:
                for _ in range(5):
                    algo.order(self.asset, 1)
                    algo.order_count += 1

            algo.minute_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            max_orders_per_day=9,
            sim_params=sim_params,
        )

        with pytest.raises(TradingControlViolation):
            algo.run()

        assert algo.order_count == 9

        # Set a limit of 5 orders per day, and order 5 times in the first
        # minute of each day. This should succeed because the counter gets
        # reset each day.
        def handle_data(algo, data):
            if (algo.minute_count % 390) == 0:
                for _ in range(5):
                    algo.order(self.asset, 1)
                    algo.order_count += 1

            algo.minute_count += 1

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            max_orders_per_day=5,
            sim_params=sim_params,
        )
        algo.run()

        # 5 orders per day times 4 days.
        assert algo.order_count == 20

    def test_long_only(self):
        def initialize(algo):
            algo.order_count = 0
            algo.set_long_only()

        # Sell immediately -> fail immediately.
        def handle_data(algo, data):
            algo.order(algo.sid(self.sid), -1)
            algo.order_count += 1

        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 0)

        # Buy on even days, sell on odd days.  Never takes a short position, so
        # should succeed.
        def handle_data(algo, data):
            if (algo.order_count % 2) == 0:
                algo.order(algo.sid(self.sid), 1)
            else:
                algo.order(algo.sid(self.sid), -1)
            algo.order_count += 1

        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)

        # Buy on first three days, then sell off holdings.  Should succeed.
        def handle_data(algo, data):
            amounts = [1, 1, 1, -3]
            algo.order(algo.sid(self.sid), amounts[algo.order_count])
            algo.order_count += 1

        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)

        # Buy on first three days, then sell off holdings plus an extra share.
        # Should fail on the last sale.
        def handle_data(algo, data):
            amounts = [1, 1, 1, -4]
            algo.order(algo.sid(self.sid), amounts[algo.order_count])
            algo.order_count += 1

        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 3)

    def test_register_post_init(self):
        def initialize(algo):
            algo.initialized = True

        def handle_data(algo, data):
            with pytest.raises(RegisterTradingControlPostInit):
                algo.set_max_position_size(self.sid, 1, 1)
            with pytest.raises(RegisterTradingControlPostInit):
                algo.set_max_order_size(self.sid, 1, 1)
            with pytest.raises(RegisterTradingControlPostInit):
                algo.set_max_order_count(1)
            with pytest.raises(RegisterTradingControlPostInit):
                algo.set_long_only()

        self.run_algorithm(initialize=initialize, handle_data=handle_data)


class TestAssetDateBounds(zf.WithMakeAlgo, zf.ZiplineTestCase):

    START_DATE = pd.Timestamp("2014-01-02")
    END_DATE = pd.Timestamp("2014-01-03")
    SIM_PARAMS_START_DATE = END_DATE  # Only run for one day.

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    BENCHMARK_SID = 3

    @classmethod
    def make_equity_info(cls):
        T = partial(pd.Timestamp)
        return pd.DataFrame.from_records(
            [
                {
                    "sid": 1,
                    "symbol": "OLD",
                    "start_date": T("1990"),
                    "end_date": T("1991"),
                    "exchange": "TEST",
                },
                {
                    "sid": 2,
                    "symbol": "NEW",
                    "start_date": T("2017"),
                    "end_date": T("2018"),
                    "exchange": "TEST",
                },
                {
                    "sid": 3,
                    "symbol": "GOOD",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE,
                    "exchange": "TEST",
                },
            ]
        )

    def test_asset_date_bounds(self):
        def initialize(algo):
            algo.ran = False
            algo.register_trading_control(AssetDateBounds(on_error="fail"))

        def handle_data(algo, data):
            # This should work because sid 3 is valid during the algo lifetime.
            algo.order(algo.sid(3), 1)

            # Sid already expired.
            with pytest.raises(TradingControlViolation):
                algo.order(algo.sid(1), 1)

            # Sid doesn't exist yet.
            with pytest.raises(TradingControlViolation):
                algo.order(algo.sid(2), 1)

            algo.ran = True

        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        algo.run()
        assert algo.ran


class TestAccountControls(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-01-06")

    (sidint,) = ASSET_FINDER_EQUITY_SIDS = (133,)
    BENCHMARK_SID = None
    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        frame = pd.DataFrame(
            data={
                "close": [10.0, 10.0, 11.0, 11.0],
                "open": [10.0, 10.0, 11.0, 11.0],
                "low": [9.5, 9.5, 10.45, 10.45],
                "high": [10.5, 10.5, 11.55, 11.55],
                "volume": [100, 100, 100, 300],
            },
            index=cls.equity_daily_bar_days,
        )
        yield cls.sidint, frame

    def _check_algo(self, algo, expected_exc):
        with pytest.raises(expected_exc) if expected_exc else nop_context:
            algo.run()

    def check_algo_succeeds(self, algo):
        # Default for order_count assumes one order per handle_data call.
        self._check_algo(algo, None)

    def check_algo_fails(self, algo):
        self._check_algo(algo, AccountControlViolation)

    def test_set_max_leverage(self):
        def initialize(algo, max_leverage):
            algo.set_max_leverage(max_leverage=max_leverage)

        def handle_data(algo, data):
            algo.order(algo.sid(self.sidint), 1)
            algo.record(latest_time=algo.get_datetime())

        # Set max leverage to 0 so buying one share fails.
        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            max_leverage=0,
        )
        self.check_algo_fails(algo)
        assert algo.recorded_vars["latest_time"] == pd.Timestamp(
            "2006-01-04 21:00:00", tz="UTC"
        )

        # Set max leverage to 1 so buying one share passes
        def handle_data(algo, data):
            algo.order(algo.sid(self.sidint), 1)

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            max_leverage=1,
        )
        self.check_algo_succeeds(algo)

    def test_set_min_leverage(self):
        def initialize(algo, min_leverage, grace_period):
            algo.set_min_leverage(min_leverage=min_leverage, grace_period=grace_period)

        def handle_data(algo, data):
            algo.order_target_percent(algo.sid(self.sidint), 0.5)
            algo.record(latest_time=algo.get_datetime())

        # Helper for not having to pass init/handle_data at each callsite.
        def make_algo(min_leverage, grace_period):
            return self.make_algo(
                initialize=initialize,
                handle_data=handle_data,
                min_leverage=min_leverage,
                grace_period=grace_period,
            )

        # Set min leverage to 1.
        # The algorithm will succeed because it doesn't run for more
        # than 10 days.
        offset = pd.Timedelta("10 days")
        algo = make_algo(min_leverage=1, grace_period=offset)
        self.check_algo_succeeds(algo)

        # The algorithm will fail because it doesn't reach a min leverage of 1
        # after 1 day.
        offset = pd.Timedelta("1 days")
        algo = make_algo(min_leverage=1, grace_period=offset)
        self.check_algo_fails(algo)
        assert algo.recorded_vars["latest_time"] == pd.Timestamp(
            "2006-01-04 21:00:00", tz="UTC"
        )

        # Increase the offset to 2 days, and the algorithm fails a day later
        offset = pd.Timedelta("2 days")
        algo = make_algo(min_leverage=1, grace_period=offset)
        self.check_algo_fails(algo)
        assert algo.recorded_vars["latest_time"] == pd.Timestamp(
            "2006-01-05 21:00:00", tz="UTC"
        )

        # Set the min_leverage to .0001 and the algorithm succeeds.
        algo = make_algo(min_leverage=0.0001, grace_period=offset)
        self.check_algo_succeeds(algo)


class TestFuturesAlgo(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2016-01-06")
    END_DATE = pd.Timestamp("2016-01-07")
    FUTURE_MINUTE_BAR_START_DATE = pd.Timestamp("2016-01-05")

    SIM_PARAMS_DATA_FREQUENCY = "minute"

    TRADING_CALENDAR_STRS = ("us_futures",)
    TRADING_CALENDAR_PRIMARY_CAL = "us_futures"
    BENCHMARK_SID = None

    @classmethod
    def make_futures_info(cls):
        return pd.DataFrame.from_dict(
            {
                1: {
                    "symbol": "CLG16",
                    "root_symbol": "CL",
                    "start_date": pd.Timestamp("2015-12-01", tz="UTC"),
                    "notice_date": pd.Timestamp("2016-01-20", tz="UTC"),
                    "expiration_date": pd.Timestamp("2016-02-19", tz="UTC"),
                    "auto_close_date": pd.Timestamp("2016-01-18", tz="UTC"),
                    "exchange": "TEST",
                },
            },
            orient="index",
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

        algo = self.make_algo(
            script=algo_code,
            trading_calendar=get_calendar("us_futures"),
        )
        algo.run()

        # Assert that we were able to retrieve history data for minutes outside
        # of the 6:31am US/Eastern to 5:00pm US/Eastern futures open times.
        np.testing.assert_array_equal(
            algo.history_values[0].index,
            pd.date_range(
                "2016-01-06 6:27",
                "2016-01-06 6:31",
                freq="min",
                tz="US/Eastern",
            ),
        )
        np.testing.assert_array_equal(
            algo.history_values[1].index,
            pd.date_range(
                "2016-01-07 6:27",
                "2016-01-07 6:31",
                freq="min",
                tz="US/Eastern",
            ),
        )

        # Expected prices here are given by the range values created by the
        # default `make_future_minute_bar_data` method.
        np.testing.assert_array_equal(
            algo.history_values[0].values,
            list(map(float, range(2196, 2201))),
        )
        np.testing.assert_array_equal(
            algo.history_values[1].values,
            list(map(float, range(3636, 3641))),
        )

    @staticmethod
    def algo_with_slippage(slippage_model):
        return dedent(
            """
            from zipline.api import (
                commission,
                order,
                set_commission,
                set_slippage,
                sid,
                slippage,
                get_datetime,
            )

            def initialize(context):
                commission_model = commission.PerFutureTrade(0)
                set_commission(us_futures=commission_model)
                slippage_model = slippage.{model}
                set_slippage(us_futures=slippage_model)
                context.ordered = False

            def handle_data(context, data):
                if not context.ordered:
                    order(sid(1), 10)
                    context.ordered = True
                    context.order_price = data.current(sid(1), 'price')
            """
        ).format(model=slippage_model)

    def test_fixed_future_slippage(self):
        algo_code = self.algo_with_slippage("FixedSlippage(spread=0.10)")
        algo = self.make_algo(
            script=algo_code,
            trading_calendar=get_calendar("us_futures"),
        )
        results = algo.run()

        # Flatten the list of transactions.
        all_txns = [
            val for sublist in results["transactions"].tolist() for val in sublist
        ]

        assert len(all_txns) == 1
        txn = all_txns[0]

        # Add 1 to the expected price because the order does not fill until the
        # bar after the price is recorded.
        expected_spread = 0.05
        expected_price = (algo.order_price + 1) + expected_spread

        assert txn["price"] == expected_price
        assert results["orders"][0][0]["commission"] == 0.0

    def test_volume_contract_slippage(self):
        algo_code = self.algo_with_slippage(
            "VolumeShareSlippage(volume_limit=0.05, price_impact=0.1)",
        )
        algo = self.make_algo(
            script=algo_code,
            trading_calendar=get_calendar("us_futures"),
        )
        results = algo.run()

        # There should be no commissions.
        assert results["orders"][0][0]["commission"] == 0.0

        # Flatten the list of transactions.
        all_txns = [
            val for sublist in results["transactions"].tolist() for val in sublist
        ]

        # With a volume limit of 0.05, and a total volume of 100 contracts
        # traded per minute, we should require 2 transactions to order 10
        # contracts.
        assert len(all_txns) == 2

        for i, txn in enumerate(all_txns):
            # Add 1 to the order price because the order does not fill until
            # the bar after the price is recorded.
            order_price = algo.order_price + i + 1
            expected_impact = order_price * 0.1 * (0.05**2)
            expected_price = order_price + expected_impact
            assert txn["price"] == expected_price


class TestAnalyzeAPIMethod(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2016-01-05")
    END_DATE = pd.Timestamp("2016-01-05")
    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_analyze_called(self):
        self.perf_ref = None

        def initialize(context):
            pass

        def handle_data(context, data):
            pass

        def analyze(context, perf):
            self.perf_ref = perf

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
        )
        results = algo.run()
        assert results is self.perf_ref


class TestOrderCancelation(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2016-01-05")
    END_DATE = pd.Timestamp("2016-01-07")

    ASSET_FINDER_EQUITY_SIDS = (1,)
    ASSET_FINDER_EQUITY_SYMBOLS = ("ASSET1",)
    BENCHMARK_SID = None

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

    # https://stackoverflow.com/questions/50373916/pytest-to-insert-caplog-fixture-in-test-method
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    @classmethod
    def make_equity_minute_bar_data(cls):
        asset_minutes = cls.trading_calendar.sessions_minutes(
            cls.START_DATE,
            cls.END_DATE,
        )

        minutes_count = len(asset_minutes)
        minutes_arr = np.arange(1, 1 + minutes_count)

        # normal test data, but volume is pinned at 1 share per minute
        yield 1, pd.DataFrame(
            {
                "open": minutes_arr + 1,
                "high": minutes_arr + 2,
                "low": minutes_arr - 1,
                "close": minutes_arr,
                "volume": np.full(minutes_count, 1.0),
            },
            index=asset_minutes,
        )

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        yield 1, pd.DataFrame(
            {
                "open": np.full(3, 1, dtype=np.float64),
                "high": np.full(3, 1, dtype=np.float64),
                "low": np.full(3, 1, dtype=np.float64),
                "close": np.full(3, 1, dtype=np.float64),
                "volume": np.full(3, 1, dtype=np.float64),
            },
            index=cls.equity_daily_bar_days,
        )

    def prep_algo(
        self,
        cancelation_string,
        data_frequency="minute",
        amount=1000,
        minute_emission=False,
    ):
        code = self.code.format(cancelation_string, amount)
        return self.make_algo(
            script=code,
            sim_params=self.make_simparams(
                data_frequency=data_frequency,
                emission_rate="minute" if minute_emission else "daily",
            ),
        )

    @parameter_space(
        direction=[1, -1],
        minute_emission=[True, False],
    )
    def test_eod_order_cancel_minute(self, direction, minute_emission):
        """Test that EOD order cancel works in minute mode for both shorts and
        longs, and both daily emission and minute emission
        """
        # order 1000 shares of asset1.  the volume is only 1 share per bar,
        # so the order should be cancelled at the end of the day.
        algo = self.prep_algo(
            "set_cancel_policy(cancel_policy.EODCancel())",
            amount=np.copysign(1000, direction),
            minute_emission=minute_emission,
        )

        results = algo.run()

        for daily_positions in results.positions:
            assert 1 == len(daily_positions)
            assert np.copysign(389, direction) == daily_positions[0]["amount"]
            assert 1 == results.positions[0][0]["sid"]

        # should be an order on day1, but no more orders afterwards
        np.testing.assert_array_equal([1, 0, 0], list(map(len, results.orders)))

        # should be 389 txns on day 1, but no more afterwards
        np.testing.assert_array_equal([389, 0, 0], list(map(len, results.transactions)))

        the_order = results.orders[0][0]

        assert ORDER_STATUS.CANCELLED == the_order["status"]
        assert np.copysign(389, direction) == the_order["filled"]

        with self._caplog.at_level(logging.WARNING):

            assert 1 == len(self._caplog.messages)

            if direction == 1:
                expected = [
                    "Your order for 1000 shares of ASSET1 has been partially "
                    "filled. 389 shares were successfully purchased. "
                    "611 shares were not filled by the end of day and "
                    "were canceled."
                ]
                assert expected == self._caplog.messages
            elif direction == -1:
                expected = [
                    "Your order for -1000 shares of ASSET1 has been partially "
                    "filled. 389 shares were successfully sold. "
                    "611 shares were not filled by the end of day and "
                    "were canceled."
                ]
                assert expected == self._caplog.messages
            self._caplog.clear()

    def test_default_cancelation_policy(self):
        algo = self.prep_algo("")

        results = algo.run()

        # order stays open throughout simulation
        np.testing.assert_array_equal([1, 1, 1], list(map(len, results.orders)))

        # one txn per minute.  389 the first day (since no order until the
        # end of the first minute).  390 on the second day.  221 on the
        # the last day, sum = 1000.
        np.testing.assert_array_equal(
            [389, 390, 221], list(map(len, results.transactions))
        )

        with self._caplog.at_level(logging.WARNING):
            assert len(self._caplog.messages) == 0

    def test_eod_order_cancel_daily(self):
        # in daily mode, EODCancel does nothing.
        algo = self.prep_algo("set_cancel_policy(cancel_policy.EODCancel())", "daily")

        results = algo.run()

        # order stays open throughout simulation
        np.testing.assert_array_equal([1, 1, 1], list(map(len, results.orders)))

        # one txn per day
        np.testing.assert_array_equal([0, 1, 1], list(map(len, results.transactions)))

        with self._caplog.at_level(logging.WARNING):
            assert len(self._caplog.messages) == 0


class TestDailyEquityAutoClose(zf.WithMakeAlgo, zf.ZiplineTestCase):
    """Tests if delisted equities are properly removed from a portfolio holding
    positions in said equities.
    """

    #     January 2015
    # Su Mo Tu We Th Fr Sa
    #              1  2  3
    #  4  5  6  7  8  9 10
    # 11 12 13 14 15 16 17
    # 18 19 20 21 22 23 24
    # 25 26 27 28 29 30 31
    START_DATE = pd.Timestamp("2015-01-05")
    END_DATE = pd.Timestamp("2015-01-13")

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = None

    @classmethod
    def init_class_fixtures(cls):
        super(TestDailyEquityAutoClose, cls).init_class_fixtures()
        cls.assets = cls.asset_finder.retrieve_all(cls.asset_finder.equities_sids)

    @classmethod
    def make_equity_info(cls):
        cls.test_days = cls.trading_calendar.sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )
        assert len(cls.test_days) == 7, "Number of days in test changed!"
        cls.first_asset_expiration = cls.test_days[2]

        # Assets start on start date and delist every two days:
        #
        #     start_date   end_date auto_close_date
        #   0 2015-01-05 2015-01-07      2015-01-09
        #   1 2015-01-05 2015-01-09      2015-01-13
        #   2 2015-01-05 2015-01-13      2015-01-15
        cls.asset_info = make_jagged_equity_info(
            num_assets=3,
            start_date=cls.test_days[0],
            first_end=cls.first_asset_expiration,
            frequency=cls.trading_calendar.day,
            periods_between_ends=2,
            auto_close_delta=2 * cls.trading_calendar.day,
        )
        return cls.asset_info

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        cls.daily_data = make_trade_data_for_asset_info(
            dates=cls.test_days,
            asset_info=cls.asset_info,
            price_start=10,
            price_step_by_sid=10,
            price_step_by_date=1,
            volume_start=100,
            volume_step_by_sid=100,
            volume_step_by_date=10,
        )
        return cls.daily_data.items()

    def daily_prices_on_tick(self, row):
        return [trades.iloc[row].close for trades in self.daily_data.values()]

    def final_daily_price(self, asset):
        return self.daily_data[asset.sid].loc[asset.end_date].close

    def default_initialize(self):
        """Initialize function shared between test algos."""

        def initialize(context):
            context.ordered = False
            context.set_commission(PerShare(0, 0))
            context.set_slippage(FixedSlippage(spread=0))
            context.num_positions = []
            context.cash = []

        return initialize

    def default_handle_data(self, assets, order_size):
        """Handle data function shared between test algos."""

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
        capital_base=[1, 100000],
        __fail_fast=True,
    )
    def test_daily_delisted_equities(self, order_size, capital_base):
        """Make sure that after an equity gets delisted, our portfolio holds the
        correct number of equities and correct amount of cash.
        """
        assets = self.assets
        final_prices = {asset.sid: self.final_daily_price(asset) for asset in assets}

        # Prices at which we expect our orders to be filled.
        initial_fill_prices = self.daily_prices_on_tick(1)
        cost_basis = sum(initial_fill_prices) * order_size

        # Last known prices of assets that will be auto-closed.
        fp0 = final_prices[0]
        fp1 = final_prices[1]

        algo = self.make_algo(
            initialize=self.default_initialize(),
            handle_data=self.default_handle_data(assets, order_size),
            sim_params=self.make_simparams(
                capital_base=capital_base,
                data_frequency="daily",
            ),
        )
        output = algo.run()

        initial_cash = capital_base
        after_fills = initial_cash - cost_basis
        after_first_auto_close = after_fills + fp0 * (order_size)
        after_second_auto_close = after_first_auto_close + fp1 * (order_size)

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

        # Check expected cash.
        assert expected_cash == list(output["ending_cash"])

        # The cash recorded by the algo should be behind by a day from the
        # computed ending cash.
        expected_cash.insert(3, after_fills)
        assert algo.cash == expected_cash[:-1]

        # Check expected long/short counts.
        # We have longs if order_size > 0.
        # We have shorts if order_size < 0.
        if order_size > 0:
            assert expected_num_positions == list(output["longs_count"])
            assert [0] * len(self.test_days) == list(output["shorts_count"])
        else:
            assert expected_num_positions == list(output["shorts_count"])
            assert [0] * len(self.test_days) == list(output["longs_count"])

        # The number of positions recorded by the algo should be behind by a
        # day from the computed long/short counts.
        expected_num_positions.insert(3, 3)
        assert algo.num_positions == expected_num_positions[:-1]

        # Check expected transactions.
        # We should have a transaction of order_size shares per sid.
        transactions = output["transactions"]
        initial_fills = transactions.iloc[1]
        assert len(initial_fills) == len(assets)

        last_minute_of_session = self.trading_calendar.session_close(self.test_days[1])

        for asset, txn in zip(assets, initial_fills):
            assert (
                dict(
                    txn,
                    **{
                        "amount": order_size,
                        "commission": None,
                        "dt": last_minute_of_session,
                        "price": initial_fill_prices[asset],
                        "sid": asset,
                    },
                )
                == txn
            )
            # This will be a UUID.
            assert isinstance(txn["order_id"], str)

        def transactions_for_date(date):
            return transactions.iloc[self.test_days.get_loc(date)]

        # We should have exactly one auto-close transaction on the close date
        # of asset 0.
        (first_auto_close_transaction,) = transactions_for_date(
            assets[0].auto_close_date
        )
        assert first_auto_close_transaction == {
            "amount": -order_size,
            "commission": None,
            "dt": self.trading_calendar.session_close(
                assets[0].auto_close_date,
            ),
            "price": fp0,
            "sid": assets[0],
            "order_id": None,  # Auto-close txns emit Nones for order_id.
        }

        (second_auto_close_transaction,) = transactions_for_date(
            assets[1].auto_close_date
        )
        assert second_auto_close_transaction == {
            "amount": -order_size,
            "commission": None,
            "dt": self.trading_calendar.session_close(
                assets[1].auto_close_date,
            ),
            "price": fp1,
            "sid": assets[1],
            "order_id": None,  # Auto-close txns emit Nones for order_id.
        }

    def test_cancel_open_orders(self):
        """Test that any open orders for an equity that gets delisted are
        canceled.  Unless an equity is auto closed, any open orders for that
        equity will persist indefinitely.
        """
        assets = self.assets
        first_asset_end_date = assets[0].end_date
        first_asset_auto_close_date = assets[0].auto_close_date

        def initialize(context):
            pass

        def handle_data(context, data):
            # The only order we place in this test should never be filled.
            assert context.portfolio.cash == context.portfolio.starting_cash

            today_session = self.trading_calendar.minute_to_session(
                context.get_datetime()
            )
            day_after_auto_close = self.trading_calendar.next_session(
                first_asset_auto_close_date,
            )

            if today_session == first_asset_end_date:
                # Equity 0 will no longer exist tomorrow, so this order will
                # never be filled.
                assert len(context.get_open_orders()) == 0
                context.order(context.sid(0), 10)
                assert len(context.get_open_orders()) == 1
            elif today_session == first_asset_auto_close_date:
                # We do not cancel open orders until the end of the auto close
                # date, so our open order should still exist at this point.
                assert len(context.get_open_orders()) == 1
            elif today_session == day_after_auto_close:
                assert len(context.get_open_orders()) == 0

        algo = self.make_algo(
            initialize=initialize,
            handle_data=handle_data,
            sim_params=self.make_simparams(
                data_frequency="daily",
            ),
        )
        results = algo.run()

        orders = results["orders"]

        def orders_for_date(date):
            return orders.iloc[self.test_days.get_loc(date)]

        original_open_orders = orders_for_date(first_asset_end_date)
        assert len(original_open_orders) == 1

        last_close_for_asset = algo.trading_calendar.session_close(first_asset_end_date)

        assert (
            dict(
                original_open_orders[0],
                **{
                    "amount": 10,
                    "commission": 0.0,
                    "created": last_close_for_asset,
                    "dt": last_close_for_asset,
                    "sid": assets[0],
                    "status": ORDER_STATUS.OPEN,
                    "filled": 0,
                },
            )
            == original_open_orders[0]
        )

        orders_after_auto_close = orders_for_date(first_asset_auto_close_date)
        assert len(orders_after_auto_close) == 1
        assert (
            dict(
                orders_after_auto_close[0],
                **{
                    "amount": 10,
                    "commission": 0.0,
                    "created": last_close_for_asset,
                    "dt": algo.trading_calendar.session_close(
                        first_asset_auto_close_date,
                    ),
                    "sid": assets[0],
                    "status": ORDER_STATUS.CANCELLED,
                    "filled": 0,
                },
            )
            == orders_after_auto_close[0]
        )


# NOTE: This suite is almost the same as TestDailyEquityAutoClose, except it
# uses minutely data instead of daily data, and the auto_close_date for
# equities is one day after their end_date instead of two.
class TestMinutelyEquityAutoClose(zf.WithMakeAlgo, zf.ZiplineTestCase):
    #     January 2015
    # Su Mo Tu We Th Fr Sa
    #              1  2  3
    #  4  5  6  7  8  9 10
    # 11 12 13 14 15 16 17
    # 18 19 20 21 22 23 24
    # 25 26 27 28 29 30 31
    START_DATE = pd.Timestamp("2015-01-05")
    END_DATE = pd.Timestamp("2015-01-13")

    BENCHMARK_SID = None

    @classmethod
    def init_class_fixtures(cls):
        super(TestMinutelyEquityAutoClose, cls).init_class_fixtures()
        cls.assets = cls.asset_finder.retrieve_all(cls.asset_finder.equities_sids)

    @classmethod
    def make_equity_info(cls):
        cls.test_days = cls.trading_calendar.sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )
        cls.test_minutes = cls.trading_calendar.sessions_minutes(
            cls.START_DATE,
            cls.END_DATE,
        )
        cls.first_asset_expiration = cls.test_days[2]

        # Assets start on start date and delist every two days:
        #
        #     start_date   end_date auto_close_date
        #   0 2015-01-05 2015-01-07      2015-01-09
        #   1 2015-01-05 2015-01-09      2015-01-13
        #   2 2015-01-05 2015-01-13      2015-01-15
        cls.asset_info = make_jagged_equity_info(
            num_assets=3,
            start_date=cls.test_days[0],
            first_end=cls.first_asset_expiration,
            frequency=cls.trading_calendar.day,
            periods_between_ends=2,
            auto_close_delta=1 * cls.trading_calendar.day,
        )
        return cls.asset_info

    # XXX: This test suite uses inconsistent data for minutely and daily bars.
    @classmethod
    def make_equity_minute_bar_data(cls):
        cls.minute_data = make_trade_data_for_asset_info(
            dates=cls.test_minutes,
            asset_info=cls.asset_info,
            price_start=10,
            price_step_by_sid=10,
            price_step_by_date=1,
            volume_start=100,
            volume_step_by_sid=100,
            volume_step_by_date=10,
        )
        return cls.minute_data.items()

    def minute_prices_on_tick(self, row):
        return [trades.iloc[row].close for trades in self.minute_data.values()]

    def final_minute_price(self, asset):
        return (
            self.minute_data[asset.sid]
            .loc[self.trading_calendar.session_close(asset.end_date)]
            .close
        )

    def default_initialize(self):
        """Initialize function shared between test algos."""

        def initialize(context):
            context.ordered = False
            context.set_commission(PerShare(0, 0))
            context.set_slippage(FixedSlippage(spread=0))
            context.num_positions = []
            context.cash = []

        return initialize

    def default_handle_data(self, assets, order_size):
        """Handle data function shared between test algos."""

        def handle_data(context, data):
            if not context.ordered:
                for asset in assets:
                    context.order(asset, order_size)
                context.ordered = True

            context.cash.append(context.portfolio.cash)
            context.num_positions.append(len(context.portfolio.positions))

        return handle_data

    def test_minutely_delisted_equities(self):
        assets = self.assets
        final_prices = {asset.sid: self.final_minute_price(asset) for asset in assets}
        backtest_minutes = self.minute_data[0].index.tolist()

        order_size = 10

        capital_base = 100000
        algo = self.make_algo(
            initialize=self.default_initialize(),
            handle_data=self.default_handle_data(assets, order_size),
            sim_params=self.make_simparams(
                capital_base=capital_base,
                data_frequency="minute",
            ),
        )

        output = algo.run()
        initial_fill_prices = self.minute_prices_on_tick(1)
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

        # We have the rest of the first sim day, plus the second, third and
        # fourth days' worth of minutes with cash spent.
        expected_cash.extend([after_fills] * (389 + 390 + 390 + 390))
        expected_position_counts.extend([3] * (389 + 390 + 390 + 390))

        # We then have two days with the cash refunded from asset 0.
        expected_cash.extend([after_first_auto_close] * (390 + 390))
        expected_position_counts.extend([2] * (390 + 390))

        # We then have one day with cash refunded from asset 1.
        expected_cash.extend([after_second_auto_close] * 390)
        expected_position_counts.extend([1] * 390)

        # Check list lengths first to avoid expensive comparison
        assert len(algo.cash) == len(expected_cash)
        # TODO find more efficient way to compare these lists
        assert algo.cash == expected_cash
        assert list(output["ending_cash"]) == [
            after_fills,
            after_fills,
            after_fills,
            after_first_auto_close,
            after_first_auto_close,
            after_second_auto_close,
            after_second_auto_close,
        ]

        assert algo.num_positions == expected_position_counts
        assert list(output["longs_count"]) == [3, 3, 3, 2, 2, 1, 1]

        # Check expected transactions.
        # We should have a transaction of order_size shares per sid.
        transactions = output["transactions"]

        # Note that the transactions appear on the first day rather than the
        # second in minute mode, because the fills happen on the second tick of
        # the backtest, which is still on the first day in minute mode.
        initial_fills = transactions.iloc[0]
        assert len(initial_fills) == len(assets)
        for asset, txn in zip(assets, initial_fills):
            assert (
                dict(
                    txn,
                    **{
                        "amount": order_size,
                        "commission": None,
                        "dt": backtest_minutes[1],
                        "price": initial_fill_prices[asset],
                        "sid": asset,
                    },
                )
                == txn
            )
            # This will be a UUID.
            assert isinstance(txn["order_id"], str)

        def transactions_for_date(date):
            return transactions.iloc[self.test_days.get_loc(date)]

        # We should have exactly one auto-close transaction on the close date
        # of asset 0.
        (first_auto_close_transaction,) = transactions_for_date(
            assets[0].auto_close_date
        )
        assert first_auto_close_transaction == {
            "amount": -order_size,
            "commission": None,
            "dt": algo.trading_calendar.session_close(
                assets[0].auto_close_date,
            ),
            "price": fp0,
            "sid": assets[0],
            "order_id": None,  # Auto-close txns emit Nones for order_id.
        }

        (second_auto_close_transaction,) = transactions_for_date(
            assets[1].auto_close_date
        )
        assert second_auto_close_transaction == {
            "amount": -order_size,
            "commission": None,
            "dt": algo.trading_calendar.session_close(
                assets[1].auto_close_date,
            ),
            "price": fp1,
            "sid": assets[1],
            "order_id": None,  # Auto-close txns emit Nones for order_id.
        }


class TestOrderAfterDelist(zf.WithMakeAlgo, zf.ZiplineTestCase):
    start = pd.Timestamp("2016-01-05")
    day_1 = pd.Timestamp("2016-01-06")
    day_4 = pd.Timestamp("2016-01-11")
    end = pd.Timestamp("2016-01-15")

    # FIXME: Pass a benchmark source here.
    BENCHMARK_SID = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    @classmethod
    def make_equity_info(cls):
        return pd.DataFrame.from_dict(
            {
                # Asset whose auto close date is after its end date.
                1: {
                    "start_date": cls.start,
                    "end_date": cls.day_1,
                    "auto_close_date": cls.day_4,
                    "symbol": "ASSET1",
                    "exchange": "TEST",
                },
                # Asset whose auto close date is before its end date.
                2: {
                    "start_date": cls.start,
                    "end_date": cls.day_4,
                    "auto_close_date": cls.day_1,
                    "symbol": "ASSET2",
                    "exchange": "TEST",
                },
            },
            orient="index",
        )

    # XXX: This suite doesn't use the data in its DataPortal; it uses a
    # FakeDataPortal with different mock data.
    def init_instance_fixtures(self):
        super(TestOrderAfterDelist, self).init_instance_fixtures()
        self.data_portal = FakeDataPortal(self.asset_finder)

    @parameterized.expand(
        [
            ("auto_close_after_end_date", 1),
            ("auto_close_before_end_date", 2),
        ]
    )
    def test_order_in_quiet_period(self, name, sid):
        asset = self.asset_finder.retrieve_asset(sid)

        algo_code = dedent(
            """
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
        """
        ).format(sid=sid)

        # run algo from 1/6 to 1/7
        algo = self.make_algo(
            script=algo_code,
            sim_params=SimulationParameters(
                start_session=pd.Timestamp("2016-01-06"),
                end_session=pd.Timestamp("2016-01-07"),
                trading_calendar=self.trading_calendar,
                data_frequency="minute",
            ),
        )

        algo.run()

        with self._caplog.at_level(logging.WARNING):

            # one warning per order on the second day
            assert 6 * 390 == len(self._caplog.messages)

            expected_message = (
                "Cannot place order for ASSET{sid}, as it has de-listed. "
                "Any existing positions for this asset will be liquidated "
                "on {date}.".format(sid=sid, date=asset.auto_close_date)
            )
            for w in self._caplog.messages:
                assert expected_message == w


class AlgoInputValidationTestCase(zf.WithMakeAlgo, zf.ZiplineTestCase):
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
        for method in ("initialize", "handle_data", "before_trading_start", "analyze"):

            with pytest.raises(ValueError):
                self.make_algo(script=script, **{method: lambda *args, **kwargs: None})
