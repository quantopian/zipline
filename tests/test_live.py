"""
Tests for live trading.
"""
from unittest import TestCase
from datetime import time
from collections import defaultdict

import pandas as pd
import numpy as np

# fix to allow zip_longest on Python 2.X and 3.X
try:                                    # Python 3
    from itertools import zip_longest
except ImportError:                     # Python 2
    from itertools import izip_longest as zip_longest

import os

from mock import patch, sentinel, Mock, MagicMock
from testfixtures import tempdir

from zipline.algorithm import TradingAlgorithm
from zipline.algorithm_live import LiveTradingAlgorithm, LiveAlgorithmExecutor
from zipline.data.data_portal_live import DataPortalLive
from zipline.gens.realtimeclock import (RealtimeClock,
                                        SESSION_START,
                                        BEFORE_TRADING_START_BAR)
from zipline.gens.sim_engine import MinuteSimulationClock
from zipline.gens.brokers.broker import Broker
from zipline.gens.brokers.ib_broker import IBBroker, TWSConnection
from zipline.testing.fixtures import WithSimParams
from zipline.utils.calendars import get_calendar
from zipline.utils.calendars.trading_calendar import days_at_time
from zipline.utils.serialization_utils import load_context, store_context
from zipline.testing.fixtures import (ZiplineTestCase,
                                      WithTradingEnvironment,
                                      WithDataPortal)
from zipline.errors import CannotOrderDelistedAsset


class TestRealtimeClock(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nyse_calendar = get_calendar("NYSE")

        cls.sessions = cls.nyse_calendar.sessions_in_range(
            pd.Timestamp("2017-04-20"),
            pd.Timestamp("2017-04-20")
        )

        trading_o_and_c = cls.nyse_calendar.schedule.ix[cls.sessions]
        cls.opens = trading_o_and_c['market_open']
        cls.closes = trading_o_and_c['market_close']

    def setUp(self):
        self.internal_clock = None
        self.events = defaultdict(list)

    def advance_clock(self, x):
        """Mock function for sleep. Advances the internal clock by 1 min"""
        # The internal clock advance time must be 1 minute to match
        # MinutesSimulationClock's update frequency
        self.internal_clock += pd.Timedelta('1 min')

    def get_clock(self, arg, *args, **kwargs):
        """Mock function for pandas.to_datetime which is used to query the
        current time in RealtimeClock"""
        assert arg == "now"
        return self.internal_clock

    def test_crosscheck_realtimeclock_with_minutesimulationclock(self):
        """Tests that RealtimeClock behaves like MinuteSimulationClock"""
        for minute_emission in (False, True):
            # MinuteSimulationClock also relies on to_datetime, shall not be
            # created in the patch block
            msc = MinuteSimulationClock(
                self.sessions,
                self.opens,
                self.closes,
                days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                minute_emission
            )
            msc_events = list(msc)

            with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
                    patch('zipline.gens.realtimeclock.sleep') as sleep:
                rtc = iter(RealtimeClock(
                    self.sessions,
                    self.opens,
                    self.closes,
                    days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                    minute_emission
                ))
                self.internal_clock = \
                    pd.Timestamp("2017-04-20 00:00", tz='UTC')
                to_dt.side_effect = self.get_clock
                sleep.side_effect = self.advance_clock

                rtc_events = list(rtc)

            for rtc_event, msc_event in zip_longest(rtc_events, msc_events):
                self.assertEquals(rtc_event, msc_event)

            self.assertEquals(len(rtc_events), len(msc_events))

    def test_time_skew(self):
        """Tests that RealtimeClock's time_skew parameter behaves as
        expected"""
        for time_skew in (pd.Timedelta("2 hour"), pd.Timedelta("-120 sec")):
            with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
                    patch('zipline.gens.realtimeclock.sleep') as sleep:
                clock = RealtimeClock(
                    self.sessions,
                    self.opens,
                    self.closes,
                    days_at_time(self.sessions, time(11, 31), "US/Eastern"),
                    False,
                    time_skew
                )
                to_dt.side_effect = self.get_clock
                sleep.side_effect = self.advance_clock
                start_time = pd.Timestamp("2017-04-20 15:31", tz='UTC')
                self.internal_clock = start_time

                events = list(clock)

                # Event 0 is SESSION_START which always happens at 00:00.
                ts, event_type = events[1]
                self.assertEquals(ts, start_time + time_skew)

    def test_midday_start(self):
        """Tests that RealtimeClock is able to execute if started mid-day"""
        msc = MinuteSimulationClock(
            self.sessions,
            self.opens,
            self.closes,
            days_at_time(self.sessions, time(8, 45), "US/Eastern"),
            False
        )
        msc_events = list(msc)

        with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
                patch('zipline.gens.realtimeclock.sleep') as sleep:
            rtc = RealtimeClock(
                self.sessions,
                self.opens,
                self.closes,
                days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                False
            )

            to_dt.side_effect = self.get_clock
            sleep.side_effect = self.advance_clock
            self.internal_clock = pd.Timestamp("2017-04-20 15:00", tz='UTC')

            rtc_events = list(rtc)

        # Count the mid-day position in the MinuteSimulationClock's events:
        # Simulation Tick: 2017-04-20 00:00:00+00:00 - 1 (SESSION_START)
        # Simulation Tick: 2017-04-20 12:45:00+00:00 - 4 (BEFORE_TRADING_START)
        # Simulation Tick: 2017-04-20 13:31:00+00:00 - 0 (BAR)
        msc_midday_position = 2 + 90
        self.assertEquals(rtc_events[0], msc_events[0])  # Session start bar

        # before_trading_start is fired immediately if we're after 8:45 EDT
        event_time, event_type = rtc_events[1]
        self.assertEquals(event_time,
                          pd.Timestamp("2017-04-20 15:00", tz='UTC'))
        self.assertEquals(event_type, BEFORE_TRADING_START_BAR)

        self.assertEquals(rtc_events[2:], msc_events[msc_midday_position:])

    def test_afterhours_start(self):
        """Tests that RealtimeClock returns immediately if started after RTH"""
        with patch('zipline.gens.realtimeclock.pd.to_datetime') as to_dt, \
                patch('zipline.gens.realtimeclock.sleep') as sleep:
            rtc = RealtimeClock(
                self.sessions,
                self.opens,
                self.closes,
                days_at_time(self.sessions, time(8, 45), "US/Eastern"),
                False
            )

            to_dt.side_effect = self.get_clock
            sleep.side_effect = self.advance_clock
            self.internal_clock = pd.Timestamp("2017-04-20 20:05", tz='UTC')

            events = list(rtc)
            self.assertEquals(len(events), 2)

            # SESSION_START & which always triggered.
            _, event_type = events[0]
            self.assertEquals(event_type, SESSION_START)

            event_time, event_type = events[1]
            self.assertEquals(event_time,
                              pd.Timestamp("2017-04-20 20:05", tz='UTC'))
            self.assertEquals(event_type, BEFORE_TRADING_START_BAR)


class TestPersistence(WithSimParams, WithTradingEnvironment, ZiplineTestCase):
    def noop(*args, **kwargs):
        pass

    def make_trading_algo(self, state_filename, algo_filename=None,
                          initialize=noop, handle_data=noop):
        return LiveTradingAlgorithm(
            namespace={},
            env=self.make_trading_environment(),
            get_pipeline_loader=self.make_load_function(),
            sim_params=self.make_simparams(),
            state_filename=state_filename,
            algo_filename=algo_filename,
            initialize=initialize,
            handle_data=handle_data,
            script=None)

    @tempdir()
    def test_live_trading_algorithm_creates_state_file(self, tmpdir):
        algo_text = b"""
        def initialize(context):
            pass

        def handle_data(context, data):
            pass
        """
        algo_filename = "algo.py"
        algo_path = tmpdir.write(algo_filename, algo_text)
        state_filename = os.path.join(tmpdir.path, "state_file")

        algo = self.make_trading_algo(state_filename, algo_path)

        assert not os.path.exists(state_filename)

        algo.initialize()

        assert os.path.getsize(state_filename) > 0

    @tempdir()
    def test_live_trading_algorithm_loads_state_file(self, tmpdir):
        state_filename = os.path.join(tmpdir.path, "state_file")

        def initialize_1(context):
            context.state_from_initialize = 7

        def handle_data_1(context, data):
            context.state_from_handle_data = 11

        algo_1 = self.make_trading_algo(state_filename,
                                        initialize=initialize_1,
                                        handle_data=handle_data_1)

        algo_1.initialize()
        algo_1.handle_data(data=sentinel.data)

        def initialize_2(context):
            assert False, "initialize shouldn't be called if state is loaded"

        def handle_data_2(context, data):
            assert False, "handle_data shouldn't be called"

        algo_2 = self.make_trading_algo(state_filename,
                                        initialize=initialize_2,
                                        handle_data=handle_data_2)
        algo_2.initialize()

        assert algo_2.state_from_initialize == 7
        assert algo_2.state_from_handle_data == 11

    @tempdir()
    def test_state_load_with_corrupt_state(self, tmpdir):
        state_filename = os.path.join(tmpdir.path, "state_file")

        algo_1 = self.make_trading_algo(state_filename,
                                        initialize=TestPersistence.noop,
                                        handle_data=TestPersistence.noop)

        tmpdir.write("state_file", b"roken")

        with self.assertRaises(ValueError) as e:
            algo_1.initialize()
        assert "state file" in str(e.exception)

    @tempdir()
    def test_context_persistence_checksum(self, tmpdir):
        algo_text_1 = b"""
        def initialize(context):
            context.state_from_initialize = 11

        def handle_data(context, data):
            context.state_from_handle_data = 13
        """
        algo_filename_1 = "algo_1.py"
        algo_path_1 = tmpdir.write(algo_filename_1, algo_text_1)

        state_filename_1 = os.path.join(tmpdir.path, "state_file_1")
        algo_1 = self.make_trading_algo(state_filename_1,
                                        algo_filename=algo_path_1)

        algo_1.initialize()
        algo_1.handle_data(data=sentinel.data)

        algo_text_2 = b"""
        def initialize(context):
            context.state_from_initialize = 7

        def handle_data(context, data):
            context.state_from_handle_data = 5
        """
        algo_filename_2 = "algo_2.py"
        algo_path_2 = tmpdir.write(algo_filename_2, algo_text_2)

        state_filename_2 = os.path.join(tmpdir.path, "state_file_2")
        algo_2 = self.make_trading_algo(state_filename_2,
                                        algo_filename=algo_path_2)

        algo_2.initialize()
        algo_2.handle_data(data=sentinel.data)

        algo_1_wrong_state = self.make_trading_algo(state_filename_2,
                                                    algo_filename=algo_path_1)

        algo_2_wrong_state = self.make_trading_algo(state_filename_1,
                                                    algo_filename=algo_path_2)

        with self.assertRaises(TypeError) as e1:
            algo_1_wrong_state.initialize()
        assert "state file" in str(e1.exception)

        with self.assertRaises(TypeError) as e2:
            algo_2_wrong_state.initialize()
        assert "state file" in str(e2.exception)

    @tempdir()
    def test_context_persistence_exclude_list(self, tmpdir):
        class Context(object):
            def __init__(self, rsi=None, sma=None,
                         trading_client=None, event_manager=None):
                self.rsi = rsi
                self.sma = sma
                self.trading_client = trading_client
                self.event_manager = event_manager

        context = Context(rsi=17.2, sma=40.4, trading_client=lambda x: x+3,
                          event_manager=[None, False])

        exclude_list = ['trading_client', 'event_manager']
        checksum = 'robocop'

        state_file_path = os.path.join(tmpdir.path, "state_file")

        store_context(state_file_path, context, checksum, exclude_list)

        restored_context = Context()
        load_context(state_file_path, restored_context, checksum)

        assert restored_context.__dict__.keys() == context.__dict__.keys()
        assert restored_context.rsi == context.rsi
        assert restored_context.sma == context.sma
        assert restored_context.trading_client is None
        assert restored_context.event_manager is None


class TestLiveTradingAlgorithm(WithSimParams,
                               WithDataPortal,
                               WithTradingEnvironment,
                               ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1, 2)
    ASSET_FINDER_EQUITY_SYMBOLS = ("SPY", "XIV")
    START_DATE = pd.to_datetime('2017-01-03', utc=True)
    END_DATE = pd.to_datetime('2017-04-26', utc=True)
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    SIM_PARAMS_EMISSION_RATE = 'minute'

    def test_live_trading_supports_orders_outside_ingested_period(self):
        def create_initialized_algo(trading_algorithm_class, current_dt):
            def initialize(context):
                pass

            def handle_data(context, data):
                context.order_value(context.symbol("SPY"), 100)

            algo = trading_algorithm_class(
                namespace={},
                env=self.make_trading_environment(),
                get_pipeline_loader=self.make_load_function(),
                sim_params=self.make_simparams(),
                state_filename='blah',
                algo_filename='foo',
                initialize=initialize,
                handle_data=handle_data,
                script=None)

            algo.initialize()
            algo.initialized = True  # Normally this is set through algo.run()
            algo.datetime = current_dt

            return algo

        current_dt = self.END_DATE + pd.Timedelta("1 day")

        backtest_algo = create_initialized_algo(TradingAlgorithm, current_dt)

        with self.assertRaises(CannotOrderDelistedAsset):
            backtest_algo.handle_data(data=sentinel.data)

        live_algo = create_initialized_algo(LiveTradingAlgorithm, current_dt)
        live_algo.trading_client = MagicMock(spec=LiveAlgorithmExecutor)
        live_algo.trading_client.current_data = Mock()
        live_algo.trading_client.current_data.current.return_value = 12
        live_algo.broker = MagicMock(spec=Broker)

        live_algo.handle_data(data=sentinel.data)
        assert live_algo.broker.order.called
        assert live_algo.trading_client.current_data.current.called

    def test_data_portal_live_extends_ingested_data(self):
        assets = [self.asset_finder.retrieve_asset(1), ]
        rt_bars = pd.DataFrame(
            index=pd.date_range(start='2017-09-28 10:11:00',
                                end='2017-09-28 10:45:00',
                                freq='1 Min', tz='utc'),
            columns=pd.MultiIndex.from_product(
                [assets,
                 ['open', 'high', 'low', 'close', 'volume']]),
            data=np.random.randn(35, 5)
        )
        broker = MagicMock(Broker)
        broker.get_realtime_bars.return_value = rt_bars
        data_portal_live = DataPortalLive(
            broker,
            asset_finder=self.data_portal.asset_finder,
            trading_calendar=self.data_portal.trading_calendar,
            first_trading_day=self.data_portal._first_available_session,
            equity_daily_reader=(
              self.bcolz_equity_daily_bar_reader
              if self.DATA_PORTAL_USE_DAILY_DATA else
              None
            ),
            equity_minute_reader=(
              self.bcolz_equity_minute_bar_reader
              if self.DATA_PORTAL_USE_MINUTE_DATA else
              None
            ),
            adjustment_reader=(
              self.adjustment_reader
              if self.DATA_PORTAL_USE_ADJUSTMENTS else
              None
            ),
        )

        # Test with overall bar count > available realtime bar count
        end_dt = pd.to_datetime('2017-03-03 10:00:00', utc=True)
        bar_count = 1000
        combined_data = data_portal_live.get_history_window(
            assets, end_dt, bar_count=bar_count, frequency='1m',
            field='price', data_frequency='1m')

        expected_bars = rt_bars[-bar_count:].swaplevel(0, 1, axis=1)['close']
        assert len(combined_data) == bar_count
        assert expected_bars.isin(combined_data).all().all()

        # Test with overall bar count < available realtime bar count
        end_dt = pd.to_datetime('2017-03-03 10:00:00', utc=True)
        bar_count = 10
        combined_data = data_portal_live.get_history_window(
            assets, end_dt, bar_count=bar_count, frequency='1m',
            field='price', data_frequency='1m')

        expected_bars = rt_bars[-bar_count:].swaplevel(0, 1, axis=1)['close']
        assert len(combined_data) == bar_count
        assert expected_bars.isin(combined_data).all().all()


class TestIBBroker(WithSimParams, ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1, 2)
    ASSET_FINDER_EQUITY_SYMBOLS = ("SPY", "XIV")

    def tws_bars(self):
        with patch('zipline.gens.brokers.ib_broker.TWSConnection.connect'):
            tws = TWSConnection("localhost:9999:1111",
                                sentinel.order_update_callback)

        tws._add_bar('SPY', 12.4, 10,
                     pd.to_datetime('2017-09-27 10:30:00', utc=True),
                     10, 12.401, False)
        tws._add_bar('SPY', 12.41, 10,
                     pd.to_datetime('2017-09-27 10:30:40', utc=True),
                     20, 12.411, False)
        tws._add_bar('SPY', 12.44, 20,
                     pd.to_datetime('2017-09-27 10:31:10', utc=True),
                     40, 12.441, False)
        tws._add_bar('SPY', 12.74, 5,
                     pd.to_datetime('2017-09-27 10:37:10', utc=True),
                     45, 12.741, True)
        tws._add_bar('SPY', 12.99, 15,
                     pd.to_datetime('2017-09-27 12:10:00', utc=True),
                     60, 12.991, False)
        tws._add_bar('XIV', 100.4, 100,
                     pd.to_datetime('2017-09-27 9:32:00', utc=True),
                     100, 100.401, False)
        tws._add_bar('XIV', 100.41, 100,
                     pd.to_datetime('2017-09-27 9:32:20', utc=True),
                     200, 100.411, True)
        tws._add_bar('XIV', 100.44, 200,
                     pd.to_datetime('2017-09-27 9:41:10', utc=True),
                     400, 100.441, False)
        tws._add_bar('XIV', 100.74, 50,
                     pd.to_datetime('2017-09-27 11:42:10', utc=True),
                     450, 100.741, False)

        return tws.bars

    @patch('zipline.gens.brokers.ib_broker.TWSConnection')
    def test_get_spot_value(self, tws):
        dt = None  # dt is not used in real broker
        data_freq = 'minute'
        asset = self.env.asset_finder.retrieve_asset(1)
        bars = {'last_trade_price': [12, 10, 11, 14],
                'last_trade_size': [1, 2, 3, 4],
                'total_volume': [10, 10, 10, 10],
                'vwap': [12.1, 10.1, 11.1, 14.1],
                'single_trade_flag': [0, 1, 0, 1]}
        last_trade_times = [pd.to_datetime('2017-06-16 10:30:00', utc=True),
                            pd.to_datetime('2017-06-16 10:30:11', utc=True),
                            pd.to_datetime('2017-06-16 10:30:30', utc=True),
                            pd.to_datetime('2017-06-17 10:31:9', utc=True)]
        index = pd.DatetimeIndex(last_trade_times)
        broker = IBBroker(sentinel.tws_uri)
        tws.return_value.bars = {asset.symbol: pd.DataFrame(
            index=index, data=bars)}

        price = broker.get_spot_value(asset, 'price', dt, data_freq)
        last_trade = broker.get_spot_value(asset, 'last_traded', dt, data_freq)
        open_ = broker.get_spot_value(asset, 'open', dt, data_freq)
        high = broker.get_spot_value(asset, 'high', dt, data_freq)
        low = broker.get_spot_value(asset, 'low', dt, data_freq)
        close = broker.get_spot_value(asset, 'close', dt, data_freq)
        volume = broker.get_spot_value(asset, 'volume', dt, data_freq)

        # Only the last minute is taken into account, therefore
        # the first bar is ignored
        assert price == bars['last_trade_price'][-1]
        assert last_trade == last_trade_times[-1]
        assert open_ == bars['last_trade_price'][1]
        assert high == max(bars['last_trade_price'][1:])
        assert low == min(bars['last_trade_price'][1:])
        assert close == bars['last_trade_price'][-1]
        assert volume == sum(bars['last_trade_size'][1:])

    def test_get_realtime_bars_produces_correct_df(self):
        bars = self.tws_bars()

        with patch('zipline.gens.brokers.ib_broker.TWSConnection'):
            broker = IBBroker(sentinel.tws_uri)
            broker._tws.bars = bars

        assets = (self.env.asset_finder.retrieve_asset(1),
                  self.env.asset_finder.retrieve_asset(2))

        realtime_history = broker.get_realtime_bars(assets, '1m')

        asset_spy = self.env.asset_finder.retrieve_asset(1)
        asset_xiv = self.env.asset_finder.retrieve_asset(2)

        assert asset_spy in realtime_history
        assert asset_xiv in realtime_history

        spy = realtime_history[asset_spy]
        xiv = realtime_history[asset_xiv]

        assert list(spy.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert list(xiv.columns) == ['open', 'high', 'low', 'close', 'volume']

        # There are 159 minutes between the first (XIV @ 2017-09-27 9:32:00)
        # and the last bar (SPY @ 2017-09-27 12:10:00)
        assert len(realtime_history) == 159

        spy_non_na = spy.dropna()
        xiv_non_na = xiv.dropna()
        assert len(spy_non_na) == 4
        assert len(xiv_non_na) == 3

        assert spy_non_na.iloc[0].name == pd.to_datetime(
            '2017-09-27 10:30:00', utc=True)
        assert spy_non_na.iloc[0].open == 12.40
        assert spy_non_na.iloc[0].high == 12.41
        assert spy_non_na.iloc[0].low == 12.40
        assert spy_non_na.iloc[0].close == 12.41
        assert spy_non_na.iloc[0].volume == 20

        assert spy_non_na.iloc[1].name == pd.to_datetime(
            '2017-09-27 10:31:00', utc=True)
        assert spy_non_na.iloc[1].open == 12.44
        assert spy_non_na.iloc[1].high == 12.44
        assert spy_non_na.iloc[1].low == 12.44
        assert spy_non_na.iloc[1].close == 12.44
        assert spy_non_na.iloc[1].volume == 20

        assert spy_non_na.iloc[-1].name == pd.to_datetime(
            '2017-09-27 12:10:00', utc=True)
        assert spy_non_na.iloc[-1].open == 12.99
        assert spy_non_na.iloc[-1].high == 12.99
        assert spy_non_na.iloc[-1].low == 12.99
        assert spy_non_na.iloc[-1].close == 12.99
        assert spy_non_na.iloc[-1].volume == 15

        assert xiv_non_na.iloc[0].name == pd.to_datetime(
            '2017-09-27 9:32:00', utc=True)
        assert xiv_non_na.iloc[0].open == 100.4
        assert xiv_non_na.iloc[0].high == 100.41
        assert xiv_non_na.iloc[0].low == 100.4
        assert xiv_non_na.iloc[0].close == 100.41
        assert xiv_non_na.iloc[0].volume == 200
