"""
Tests for live trading.
"""
import pandas as pd
import numpy as np

try:                                    # Python 3
    from itertools import zip_longest
except ImportError:                     # Python 2
    from itertools import izip_longest as zip_longest

from functools import partial

from mock import sentinel, Mock, MagicMock
from zipline.algorithm import TradingAlgorithm
from zipline.algorithm_live import LiveTradingAlgorithm, LiveAlgorithmExecutor
from zipline.data.data_portal_live import DataPortalLive
from zipline.gens.brokers.broker import Broker
from zipline.testing.fixtures import WithSimParams
from zipline.testing.fixtures import (ZiplineTestCase,
                                      WithDataPortal)
from zipline.errors import CannotOrderDelistedAsset


class TestLiveTradingAlgorithm(WithSimParams,
                               WithDataPortal,
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
                asset_finder=self.asset_finder,
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

        broker = MagicMock(spec=Broker)
        live_algo = create_initialized_algo(
            partial(LiveTradingAlgorithm, broker=broker), current_dt)
        live_algo.trading_client = MagicMock(spec=LiveAlgorithmExecutor)
        live_algo.trading_client.current_data = Mock()
        live_algo.trading_client.current_data.current.return_value = 12

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