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

"""
Tests for the zipline.finance package
"""
from pathlib import Path
from datetime import datetime, timedelta
from functools import partial
import numpy as np
import pandas as pd
import pytest
import pytz
import zipline.utils.factory as factory
from testfixtures import TempDirectory
from zipline.data.bcolz_daily_bars import BcolzDailyBarReader, BcolzDailyBarWriter
from zipline.data.data_portal import DataPortal
from zipline.data.bcolz_minute_bars import BcolzMinuteBarReader, BcolzMinuteBarWriter
from zipline.finance.asset_restrictions import NoRestrictions
from zipline.finance.blotter.simulation_blotter import SimulationBlotter
from zipline.finance.execution import LimitOrder, MarketOrder
from zipline.finance.metrics import MetricsTracker
from zipline.finance.metrics import load as load_metrics_set
from zipline.finance.slippage import FixedBasisPointsSlippage, FixedSlippage
from zipline.finance.trading import SimulationParameters
from zipline.protocol import BarData
from zipline.testing import write_bcolz_minute_data

DEFAULT_TIMEOUT = 15  # seconds
EXTENDED_TIMEOUT = 90

_multiprocess_can_split_ = False


@pytest.fixture(scope="class")
def set_test_finance(request, with_asset_finder):
    ASSET_FINDER_COUNTRY_CODE = "??"

    START_DATES = [
        pd.Timestamp("2006-01-03"),
    ] * 3
    END_DATES = [
        pd.Timestamp("2006-12-29"),
    ] * 3

    equities = pd.DataFrame(
        list(
            zip(
                [1, 2, 133],
                ["A", "B", "C"],
                START_DATES,
                END_DATES,
                [
                    "NYSE",
                ]
                * 3,
            )
        ),
        columns=["sid", "symbol", "start_date", "end_date", "exchange"],
    )

    exchange_names = [df["exchange"] for df in (equities,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    request.cls.asset_finder = with_asset_finder(
        **dict(equities=equities, exchanges=exchanges)
    )


@pytest.mark.usefixtures("set_test_finance", "with_trading_calendars")
class TestFinance:
    start = pd.Timestamp("2006-01-01")
    end = pd.Timestamp("2006-12-31")

    # TODO: write tests for short sales
    # TODO: write a test to do massive buying or shorting.

    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    def test_partially_filled_orders(self):
        # create a scenario where order size and trade size are equal
        # so that orders must be spread out over several trades.
        params = {
            "trade_count": 360,
            "trade_interval": timedelta(minutes=1),
            "order_count": 2,
            "order_amount": 100,
            "order_interval": timedelta(minutes=1),
            # because we placed two orders for 100 shares each, and the volume
            # of each trade is 100, and by default you can take up 10% of the
            # bar's volume (per FixedBasisPointsSlippage, the default slippage
            # model), the simulator should spread the order into 20 trades of
            # 10 shares per order.
            "expected_txn_count": 20,
            "expected_txn_volume": 2 * 100,
            "default_slippage": True,
        }

        self.transaction_sim(**params)

        # same scenario, but with short sales
        params2 = {
            "trade_count": 360,
            "trade_interval": timedelta(minutes=1),
            "order_count": 2,
            "order_amount": -100,
            "order_interval": timedelta(minutes=1),
            "expected_txn_count": 20,
            "expected_txn_volume": 2 * -100,
            "default_slippage": True,
        }

        self.transaction_sim(**params2)

    # @pytest.mark.timeout(DEFAULT_TIMEOUT)
    def test_collapsing_orders(self):
        # create a scenario where order.amount <<< trade.volume
        # to test that several orders can be covered properly by one trade,
        # but are represented by multiple transactions.
        params1 = {
            "trade_count": 6,
            "trade_interval": timedelta(hours=1),
            "order_count": 24,
            "order_amount": 1,
            "order_interval": timedelta(minutes=1),
            # because we placed an orders totaling less than 25% of one trade
            # the simulator should produce just one transaction.
            "expected_txn_count": 24,
            "expected_txn_volume": 24,
        }
        self.transaction_sim(**params1)

        # second verse, same as the first. except short!
        params2 = {
            "trade_count": 6,
            "trade_interval": timedelta(hours=1),
            "order_count": 24,
            "order_amount": -1,
            "order_interval": timedelta(minutes=1),
            "expected_txn_count": 24,
            "expected_txn_volume": -24,
        }
        self.transaction_sim(**params2)

        # Runs the collapsed trades over daily trade intervals.
        # Ensuring that our delay works for daily intervals as well.
        params3 = {
            "trade_count": 6,
            "trade_interval": timedelta(days=1),
            "order_count": 24,
            "order_amount": 1,
            "order_interval": timedelta(minutes=1),
            "expected_txn_count": 24,
            "expected_txn_volume": 24,
        }
        self.transaction_sim(**params3)

    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    def test_alternating_long_short(self):
        # create a scenario where we alternate buys and sells
        params1 = {
            "trade_count": int(6.5 * 60 * 4),
            "trade_interval": timedelta(minutes=1),
            "order_count": 4,
            "order_amount": 10,
            "order_interval": timedelta(hours=24),
            "alternate": True,
            "complete_fill": True,
            "expected_txn_count": 4,
            "expected_txn_volume": 0,  # equal buys and sells
        }
        self.transaction_sim(**params1)

    def transaction_sim(self, **params):
        """This is a utility method that asserts expected
        results for conversion of orders to transactions given a
        trade history
        """
        trade_count = params["trade_count"]
        trade_interval = params["trade_interval"]
        order_count = params["order_count"]
        order_amount = params["order_amount"]
        order_interval = params["order_interval"]
        expected_txn_count = params["expected_txn_count"]
        expected_txn_volume = params["expected_txn_volume"]

        # optional parameters
        # ---------------------
        # if present, alternate between long and short sales
        alternate = params.get("alternate")

        # if present, expect transaction amounts to match orders exactly.
        complete_fill = params.get("complete_fill")

        asset1 = self.asset_finder.retrieve_asset(1)

        with TempDirectory() as tempdir:

            if trade_interval < timedelta(days=1):

                sim_params = factory.create_simulation_parameters(
                    start=self.start, end=self.end, data_frequency="minute"
                )

                minutes = self.trading_calendar.minutes_window(
                    sim_params.first_open,
                    int((trade_interval.total_seconds() / 60) * trade_count) + 100,
                )

                price_data = np.array([10.1] * len(minutes))
                assets = {
                    asset1.sid: pd.DataFrame(
                        {
                            "open": price_data,
                            "high": price_data,
                            "low": price_data,
                            "close": price_data,
                            "volume": np.array([100] * len(minutes)),
                            "dt": minutes,
                        }
                    ).set_index("dt")
                }

                write_bcolz_minute_data(
                    self.trading_calendar,
                    self.trading_calendar.sessions_in_range(
                        self.trading_calendar.minute_to_session(minutes[0]),
                        self.trading_calendar.minute_to_session(minutes[-1]),
                    ),
                    tempdir.path,
                    assets.items(),
                )

                equity_minute_reader = BcolzMinuteBarReader(tempdir.path)

                data_portal = DataPortal(
                    self.asset_finder,
                    self.trading_calendar,
                    first_trading_day=equity_minute_reader.first_trading_day,
                    equity_minute_reader=equity_minute_reader,
                )
            else:
                sim_params = factory.create_simulation_parameters(
                    data_frequency="daily"
                )

                days = sim_params.sessions

                assets = {
                    1: pd.DataFrame(
                        {
                            "open": [10.1] * len(days),
                            "high": [10.1] * len(days),
                            "low": [10.1] * len(days),
                            "close": [10.1] * len(days),
                            "volume": [100] * len(days),
                            "day": [day.value for day in days],
                        },
                        index=days,
                    )
                }

                path = Path(tempdir.path) / "testdata.bcolz"
                BcolzDailyBarWriter(
                    path, self.trading_calendar, days[0], days[-1]
                ).write(assets.items())

                equity_daily_reader = BcolzDailyBarReader(path)

                data_portal = DataPortal(
                    self.asset_finder,
                    self.trading_calendar,
                    first_trading_day=equity_daily_reader.first_trading_day,
                    equity_daily_reader=equity_daily_reader,
                )

            if "default_slippage" not in params or not params["default_slippage"]:
                slippage_func = FixedBasisPointsSlippage()
            else:
                slippage_func = None

            blotter = SimulationBlotter(slippage_func)

            start_date = sim_params.first_open

            alternator = -1 if alternate else 1

            tracker = MetricsTracker(
                trading_calendar=self.trading_calendar,
                first_session=sim_params.start_session,
                last_session=sim_params.end_session,
                capital_base=sim_params.capital_base,
                emission_rate=sim_params.emission_rate,
                data_frequency=sim_params.data_frequency,
                asset_finder=self.asset_finder,
                metrics=load_metrics_set("none"),
            )

            # replicate what tradesim does by going through every minute or day
            # of the simulation and processing open orders each time
            if sim_params.data_frequency == "minute":
                ticks = minutes
            else:
                ticks = days.tz_localize("UTC")

            transactions = []

            order_list = []
            order_date = start_date
            for tick in ticks:
                blotter.current_dt = tick
                if tick >= order_date and len(order_list) < order_count:
                    # place an order
                    direction = alternator ** len(order_list)
                    order_id = blotter.order(
                        asset1,
                        order_amount * direction,
                        MarketOrder(),
                    )
                    order_list.append(blotter.orders[order_id])
                    order_date = order_date + order_interval
                    # move after market orders to just after market next
                    # market open.
                    if order_date.hour >= 21:
                        if order_date.minute >= 00:
                            order_date = order_date + timedelta(days=1)
                            order_date = order_date.replace(hour=14, minute=30)
                else:
                    bar_data = BarData(
                        data_portal=data_portal,
                        simulation_dt_func=lambda: tick,
                        data_frequency=sim_params.data_frequency,
                        trading_calendar=self.trading_calendar,
                        restrictions=NoRestrictions(),
                    )
                    txns, _, closed_orders = blotter.get_transactions(bar_data)
                    for txn in txns:
                        tracker.process_transaction(txn)
                        transactions.append(txn)

                    blotter.prune_orders(closed_orders)

            for i in range(order_count):
                order = order_list[i]
                assert order.asset == asset1
                assert order.amount == order_amount * alternator**i

            if complete_fill:
                assert len(transactions) == len(order_list)

            total_volume = 0
            for i in range(len(transactions)):
                txn = transactions[i]
                total_volume += txn.amount
                if complete_fill:
                    order = order_list[i]
                    assert order.amount == txn.amount

            assert total_volume == expected_txn_volume

            assert len(transactions) == expected_txn_count

            if total_volume == 0:
                with pytest.raises(KeyError):
                    tracker.positions[asset1]
            else:
                cumulative_pos = tracker.positions[asset1]
                assert total_volume == cumulative_pos.amount

            # the open orders should not contain the asset.
            oo = blotter.open_orders
            assert asset1 not in oo, "Entry is removed when no open orders"

    def test_blotter_processes_splits(self):
        blotter = SimulationBlotter(equity_slippage=FixedSlippage())

        # set up two open limit orders with very low limit prices,
        # one for sid 1 and one for sid 2
        asset1 = self.asset_finder.retrieve_asset(1)
        asset2 = self.asset_finder.retrieve_asset(2)
        asset133 = self.asset_finder.retrieve_asset(133)

        blotter.order(asset1, 100, LimitOrder(10, asset=asset1))
        blotter.order(asset2, 100, LimitOrder(10, asset=asset2))

        # send in splits for assets 133 and 2.  We have no open orders for
        # asset 133 so it should be ignored.
        blotter.process_splits([(asset133, 0.5), (asset2, 0.3333)])

        for asset in [asset1, asset2]:
            order_lists = blotter.open_orders[asset]
            assert order_lists is not None
            assert 1 == len(order_lists)

        asset1_order = blotter.open_orders[1][0]
        asset2_order = blotter.open_orders[2][0]

        # make sure the asset1 order didn't change
        assert 100 == asset1_order.amount
        assert 10 == asset1_order.limit
        assert 1 == asset1_order.asset

        # make sure the asset2 order did change
        # to 300 shares at 3.33
        assert 300 == asset2_order.amount
        assert 3.33 == asset2_order.limit
        assert 2 == asset2_order.asset


@pytest.mark.usefixtures("with_trading_calendars")
class TestSimulationParameters:
    """
    Tests for date management utilities in zipline.finance.trading.
    """

    def test_simulation_parameters(self):
        sp = SimulationParameters(
            start_session=pd.Timestamp("2008-01-01"),
            end_session=pd.Timestamp("2008-12-31"),
            capital_base=100000,
            trading_calendar=self.trading_calendar,
        )

        assert sp.last_close.month == 12
        assert sp.last_close.day == 31

    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    def test_sim_params_days_in_period(self):

        #     January 2008
        #  Su Mo Tu We Th Fr Sa
        #         1  2  3  4  5
        #   6  7  8  9 10 11 12
        #  13 14 15 16 17 18 19
        #  20 21 22 23 24 25 26
        #  27 28 29 30 31

        params = SimulationParameters(
            start_session=pd.Timestamp("2007-12-31"),
            end_session=pd.Timestamp("2008-01-07"),
            capital_base=100_000,
            trading_calendar=self.trading_calendar,
        )

        expected_trading_days = (
            datetime(2007, 12, 31),
            # Skip new years
            # holidays taken from: http://www.nyse.com/press/1191407641943.html
            datetime(2008, 1, 2),
            datetime(2008, 1, 3),
            datetime(2008, 1, 4),
            # Skip Saturday
            # Skip Sunday
            datetime(2008, 1, 7),
        )

        num_expected_trading_days = 5
        assert num_expected_trading_days == len(params.sessions)
        np.testing.assert_array_equal(expected_trading_days, params.sessions.tolist())
