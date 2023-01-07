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
from collections import deque
from functools import partial
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

import zipline.testing.fixtures as zf
from zipline.assets.continuous_futures import OrderedContracts, delivery_predicate
from zipline.assets.roll_finder import ROLL_DAYS_FOR_CURRENT_CONTRACT, VolumeRollFinder
from zipline.data.bcolz_minute_bars import FUTURES_MINUTES_PER_DAY
from zipline.errors import SymbolNotFound


@pytest.fixture(scope="class")
def set_test_ordered_futures_contracts(request, with_asset_finder):
    ASSET_FINDER_COUNTRY_CODE = "??"

    root_symbols = pd.DataFrame(
        {
            "root_symbol": ["FO", "BA", "BZ"],
            "root_symbol_id": [1, 2, 3],
            "exchange": ["CMES", "CMES", "CMES"],
        }
    )

    fo_frame = pd.DataFrame(
        {
            "root_symbol": ["FO"] * 4,
            "asset_name": ["Foo"] * 4,
            "symbol": ["FOF16", "FOG16", "FOH16", "FOJ16"],
            "sid": range(1, 5),
            "start_date": pd.date_range("2015-01-01", periods=4),
            "end_date": pd.date_range("2016-01-01", periods=4),
            "notice_date": pd.date_range("2016-01-01", periods=4),
            "expiration_date": pd.date_range("2016-01-01", periods=4),
            "auto_close_date": pd.date_range("2016-01-01", periods=4),
            "tick_size": [0.001] * 4,
            "multiplier": [1000.0] * 4,
            "exchange": ["CMES"] * 4,
        }
    )
    # BA is set up to test a quarterly roll, to test Eurodollar-like
    # behavior
    # The roll should go from BAH16 -> BAM16
    ba_frame = pd.DataFrame(
        {
            "root_symbol": ["BA"] * 3,
            "asset_name": ["Bar"] * 3,
            "symbol": ["BAF16", "BAG16", "BAH16"],
            "sid": range(5, 8),
            "start_date": pd.date_range("2015-01-01", periods=3),
            "end_date": pd.date_range("2016-01-01", periods=3),
            "notice_date": pd.date_range("2016-01-01", periods=3),
            "expiration_date": pd.date_range("2016-01-01", periods=3),
            "auto_close_date": pd.date_range("2016-01-01", periods=3),
            "tick_size": [0.001] * 3,
            "multiplier": [1000.0] * 3,
            "exchange": ["CMES"] * 3,
        }
    )
    # BZ is set up to test the case where the first contract in a chain has
    # an auto close date before its start date. It also tests the case
    # where a contract in the chain has a start date after the auto close
    # date of the previous contract, leaving a gap with no active contract.
    bz_frame = pd.DataFrame(
        {
            "root_symbol": ["BZ"] * 4,
            "asset_name": ["Baz"] * 4,
            "symbol": ["BZF15", "BZG15", "BZH15", "BZJ16"],
            "sid": range(8, 12),
            "start_date": [
                pd.Timestamp("2015-01-02"),
                pd.Timestamp("2015-01-03"),
                pd.Timestamp("2015-02-23"),
                pd.Timestamp("2015-02-24"),
            ],
            "end_date": pd.date_range(
                "2015-02-01",
                periods=4,
                freq="MS",
            ),
            "notice_date": [
                pd.Timestamp("2014-12-31"),
                pd.Timestamp("2015-02-18"),
                pd.Timestamp("2015-03-18"),
                pd.Timestamp("2015-04-17"),
            ],
            "expiration_date": pd.date_range(
                "2015-02-01",
                periods=4,
                freq="MS",
            ),
            "auto_close_date": [
                pd.Timestamp("2014-12-29"),
                pd.Timestamp("2015-02-16"),
                pd.Timestamp("2015-03-16"),
                pd.Timestamp("2015-04-15"),
            ],
            "tick_size": [0.001] * 4,
            "multiplier": [1000.0] * 4,
            "exchange": ["CMES"] * 4,
        }
    )

    futures = pd.concat([fo_frame, ba_frame, bz_frame])

    exchange_names = [df["exchange"] for df in (futures,) if df is not None]
    if exchange_names:
        exchanges = pd.DataFrame(
            {
                "exchange": pd.concat(exchange_names).unique(),
                "country_code": ASSET_FINDER_COUNTRY_CODE,
            }
        )

    request.cls.asset_finder = with_asset_finder(
        **dict(futures=futures, exchanges=exchanges, root_symbols=root_symbols)
    )


class ContinuousFuturesTestCase(
    zf.WithCreateBarData, zf.WithMakeAlgo, zf.ZiplineTestCase
):
    START_DATE = pd.Timestamp("2015-01-05")
    END_DATE = pd.Timestamp("2016-10-19")

    SIM_PARAMS_START = pd.Timestamp("2016-01-26")
    SIM_PARAMS_END = pd.Timestamp("2016-01-28")
    SIM_PARAMS_DATA_FREQUENCY = "minute"
    TRADING_CALENDAR_STRS = ("us_futures",)
    TRADING_CALENDAR_PRIMARY_CAL = "us_futures"

    ASSET_FINDER_FUTURE_CHAIN_PREDICATES = {
        "BZ": partial(delivery_predicate, set(["F", "H"])),
    }

    @classmethod
    def make_root_symbols_info(cls):
        return pd.DataFrame(
            {
                "root_symbol": ["FOOBAR", "BZ", "MA", "DF"],
                "root_symbol_id": [1, 2, 3, 4],
                "exchange": ["CMES", "CMES", "CMES", "CMES"],
            }
        )

    @classmethod
    def make_futures_info(cls):
        fo_frame = pd.DataFrame(
            {
                "symbol": [
                    "FOOBARF16",
                    "FOOBARG16",
                    "FOOBARH16",
                    "FOOBARJ16",
                    "FOOBARK16",
                    "FOOBARF22",
                    "FOOBARG22",
                ],
                "sid": range(0, 7),
                "root_symbol": ["FOOBAR"] * 7,
                "asset_name": ["Foo"] * 7,
                "start_date": [
                    pd.Timestamp("2015-01-05"),
                    pd.Timestamp("2015-02-05"),
                    pd.Timestamp("2015-03-05"),
                    pd.Timestamp("2015-04-05"),
                    pd.Timestamp("2015-05-05"),
                    pd.Timestamp("2021-01-05"),
                    pd.Timestamp("2015-01-05"),
                ],
                "end_date": [
                    pd.Timestamp("2016-08-19"),
                    pd.Timestamp("2016-09-19"),
                    pd.Timestamp("2016-10-19"),
                    pd.Timestamp("2016-11-19"),
                    pd.Timestamp("2022-08-19"),
                    pd.Timestamp("2022-09-19"),
                    # Set the last contract's end date (which is the last
                    # date for which there is data to a value that is
                    # within the range of the dates being tested.  This
                    # models real life scenarios where the end date of the
                    # furthest out contract is not necessarily the
                    # greatest end date all contracts in the chain.
                    pd.Timestamp("2015-02-05"),
                ],
                "notice_date": [
                    pd.Timestamp("2016-01-27"),
                    pd.Timestamp("2016-02-26"),
                    pd.Timestamp("2016-03-24"),
                    pd.Timestamp("2016-04-26"),
                    pd.Timestamp("2016-05-26"),
                    pd.Timestamp("2022-01-26"),
                    pd.Timestamp("2022-02-26"),
                ],
                "expiration_date": [
                    pd.Timestamp("2016-01-27"),
                    pd.Timestamp("2016-02-26"),
                    pd.Timestamp("2016-03-24"),
                    pd.Timestamp("2016-04-26"),
                    pd.Timestamp("2016-05-26"),
                    pd.Timestamp("2022-01-26"),
                    pd.Timestamp("2022-02-26"),
                ],
                "auto_close_date": [
                    pd.Timestamp("2016-01-27"),
                    pd.Timestamp("2016-02-26"),
                    pd.Timestamp("2016-03-24"),
                    pd.Timestamp("2016-04-26"),
                    pd.Timestamp("2016-05-26"),
                    pd.Timestamp("2022-01-26"),
                    pd.Timestamp("2022-02-26"),
                ],
                "tick_size": [0.001] * 7,
                "multiplier": [1000.0] * 7,
                "exchange": ["CMES"] * 7,
            }
        )

        # BZ is set up to test chain predicates, for futures such as PL which
        # only use a subset of contracts for the roll chain.
        bz_frame = pd.DataFrame(
            {
                "symbol": ["BZF16", "BZG16", "BZH16"],
                "root_symbol": ["BZ"] * 3,
                "asset_name": ["Baz"] * 3,
                "sid": range(10, 13),
                "start_date": [
                    pd.Timestamp("2005-01-01"),
                    pd.Timestamp("2005-01-21"),
                    pd.Timestamp("2005-01-21"),
                ],
                "end_date": [
                    pd.Timestamp("2016-08-19"),
                    pd.Timestamp("2016-11-21"),
                    pd.Timestamp("2016-10-19"),
                ],
                "notice_date": [
                    pd.Timestamp("2016-01-11"),
                    pd.Timestamp("2016-02-08"),
                    pd.Timestamp("2016-03-09"),
                ],
                "expiration_date": [
                    pd.Timestamp("2016-01-11"),
                    pd.Timestamp("2016-02-08"),
                    pd.Timestamp("2016-03-09"),
                ],
                "auto_close_date": [
                    pd.Timestamp("2016-01-11"),
                    pd.Timestamp("2016-02-08"),
                    pd.Timestamp("2016-03-09"),
                ],
                "tick_size": [0.001] * 3,
                "multiplier": [1000.0] * 3,
                "exchange": ["CMES"] * 3,
            }
        )

        # MA is set up to test a contract which is has no active volume.
        ma_frame = pd.DataFrame(
            {
                "symbol": ["MAG16", "MAH16", "MAJ16"],
                "root_symbol": ["MA"] * 3,
                "asset_name": ["Most Active"] * 3,
                "sid": range(14, 17),
                "start_date": [
                    pd.Timestamp("2005-01-01"),
                    pd.Timestamp("2005-01-21"),
                    pd.Timestamp("2005-01-21"),
                ],
                "end_date": [
                    pd.Timestamp("2016-08-19"),
                    pd.Timestamp("2016-11-21"),
                    pd.Timestamp("2016-10-19"),
                ],
                "notice_date": [
                    pd.Timestamp("2016-02-17"),
                    pd.Timestamp("2016-03-16"),
                    pd.Timestamp("2016-04-13"),
                ],
                "expiration_date": [
                    pd.Timestamp("2016-02-17"),
                    pd.Timestamp("2016-03-16"),
                    pd.Timestamp("2016-04-13"),
                ],
                "auto_close_date": [
                    pd.Timestamp("2016-02-17"),
                    pd.Timestamp("2016-03-16"),
                    pd.Timestamp("2016-04-13"),
                ],
                "tick_size": [0.001] * 3,
                "multiplier": [1000.0] * 3,
                "exchange": ["CMES"] * 3,
            }
        )

        # DF is set up to have a double volume flip between the 'F' and 'G'
        # contracts, and then a really early temporary volume flip between the
        # 'G' and 'H' contracts.
        df_frame = pd.DataFrame(
            {
                "symbol": ["DFF16", "DFG16", "DFH16"],
                "root_symbol": ["DF"] * 3,
                "asset_name": ["Double Flip"] * 3,
                "sid": range(17, 20),
                "start_date": [
                    pd.Timestamp("2005-01-01"),
                    pd.Timestamp("2005-02-01"),
                    pd.Timestamp("2005-03-01"),
                ],
                "end_date": [
                    pd.Timestamp("2016-08-19"),
                    pd.Timestamp("2016-09-19"),
                    pd.Timestamp("2016-10-19"),
                ],
                "notice_date": [
                    pd.Timestamp("2016-02-19"),
                    pd.Timestamp("2016-03-18"),
                    pd.Timestamp("2016-04-22"),
                ],
                "expiration_date": [
                    pd.Timestamp("2016-02-19"),
                    pd.Timestamp("2016-03-18"),
                    pd.Timestamp("2016-04-22"),
                ],
                "auto_close_date": [
                    pd.Timestamp("2016-02-17"),
                    pd.Timestamp("2016-03-16"),
                    pd.Timestamp("2016-04-20"),
                ],
                "tick_size": [0.001] * 3,
                "multiplier": [1000.0] * 3,
                "exchange": ["CMES"] * 3,
            }
        )

        return pd.concat([fo_frame, bz_frame, ma_frame, df_frame])

    @classmethod
    def make_future_minute_bar_data(cls):
        tc = cls.trading_calendar
        start = pd.Timestamp("2016-01-26")
        end = pd.Timestamp("2016-04-29")
        dts = tc.sessions_minutes(start, end)
        sessions = tc.sessions_in_range(start, end)
        # Generate values in the XXY.YYY space, with XX representing the
        # session and Y.YYY representing the minute within the session.
        # e.g. the close of the 23rd session would be 231.440.
        r = 10.0
        day_markers = np.repeat(
            np.arange(r, r * len(sessions) + r, r), FUTURES_MINUTES_PER_DAY
        )
        r = 0.001
        min_markers = np.tile(
            np.arange(r, r * FUTURES_MINUTES_PER_DAY + r, r), len(sessions)
        )

        markers = day_markers + min_markers

        # Volume uses a similar scheme as above but times 1000.
        r = 10.0 * 1000
        vol_day_markers = np.repeat(
            np.arange(r, r * len(sessions) + r, r, dtype=np.int64),
            FUTURES_MINUTES_PER_DAY,
        )
        r = 0.001 * 1000
        vol_min_markers = np.tile(
            np.arange(r, r * FUTURES_MINUTES_PER_DAY + r, r, dtype=np.int64),
            len(sessions),
        )
        vol_markers = vol_day_markers + vol_min_markers
        base_df = pd.DataFrame(
            {
                "open": np.full(len(dts), 102000.0) + markers,
                "high": np.full(len(dts), 109000.0) + markers,
                "low": np.full(len(dts), 101000.0) + markers,
                "close": np.full(len(dts), 105000.0) + markers,
                "volume": np.full(len(dts), 10000, dtype=np.int64) + vol_markers,
            },
            index=dts,
        )
        # Add the sid to the ones place of the prices, so that the ones
        # place can be used to eyeball the source contract.

        # For volume roll tests end sid volume early.
        # FOOBARF16 cuts out day before autoclose of 01-26
        # FOOBARG16 cuts out on autoclose
        # FOOBARH16 cuts out 4 days before autoclose
        # FOOBARJ16 cuts out 3 days before autoclose
        # Make FOOBARG22 have a blip of trading, but not be the actively trading,
        # so that it does not particpate in volume rolls.

        sid_to_vol_stop_session = {
            0: pd.Timestamp("2016-01-26"),
            1: pd.Timestamp("2016-02-26"),
            2: pd.Timestamp("2016-03-18"),
            3: pd.Timestamp("2016-04-20"),
            6: pd.Timestamp("2016-01-27"),
        }
        for i in range(20):
            df = base_df.copy()
            df += i * 10000
            if i in sid_to_vol_stop_session:
                vol_stop_session = sid_to_vol_stop_session[i]
                m_open = tc.session_first_minute(vol_stop_session)
                loc = dts.searchsorted(m_open)
                # Add a little bit of noise to roll. So that predicates that
                # check for exactly 0 do not work, since there may be
                # stragglers after a roll.
                df.volume.values[loc] = 1000
                df.volume.values[loc + 1 :] = 0
            j = i - 1
            if j in sid_to_vol_stop_session:
                non_primary_end = sid_to_vol_stop_session[j]
                m_close = tc.session_close(non_primary_end)
                if m_close > dts[0]:
                    loc = dts.get_loc(m_close)
                    # Add some volume before a roll, since a contract may be
                    # entered earlier than when it is the primary.
                    df.volume.values[: loc + 1] = 10
            if i == 15:  # No volume for MAH16
                df.volume.values[:] = 0
            if i == 17:
                end_loc = dts.searchsorted(pd.Timestamp("2016-02-16 23:00:00+00:00"))
                df.volume.values[:end_loc] = 10
                df.volume.values[end_loc:] = 0
            if i == 18:
                cross_loc_1 = dts.searchsorted(
                    pd.Timestamp("2016-02-09 23:01:00+00:00")
                )
                cross_loc_2 = dts.searchsorted(
                    pd.Timestamp("2016-02-11 23:01:00+00:00")
                )
                cross_loc_3 = dts.searchsorted(
                    pd.Timestamp("2016-02-15 23:01:00+00:00")
                )
                end_loc = dts.searchsorted(pd.Timestamp("2016-03-16 23:01:00+00:00"))
                df.volume.values[:cross_loc_1] = 5
                df.volume.values[cross_loc_1:cross_loc_2] = 15
                df.volume.values[cross_loc_2:cross_loc_3] = 5
                df.volume.values[cross_loc_3:end_loc] = 15
                df.volume.values[end_loc:] = 0
            if i == 19:
                early_cross_1 = dts.searchsorted(
                    pd.Timestamp("2016-03-01 23:01:00+00:00")
                )
                early_cross_2 = dts.searchsorted(
                    pd.Timestamp("2016-03-03 23:01:00+00:00")
                )
                end_loc = dts.searchsorted(pd.Timestamp("2016-04-19 23:01:00+00:00"))
                df.volume.values[:early_cross_1] = 1
                df.volume.values[early_cross_1:early_cross_2] = 20
                df.volume.values[early_cross_2:end_loc] = 10
                df.volume.values[end_loc:] = 0
            yield i, df

    def test_double_volume_switch(self):
        """Test that when a double volume switch occurs we treat the first switch
        as the roll, assuming it is within a certain distance of the next auto
        close date. See `VolumeRollFinder._active_contract` for a full
        explanation and example.
        """
        cf = self.asset_finder.create_continuous_future(
            "DF",
            0,
            "volume",
            None,
        )

        sessions = self.trading_calendar.sessions_in_range("2016-02-09", "2016-02-17")

        for session in sessions:
            bar_data = self.create_bardata(lambda: session)
            contract = bar_data.current(cf, "contract")

            # The 'G' contract surpasses the 'F' contract in volume on
            # 2016-02-10, which means that the 'G' contract should become the
            # front contract starting on 2016-02-11.
            if session < pd.Timestamp("2016-02-11"):
                assert contract.symbol == "DFF16"
            else:
                assert contract.symbol == "DFG16"

        # This test asserts behavior about a back contract briefly spiking in
        # volume, but more than a week before the front contract's auto close
        # date, meaning it does not fall in the 'grace' period used by
        # `VolumeRollFinder._active_contract`. Therefore we should not roll to
        # the back contract and the front contract should remain current until
        # its auto close date.
        sessions = self.trading_calendar.sessions_in_range("2016-03-01", "2016-03-21")

        for session in sessions:
            bar_data = self.create_bardata(lambda: session)
            contract = bar_data.current(cf, "contract")

            if session < pd.Timestamp("2016-03-17"):
                assert contract.symbol == "DFG16"
            else:
                assert contract.symbol == "DFH16"

    def test_create_continuous_future(self):
        cf_primary = self.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )

        assert cf_primary.root_symbol == "FOOBAR"
        assert cf_primary.offset == 0
        assert cf_primary.roll_style == "calendar"
        assert cf_primary.start_date == pd.Timestamp("2015-01-05")
        assert cf_primary.end_date == pd.Timestamp("2022-09-19")

        retrieved_primary = self.asset_finder.retrieve_asset(cf_primary.sid)

        assert retrieved_primary == cf_primary

        cf_secondary = self.asset_finder.create_continuous_future(
            "FOOBAR", 1, "calendar", None
        )

        assert cf_secondary.root_symbol == "FOOBAR"
        assert cf_secondary.offset == 1
        assert cf_secondary.roll_style == "calendar"
        assert cf_primary.start_date == pd.Timestamp("2015-01-05")
        assert cf_primary.end_date == pd.Timestamp("2022-09-19")

        retrieved = self.asset_finder.retrieve_asset(cf_secondary.sid)

        assert retrieved == cf_secondary

        assert cf_primary != cf_secondary

        # Assert that the proper exception is raised if the given root symbol
        # does not exist.
        with pytest.raises(SymbolNotFound):
            self.asset_finder.create_continuous_future("NO", 0, "calendar", None)

    def test_current_contract(self):
        cf_primary = self.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        bar_data = self.create_bardata(lambda: pd.Timestamp("2016-01-26"))
        contract = bar_data.current(cf_primary, "contract")

        assert contract.symbol == "FOOBARF16"

        bar_data = self.create_bardata(lambda: pd.Timestamp("2016-01-27"))
        contract = bar_data.current(cf_primary, "contract")

        assert contract.symbol == "FOOBARG16", (
            "Auto close at beginning of session so FOOBARG16 is now "
            "the current contract."
        )

    def test_get_value_contract_daily(self):
        cf_primary = self.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )

        contract = self.data_portal.get_spot_value(
            cf_primary,
            "contract",
            pd.Timestamp("2016-01-26"),
            "daily",
        )

        assert contract.symbol == "FOOBARF16"

        contract = self.data_portal.get_spot_value(
            cf_primary,
            "contract",
            pd.Timestamp("2016-01-27"),
            "daily",
        )

        assert contract.symbol == "FOOBARG16", (
            "Auto close at beginning of session so FOOBARG16 is now "
            "the current contract."
        )

        # Test that the current contract outside of the continuous future's
        # start and end dates is None.
        contract = self.data_portal.get_spot_value(
            cf_primary,
            "contract",
            self.START_DATE - self.trading_calendar.day,
            "daily",
        )
        assert contract is None

    def test_get_value_close_daily(self):
        cf_primary = self.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )

        value = self.data_portal.get_spot_value(
            cf_primary, "close", pd.Timestamp("2016-01-26"), "daily"
        )

        assert value == 105011.44

        value = self.data_portal.get_spot_value(
            cf_primary, "close", pd.Timestamp("2016-01-27"), "daily"
        )

        assert value == 115021.44, (
            "Auto close at beginning of session so FOOBARG16 is now "
            "the current contract."
        )

        # Check a value which occurs after the end date of the last known
        # contract, to prevent a regression where the end date of the last
        # contract was used instead of the max date of all contracts.
        value = self.data_portal.get_spot_value(
            cf_primary, "close", pd.Timestamp("2016-03-26"), "daily"
        )

        assert value == 135441.44, (
            "Value should be for FOOBARJ16, even though last "
            "contract ends before query date."
        )

    def test_current_contract_volume_roll(self):
        cf_primary = self.asset_finder.create_continuous_future(
            "FOOBAR", 0, "volume", None
        )
        bar_data = self.create_bardata(lambda: pd.Timestamp("2016-01-26"))
        contract = bar_data.current(cf_primary, "contract")

        assert contract.symbol == "FOOBARF16"

        bar_data = self.create_bardata(lambda: pd.Timestamp("2016-01-27"))
        contract = bar_data.current(cf_primary, "contract")

        assert contract.symbol == "FOOBARG16", (
            "Auto close at beginning of session. FOOBARG16 is now "
            "the current contract."
        )

        bar_data = self.create_bardata(lambda: pd.Timestamp("2016-02-29"))
        contract = bar_data.current(cf_primary, "contract")
        assert (
            contract.symbol == "FOOBARH16"
        ), "Volume switch to FOOBARH16, should have triggered roll."

    def test_current_contract_in_algo(self):
        code = dedent(
            """
            from zipline.api import (
                record,
                continuous_future,
                schedule_function,
                get_datetime,
            )

            def initialize(algo):
                algo.primary_cl = continuous_future('FOOBAR', 0, 'calendar', None)
                algo.secondary_cl = continuous_future('FOOBAR', 1, 'calendar', None)
                schedule_function(record_current_contract)

            def record_current_contract(algo, data):
                record(datetime=get_datetime())
                record(primary=data.current(algo.primary_cl, 'contract'))
                record(secondary=data.current(algo.secondary_cl, 'contract'))
            """
        )
        results = self.run_algorithm(script=code)
        result = results.iloc[0]

        assert (
            result.primary.symbol == "FOOBARF16"
        ), "Primary should be FOOBARF16 on first session."
        assert (
            result.secondary.symbol == "FOOBARG16"
        ), "Secondary should be FOOBARG16 on first session."

        result = results.iloc[1]
        # Second day, primary should switch to FOOBARG
        assert result.primary.symbol == "FOOBARG16", (
            "Primary should be FOOBARG16 on second session, auto "
            "close is at beginning of the session."
        )
        assert result.secondary.symbol == "FOOBARH16", (
            "Secondary should be FOOBARH16 on second session, auto "
            "close is at beginning of the session."
        )

        result = results.iloc[2]
        # Second day, primary should switch to FOOBARG
        assert (
            result.primary.symbol == "FOOBARG16"
        ), "Primary should remain as FOOBARG16 on third session."
        assert (
            result.secondary.symbol == "FOOBARH16"
        ), "Secondary should remain as FOOBARH16 on third session."

    def test_current_chain_in_algo(self):
        code = dedent(
            """
            from zipline.api import (
                record,
                continuous_future,
                schedule_function,
                get_datetime,
            )

            def initialize(algo):
                algo.primary_cl = continuous_future('FOOBAR', 0, 'calendar', None)
                algo.secondary_cl = continuous_future('FOOBAR', 1, 'calendar', None)
                schedule_function(record_current_contract)

            def record_current_contract(algo, data):
                record(datetime=get_datetime())
                primary_chain = data.current_chain(algo.primary_cl)
                secondary_chain = data.current_chain(algo.secondary_cl)
                record(primary_len=len(primary_chain))
                record(primary_first=primary_chain[0].symbol)
                record(primary_last=primary_chain[-1].symbol)
                record(secondary_len=len(secondary_chain))
                record(secondary_first=secondary_chain[0].symbol)
                record(secondary_last=secondary_chain[-1].symbol)
            """
        )
        results = self.run_algorithm(script=code)
        result = results.iloc[0]

        assert result.primary_len == 6, (
            "There should be only 6 contracts in the chain for "
            "the primary, there are 7 contracts defined in the "
            "fixture, but one has a start after the simulation "
            "date."
        )
        assert result.secondary_len == 5, (
            "There should be only 5 contracts in the chain for "
            "the primary, there are 7 contracts defined in the "
            "fixture, but one has a start after the simulation "
            "date. And the first is not included because it is "
            "the primary on that date."
        )

        assert result.primary_first == "FOOBARF16", (
            "Front of primary chain should be FOOBARF16 on first " "session."
        )
        assert result.secondary_first == "FOOBARG16", (
            "Front of secondary chain should be FOOBARG16 on first " "session."
        )

        assert result.primary_last == "FOOBARG22", (
            "End of primary chain should be FOOBARK16 on first " "session."
        )
        assert result.secondary_last == "FOOBARG22", (
            "End of secondary chain should be FOOBARK16 on first " "session."
        )

        # Second day, primary should switch to FOOBARG
        result = results.iloc[1]

        assert result.primary_len == 5, (
            "There should be only 5 contracts in the chain for "
            "the primary, there are 7 contracts defined in the "
            "fixture, but one has a start after the simulation "
            "date. The first is not included because of roll."
        )
        assert result.secondary_len == 4, (
            "There should be only 4 contracts in the chain for "
            "the primary, there are 7 contracts defined in the "
            "fixture, but one has a start after the simulation "
            "date. The first is not included because of roll, "
            "the second is the primary on that date."
        )

        assert result.primary_first == "FOOBARG16", (
            "Front of primary chain should be FOOBARG16 on second " "session."
        )
        assert result.secondary_first == "FOOBARH16", (
            "Front of secondary chain should be FOOBARH16 on second " "session."
        )

        # These values remain FOOBARJ16 because fixture data is not exhaustive
        # enough to move the end of the chain.
        assert result.primary_last == "FOOBARG22", (
            "End of primary chain should be FOOBARK16 on second " "session."
        )
        assert result.secondary_last == "FOOBARG22", (
            "End of secondary chain should be FOOBARK16 on second " "session."
        )

    def test_history_sid_session(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-03-04 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1d",
            "sid",
            "minute",
        )

        assert (
            window.loc["2016-01-26", cf] == 0
        ), "Should be FOOBARF16 at beginning of window."

        assert (
            window.loc["2016-01-27", cf] == 1
        ), "Should be FOOBARG16 after first roll."

        assert (
            window.loc["2016-02-25", cf] == 1
        ), "Should be FOOBARG16 on session before roll."

        assert (
            window.loc["2016-02-26", cf] == 2
        ), "Should be FOOBARH16 on session with roll."

        assert (
            window.loc["2016-02-29", cf] == 2
        ), "Should be FOOBARH16 on session after roll."

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-04-06 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1d",
            "sid",
            "minute",
        )

        assert (
            window.loc["2016-02-25", cf] == 1
        ), "Should be FOOBARG16 at beginning of window."

        assert (
            window.loc["2016-02-26", cf] == 2
        ), "Should be FOOBARH16 on session with roll."

        assert (
            window.loc["2016-02-29", cf] == 2
        ), "Should be FOOBARH16 on session after roll."

        assert (
            window.loc["2016-03-24", cf] == 3
        ), "Should be FOOBARJ16 on session with roll."

        assert (
            window.loc["2016-03-28", cf] == 3
        ), "Should be FOOBARJ16 on session after roll."

    def test_history_sid_session_delivery_predicate(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "BZ", 0, "calendar", None
        )
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-01-11 18:01", tz="US/Eastern").tz_convert("UTC"),
            3,
            "1d",
            "sid",
            "minute",
        )

        assert (
            window.loc["2016-01-08", cf] == 10
        ), "Should be BZF16 at beginning of window."

        assert window.loc["2016-01-11", cf] == 12, (
            "Should be BZH16 after first roll, having skipped " "over BZG16."
        )

        assert window.loc["2016-01-12", cf] == 12, "Should have remained BZG16"

    def test_history_sid_session_secondary(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 1, "calendar", None
        )
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-03-04 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1d",
            "sid",
            "minute",
        )

        assert (
            window.loc["2016-01-26", cf] == 1
        ), "Should be FOOBARG16 at beginning of window."

        assert (
            window.loc["2016-01-27", cf] == 2
        ), "Should be FOOBARH16 after first roll."

        assert (
            window.loc["2016-02-25", cf] == 2
        ), "Should be FOOBARH16 on session before roll."

        assert (
            window.loc["2016-02-26", cf] == 3
        ), "Should be FOOBARJ16 on session with roll."

        assert (
            window.loc["2016-02-29", cf] == 3
        ), "Should be FOOBARJ16 on session after roll."

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-04-06 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1d",
            "sid",
            "minute",
        )

        assert (
            window.loc["2016-02-25", cf] == 2
        ), "Should be FOOBARH16 at beginning of window."

        assert (
            window.loc["2016-02-26", cf] == 3
        ), "Should be FOOBARJ16 on session with roll."

        assert (
            window.loc["2016-02-29", cf] == 3
        ), "Should be FOOBARJ16 on session after roll."

        assert (
            window.loc["2016-03-24", cf] == 4
        ), "Should be FOOBARK16 on session with roll."

        assert (
            window.loc["2016-03-28", cf] == 4
        ), "Should be FOOBARK16 on session after roll."

    def test_history_sid_session_volume_roll(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "volume", None
        )
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-03-04 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1d",
            "sid",
            "minute",
        )

        # Volume cuts out for FOOBARF16 on 2016-01-25
        assert (
            window.loc["2016-01-26", cf] == 0
        ), "Should be FOOBARF16 at beginning of window."

        assert window.loc["2016-01-27", cf] == 1, "Should have rolled to FOOBARG16."

        assert (
            window.loc["2016-02-26", cf] == 1
        ), "Should be FOOBARG16 on session before roll."

        assert (
            window.loc["2016-02-29", cf] == 2
        ), "Should be FOOBARH16 on session with roll."

        assert (
            window.loc["2016-03-01", cf] == 2
        ), "Should be FOOBARH16 on session after roll."

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-04-06 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1d",
            "sid",
            "minute",
        )

        assert (
            window.loc["2016-02-26", cf] == 1
        ), "Should be FOOBARG16 at beginning of window."

        assert window.loc["2016-02-29", cf] == 2, "Should be FOOBARH16 on roll session."

        assert window.loc["2016-03-01", cf] == 2, "Should remain FOOBARH16."

        assert (
            window.loc["2016-03-17", cf] == 2
        ), "Should be FOOBARH16 on session before volume cuts out."

        assert window.loc["2016-03-18", cf] == 2, (
            "Should be FOOBARH16 on session where the volume of "
            "FOOBARH16 cuts out, the roll is upcoming."
        )

        assert window.loc["2016-03-24", cf] == 3, "Should have rolled to FOOBARJ16."

        assert window.loc["2016-03-28", cf] == 3, "Should have remained FOOBARJ16."

    def test_history_sid_minute(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        window = self.data_portal.get_history_window(
            [cf.sid],
            pd.Timestamp("2016-01-26 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "sid",
            "minute",
        )

        assert window.loc[pd.Timestamp("2016-01-26 22:32", tz="UTC"), cf.sid] == 0, (
            "Should be FOOBARF16 at beginning of window. A minute "
            "which is in the 01-26 session, before the roll."
        )

        assert (
            window.loc[pd.Timestamp("2016-01-26 23:00", tz="UTC"), cf.sid] == 0
        ), "Should be FOOBARF16 on on minute before roll minute."

        assert (
            window.loc[pd.Timestamp("2016-01-26 23:01", tz="UTC"), cf.sid] == 1
        ), "Should be FOOBARG16 on minute after roll."

        # Advance the window a day.
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-01-27 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "sid",
            "minute",
        )

        assert (
            window.loc[pd.Timestamp("2016-01-27 22:32", tz="UTC"), cf.sid] == 1
        ), "Should be FOOBARG16 at beginning of window."

        assert (
            window.loc[pd.Timestamp("2016-01-27 23:01", tz="UTC"), cf.sid] == 1
        ), "Should remain FOOBARG16 on next session."

    def test_history_close_session(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        window = self.data_portal.get_history_window(
            [cf.sid], pd.Timestamp("2016-03-06"), 30, "1d", "close", "daily"
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-01-26"), cf.sid],
            105011.440,
            err_msg="At beginning of window, should be FOOBARG16's first value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-26"), cf.sid],
            125241.440,
            err_msg="On session with roll, should be FOOBARH16's 24th value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-29"), cf.sid],
            125251.440,
            err_msg="After roll, Should be FOOBARH16's 25th value.",
        )

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf.sid], pd.Timestamp("2016-04-06"), 30, "1d", "close", "daily"
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-24"), cf.sid],
            115221.440,
            err_msg="At beginning of window, should be FOOBARG16's 22nd value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-26"), cf.sid],
            125241.440,
            err_msg="On session with roll, should be FOOBARH16's 24th value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-29"), cf.sid],
            125251.440,
            err_msg="On session after roll, should be FOOBARH16's 25th value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-03-24"), cf.sid],
            135431.440,
            err_msg="On session with roll, should be FOOBARJ16's 43rd value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-03-28"), cf.sid],
            135441.440,
            err_msg="On session after roll, Should be FOOBARJ16's 44th value.",
        )

    def test_history_close_session_skip_volume(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "MA", 0, "volume", None
        )
        window = self.data_portal.get_history_window(
            [cf.sid], pd.Timestamp("2016-03-06"), 30, "1d", "close", "daily"
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-01-26"), cf.sid],
            245011.440,
            err_msg="At beginning of window, should be MAG16's first value.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-26"), cf.sid],
            265241.440,
            err_msg="Should have skipped MAH16 to MAJ16.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-29"), cf.sid],
            265251.440,
            err_msg="Should have remained MAJ16.",
        )

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf.sid], pd.Timestamp("2016-04-06"), 30, "1d", "close", "daily"
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-24"), cf.sid],
            265221.440,
            err_msg="Should be MAJ16, having skipped MAH16.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-02-29"), cf.sid],
            265251.440,
            err_msg="Should be MAJ1 for rest of window.",
        )

        assert_almost_equal(
            window.loc[pd.Timestamp("2016-03-24"), cf.sid],
            265431.440,
            err_msg="Should be MAJ16 for rest of window.",
        )

    def test_history_close_session_adjusted(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        cf_mul = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", "mul"
        )
        cf_add = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", "add"
        )
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            pd.Timestamp("2016-03-06"),
            30,
            "1d",
            "close",
            "daily",
        )

        # Unadjusted value is: 115011.44
        # Adjustment is based on hop from 115231.44 to 125231.44
        # a ratio of ~0.920
        assert_almost_equal(
            window.loc["2016-01-26", cf_mul],
            124992.348,
            err_msg="At beginning of window, should be FOOBARG16's first value, "
            "adjusted.",
        )

        # Difference of 7008.561
        assert_almost_equal(
            window.loc["2016-01-26", cf_add],
            125011.44,
            err_msg="At beginning of window, should be FOOBARG16's first value, "
            "adjusted.",
        )

        assert_almost_equal(
            window.loc["2016-02-26", cf_mul],
            125241.440,
            err_msg="On session with roll, should be FOOBARH16's 24th value, "
            "unadjusted.",
        )

        assert_almost_equal(
            window.loc["2016-02-26", cf_add],
            125241.440,
            err_msg="On session with roll, should be FOOBARH16's 24th value, "
            "unadjusted.",
        )

        assert_almost_equal(
            window.loc["2016-02-29", cf_mul],
            125251.440,
            err_msg="After roll, Should be FOOBARH16's 25th value, unadjusted.",
        )

        assert_almost_equal(
            window.loc["2016-02-29", cf_add],
            125251.440,
            err_msg="After roll, Should be FOOBARH16's 25th value, unadjusted.",
        )

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            pd.Timestamp("2016-04-06"),
            30,
            "1d",
            "close",
            "daily",
        )

        # Unadjusted value: 115221.44
        # Adjustments based on hops:
        # 2016-02-25 00:00:00+00:00
        # front 115231.440
        # back  125231.440
        # ratio: ~0.920
        # difference: 10000.0
        # and
        # 2016-03-23 00:00:00+00:00
        # front 125421.440
        # back  135421.440
        # ratio: ~1.080
        # difference: 10000.00
        assert_almost_equal(
            window.loc["2016-02-24", cf_mul],
            135236.905,
            err_msg="At beginning of window, should be FOOBARG16's 22nd value, "
            "with two adjustments.",
        )

        assert_almost_equal(
            window.loc["2016-02-24", cf_add],
            135251.44,
            err_msg="At beginning of window, should be FOOBARG16's 22nd value, "
            "with two adjustments",
        )

        # Unadjusted: 125241.44
        assert_almost_equal(
            window.loc["2016-02-26", cf_mul],
            135259.442,
            err_msg="On session with roll, should be FOOBARH16's 24th value, "
            "with one adjustment.",
        )

        assert_almost_equal(
            window.loc["2016-02-26", cf_add],
            135271.44,
            err_msg="On session with roll, should be FOOBARH16's 24th value, "
            "with one adjustment.",
        )

        # Unadjusted: 125251.44
        assert_almost_equal(
            window.loc["2016-02-29", cf_mul],
            135270.241,
            err_msg="On session after roll, should be FOOBARH16's 25th value, "
            "with one adjustment.",
        )

        assert_almost_equal(
            window.loc["2016-02-29", cf_add],
            135281.44,
            err_msg="On session after roll, should be FOOBARH16's 25th value, "
            "unadjusted.",
        )

        # Unadjusted: 135431.44
        assert_almost_equal(
            window.loc["2016-03-24", cf_mul],
            135431.44,
            err_msg="On session with roll, should be FOOBARJ16's 43rd value, "
            "unadjusted.",
        )

        assert_almost_equal(
            window.loc["2016-03-24", cf_add],
            135431.44,
            err_msg="On session with roll, should be FOOBARJ16's 43rd value.",
        )

        # Unadjusted: 135441.44
        assert_almost_equal(
            window.loc["2016-03-28", cf_mul],
            135441.44,
            err_msg="On session after roll, Should be FOOBARJ16's 44th value.",
        )

        assert_almost_equal(
            window.loc["2016-03-28", cf_add],
            135441.44,
            err_msg="On session after roll, Should be FOOBARJ16's 44th value.",
        )

    def test_history_close_minute(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        window = self.data_portal.get_history_window(
            [cf.sid],
            pd.Timestamp("2016-02-25 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "close",
            "minute",
        )

        assert (
            window.loc[pd.Timestamp("2016-02-25 22:32", tz="UTC"), cf.sid] == 115231.412
        ), (
            "Should be FOOBARG16 at beginning of window. A minute "
            "which is in the 02-25 session, before the roll."
        )

        assert (
            window.loc[pd.Timestamp("2016-02-25 23:00", tz="UTC"), cf.sid] == 115231.440
        ), "Should be FOOBARG16 on on minute before roll minute."

        assert (
            window.loc[pd.Timestamp("2016-02-25 23:01", tz="UTC"), cf.sid] == 125240.001
        ), "Should be FOOBARH16 on minute after roll."

        # Advance the window a session.
        window = self.data_portal.get_history_window(
            [cf],
            pd.Timestamp("2016-02-28 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "close",
            "minute",
        )

        assert (
            window.loc["2016-02-26 22:32", cf] == 125241.412
        ), "Should be FOOBARH16 at beginning of window."

        assert (
            window.loc["2016-02-28 23:01", cf] == 125250.001
        ), "Should remain FOOBARH16 on next session."

    def test_history_close_minute_adjusted(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", None
        )
        cf_mul = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", "mul"
        )
        cf_add = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "calendar", "add"
        )
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            pd.Timestamp("2016-02-25 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "close",
            "minute",
        )

        # Unadjusted: 115231.412
        # Adjustment based on roll:
        # 2016-02-25 23:00:00+00:00
        # front: 115231.440
        # back:  125231.440
        # Ratio: ~0.920
        # Difference: 10000.00
        assert window.loc["2016-02-25 22:32", cf_mul] == 125231.41, (
            "Should be FOOBARG16 at beginning of window. A minute "
            "which is in the 02-25 session, before the roll."
        )

        assert window.loc["2016-02-25 22:32", cf_add] == 125231.412, (
            "Should be FOOBARG16 at beginning of window. A minute "
            "which is in the 02-25 session, before the roll."
        )

        # Unadjusted: 115231.44
        # Should use same ratios as above.
        assert window.loc["2016-02-25 23:00", cf_mul] == 125231.44, (
            "Should be FOOBARG16 on on minute before roll minute, " "adjusted."
        )

        assert window.loc["2016-02-25 23:00", cf_add] == 125231.44, (
            "Should be FOOBARG16 on on minute before roll minute, " "adjusted."
        )

        assert (
            window.loc["2016-02-25 23:01", cf_mul] == 125240.001
        ), "Should be FOOBARH16 on minute after roll, unadjusted."

        assert (
            window.loc["2016-02-25 23:01", cf_add] == 125240.001
        ), "Should be FOOBARH16 on minute after roll, unadjusted."

        # Advance the window a session.
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            pd.Timestamp("2016-02-28 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "close",
            "minute",
        )

        # No adjustments in this window.
        assert (
            window.loc["2016-02-26 22:32", cf_mul] == 125241.412
        ), "Should be FOOBARH16 at beginning of window."

        assert (
            window.loc["2016-02-28 23:01", cf_mul] == 125250.001
        ), "Should remain FOOBARH16 on next session."

    def test_history_close_minute_adjusted_volume_roll(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "volume", None
        )
        cf_mul = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "volume", "mul"
        )
        cf_add = self.data_portal.asset_finder.create_continuous_future(
            "FOOBAR", 0, "volume", "add"
        )
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            pd.Timestamp("2016-02-28 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "close",
            "minute",
        )

        # Unadjusted: 115241.412
        # Adjustment based on roll:
        # 2016-02-25 23:00:00+00:00
        # front: 115241.440 (FOG16)
        # back:  125241.440 (FOH16)
        # Ratio: ~0.920
        # Difference: 10000.00
        assert window.loc["2016-02-26 22:32", cf_mul] == 125242.973, (
            "Should be FOOBARG16 at beginning of window. A minute "
            "which is in the 02-25 session, before the roll."
        )

        assert window.loc["2016-02-26 22:32", cf_add] == 125242.851, (
            "Should be FOOBARG16 at beginning of window. A minute "
            "which is in the 02-25 session, before the roll."
        )

        # Unadjusted: 115231.44
        # Should use same ratios as above.
        assert window.loc["2016-02-26 23:00", cf_mul] == 125243.004, (
            "Should be FOOBARG16 on minute before roll minute, " "adjusted."
        )

        assert window.loc["2016-02-26 23:00", cf_add] == 125242.879, (
            "Should be FOOBARG16 on minute before roll minute, " "adjusted."
        )

        assert (
            window.loc["2016-02-28 23:01", cf_mul] == 125250.001
        ), "Should be FOOBARH16 on minute after roll, unadjusted."

        assert (
            window.loc["2016-02-28 23:01", cf_add] == 125250.001
        ), "Should be FOOBARH16 on minute after roll, unadjusted."

        # Advance the window a session.
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            pd.Timestamp("2016-02-29 18:01", tz="US/Eastern").tz_convert("UTC"),
            30,
            "1m",
            "close",
            "minute",
        )

        # No adjustments in this window.
        assert (
            window.loc["2016-02-29 22:32", cf_mul] == 125251.412
        ), "Should be FOOBARH16 at beginning of window."

        assert (
            window.loc["2016-02-29 23:01", cf_mul] == 125260.001
        ), "Should remain FOOBARH16 on next session."


class RollFinderTestCase(zf.WithBcolzFutureDailyBarReader, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp("2017-01-03")
    END_DATE = pd.Timestamp("2017-05-23")

    TRADING_CALENDAR_STRS = ("us_futures",)
    TRADING_CALENDAR_PRIMARY_CAL = "us_futures"

    @classmethod
    def init_class_fixtures(cls):
        super(RollFinderTestCase, cls).init_class_fixtures()

        cls.volume_roll_finder = VolumeRollFinder(
            cls.trading_calendar,
            cls.asset_finder,
            cls.bcolz_future_daily_bar_reader,
        )

    @classmethod
    def make_futures_info(cls):
        day = cls.trading_calendar.day
        two_days = 2 * day
        end_buffer_days = ROLL_DAYS_FOR_CURRENT_CONTRACT * day

        cls.first_end_date = pd.Timestamp("2017-01-20")
        cls.second_end_date = pd.Timestamp("2017-02-17")
        cls.third_end_date = pd.Timestamp("2017-03-17")
        cls.third_auto_close_date = cls.third_end_date - two_days
        cls.fourth_start_date = cls.third_auto_close_date - two_days
        cls.fourth_end_date = pd.Timestamp("2017-04-17")
        cls.fourth_auto_close_date = cls.fourth_end_date + two_days
        cls.fifth_start_date = pd.Timestamp("2017-03-15")
        cls.fifth_end_date = cls.END_DATE
        cls.fifth_auto_close_date = cls.fifth_end_date - two_days
        cls.last_start_date = cls.fourth_end_date

        return pd.DataFrame.from_dict(
            {
                1000: {
                    "symbol": "CLF17",
                    "root_symbol": "CL",
                    "start_date": cls.START_DATE,
                    "end_date": cls.first_end_date,
                    "auto_close_date": cls.first_end_date - two_days,
                    "exchange": "CMES",
                },
                1001: {
                    "symbol": "CLG17",
                    "root_symbol": "CL",
                    "start_date": cls.START_DATE,
                    "end_date": cls.second_end_date,
                    "auto_close_date": cls.second_end_date - two_days,
                    "exchange": "CMES",
                },
                1002: {
                    "symbol": "CLH17",
                    "root_symbol": "CL",
                    "start_date": cls.START_DATE,
                    "end_date": cls.third_end_date,
                    "auto_close_date": cls.third_auto_close_date,
                    "exchange": "CMES",
                },
                1003: {
                    "symbol": "CLJ17",
                    "root_symbol": "CL",
                    "start_date": cls.fourth_start_date,
                    "end_date": cls.fourth_end_date,
                    "auto_close_date": cls.fourth_auto_close_date,
                    "exchange": "CMES",
                },
                1004: {
                    "symbol": "CLK17",
                    "root_symbol": "CL",
                    "start_date": cls.fifth_start_date,
                    "end_date": cls.fifth_end_date,
                    "auto_close_date": cls.fifth_auto_close_date,
                    "exchange": "CMES",
                },
                1005: {
                    "symbol": "CLM17",
                    "root_symbol": "CL",
                    "start_date": cls.last_start_date,
                    "end_date": cls.END_DATE,
                    "auto_close_date": cls.END_DATE + two_days,
                    "exchange": "CMES",
                },
                1006: {
                    "symbol": "CLN17",
                    "root_symbol": "CL",
                    "start_date": cls.last_start_date,
                    "end_date": cls.END_DATE,
                    "auto_close_date": cls.END_DATE + two_days,
                    "exchange": "CMES",
                },
                2000: {
                    # Using a placeholder month of 'A' to mean this is the
                    # first contract in the chain.
                    "symbol": "FVA17",
                    "root_symbol": "FV",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE + end_buffer_days,
                    "auto_close_date": cls.END_DATE + two_days,
                    "exchange": "CMES",
                },
                2001: {
                    # Using a placeholder month of 'B' to mean this is the
                    # second contract in the chain.
                    "symbol": "FVB17",
                    "root_symbol": "FV",
                    "start_date": cls.START_DATE,
                    "end_date": cls.END_DATE + end_buffer_days,
                    "auto_close_date": cls.END_DATE + end_buffer_days,
                    "exchange": "CMES",
                },
            },
            orient="index",
        )

    @classmethod
    def make_future_daily_bar_data(cls):
        """
                Volume data should look like this:

                             CLF17    CLG17    CLH17    CLJ17    CLK17    CLM17   CLN17
               2017-01-03     2000     1000        5        0        0        0       0
               2017-01-04     2000     1000        5        0        0        0       0
                   ...
               2017-01-16     2000     1000        5        0        0        0       0
               2017-01-17     2000     1000        5        0        0        0       0
        ACD -> 2017-01-18     2000_    1000        5        0        0        0       0
               2017-01-19     2000 `-> 1000        5        0        0        0       0
               2017-01-20     2000     1000        5        0        0        0       0
               2017-01-23        0     1000        5        0        0        0       0
                   ...
               2017-02-09        0     1000        5        0        0        0       0
               2017-02-10        0     1000_    5000        0        0        0       0
               2017-02-13        0     1000 `-> 5000        0        0        0       0
               2017-02-14        0     1000     5000        0        0        0       0
        ACD -> 2017-02-15        0     1000     5000        0        0        0       0
               2017-02-16        0     1000     5000        0        0        0       0
               2017-02-17        0     1000     5000        0        0        0       0
               2017-02-20        0        0     5000        0        0        0       0
                   ...
               2017-03-10        0        0     5000        0        0        0       0
               2017-03-13        0        0     5000     4000        0        0       0
               2017-03-14        0        0     5000     4000        0        0       0
        ACD -> 2017-03-15        0        0     5000_    4000     3000        0       0
               2017-03-16        0        0     5000 `-> 4000     3000        0       0
               2017-03-17        0        0     5000     4000     3000        0       0
               2017-03-20        0        0        0     4000     3000        0       0
                   ...
               2017-04-14        0        0        0     4000     3000        0       0
               2017-04-17        0        0        0     4000_    3000        0       0
               2017-04-18        0        0        0        0 `-> 3000        0       0
        ACD -> 2017-04-19        0        0        0        0     3000     1000    2000
               2017-04-20        0        0        0        0     3000     1000    2000
               2017-04-21        0        0        0        0     3000     1000    2000
                   ...
               2017-05-16        0        0        0        0     3000     1000    2000
               2017-05-17        0        0        0        0     3000     1000    2000
               2017-05-18        0        0        0        0     3000_    1000    2000
        ACD -> 2017-05-19        0        0        0        0     3000 `---1000--> 2000
               2017-05-22        0        0        0        0     3000     1000    2000
               2017-05-23        0        0        0        0     3000     1000    2000

                The first roll occurs because we reach the auto close date of CLF17.
                The second roll occurs because the volume of CLH17 overtakes CLG17.
                The third roll is testing the fact that CLJ17 has no data in the grace
                period before CLH17's auto close date.
                The fourth roll is testing that we properly handle the case where a
                contract's auto close date is *after* its end date.
                The fifth roll occurs on the auto close date of CLK17, but we skip over
                CLM17 because of it's low volume, and roll directly to CLN17. This is
                used to cover an edge case where the window passed to get_rolls end on
                the auto close date of CLK17.

                A volume of zero here is used to represent the fact that a contract no
                longer exists.
        """
        date_index = cls.trading_calendar.sessions_in_range(
            cls.START_DATE,
            cls.END_DATE,
        )

        def create_contract_data(volume):
            # The prices used here are arbitrary as they are irrelevant for the
            # purpose of testing roll behavior.
            return pd.DataFrame(
                {"open": 5, "high": 6, "low": 4, "close": 5, "volume": volume},
                index=date_index,
            )

        # Make a copy because we are taking a slice of a data frame.
        first_contract_data = create_contract_data(2000)
        yield 1000, first_contract_data.copy().loc[: cls.first_end_date]

        # Make a copy because we are taking a slice of a data frame.
        second_contract_data = create_contract_data(1000)
        yield 1001, second_contract_data.copy().loc[: cls.second_end_date]

        third_contract_data = create_contract_data(5)
        volume_flip_date = pd.Timestamp("2017-02-10")
        third_contract_data.loc[volume_flip_date:, "volume"] = 5000
        yield 1002, third_contract_data

        # Make a copy because we are taking a slice of a data frame.
        fourth_contract_data = create_contract_data(4000)
        yield (
            1003,
            fourth_contract_data.copy().loc[
                cls.fourth_start_date : cls.fourth_end_date
            ],
        )

        # Make a copy because we are taking a slice of a data frame.
        fifth_contract_data = create_contract_data(3000)
        yield 1004, fifth_contract_data.copy().loc[cls.fifth_start_date :]

        sixth_contract_data = create_contract_data(1000)
        yield 1005, sixth_contract_data.copy().loc[cls.last_start_date :]

        seventh_contract_data = create_contract_data(2000)
        yield 1006, seventh_contract_data.copy().loc[cls.last_start_date :]

        # The data for FV does not really matter except that contract 2000 has
        # higher volume than contract 2001.
        yield 2000, create_contract_data(200)
        yield 2001, create_contract_data(100)

    def test_volume_roll(self):
        """Test normally behaving rolls."""
        rolls = self.volume_roll_finder.get_rolls(
            root_symbol="CL",
            start=self.START_DATE + self.trading_calendar.day,
            end=self.second_end_date,
            offset=0,
        )
        assert rolls == [
            (1000, pd.Timestamp("2017-01-19")),
            (1001, pd.Timestamp("2017-02-13")),
            (1002, None),
        ]

    def test_no_roll(self):
        # If we call 'get_rolls' with start and end dates that do not have any
        # rolls between them, we should still expect the last roll date to be
        # computed successfully.
        date_not_near_roll = pd.Timestamp("2017-02-01")
        rolls = self.volume_roll_finder.get_rolls(
            root_symbol="CL",
            start=date_not_near_roll,
            end=date_not_near_roll + self.trading_calendar.day,
            offset=0,
        )
        assert rolls == [(1001, None)]

    def test_roll_in_grace_period(self):
        """The volume roll finder can look for data up to a week before the given
        date. This test asserts that we not only return the correct active
        contract during that previous week (grace period), but also that we do
        not go into exception if one of the contracts does not exist.
        """
        rolls = self.volume_roll_finder.get_rolls(
            root_symbol="CL",
            start=self.second_end_date,
            end=self.third_end_date,
            offset=0,
        )
        assert rolls == [
            (1002, pd.Timestamp("2017-03-16")),
            (1003, None),
        ]

    def test_end_before_auto_close(self):
        # Test that we correctly roll from CLJ17 (1003) to CLK17 (1004) even
        # though CLJ17 has an auto close date after its end date.
        rolls = self.volume_roll_finder.get_rolls(
            root_symbol="CL",
            start=self.fourth_start_date,
            end=self.fourth_auto_close_date,
            offset=0,
        )
        assert rolls == [
            (1002, pd.Timestamp("2017-03-16")),
            (1003, pd.Timestamp("2017-04-18")),
            (1004, None),
        ]

    def test_roll_window_ends_on_auto_close(self):
        """Test that when skipping over a low volume contract (CLM17), we use the
        correct roll date for the previous contract (CLK17) when that
        contract's auto close date falls on the end date of the roll window.
        """
        rolls = self.volume_roll_finder.get_rolls(
            root_symbol="CL",
            start=self.last_start_date,
            end=self.fifth_auto_close_date,
            offset=0,
        )
        assert rolls == [
            (1003, pd.Timestamp("2017-04-18")),
            (1004, pd.Timestamp("2017-05-19")),
            (1006, None),
        ]

    def test_get_contract_center(self):
        asset_finder = self.asset_finder
        get_contract_center = partial(
            self.volume_roll_finder.get_contract_center,
            offset=0,
        )

        # Test that the current contract adheres to the rolls.
        assert get_contract_center(
            "CL", dt=pd.Timestamp("2017-01-18")
        ) == asset_finder.retrieve_asset(1000)
        assert get_contract_center(
            "CL", dt=pd.Timestamp("2017-01-19")
        ) == asset_finder.retrieve_asset(1001)

        # Test that we still get the correct current contract close to or at
        # the max day boundary. Contracts 2000 and 2001 both have auto close
        # dates after `self.END_DATE` so 2000 should always be the current
        # contract. However, they do not have any volume data after this point
        # so this test ensures that we do not fail to calculate the forward
        # looking rolls required for `VolumeRollFinder.get_contract_center`.
        near_end = self.END_DATE - self.trading_calendar.day
        assert get_contract_center("FV", dt=near_end) == asset_finder.retrieve_asset(
            2000
        )
        assert get_contract_center(
            "FV", dt=self.END_DATE
        ) == asset_finder.retrieve_asset(2000)


@pytest.mark.usefixtures("set_test_ordered_futures_contracts")
class TestOrderedContracts:
    def test_contract_at_offset(self):
        contract_sids = np.array([1, 2, 3, 4], dtype=np.int64)
        start_dates = pd.date_range("2015-01-01", periods=4)

        contracts = deque(self.asset_finder.retrieve_all(contract_sids))

        oc = OrderedContracts("FOOBAR", contracts)

        assert 1 == oc.contract_at_offset(
            1, 0, start_dates[-1].value
        ), "Offset of 0 should return provided sid"

        assert 2 == oc.contract_at_offset(
            1, 1, start_dates[-1].value
        ), "Offset of 1 should return next sid in chain."

        assert None is oc.contract_at_offset(
            4, 1, start_dates[-1].value
        ), "Offset at end of chain should not crash."

    def test_active_chain(self):
        contract_sids = np.array([1, 2, 3, 4], dtype=np.int64)

        contracts = deque(self.asset_finder.retrieve_all(contract_sids))

        oc = OrderedContracts("FOOBAR", contracts)

        # Test sid 1 as days increment, as the sessions march forward
        # a contract should be added per day, until all defined contracts
        # are returned.
        chain = oc.active_chain(1, pd.Timestamp("2014-12-31").value)
        assert [] == list(chain), (
            "On session before first start date, no contracts "
            "in chain should be active."
        )
        chain = oc.active_chain(1, pd.Timestamp("2015-01-01").value)
        assert [1] == list(chain), (
            "[1] should be the active chain on 01-01, since all "
            "other start dates occur after 01-01."
        )

        chain = oc.active_chain(1, pd.Timestamp("2015-01-02").value)
        assert [1, 2] == list(chain), "[1, 2] should be the active contracts on 01-02."

        chain = oc.active_chain(1, pd.Timestamp("2015-01-03").value)
        assert [1, 2, 3] == list(
            chain
        ), "[1, 2, 3] should be the active contracts on 01-03."

        chain = oc.active_chain(1, pd.Timestamp("2015-01-04").value)
        assert 4 == len(chain), (
            "[1, 2, 3, 4] should be the active contracts on "
            "01-04, this is all defined contracts in the test "
            "case."
        )

        chain = oc.active_chain(1, pd.Timestamp("2015-01-05").value)
        assert 4 == len(chain), (
            "[1, 2, 3, 4] should be the active contracts on "
            "01-05. This tests the case where all start dates "
            "are before the query date."
        )

        # Test querying each sid at a time when all should be alive.
        chain = oc.active_chain(2, pd.Timestamp("2015-01-05").value)
        assert [2, 3, 4] == list(chain)

        chain = oc.active_chain(3, pd.Timestamp("2015-01-05").value)
        assert [3, 4] == list(chain)

        chain = oc.active_chain(4, pd.Timestamp("2015-01-05").value)
        assert [4] == list(chain)

        # Test defined contract to check edge conditions.
        chain = oc.active_chain(4, pd.Timestamp("2015-01-03").value)
        assert [] == list(chain), (
            "No contracts should be active, since 01-03 is " "before 4's start date."
        )

        chain = oc.active_chain(4, pd.Timestamp("2015-01-04").value)
        assert [4] == list(chain), "[4] should be active beginning at its start date."

    def test_delivery_predicate(self):
        contract_sids = range(5, 8)
        contracts = deque(self.asset_finder.retrieve_all(contract_sids))

        oc = OrderedContracts(
            "BA",
            contracts,
            chain_predicate=partial(delivery_predicate, set(["F", "H"])),
        )

        # Test sid 1 as days increment, as the sessions march forward
        # a contract should be added per day, until all defined contracts
        # are returned.
        chain = oc.active_chain(5, pd.Timestamp("2015-01-05").value)
        assert [5, 7] == list(chain), (
            "Contract BAG16 (sid=6) should be ommitted from chain, since "
            "it does not satisfy the roll predicate."
        )

    def test_auto_close_before_start(self):
        contract_sids = np.array([8, 9, 10, 11], dtype=np.int64)
        contracts = self.asset_finder.retrieve_all(contract_sids)
        oc = OrderedContracts("BZ", deque(contracts))

        # The OrderedContracts chain should omit BZF16 and start with BZG16.
        assert oc.start_date == contracts[1].start_date
        assert oc.end_date == contracts[-1].end_date
        assert oc.contract_before_auto_close(oc.start_date.value) == 9

        # The OrderedContracts chain should end on the last contract even
        # though there is a gap between the auto close date of BZG16 and the
        # start date of BZH16. During this period, BZH16 should be considered
        # the center contract, as a placeholder of sorts.
        assert oc.contract_before_auto_close(contracts[1].notice_date.value) == 10
        assert oc.contract_before_auto_close(contracts[2].start_date.value) == 10


class NoPrefetchContinuousFuturesTestCase(ContinuousFuturesTestCase):
    DATA_PORTAL_MINUTE_HISTORY_PREFETCH = 0
    DATA_PORTAL_DAILY_HISTORY_PREFETCH = 0
