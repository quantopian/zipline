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

from textwrap import dedent

from numpy import (
    arange,
    array,
    int64,
    full,
    repeat,
    tile,
)
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas import Timestamp, DataFrame

from zipline import TradingAlgorithm
from zipline.assets.continuous_futures import OrderedContracts
from zipline.data.minute_bars import FUTURES_MINUTES_PER_DAY
from zipline.testing.fixtures import (
    WithCreateBarData,
    WithDataPortal,
    WithBcolzFutureMinuteBarReader,
    WithSimParams,
    ZiplineTestCase,
)


class ContinuousFuturesTestCase(WithCreateBarData,
                                WithDataPortal,
                                WithSimParams,
                                WithBcolzFutureMinuteBarReader,
                                ZiplineTestCase):

    START_DATE = pd.Timestamp('2015-01-05', tz='UTC')
    END_DATE = pd.Timestamp('2016-10-19', tz='UTC')

    SIM_PARAMS_START = pd.Timestamp('2016-01-26', tz='UTC')
    SIM_PARAMS_END = pd.Timestamp('2016-01-28', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    TRADING_CALENDAR_STRS = ('us_futures',)
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

    @classmethod
    def make_root_symbols_info(self):
        return pd.DataFrame({
            'root_symbol': ['FO'],
            'root_symbol_id': [1],
            'exchange': ['CME']})

    @classmethod
    def make_futures_info(self):
        return DataFrame({
            'symbol': ['FOF16', 'FOG16', 'FOH16', 'FOJ16', 'FOK16', 'FOF22'],
            'root_symbol': ['FO'] * 6,
            'asset_name': ['Foo'] * 6,
            'start_date': [Timestamp('2015-01-05', tz='UTC'),
                           Timestamp('2015-02-05', tz='UTC'),
                           Timestamp('2015-03-05', tz='UTC'),
                           Timestamp('2015-04-05', tz='UTC'),
                           Timestamp('2015-05-05', tz='UTC'),
                           Timestamp('2021-01-05', tz='UTC')],
            'end_date': [Timestamp('2016-08-19', tz='UTC'),
                         Timestamp('2016-09-19', tz='UTC'),
                         Timestamp('2016-10-19', tz='UTC'),
                         Timestamp('2016-11-19', tz='UTC'),
                         Timestamp('2016-12-19', tz='UTC'),
                         Timestamp('2022-08-19', tz='UTC')],
            'notice_date': [Timestamp('2016-01-27', tz='UTC'),
                            Timestamp('2016-02-26', tz='UTC'),
                            Timestamp('2016-03-24', tz='UTC'),
                            Timestamp('2016-04-26', tz='UTC'),
                            Timestamp('2016-05-26', tz='UTC'),
                            Timestamp('2022-01-26', tz='UTC')],
            'expiration_date': [Timestamp('2016-01-27', tz='UTC'),
                                Timestamp('2016-02-26', tz='UTC'),
                                Timestamp('2016-03-24', tz='UTC'),
                                Timestamp('2016-04-26', tz='UTC'),
                                Timestamp('2016-05-26', tz='UTC'),
                                Timestamp('2022-01-26', tz='UTC')],
            'auto_close_date': [Timestamp('2016-01-27', tz='UTC'),
                                Timestamp('2016-02-26', tz='UTC'),
                                Timestamp('2016-03-24', tz='UTC'),
                                Timestamp('2016-04-26', tz='UTC'),
                                Timestamp('2016-05-26', tz='UTC'),
                                Timestamp('2022-01-26', tz='UTC')],
            'tick_size': [0.001] * 6,
            'multiplier': [1000.0] * 6,
            'exchange': ['CME'] * 6,
        })

    @classmethod
    def make_future_minute_bar_data(cls):
        tc = cls.trading_calendar
        start = pd.Timestamp('2016-01-26', tz='UTC')
        end = pd.Timestamp('2016-04-29', tz='UTC')
        dts = tc.minutes_for_sessions_in_range(start, end)
        sessions = tc.sessions_in_range(start, end)
        # Generate values in the XXY.YYY space, with XX representing the
        # session and Y.YYY representing the minute within the session.
        # e.g. the close of the 23rd session would be 231.440.
        r = 10.0
        day_markers = repeat(
            arange(r, r * len(sessions) + r, r),
            FUTURES_MINUTES_PER_DAY)
        r = 0.001
        min_markers = tile(
            arange(r, r * FUTURES_MINUTES_PER_DAY + r, r),
            len(sessions))

        markers = day_markers + min_markers

        # Volume uses a similar scheme as above but times 1000.
        r = 10.0 * 1000
        vol_day_markers = repeat(
            arange(r, r * len(sessions) + r, r, dtype=int64),
            FUTURES_MINUTES_PER_DAY)
        r = 0.001 * 1000
        vol_min_markers = tile(
            arange(r, r * FUTURES_MINUTES_PER_DAY + r, r, dtype=int64),
            len(sessions))
        vol_markers = vol_day_markers + vol_min_markers
        base_df = pd.DataFrame(
            {
                'open': full(len(dts), 102000.0) + markers,
                'high': full(len(dts), 109000.0) + markers,
                'low': full(len(dts), 101000.0) + markers,
                'close': full(len(dts), 105000.0) + markers,
                'volume': full(len(dts), 10000, dtype=int64) + vol_markers,
            },
            index=dts)
        # Add the sid to the ones place of the prices, so that the ones
        # place can be used to eyeball the source contract.

        # For volume roll tests end sid volume early.
        # FOF16 cuts out day before autoclose of 01-26
        # FOG16 cuts out on autoclose
        # FOH16 cuts out 4 days before autoclose
        # FOJ16 cuts out 3 days before autoclose

        sid_to_vol_stop_session = {
            0: Timestamp('2016-01-26', tz='UTC'),
            1: Timestamp('2016-02-26', tz='UTC'),
            2: Timestamp('2016-03-18', tz='UTC'),
            3: Timestamp('2016-04-20', tz='UTC'),
        }
        for i in range(6):
            df = base_df.copy()
            df += i * 10000
            if i in sid_to_vol_stop_session:
                vol_stop_session = sid_to_vol_stop_session[i]
                m_open = tc.open_and_close_for_session(vol_stop_session)[0]
                loc = dts.searchsorted(m_open)
                # Add a little bit of noise to roll. So that checks for exacly
                # 0 do not work, since there may be stragglers after a roll.
                df.volume.values[loc] = 1000
                df.volume.values[loc + 1:] = 0
            j = i - 1
            if j in sid_to_vol_stop_session:
                non_primary_end = sid_to_vol_stop_session[j] - sessions.freq
                m_close = tc.open_and_close_for_session(non_primary_end)[1]
                loc = dts.searchsorted(m_close)
                # Add some volume before a roll, since a contracted may be
                # entered earlier than when it is the primary.
                df.volume.values[:loc] = 2000
            yield i, df

    def test_create_continuous_future(self):
        cf_primary = self.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')

        self.assertEqual(cf_primary.root_symbol, 'FO')
        self.assertEqual(cf_primary.offset, 0)
        self.assertEqual(cf_primary.roll_style, 'calendar')

        retrieved_primary = self.asset_finder.retrieve_asset(
            cf_primary.sid)

        self.assertEqual(retrieved_primary, cf_primary)

        cf_secondary = self.asset_finder.create_continuous_future(
            'FO', 1, 'calendar')

        self.assertEqual(cf_secondary.root_symbol, 'FO')
        self.assertEqual(cf_secondary.offset, 1)
        self.assertEqual(cf_secondary.roll_style, 'calendar')

        retrieved = self.asset_finder.retrieve_asset(
            cf_secondary.sid)

        self.assertEqual(retrieved, cf_secondary)

        self.assertNotEqual(cf_primary, cf_secondary)

    def test_current_contract(self):
        cf_primary = self.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-01-26', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')

        self.assertEqual(contract.symbol, 'FOF16')

        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-01-27', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')

        self.assertEqual(contract.symbol, 'FOG16',
                         'Auto close at beginning of session so FOG16 is now '
                         'the current contract.')

    def test_get_spot_value_contract_daily(self):
        cf_primary = self.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')

        contract = self.data_portal.get_spot_value(
            cf_primary,
            'contract',
            pd.Timestamp('2016-01-26', tz='UTC'),
            'daily',
        )

        self.assertEqual(contract.symbol, 'FOF16')

        contract = self.data_portal.get_spot_value(
            cf_primary,
            'contract',
            pd.Timestamp('2016-01-27', tz='UTC'),
            'daily',
        )

        self.assertEqual(contract.symbol, 'FOG16',
                         'Auto close at beginning of session so FOG16 is now '
                         'the current contract.')

    def test_current_contract_volume_roll(self):
        cf_primary = self.asset_finder.create_continuous_future(
            'FO', 0, 'volume')
        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-01-26', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')

        self.assertEqual(contract.symbol, 'FOG16')

        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-01-26', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')

        self.assertEqual(contract.symbol, 'FOG16',
                         'Auto close at beginning of session. FOG16 remains '
                         'the current contract.')

        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-02-26', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')
        self.assertEqual(contract.symbol, 'FOH16',
                         'Volume switch to FOH16, should have triggered roll.')

    def test_current_contract_in_algo(self):
        code = dedent("""
from zipline.api import (
    record,
    continuous_future,
    schedule_function,
    get_datetime,
)

def initialize(algo):
    algo.primary_cl = continuous_future('FO', 0, 'calendar')
    algo.secondary_cl = continuous_future('FO', 1, 'calendar')
    schedule_function(record_current_contract)

def record_current_contract(algo, data):
    record(datetime=get_datetime())
    record(primary=data.current(algo.primary_cl, 'contract'))
    record(secondary=data.current(algo.secondary_cl, 'contract'))
""")
        algo = TradingAlgorithm(script=code,
                                sim_params=self.sim_params,
                                trading_calendar=self.trading_calendar,
                                env=self.env)
        results = algo.run(self.data_portal)
        result = results.iloc[0]

        self.assertEqual(result.primary.symbol,
                         'FOF16',
                         'Primary should be FOF16 on first session.')
        self.assertEqual(result.secondary.symbol,
                         'FOG16',
                         'Secondary should be FOG16 on first session.')

        result = results.iloc[1]
        # Second day, primary should switch to FOG
        self.assertEqual(result.primary.symbol,
                         'FOG16',
                         'Primary should be FOG16 on second session, auto '
                         'close is at beginning of the session.')
        self.assertEqual(result.secondary.symbol,
                         'FOH16',
                         'Secondary should be FOH16 on second session, auto '
                         'close is at beginning of the session.')

        result = results.iloc[2]
        # Second day, primary should switch to FOG
        self.assertEqual(result.primary.symbol,
                         'FOG16',
                         'Primary should remain as FOG16 on third session.')
        self.assertEqual(result.secondary.symbol,
                         'FOH16',
                         'Secondary should remain as FOH16 on third session.')

    def test_current_chain_in_algo(self):
        code = dedent("""
from zipline.api import (
    record,
    continuous_future,
    schedule_function,
    get_datetime,
)

def initialize(algo):
    algo.primary_cl = continuous_future('FO', 0, 'calendar')
    algo.secondary_cl = continuous_future('FO', 1, 'calendar')
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
""")
        algo = TradingAlgorithm(script=code,
                                sim_params=self.sim_params,
                                trading_calendar=self.trading_calendar,
                                env=self.env)
        results = algo.run(self.data_portal)
        result = results.iloc[0]

        self.assertEqual(result.primary_len,
                         5,
                         'There should be only 5 contracts in the chain for '
                         'the primary, there are 6 contracts defined in the '
                         'fixture, but one has a start after the simulation '
                         'date.')
        self.assertEqual(result.secondary_len,
                         4,
                         'There should be only 4 contracts in the chain for '
                         'the primary, there are 6 contracts defined in the '
                         'fixture, but one has a start after the simulation '
                         'date. And the first is not included because it is '
                         'the primary on that date.')

        self.assertEqual(result.primary_first,
                         'FOF16',
                         'Front of primary chain should be FOF16 on first '
                         'session.')
        self.assertEqual(result.secondary_first,
                         'FOG16',
                         'Front of secondary chain should be FOG16 on first '
                         'session.')

        self.assertEqual(result.primary_last,
                         'FOK16',
                         'End of primary chain should be FOK16 on first '
                         'session.')
        self.assertEqual(result.secondary_last,
                         'FOK16',
                         'End of secondary chain should be FOK16 on first '
                         'session.')

        # Second day, primary should switch to FOG
        result = results.iloc[1]

        self.assertEqual(result.primary_len,
                         4,
                         'There should be only 4 contracts in the chain for '
                         'the primary, there are 6 contracts defined in the '
                         'fixture, but one has a start after the simulation '
                         'date. The first is not included because of roll.')
        self.assertEqual(result.secondary_len,
                         3,
                         'There should be only 3 contracts in the chain for '
                         'the primary, there are 6 contracts defined in the '
                         'fixture, but one has a start after the simulation '
                         'date. The first is not included because of roll, '
                         'the second is the primary on that date.')

        self.assertEqual(result.primary_first,
                         'FOG16',
                         'Front of primary chain should be FOG16 on second '
                         'session.')
        self.assertEqual(result.secondary_first,
                         'FOH16',
                         'Front of secondary chain should be FOH16 on second '
                         'session.')

        # These values remain FOJ16 because fixture data is not exhaustive
        # enough to move the end of the chain.
        self.assertEqual(result.primary_last,
                         'FOK16',
                         'End of primary chain should be FOK16 on second '
                         'session.')
        self.assertEqual(result.secondary_last,
                         'FOK16',
                         'End of secondary chain should be FOK16 on second '
                         'session.')

    def test_history_sid_session(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-03-04 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1d', 'sid')

        self.assertEqual(window.loc['2016-01-26', cf],
                         0,
                         "Should be FOF16 at beginning of window.")

        self.assertEqual(window.loc['2016-01-27', cf],
                         1,
                         "Should be FOG16 after first roll.")

        self.assertEqual(window.loc['2016-02-25', cf],
                         1,
                         "Should be FOG16 on session before roll.")

        self.assertEqual(window.loc['2016-02-26', cf],
                         2,
                         "Should be FOH16 on session with roll.")

        self.assertEqual(window.loc['2016-02-29', cf],
                         2,
                         "Should be FOH16 on session after roll.")

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-04-06 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1d', 'sid')

        self.assertEqual(window.loc['2016-02-25', cf],
                         1,
                         "Should be FOG16 at beginning of window.")

        self.assertEqual(window.loc['2016-02-26', cf],
                         2,
                         "Should be FOH16 on session with roll.")

        self.assertEqual(window.loc['2016-02-29', cf],
                         2,
                         "Should be FOH16 on session after roll.")

        self.assertEqual(window.loc['2016-03-24', cf],
                         3,
                         "Should be FOJ16 on session with roll.")

        self.assertEqual(window.loc['2016-03-28', cf],
                         3,
                         "Should be FOJ16 on session after roll.")

    def test_history_sid_session_secondary(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 1, 'calendar')
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-03-04 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1d', 'sid')

        self.assertEqual(window.loc['2016-01-26', cf],
                         1,
                         "Should be FOG16 at beginning of window.")

        self.assertEqual(window.loc['2016-01-27', cf],
                         2,
                         "Should be FOH16 after first roll.")

        self.assertEqual(window.loc['2016-02-25', cf],
                         2,
                         "Should be FOH16 on session before roll.")

        self.assertEqual(window.loc['2016-02-26', cf],
                         3,
                         "Should be FOJ16 on session with roll.")

        self.assertEqual(window.loc['2016-02-29', cf],
                         3,
                         "Should be FOJ16 on session after roll.")

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-04-06 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1d', 'sid')

        self.assertEqual(window.loc['2016-02-25', cf],
                         2,
                         "Should be FOH16 at beginning of window.")

        self.assertEqual(window.loc['2016-02-26', cf],
                         3,
                         "Should be FOJ16 on session with roll.")

        self.assertEqual(window.loc['2016-02-29', cf],
                         3,
                         "Should be FOJ16 on session after roll.")

        self.assertEqual(window.loc['2016-03-24', cf],
                         4,
                         "Should be FOK16 on session with roll.")

        self.assertEqual(window.loc['2016-03-28', cf],
                         4,
                         "Should be FOK16 on session after roll.")

    def test_history_sid_session_volume_roll(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'volume')
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-03-04 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1d', 'sid')

        # Volume cuts out for FOF16 on 2016-01-25
        self.assertEqual(window.loc['2016-01-26', cf],
                         1,
                         "Should be FOG16 at beginning of window.")

        self.assertEqual(window.loc['2016-01-27', cf],
                         1,
                         "Should have remained FOG16.")

        self.assertEqual(window.loc['2016-02-25', cf],
                         1,
                         "Should be FOG16 on session before roll.")

        self.assertEqual(window.loc['2016-02-26', cf],
                         2,
                         "Should be FOH16 on session with roll.")

        self.assertEqual(window.loc['2016-02-29', cf],
                         2,
                         "Should be FOH16 on session after roll.")

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-04-06 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1d', 'sid')

        self.assertEqual(window.loc['2016-02-25', cf],
                         1,
                         "Should be FOG16 at beginning of window.")

        self.assertEqual(window.loc['2016-02-26', cf],
                         2,
                         "Should be FOH16 on session with roll.")

        self.assertEqual(window.loc['2016-02-29', cf],
                         2,
                         "Should be FOH16 on session after roll.")

        self.assertEqual(window.loc['2016-03-17', cf],
                         2,
                         "Should be FOH16 on session before volume cuts out.")

        self.assertEqual(window.loc['2016-03-18', cf],
                         3,
                         "Should be FOJ16 on session where the volume of "
                         "FOH16 cuts out.")

        self.assertEqual(window.loc['2016-03-24', cf],
                         3,
                         "Should have remained FOJ16.")

        self.assertEqual(window.loc['2016-03-28', cf],
                         3,
                         "Should have remained FOJ16.")

    def test_history_sid_minute(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        window = self.data_portal.get_history_window(
            [cf.sid],
            Timestamp('2016-01-26 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1m', 'sid')

        self.assertEqual(window.loc['2016-01-26 22:32', cf],
                         0,
                         "Should be FOF16 at beginning of window. A minute "
                         "which is in the 01-26 session, before the roll.")

        self.assertEqual(window.loc['2016-01-26 23:00', cf],
                         0,
                         "Should be FOF16 on on minute before roll minute.")

        self.assertEqual(window.loc['2016-01-26 23:01', cf],
                         1,
                         "Should be FOG16 on minute after roll.")

        # Advance the window a day.
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-01-27 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1m', 'sid')

        self.assertEqual(window.loc['2016-01-27 22:32', cf],
                         1,
                         "Should be FOG16 at beginning of window.")

        self.assertEqual(window.loc['2016-01-27 23:01', cf],
                         1,
                         "Should remain FOG16 on next session.")

    def test_history_close_session(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        window = self.data_portal.get_history_window(
            [cf.sid], Timestamp('2016-03-06', tz='UTC'), 30, '1d', 'close')

        assert_almost_equal(
            window.loc['2016-01-26', cf],
            105011.440,
            err_msg="At beginning of window, should be FOG16's first value.")

        assert_almost_equal(
            window.loc['2016-02-26', cf],
            125241.440,
            err_msg="On session with roll, should be FOH16's 24th value.")

        assert_almost_equal(
            window.loc['2016-02-29', cf],
            125251.440,
            err_msg="After roll, Should be FOH16's 25th value.")

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf.sid], Timestamp('2016-04-06', tz='UTC'), 30, '1d', 'close')

        assert_almost_equal(
            window.loc['2016-02-24', cf],
            115221.440,
            err_msg="At beginning of window, should be FOG16's 22nd value.")

        assert_almost_equal(
            window.loc['2016-02-26', cf],
            125241.440,
            err_msg="On session with roll, should be FOH16's 24th value.")

        assert_almost_equal(
            window.loc['2016-02-29', cf],
            125251.440,
            err_msg="On session after roll, should be FOH16's 25th value.")

        assert_almost_equal(
            window.loc['2016-03-24', cf],
            135431.440,
            err_msg="On session with roll, should be FOJ16's 43rd value.")

        assert_almost_equal(
            window.loc['2016-03-28', cf],
            135441.440,
            err_msg="On session after roll, Should be FOJ16's 44th value.")

    def test_history_close_session_adjusted(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        cf_mul = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar').adj('mul')
        cf_add = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar').adj('add')
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            Timestamp('2016-03-06', tz='UTC'), 30, '1d', 'close')

        # Unadjusted value is: 115011.44
        # Adjustment is based on hop from 115231.44 to 125231.44
        # a ratio of ~0.920
        assert_almost_equal(
            window.loc['2016-01-26', cf_mul],
            124992.348,
            err_msg="At beginning of window, should be FOG16's first value, "
            "adjusted.")

        # Difference of 7008.561
        assert_almost_equal(
            window.loc['2016-01-26', cf_add],
            125011.44,
            err_msg="At beginning of window, should be FOG16's first value, "
            "adjusted.")

        assert_almost_equal(
            window.loc['2016-02-26', cf_mul],
            125241.440,
            err_msg="On session with roll, should be FOH16's 24th value, "
            "unadjusted.")

        assert_almost_equal(
            window.loc['2016-02-26', cf_add],
            125241.440,
            err_msg="On session with roll, should be FOH16's 24th value, "
            "unadjusted.")

        assert_almost_equal(
            window.loc['2016-02-29', cf_mul],
            125251.440,
            err_msg="After roll, Should be FOH16's 25th value, unadjusted.")

        assert_almost_equal(
            window.loc['2016-02-29', cf_add],
            125251.440,
            err_msg="After roll, Should be FOH16's 25th value, unadjusted.")

        # Advance the window a month.
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            Timestamp('2016-04-06', tz='UTC'), 30, '1d', 'close')

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
            window.loc['2016-02-24', cf_mul],
            135236.905,
            err_msg="At beginning of window, should be FOG16's 22nd value, "
            "with two adjustments.")

        assert_almost_equal(
            window.loc['2016-02-24', cf_add],
            135251.44,
            err_msg="At beginning of window, should be FOG16's 22nd value, "
            "with two adjustments")

        # Unadjusted: 125241.44
        assert_almost_equal(
            window.loc['2016-02-26', cf_mul],
            135259.442,
            err_msg="On session with roll, should be FOH16's 24th value, "
            "with one adjustment.")

        assert_almost_equal(
            window.loc['2016-02-26', cf_add],
            135271.44,
            err_msg="On session with roll, should be FOH16's 24th value, "
            "with one adjustment.")

        # Unadjusted: 125251.44
        assert_almost_equal(
            window.loc['2016-02-29', cf_mul],
            135270.241,
            err_msg="On session after roll, should be FOH16's 25th value, "
            "with one adjustment.")

        assert_almost_equal(
            window.loc['2016-02-29', cf_add],
            135281.44,
            err_msg="On session after roll, should be FOH16's 25th value, "
            "unadjusted.")

        # Unadjusted: 135431.44
        assert_almost_equal(
            window.loc['2016-03-24', cf_mul],
            135431.44,
            err_msg="On session with roll, should be FOJ16's 43rd value, "
            "unadjusted.")

        assert_almost_equal(
            window.loc['2016-03-24', cf_add],
            135431.44,
            err_msg="On session with roll, should be FOJ16's 43rd value.")

        # Unadjusted: 135441.44
        assert_almost_equal(
            window.loc['2016-03-28', cf_mul],
            135441.44,
            err_msg="On session after roll, Should be FOJ16's 44th value.")

        assert_almost_equal(
            window.loc['2016-03-28', cf_add],
            135441.44,
            err_msg="On session after roll, Should be FOJ16's 44th value.")

    def test_history_close_minute(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        window = self.data_portal.get_history_window(
            [cf.sid],
            Timestamp('2016-02-25 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1m', 'close')

        self.assertEqual(window.loc['2016-02-25 22:32', cf],
                         115231.412,
                         "Should be FOG16 at beginning of window. A minute "
                         "which is in the 02-25 session, before the roll.")

        self.assertEqual(window.loc['2016-02-25 23:00', cf],
                         115231.440,
                         "Should be FOG16 on on minute before roll minute.")

        self.assertEqual(window.loc['2016-02-25 23:01', cf],
                         125240.001,
                         "Should be FOH16 on minute after roll.")

        # Advance the window a session.
        window = self.data_portal.get_history_window(
            [cf],
            Timestamp('2016-02-28 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1m', 'close')

        self.assertEqual(window.loc['2016-02-26 22:32', cf],
                         125241.412,
                         "Should be FOH16 at beginning of window.")

        self.assertEqual(window.loc['2016-02-28 23:01', cf],
                         125250.001,
                         "Should remain FOH16 on next session.")

    def test_history_close_minute_adjusted(self):
        cf = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar')
        cf_mul = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar').adj('mul')
        cf_add = self.data_portal.asset_finder.create_continuous_future(
            'FO', 0, 'calendar').adj('add')
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            Timestamp('2016-02-25 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1m', 'close')

        # Unadjusted: 115231.412
        # Adjustment based on roll:
        # 2016-02-25 23:00:00+00:00
        # front: 115231.440
        # back:  125231.440
        # Ratio: ~0.920
        # Difference: 10000.00
        self.assertEqual(window.loc['2016-02-25 22:32', cf_mul],
                         125231.41,
                         "Should be FOG16 at beginning of window. A minute "
                         "which is in the 02-25 session, before the roll.")

        self.assertEqual(window.loc['2016-02-25 22:32', cf_add],
                         125231.412,
                         "Should be FOG16 at beginning of window. A minute "
                         "which is in the 02-25 session, before the roll.")

        # Unadjusted: 115231.44
        # Should use same ratios as above.
        self.assertEqual(window.loc['2016-02-25 23:00', cf_mul],
                         125231.44,
                         "Should be FOG16 on on minute before roll minute, "
                         "adjusted.")

        self.assertEqual(window.loc['2016-02-25 23:00', cf_add],
                         125231.44,
                         "Should be FOG16 on on minute before roll minute, "
                         "adjusted.")

        self.assertEqual(window.loc['2016-02-25 23:01', cf_mul],
                         125240.001,
                         "Should be FOH16 on minute after roll, unadjusted.")

        self.assertEqual(window.loc['2016-02-25 23:01', cf_add],
                         125240.001,
                         "Should be FOH16 on minute after roll, unadjusted.")

        # Advance the window a session.
        window = self.data_portal.get_history_window(
            [cf, cf_mul, cf_add],
            Timestamp('2016-02-28 18:01', tz='US/Eastern').tz_convert('UTC'),
            30, '1m', 'close')

        # No adjustments in this window.
        self.assertEqual(window.loc['2016-02-26 22:32', cf_mul],
                         125241.412,
                         "Should be FOH16 at beginning of window.")

        self.assertEqual(window.loc['2016-02-28 23:01', cf_mul],
                         125250.001,
                         "Should remain FOH16 on next session.")


class OrderedContractsTestCase(ZiplineTestCase):

    def test_contract_at_offset(self):
        contract_sids = array([1, 2, 3, 4], dtype=int64)
        start_dates = pd.date_range('2015-01-01', periods=4, tz="UTC")
        auto_close_dates = pd.date_range('2016-04-01', periods=4, tz="UTC")

        oc = OrderedContracts('FO',
                              contract_sids,
                              start_dates.astype('int64'),
                              auto_close_dates.astype('int64'))

        self.assertEquals(1,
                          oc.contract_at_offset(1, 0, start_dates[-1].value),
                          "Offset of 0 should return provided sid")

        self.assertEquals(2,
                          oc.contract_at_offset(1, 1, start_dates[-1].value),
                          "Offset of 1 should return next sid in chain.")

        self.assertEquals(None,
                          oc.contract_at_offset(4, 1, start_dates[-1].value),
                          "Offset at end of chain should not crash.")

    def test_active_chain(self):
        contract_sids = array([1, 2, 3, 4], dtype=int64)
        start_dates = pd.date_range('2015-01-01', periods=4, tz="UTC")
        auto_close_dates = pd.date_range('2016-04-01', periods=4, tz="UTC")

        oc = OrderedContracts('FO',
                              contract_sids,
                              start_dates.astype('int64'),
                              auto_close_dates.astype('int64'))

        # Test sid 1 as days increment, as the sessions march forward
        # a contract should be added per day, until all defined contracts
        # are returned.
        chain = oc.active_chain(1, pd.Timestamp('2014-12-31', tz='UTC').value)
        self.assertEquals([], list(chain),
                          "On session before first start date, no contracts "
                          "in chain should be active.")
        chain = oc.active_chain(1, pd.Timestamp('2015-01-01', tz='UTC').value)
        self.assertEquals([1], list(chain),
                          "[1] should be the active chain on 01-01, since all "
                          "other start dates occur after 01-01.")

        chain = oc.active_chain(1, pd.Timestamp('2015-01-02', tz='UTC').value)
        self.assertEquals([1, 2], list(chain),
                          "[1, 2] should be the active contracts on 01-02.")

        chain = oc.active_chain(1, pd.Timestamp('2015-01-03', tz='UTC').value)
        self.assertEquals([1, 2, 3], list(chain),
                          "[1, 2, 3] should be the active contracts on 01-03.")

        chain = oc.active_chain(1, pd.Timestamp('2015-01-04', tz='UTC').value)
        self.assertEquals(4, len(chain),
                          "[1, 2, 3, 4] should be the active contracts on "
                          "01-04, this is all defined contracts in the test "
                          "case.")

        chain = oc.active_chain(1, pd.Timestamp('2015-01-05', tz='UTC').value)
        self.assertEquals(4, len(chain),
                          "[1, 2, 3, 4] should be the active contracts on "
                          "01-05. This tests the case where all start dates "
                          "are before the query date.")

        # Test querying each sid at a time when all should be alive.
        chain = oc.active_chain(2, pd.Timestamp('2015-01-05', tz='UTC').value)
        self.assertEquals([2, 3, 4], list(chain))

        chain = oc.active_chain(3, pd.Timestamp('2015-01-05', tz='UTC').value)
        self.assertEquals([3, 4], list(chain))

        chain = oc.active_chain(4, pd.Timestamp('2015-01-05', tz='UTC').value)
        self.assertEquals([4], list(chain))

        # Test defined contract to check edge conditions.
        chain = oc.active_chain(4, pd.Timestamp('2015-01-03', tz='UTC').value)
        self.assertEquals([], list(chain),
                          "No contracts should be active, since 01-03 is "
                          "before 4's start date.")

        chain = oc.active_chain(4, pd.Timestamp('2015-01-04', tz='UTC').value)
        self.assertEquals([4], list(chain),
                          "[4] should be active beginning at its start date.")


class NoPrefetchContinuousFuturesTestCase(ContinuousFuturesTestCase):
    DATA_PORTAL_MINUTE_HISTORY_PREFETCH = 0
    DATA_PORTAL_DAILY_HISTORY_PREFETCH = 0
