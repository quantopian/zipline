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

import pandas as pd
from pandas import Timestamp, DataFrame

from zipline import TradingAlgorithm
from zipline.testing.fixtures import (
    WithCreateBarData,
    WithSimParams,
    ZiplineTestCase,
)


class ContinuousFuturesTestCase(WithCreateBarData,
                                WithSimParams,
                                ZiplineTestCase):

    START_DATE = pd.Timestamp('2015-01-05', tz='UTC')
    END_DATE = pd.Timestamp('2016-10-19', tz='UTC')

    SIM_PARAMS_START = pd.Timestamp('2016-01-25', tz='UTC')
    SIM_PARAMS_END = pd.Timestamp('2016-01-27', tz='UTC')
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
            'symbol': ['FOF', 'FOG', 'FOH'],
            'root_symbol': ['FO', 'FO', 'FO'],
            'asset_name': ['Foo'] * 3,
            'start_date': [Timestamp('2015-01-05', tz='UTC'),
                           Timestamp('2015-02-05', tz='UTC'),
                           Timestamp('2015-03-05', tz='UTC')],
            'end_date': [Timestamp('2016-08-19', tz='UTC'),
                         Timestamp('2016-09-19', tz='UTC'),
                         Timestamp('2016-10-19', tz='UTC')],
            'notice_date': [Timestamp('2016-01-26', tz='UTC'),
                            Timestamp('2016-02-26', tz='UTC'),
                            Timestamp('2016-03-26', tz='UTC')],
            'expiration_date': [Timestamp('2016-01-26', tz='UTC'),
                                Timestamp('2016-02-26', tz='UTC'),
                                Timestamp('2016-03-26', tz='UTC')],
            'auto_close_date': [Timestamp('2016-01-26', tz='UTC'),
                                Timestamp('2016-02-26', tz='UTC'),
                                Timestamp('2016-03-26', tz='UTC')],
            'tick_size': [0.001] * 3,
            'multiplier': [1000.0] * 3,
            'exchange': ['CME'] * 3,
        })

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
            lambda: pd.Timestamp('2016-01-25', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')

        self.assertEqual(contract.symbol, 'FOF')

        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-01-26', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')

        self.assertEqual(contract.symbol, 'FOG',
                         'Auto close at beginning of session so FOG is now '
                         'the current contract.')

        bar_data = self.create_bardata(
            lambda: pd.Timestamp('2016-01-27', tz='UTC'))
        contract = bar_data.current(cf_primary, 'contract')
        self.assertEqual(contract.symbol, 'FOG')

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

        self.assertEqual(results.iloc[0].primary.symbol,
                         'FOF',
                         'Primary should be FOF on first session.')
        self.assertEqual(results.iloc[0].secondary.symbol,
                         'FOG',
                         'Secondary should be FOG on first session.')

        # Second day, primary should switch to FOG
        self.assertEqual(results.iloc[1].primary.symbol,
                         'FOG',
                         'Primary should be FOG on second session, auto close '
                         'is at beginning of the session.')
        self.assertEqual(results.iloc[1].secondary.symbol,
                         'FOH',
                         'Secondary should be FOH on second session, auto '
                         'close is at beginning of the session.')

        # Second day, primary should switch to FOG
        self.assertEqual(results.iloc[2].primary.symbol,
                         'FOG',
                         'Primary should remain as FOG on third session.')
        self.assertEqual(results.iloc[2].secondary.symbol,
                         'FOH',
                         'Secondary should remain as FOG on third session.')
