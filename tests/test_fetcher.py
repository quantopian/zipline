#
# Copyright 2015 Quantopian, Inc.
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

from unittest import TestCase
from nose_parameterized import parameterized

import pandas as pd
import numpy as np
import responses
from mock import patch
from zipline import TradingAlgorithm
from zipline.errors import UnsupportedOrderParameters
from zipline.finance.trading import TradingEnvironment
from zipline.sources.requests_csv import mask_requests_args

from zipline.utils import factory
from zipline.utils.test_utils import FetcherDataPortal

from .resources.fetcher_inputs.fetcher_test_data import (
    MULTI_SIGNAL_CSV_DATA,
    AAPL_CSV_DATA,
    AAPL_MINUTE_CSV_DATA,
    IBM_CSV_DATA,
    ANNUAL_AAPL_CSV_DATA,
    AAPL_IBM_CSV_DATA,
    NOMATCH_CSV_DATA,
    CPIAUCSL_DATA,
    PALLADIUM_DATA,
    FETCHER_UNIVERSE_DATA,
    NON_ASSET_FETCHER_UNIVERSE_DATA,
    FETCHER_UNIVERSE_DATA_TICKER_COLUMN, FETCHER_ALTERNATE_COLUMN_HEADER)


class FetcherTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        responses.start()
        responses.add(responses.GET,
                      'https://fake.urls.com/aapl_minute_csv_data.csv',
                      body=AAPL_MINUTE_CSV_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/aapl_csv_data.csv',
                      body=AAPL_CSV_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/multi_signal_csv_data.csv',
                      body=MULTI_SIGNAL_CSV_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/nomatch_csv_data.csv',
                      body=NOMATCH_CSV_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/cpiaucsl_data.csv',
                      body=CPIAUCSL_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/ibm_csv_data.csv',
                      body=IBM_CSV_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/aapl_ibm_csv_data.csv',
                      body=AAPL_IBM_CSV_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/palladium_data.csv',
                      body=PALLADIUM_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/fetcher_universe_data.csv',
                      body=FETCHER_UNIVERSE_DATA, content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/bad_fetcher_universe_data.csv',
                      body=NON_ASSET_FETCHER_UNIVERSE_DATA,
                      content_type='text/csv')
        responses.add(responses.GET,
                      'https://fake.urls.com/annual_aapl_csv_data.csv',
                      body=ANNUAL_AAPL_CSV_DATA, content_type='text/csv')

        cls.sim_params = factory.create_simulation_parameters()
        cls.env = TradingEnvironment()
        cls.env.write_data(
            equities_data={
                24: {
                    "start_date": pd.Timestamp("2006-01-01", tz='UTC'),
                    "end_date": pd.Timestamp("2007-01-01", tz='UTC'),
                    'symbol': "AAPL",
                    "asset_type": "equity",
                    "exchange": "nasdaq"
                },
                3766: {
                    "start_date": pd.Timestamp("2006-01-01", tz='UTC'),
                    "end_date": pd.Timestamp("2007-01-01", tz='UTC'),
                    'symbol': "IBM",
                    "asset_type": "equity",
                    "exchange": "nasdaq"
                },
                5061: {
                    "start_date": pd.Timestamp("2006-01-01", tz='UTC'),
                    "end_date": pd.Timestamp("2007-01-01", tz='UTC'),
                    'symbol': "MSFT",
                    "asset_type": "equity",
                    "exchange": "nasdaq"
                },
                14848: {
                    "start_date": pd.Timestamp("2006-01-01", tz='UTC'),
                    "end_date": pd.Timestamp("2007-01-01", tz='UTC'),
                    'symbol': "YHOO",
                    "asset_type": "equity",
                    "exchange": "nasdaq"
                },
                25317: {
                    "start_date": pd.Timestamp("2006-01-01", tz='UTC'),
                    "end_date": pd.Timestamp("2007-01-01", tz='UTC'),
                    'symbol': "DELL",
                    "asset_type": "equity",
                    "exchange": "nasdaq"
                }
            }

        )

    def run_algo(self, code, sim_params=None, data_frequency="daily"):
        if sim_params is None:
            sim_params = self.sim_params

        test_algo = TradingAlgorithm(
            script=code,
            sim_params=sim_params,
            env=self.env,
            data_frequency=data_frequency
        )

        results = test_algo.run(
            data_portal=FetcherDataPortal(self.env, self.sim_params)
        )

        return results

    def test_minutely_fetcher(self):
        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-03", tz='UTC'),
            end=pd.Timestamp("2006-01-31", tz='UTC'),
            emission_rate="minute",
            data_frequency="minute"
        )

        test_algo = TradingAlgorithm(
            script="""
from zipline.api import fetch_csv, record, sid

def initialize(context):
    fetch_csv('https://fake.urls.com/aapl_minute_csv_data.csv')

def handle_data(context, data):
    record(aapl_signal=data[sid(24)].signal)
""", sim_params=sim_params, data_frequency="minute", env=self.env)

        # manually setting data portal and getting generator because we need
        # the minutely emission packets here.  TradingAlgorithm.run() only
        # returns daily packets.
        test_algo.data_portal = FetcherDataPortal(self.env, sim_params)
        gen = test_algo.get_generator()
        perf_packets = list(gen)

        signal = [result["minute_perf"]["recorded_vars"]["aapl_signal"] for
                  result in perf_packets if "minute_perf" in result]

        self.assertEqual(20 * 390, len(signal))

        # csv data is:
        # symbol,date,signal
        # aapl,1/4/06 4:01PM,-1
        # aapl,1/5/06 4:00PM,5
        # aapl,1/6/06 9:30AM,6
        # aapl,1/9/06 12:01PM,9

        # dates are interpreted as UTC time
        # market hours are 14:31-21:00 UTC each of those days

        # day1 starts at 2006-01-04 14:31
        # day1 ends at 2006-01-04 21:00
        # day2 starts at 2006-01-04 14:31
        # -1 starts at 2006-01-04 16:01
        # day2 ends at 2006-01-04 14:31
        # day3 starts at 2006-01-05 14:31
        # 5 starts at 2006-01-05 16:00
        # day3 ends at 2006-01-05 21:00
        # 6 starts at 2006-01-06 9:30
        # day4 starts at 2006-01-06 14:31
        # day4 ends at 2006-01-06 21:00
        # 9 starts at 2006-01-09 12:01
        # day5 starts at 2006-01-09 14:31
        # day5 ends at 2006-01-09 21:00
        # ...
        # day20 ends at 2006-01-31 21:00

        # 480 NaNs
        # 389 -1s
        # 301 5s
        # 390 6s
        # 6240 9s

        values = [result["minute_perf"]["recorded_vars"]["aapl_signal"]
                  for result in perf_packets if "minute_perf" in result]

        np.testing.assert_array_equal([np.NaN] * 480, values[0:480])
        np.testing.assert_array_equal([-1.0] * 389, values[480:869])
        np.testing.assert_array_equal([5.0] * 301, values[869:1170])
        np.testing.assert_array_equal([6.0] * 390, values[1170:1560])
        np.testing.assert_array_equal([9.0] * 6240, values[1560:])

    def test_fetch_csv_with_multi_symbols(self):
        results = self.run_algo(
            """
from zipline.api import fetch_csv, record, sid

def initialize(context):
    fetch_csv('https://fake.urls.com/multi_signal_csv_data.csv')
    context.stocks = [sid(3766), sid(25317)]

def handle_data(context, data):
    record(ibm_signal=data[sid(3766)]["signal"])
    record(dell_signal=data[sid(25317)]["signal"])

    assert "signal" not in data[sid(24)]
    """)

        self.assertEqual(5, results["ibm_signal"].iloc[-1])
        self.assertEqual(5, results["dell_signal"].iloc[-1])

    def test_fetch_csv_with_nomatch_symbol(self):
        """
        The algorithm is loading data with a symbol column
        that contains a symbol that doesn't match anything in
        our database. Letting these types events through
        creates complications for the order method. So we drop these
        values, instead of just letting the securities roll through.

        This test also ensures that if a given symbol has *multiple*
        matches, but *none* of the matches are inside the test range,
        that none of them are found (nor does the algo crash)
        """
        results = self.run_algo(
            """
from zipline.api import fetch_csv, sid, record

def initialize(context):
    fetch_csv('https://fake.urls.com/nomatch_csv_data.csv',
              mask=True)
    context.stocks = [sid(3766), sid(25317)]

def handle_data(context, data):
    if "signal" in data[sid(3766)]:
        record(ibm_signal=data[sid(3766)]["signal"])

    if "signal" in data[sid(25317)]:
        record(dell_signal=data[sid(25317)]["signal"])
            """)

        self.assertNotIn("dell_signal", results.columns)
        self.assertNotIn("ibm_signal", results.columns)

    def test_fetch_csv_with_pure_signal_file(self):
        results = self.run_algo(
            """
from zipline.api import fetch_csv, sid, record

def clean(df):
    return df.rename(columns={'Value':'cpi', 'Date':'date'})

def initialize(context):
    fetch_csv(
        'https://fake.urls.com/cpiaucsl_data.csv',
        symbol='urban',
        pre_func=clean,
        date_format='%Y-%m-%d'
        )
    context.stocks = [sid(3766), sid(25317)]

def handle_data(context, data):
    cur_cpi = data['urban']['cpi']
    record(cpi=cur_cpi)
            """)

        self.assertEqual(results["cpi"][-1], 203.1)

    def test_algo_fetch_csv(self):
        results = self.run_algo(
            """
from zipline.api import fetch_csv, record, sid

def normalize(df):
    df['scaled'] = df['signal'] * 10
    return df

def initialize(context):
    fetch_csv('https://fake.urls.com/aapl_csv_data.csv',
            post_func=normalize)
    context.checked_name = False

def handle_data(context, data):
    record(
        signal=data[sid(24)]['signal'],
        scaled=data[sid(24)]['scaled'],
        price=data[sid(24)].price)
        """)

        self.assertEqual(5, results["signal"][-1])
        self.assertEqual(50, results["scaled"][-1])
        self.assertEqual(24, results["price"][-1])  # fake value

    def test_algo_fetch_csv_with_extra_symbols(self):
        results = self.run_algo(
            """
from zipline.api import fetch_csv, record, sid

def normalize(df):
    df['scaled'] = df['signal'] * 10
    return df

def initialize(context):
    fetch_csv('https://fake.urls.com/aapl_ibm_csv_data.csv',
            post_func=normalize,
            mask=True)

def handle_data(context, data):
    if 'signal' in data[sid(24)]:
        record(
            signal=data[sid(24)]['signal'],
            scaled=data[sid(24)]['scaled'],
            price=data[sid(24)].price)
            """
        )

        self.assertEqual(5, results["signal"][-1])
        self.assertEqual(50, results["scaled"][-1])
        self.assertEqual(24, results["price"][-1])  # fake value

    @parameterized.expand([("unspecified", ""),
                           ("none", "usecols=None"),
                           ("empty", "usecols=[]"),
                           ("without date", "usecols=['Value']"),
                           ("with date", "usecols=('Value', 'Date')")])
    def test_usecols(self, testname, usecols):
        code = """
from zipline.api import fetch_csv, sid, record

def clean(df):
    return df.rename(columns={{'Value':'cpi'}})

def initialize(context):
    fetch_csv(
        'https://fake.urls.com/cpiaucsl_data.csv',
        symbol='urban',
        pre_func=clean,
        date_column='Date',
        date_format='%Y-%m-%d',{usecols}
        )
    context.stocks = [sid(3766), sid(25317)]

def handle_data(context, data):
    if {should_have_data}:
        assert 'cpi' in data['urban']
    else:
        assert 'cpi' not in data['urban']
        """

        results = self.run_algo(
            code.format(
                usecols=usecols,
                should_have_data=testname in [
                    'none',
                    'unspecified',
                    'without date',
                    'with date',
                ],
            )
        )

        # 251 trading days in 2006
        self.assertEqual(len(results), 251)

    def test_sources_merge_custom_ticker(self):
        requests_kwargs = {}

        def capture_kwargs(zelf, url, **kwargs):
            requests_kwargs.update(
                mask_requests_args(url, kwargs).requests_kwargs
            )
            return PALLADIUM_DATA

        # Patching fetch_url instead of using responses in this test so that we
        # can intercept the requests keyword arguments and confirm that they're
        # correct.
        with patch('zipline.sources.requests_csv.PandasRequestsCSV.fetch_url',
                   new=capture_kwargs):
            results = self.run_algo(
                """
from zipline.api import fetch_csv, record, sid

def rename_col(df):
    df = df.rename(columns={'New York 15:00': 'price'})
    df = df.fillna(method='ffill')
    return df[['price', 'sid']]

def initialize(context):
    fetch_csv('https://dl.dropbox.com/u/16705795/PALL.csv',
        date_column='Date',
        symbol='palladium',
        post_func=rename_col,
        date_format='%Y-%m-%d'
        )
    context.stock = sid(24)

def handle_data(context, data):
    palladium = data['palladium']
    aapl = data[context.stock]
    if 'price' in palladium:
        record(palladium=palladium.price)
    if 'price' in aapl:
        record(aapl=aapl.price)
        """)

            np.testing.assert_array_equal([24] * 251, results["aapl"])
            self.assertEqual(337, results["palladium"].iloc[-1])

            expected = {
                'allow_redirects': False,
                'stream': True,
                'timeout': 30.0,
            }

            self.assertEqual(expected, requests_kwargs)

    @parameterized.expand([("symbol", FETCHER_UNIVERSE_DATA, None),
                           ("arglebargle", FETCHER_UNIVERSE_DATA_TICKER_COLUMN,
                            FETCHER_ALTERNATE_COLUMN_HEADER)])
    def test_fetcher_universe(self, name, data, column_name):
        # Patching fetch_url here rather than using responses because (a) it's
        # easier given the paramaterization, and (b) there are enough tests
        # using responses that the fetch_url code is getting a good workout so
        # we don't have to use it in every test.
        with patch('zipline.sources.requests_csv.PandasRequestsCSV.fetch_url',
                   new=lambda *a, **k: data):
            sim_params = factory.create_simulation_parameters(
                start=pd.Timestamp("2006-01-09", tz='UTC'),
                end=pd.Timestamp("2006-01-11", tz='UTC')
            )

            algocode = """
from pandas import Timestamp
from zipline.api import fetch_csv, record, sid, get_datetime

def initialize(context):
    fetch_csv(
        'https://dl.dropbox.com/u/16705795/dtoc_history.csv',
        date_format='%m/%d/%Y'{token}
    )
    context.expected_sids = {{
        Timestamp('2006-01-09 00:00:00+0000', tz='UTC'):[24, 3766, 5061],
        Timestamp('2006-01-10 00:00:00+0000', tz='UTC'):[24, 3766, 5061],
        Timestamp('2006-01-11 00:00:00+0000', tz='UTC'):[24, 3766, 5061, 14848]
    }}
    context.bar_count = 0

def handle_data(context, data):
    expected = context.expected_sids[get_datetime()]
    actual = data.fetcher_assets
    for stk in expected:
        if stk not in actual:
            raise Exception(
                "{{stk}} is missing on dt={{dt}}".format(
                    stk=stk, dt=get_datetime()))

    record(sid_count=len(actual))
    record(bar_count=context.bar_count)
    context.bar_count += 1
            """
            replacement = ""
            if column_name:
                replacement = ",symbol_column='%s'\n" % column_name
            real_algocode = algocode.format(token=replacement)

            results = self.run_algo(real_algocode, sim_params=sim_params)

            self.assertEqual(len(results), 3)
            self.assertEqual(3, results["sid_count"].iloc[0])
            self.assertEqual(3, results["sid_count"].iloc[1])
            self.assertEqual(4, results["sid_count"].iloc[2])

    def test_fetcher_universe_non_security_return(self):
        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-09", tz='UTC'),
            end=pd.Timestamp("2006-01-10", tz='UTC')
        )

        self.run_algo(
            """
from zipline.api import fetch_csv

def initialize(context):
    fetch_csv(
        'https://fake.urls.com/bad_fetcher_universe_data.csv',
        date_format='%m/%d/%Y'
    )

def handle_data(context, data):
    if len(data.fetcher_assets) > 0:
        raise Exception("Shouldn't be any assets in fetcher_assets!")
            """,
            sim_params=sim_params,
        )

    def test_order_against_data(self):
        with self.assertRaises(UnsupportedOrderParameters):
            self.run_algo("""
from zipline.api import fetch_csv, order, sid

def rename_col(df):
    return df.rename(columns={'New York 15:00': 'price'})

def initialize(context):
    fetch_csv('https://fake.urls.com/palladium_data.csv',
        date_column='Date',
        symbol='palladium',
        post_func=rename_col,
        date_format='%Y-%m-%d'
        )
    context.stock = sid(24)

def handle_data(context, data):
    order('palladium', 100)
            """)

    def test_fetcher_universe_minute(self):
        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-09", tz='UTC'),
            end=pd.Timestamp("2006-01-11", tz='UTC'),
            data_frequency="minute"
        )

        results = self.run_algo(
            """
from pandas import Timestamp
from zipline.api import fetch_csv, record, get_datetime

def initialize(context):
    fetch_csv(
        'https://fake.urls.com/fetcher_universe_data.csv',
        date_format='%m/%d/%Y'
    )
    context.expected_sids = {
        Timestamp('2006-01-09 00:00:00+0000', tz='UTC'):[24, 3766, 5061],
        Timestamp('2006-01-10 00:00:00+0000', tz='UTC'):[24, 3766, 5061],
        Timestamp('2006-01-11 00:00:00+0000', tz='UTC'):[24, 3766, 5061, 14848]
    }
    context.bar_count = 0

def handle_data(context, data):
    expected = context.expected_sids[get_datetime().replace(hour=0, minute=0)]
    actual = data.fetcher_assets
    for stk in expected:
        if stk not in actual:
            raise Exception("{stk} is missing".format(stk=stk))

    record(sid_count=len(actual))
    record(bar_count=context.bar_count)
    context.bar_count += 1
        """, sim_params=sim_params, data_frequency="minute"
        )

        self.assertEqual(3, len(results))
        self.assertEqual(3, results["sid_count"].iloc[0])
        self.assertEqual(3, results["sid_count"].iloc[1])
        self.assertEqual(4, results["sid_count"].iloc[2])
