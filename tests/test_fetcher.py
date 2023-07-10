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
from parameterized import parameterized

import pandas as pd
import numpy as np
from unittest import mock
from zipline.errors import UnsupportedOrderParameters
from zipline.sources.requests_csv import mask_requests_args
from zipline.utils import factory
from zipline.testing import FetcherDataPortal
from zipline.testing.fixtures import (
    WithResponses,
    WithMakeAlgo,
    ZiplineTestCase,
)

from tests.resources.fetcher_inputs.fetcher_test_data import (
    AAPL_CSV_DATA,
    AAPL_IBM_CSV_DATA,
    AAPL_MINUTE_CSV_DATA,
    CPIAUCSL_DATA,
    FETCHER_ALTERNATE_COLUMN_HEADER,
    FETCHER_UNIVERSE_DATA,
    FETCHER_UNIVERSE_DATA_TICKER_COLUMN,
    MULTI_SIGNAL_CSV_DATA,
    NON_ASSET_FETCHER_UNIVERSE_DATA,
    PALLADIUM_DATA,
    NFLX_DATA,
)
import pytest


# XXX: The algorithms in this suite do way more work than they should have to.
class FetcherTestCase(WithResponses, WithMakeAlgo, ZiplineTestCase):
    START_DATE = pd.Timestamp("2006-01-03")
    END_DATE = pd.Timestamp("2006-12-29")

    SIM_PARAMS_DATA_FREQUENCY = "daily"
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        start_date = pd.Timestamp("2006-01-01")
        end_date = pd.Timestamp("2007-01-01")
        return pd.DataFrame.from_dict(
            {
                24: {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol": "AAPL",
                    "exchange": "nasdaq",
                },
                3766: {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol": "IBM",
                    "exchange": "nasdaq",
                },
                5061: {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol": "MSFT",
                    "exchange": "nasdaq",
                },
                14848: {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol": "YHOO",
                    "exchange": "nasdaq",
                },
                25317: {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol": "DELL",
                    "exchange": "nasdaq",
                },
                13: {
                    "start_date": start_date,
                    "end_date": pd.Timestamp("2010-01-01"),
                    "symbol": "NFLX",
                    "exchange": "nasdaq",
                },
                9999999: {
                    "start_date": start_date,
                    "end_date": end_date,
                    "symbol": "AAPL",
                    "exchange": "non_us_exchange",
                },
            },
            orient="index",
        )

    @classmethod
    def make_exchanges_info(cls, *args, **kwargs):
        return pd.DataFrame.from_records(
            [
                {"exchange": "nasdaq", "country_code": "US"},
                {"exchange": "non_us_exchange", "country_code": "CA"},
            ]
        )

    def run_algo(self, code, sim_params=None):
        if sim_params is None:
            sim_params = self.sim_params

        test_algo = self.make_algo(
            script=code,
            sim_params=sim_params,
            data_portal=FetcherDataPortal(
                self.asset_finder,
                self.trading_calendar,
            ),
        )
        results = test_algo.run()

        return results

    def test_minutely_fetcher(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/aapl_minute_csv_data.csv",
            body=AAPL_MINUTE_CSV_DATA,
            content_type="text/csv",
        )

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-03"),
            end=pd.Timestamp("2006-01-10"),
            emission_rate="minute",
            data_frequency="minute",
        )

        test_algo = self.make_algo(
            script="""
from zipline.api import fetch_csv, record, sid

def initialize(context):
    fetch_csv('https://fake.urls.com/aapl_minute_csv_data.csv')

def handle_data(context, data):
    record(aapl_signal=data.current(sid(24), "signal"))
""",
            sim_params=sim_params,
        )

        # manually getting generator because we need the minutely emission
        # packets here. TradingAlgorithm.run() only returns daily packets.
        signal = [
            result["minute_perf"]["recorded_vars"]["aapl_signal"]
            for result in test_algo.get_generator()
            if "minute_perf" in result
        ]

        assert 6 * 390 == len(signal)

        # csv data is:
        # symbol,date,signal
        # aapl,1/4/06 5:31AM, 1
        # aapl,1/4/06 11:30AM, 2
        # aapl,1/5/06 5:31AM, 1
        # aapl,1/5/06 11:30AM, 3
        # aapl,1/9/06 5:31AM, 1
        # aapl,1/9/06 11:30AM, 4 for dates 1/3 to 1/10

        # 2 signals per day, only last signal is taken. So we expect
        # 390 bars of signal NaN on 1/3
        # 390 bars of signal 2 on 1/4
        # 390 bars of signal 3 on 1/5
        # 390 bars of signal 3 on 1/6 (forward filled)
        # 390 bars of signal 4 on 1/9
        # 390 bars of signal 4 on 1/9 (forward filled)

        np.testing.assert_array_equal([np.NaN] * 390, signal[0:390])
        np.testing.assert_array_equal([2] * 390, signal[390:780])
        np.testing.assert_array_equal([3] * 780, signal[780:1560])
        np.testing.assert_array_equal([4] * 780, signal[1560:])

    def test_fetch_csv_with_multi_symbols(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/multi_signal_csv_data.csv",
            body=MULTI_SIGNAL_CSV_DATA,
            content_type="text/csv",
        )

        results = self.run_algo(
            """
from zipline.api import fetch_csv, record, sid

def initialize(context):
    fetch_csv('https://fake.urls.com/multi_signal_csv_data.csv')
    context.stocks = [sid(3766), sid(25317)]

def handle_data(context, data):
    record(ibm_signal=data.current(sid(3766), "signal"))
    record(dell_signal=data.current(sid(25317), "signal"))
    """
        )

        assert 5 == results["ibm_signal"].iloc[-1]
        assert 5 == results["dell_signal"].iloc[-1]

    def test_fetch_csv_with_pure_signal_file(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/cpiaucsl_data.csv",
            body=CPIAUCSL_DATA,
            content_type="text/csv",
        )

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

    cur_cpi = data.current("urban", "cpi")
    record(cpi=cur_cpi)
            """
        )

        assert results["cpi"][-1] == 203.1

    def test_algo_fetch_csv(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/aapl_csv_data.csv",
            body=AAPL_CSV_DATA,
            content_type="text/csv",
        )

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
        signal=data.current(sid(24), "signal"),
        scaled=data.current(sid(24), "scaled"),
        price=data.current(sid(24), "price"))
        """
        )

        assert 5 == results["signal"][-1]
        assert 50 == results["scaled"][-1]
        assert 24 == results["price"][-1]  # fake value

    def test_algo_fetch_csv_with_extra_symbols(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/aapl_ibm_csv_data.csv",
            body=AAPL_IBM_CSV_DATA,
            content_type="text/csv",
        )

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
    record(
        signal=data.current(sid(24),"signal"),
        scaled=data.current(sid(24), "scaled"),
        price=data.current(sid(24), "price"))
            """
        )

        assert 5 == results["signal"][-1]
        assert 50 == results["scaled"][-1]
        assert 24 == results["price"][-1]  # fake value

    @parameterized.expand(
        [
            ("unspecified", ""),
            ("none", "usecols=None"),
            ("without date", "usecols=['Value']"),
            ("with date", "usecols=('Value', 'Date')"),
        ]
    )
    def test_usecols(self, testname, usecols):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/cpiaucsl_data.csv",
            body=CPIAUCSL_DATA,
            content_type="text/csv",
        )

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
    data.current("urban", "cpi")
        """
        results = self.run_algo(code.format(usecols=usecols))
        # 251 trading days in 2006
        assert len(results) == 251

    def test_sources_merge_custom_ticker(self):
        requests_kwargs = {}

        def capture_kwargs(zelf, url, **kwargs):
            requests_kwargs.update(mask_requests_args(url, kwargs).requests_kwargs)
            return PALLADIUM_DATA

        # Patching fetch_url instead of using responses in this test so that we
        # can intercept the requests keyword arguments and confirm that they're
        # correct.
        with mock.patch(
            "zipline.sources.requests_csv.PandasRequestsCSV.fetch_url",
            new=capture_kwargs,
        ):
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
    record(palladium=data.current("palladium", "price"))
    record(aapl=data.current(context.stock, "price"))
        """
            )

            np.testing.assert_array_equal([24] * 251, results["aapl"])
            assert 337 == pd.to_numeric(results["palladium"]).iloc[-1]

            expected = {
                "allow_redirects": False,
                "stream": True,
                "timeout": 30.0,
            }

            assert expected == requests_kwargs

    @parameterized.expand(
        [
            ("symbol", FETCHER_UNIVERSE_DATA, None),
            (
                "arglebargle",
                FETCHER_UNIVERSE_DATA_TICKER_COLUMN,
                FETCHER_ALTERNATE_COLUMN_HEADER,
            ),
        ]
    )
    def test_fetcher_universe(self, name, data, column_name):
        # Patching fetch_url here rather than using responses because (a) it's
        # easier given the parameterization, and (b) there are enough tests
        # using responses that the fetch_url code is getting a good workout so
        # we don't have to use it in every test.
        with mock.patch(
            "zipline.sources.requests_csv.PandasRequestsCSV.fetch_url",
            new=lambda *a, **k: data,
        ):
            sim_params = factory.create_simulation_parameters(
                start=pd.Timestamp("2006-01-09"),
                end=pd.Timestamp("2006-01-11"),
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
    expected = context.expected_sids[get_datetime().normalize()]
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

            assert len(results) == 3
            assert 3 == results["sid_count"].iloc[0]
            assert 3 == results["sid_count"].iloc[1]
            assert 4 == results["sid_count"].iloc[2]

    def test_fetcher_universe_non_security_return(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/bad_fetcher_universe_data.csv",
            body=NON_ASSET_FETCHER_UNIVERSE_DATA,
            content_type="text/csv",
        )

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-09"),
            end=pd.Timestamp("2006-01-10"),
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
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/palladium_data.csv",
            body=PALLADIUM_DATA,
            content_type="text/csv",
        )

        with pytest.raises(UnsupportedOrderParameters):
            self.run_algo(
                """
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
            """
            )

    def test_fetcher_universe_minute(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/fetcher_universe_data.csv",
            body=FETCHER_UNIVERSE_DATA,
            content_type="text/csv",
        )

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2006-01-09"),
            end=pd.Timestamp("2006-01-11"),
            data_frequency="minute",
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
        """,
            sim_params=sim_params,
        )

        assert 3 == len(results)
        assert 3 == results["sid_count"].iloc[0]
        assert 3 == results["sid_count"].iloc[1]
        assert 4 == results["sid_count"].iloc[2]

    def test_fetcher_in_before_trading_start(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/fetcher_nflx_data.csv",
            body=NFLX_DATA,
            content_type="text/csv",
        )

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2013-06-13"),
            end=pd.Timestamp("2013-11-15"),
            data_frequency="minute",
        )

        results = self.run_algo(
            """
from zipline.api import fetch_csv, record, symbol

def initialize(context):
    fetch_csv('https://fake.urls.com/fetcher_nflx_data.csv',
               date_column = 'Settlement Date',
               date_format = '%m/%d/%y')
    context.stock = symbol('NFLX')

def before_trading_start(context, data):
    record(Short_Interest = data.current(context.stock, 'dtc'))
    """,
            sim_params=sim_params,
        )

        values = results["Short_Interest"]
        np.testing.assert_array_equal(values[0:33], np.full(33, np.nan))
        np.testing.assert_array_almost_equal(values[33:44], [1.690317] * 11)
        np.testing.assert_array_almost_equal(values[44:55], [2.811858] * 11)
        np.testing.assert_array_almost_equal(values[55:64], [2.50233] * 9)
        np.testing.assert_array_almost_equal(values[64:75], [2.550829] * 11)
        np.testing.assert_array_almost_equal(values[75:], [2.64484] * 35)

    def test_fetcher_bad_data(self):
        self.responses.add(
            self.responses.GET,
            "https://fake.urls.com/fetcher_nflx_data.csv",
            body=NFLX_DATA,
            content_type="text/csv",
        )

        sim_params = factory.create_simulation_parameters(
            start=pd.Timestamp("2013-06-12"),
            end=pd.Timestamp("2013-06-14"),
            data_frequency="minute",
        )

        results = self.run_algo(
            """
from zipline.api import fetch_csv, symbol
import numpy as np
def initialize(context):
    fetch_csv('https://fake.urls.com/fetcher_nflx_data.csv',
               date_column = 'Settlement Date',
               date_format = '%m/%d/%y')
    context.nflx = symbol('NFLX')
    context.aapl = symbol('AAPL', country_code='US')
def handle_data(context, data):
    assert np.isnan(data.current(context.nflx, 'invalid_column'))
    assert np.isnan(data.current(context.aapl, 'invalid_column'))
    assert np.isnan(data.current(context.aapl, 'dtc'))
    """,
            sim_params=sim_params,
        )

        assert 3 == len(results)
