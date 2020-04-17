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
import os
import logbook
import requests
import warnings

import pandas as pd

log = logbook.Logger(__name__)


def get_benchmark_returns(symbol):
    """
    Get a Series of benchmark returns from IEX associated with `symbol`.
    Default is `SPY`.

    Parameters
    ----------
    symbol : str
        Benchmark symbol for which we're getting the returns.

    The data is provided by IEX (https://iextrading.com/), and we can
    get up to 5 years worth of data.
    """

    iex_api_key = os.environ.get('IEX_API_KEY')
    if iex_api_key is None:
        warnings.warn(
            "Please specify manually a benchmark symbol using one "
            "of the following options: \n"
            "--benchmark-file, --benchmark-symbol, --no-benchmark\n"
            "You can still retrieve market data from IEX "
            "by setting the IEX_API_KEY environment variable.\n"
            "Please note that this feature is expected to be "
            "deprecated in the future"
        )
        raise OSError("Missing environment variable IEX_API_KEY")
    r = requests.get(
        "https://cloud.iexapis.com/stable/stock/{}/chart/5y?"
        "chartCloseOnly=True&token={}".format(symbol, iex_api_key)
    )
    data = r.json()

    df = pd.DataFrame(data)

    df.index = pd.DatetimeIndex(df['date'])
    df = df['close']

    return df.sort_index().tz_localize('UTC').pct_change(1).iloc[1:]


def get_benchmark_returns_from_file(filelike):
    """
    Get a Series of benchmark returns from a file

    Parameters
    ----------
    filelike : str or file-like object
        Path to the benchmark file.
        expected csv file format:
        date,return
        2020-01-02 00:00:00+00:00,0.01
        2020-01-03 00:00:00+00:00,-0.02

    """
    log.info("Reading benchmark returns from {}", filelike)

    df = pd.read_csv(
        filelike,
        index_col=['date'],
        parse_dates=['date'],
    ).tz_localize('utc')

    if 'return' not in df.columns:
        raise ValueError("The column 'return' not found in the benchmark file \n"
                         "Expected benchmark file format :\n"
                         "date, return\n"
                         "2020-01-02 00:00:00+00:00,0.01\n"
                         "2020-01-03 00:00:00+00:00,-0.02\n")

    return df['return'].sort_index()
