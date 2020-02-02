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
import requests
import warnings

import pandas as pd


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
        raise ValueError(
            "Please set your IEX_API_KEY environment variable and retry."
            "\nPlease note that this feature will be deprecated"
        )
    r = requests.get(
        "https://cloud.iexapis.com/stable/stock/{}/chart/5y?"
        "chartCloseOnly=True&token={}".format(symbol, iex_api_key)
    )
    data = r.json()

    df = pd.DataFrame(data)

    df.index = pd.DatetimeIndex(df['date'])
    df = df['close']

    return df.sort_index().tz_localize('UTC').pct_change(1).iloc[1:]


def get_benchmark_returns_from_file(file_path):
    """
    Get a Series of benchmark returns from a file

    Parameters
    ----------
    file_path : str
        Path to the benchmark file.

    """
    try:
        df = pd.read_csv(file_path)

    except OSError:
        warnings.warn("Could not open the file %s" % file_path)
        return None

    df.index = pd.DatetimeIndex(df['date'])
    df = df['close']

    return df.sort_index().tz_localize('UTC').pct_change(1).iloc[1:]
