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

import pandas_datareader.data as web

def get_benchmark_returns(symbol, start_date, end_date):
    """
    Get a Series of benchmark returns from Google finance.

    Returns a Series with returns from (start_date, end_date].

    start_date is **not** included because we need the close from day N - 1 to
    compute the returns for day N.
    """
    df = web.DataReader(symbol, 'google', start_date, end_date)
    return df["Close"].sort_index().tz_localize('UTC').pct_change(1).iloc[1:]
