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
import pandas as pd

from six.moves.urllib_parse import urlencode


def format_yahoo_index_url(symbol, start_date, end_date):
    """
    Format a URL for querying Yahoo Finance for Index data.
    """
    return (
        'http://ichart.finance.yahoo.com/table.csv?' + urlencode({
            's': symbol,
            # start_date month, zero indexed
            'a': start_date.month - 1,
            # start_date day
            'b': start_date.day,
            # start_date year
            'c': start_date.year,
            # end_date month, zero indexed
            'd': end_date.month - 1,
            # end_date day
            'e': end_date.day,
            # end_date year
            'f': end_date.year,
            # daily frequency
            'g': 'd',
        })
    )


def get_benchmark_returns(symbol, start_date, end_date):
    """
    Get a Series of benchmark returns from Yahoo.
    """
    data = pd.read_csv(
        format_yahoo_index_url(symbol, start_date, end_date),
        parse_dates=['Date'],
        index_col='Date',
        usecols=["Open", "Close", "Date"],
    ).sort_index().tz_localize('UTC')

    returns = data["Close"].pct_change()
    # Calculate the returns for the first day using the open of that day since
    # we don't have the close of the previous day.
    first_open, first_close = data.ix[0, ["Open", "Close"]]
    returns.iloc[0] = (first_close - first_open) / first_open

    return returns
