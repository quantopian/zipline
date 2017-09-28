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
import math
import pandas as pd

import pandas_datareader.data as pd_reader


def get_benchmark_returns(symbol, first_date, last_date):
    """
    Get a Series of benchmark returns from Google associated with `symbol`.
    Default is `SPY`.

    Parameters
    ----------
    symbol : str
        Benchmark symbol for which we're getting the returns.
    first_date : pd.Timestamp
        First date for which we want to get data.
    last_date : pd.Timestamp
        Last date for which we want to get data.

    We're also now limited to 251 days worth of data per request. If we make a
    request for data that extends past 251 trading days, we'll figure out how
    many more requests we need to make and then make those requests in sequence.

    first_date is **not** included because we need the close from day N - 1 to
    compute the returns for day N.
    """
    data = pd_reader.DataReader(
        symbol,
        'google',
        first_date,
        last_date
    )

    given_first = data.index[0].tz_localize('UTC')
    _, num_requests = math.modf(
        (given_first - first_date).days / max_days
    )

    new_data = pd.Series()

    if given_first != first_date:
        for n in num_requests:
            tmp_data = pd_reader.DataReader(
                symbol,
                'google',
                first_date,
                given_first,
            )
            new_data.append(tmp_data['Close'])
            given_first = new_data.index[0].tz_localize('UTC')

    data = data['Close']
    data = new_data.append(data)
    data.fillna(method='ffill', inplace=True)

    return data.sort_index().tz_localize('UTC').pct_change().iloc[1:]
