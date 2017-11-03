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
from operator import itemgetter
import re

import numpy as np
import pandas as pd


get_unit_and_periods = itemgetter('unit', 'periods')


def parse_treasury_csv_column(column):
    """
    Parse a treasury CSV column into a more human-readable format.

    Columns start with 'RIFLGFC', followed by Y or M (year or month), followed
    by a two-digit number signifying number of years/months, followed by _N.B.
    We only care about the middle two entries, which we turn into a string like
    3month or 30year.
    """
    column_re = re.compile(
        r"^(?P<prefix>RIFLGFC)"
        "(?P<unit>[YM])"
        "(?P<periods>[0-9]{2})"
        "(?P<suffix>_N.B)$"
    )

    match = column_re.match(column)
    if match is None:
        raise ValueError("Couldn't parse CSV column %r." % column)
    unit, periods = get_unit_and_periods(match.groupdict())

    # Roundtrip through int to coerce '06' into '6'.
    return str(int(periods)) + ('year' if unit == 'Y' else 'month')


def earliest_possible_date():
    """
    The earliest date for which we can load data from this module.
    """
    # The US Treasury actually has data going back further than this, but it's
    # pretty rare to find pricing data going back that far, and there's no
    # reason to make people download benchmarks back to 1950 that they'll never
    # be able to use.
    return pd.Timestamp('1980', tz='UTC')


def get_treasury_data(start_date, end_date):
    return pd.read_csv(
        "https://www.federalreserve.gov/datadownload/Output.aspx"
        "?rel=H15"
        "&series=bf17364827e38702b42a58cf8eaa3f78"
        "&lastObs="
        "&from="  # An unbounded query is ~2x faster than specifying dates.
        "&to="
        "&filetype=csv"
        "&label=include"
        "&layout=seriescolumn"
        "&type=package",
        skiprows=5,  # First 5 rows are useless headers.
        parse_dates=['Time Period'],
        na_values=['ND'],  # Presumably this stands for "No Data".
        index_col=0,
    ).loc[
        start_date:end_date
    ].dropna(
        how='all'
    ).rename(
        columns=parse_treasury_csv_column
    ).tz_localize('UTC') * 0.01  # Convert from 2.57% to 0.0257.


def dataconverter(s):
    try:
        return float(s) / 100
    except:
        return np.nan


def get_daily_10yr_treasury_data():
    """Download daily 10 year treasury rates from the Federal Reserve and
    return a pandas.Series."""
    url = "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15" \
          "&series=bcb44e57fb57efbe90002369321bfb3f&lastObs=&from=&to=" \
          "&filetype=csv&label=include&layout=seriescolumn"
    return pd.read_csv(url, header=5, index_col=0, names=['DATE', 'BC_10YEAR'],
                       parse_dates=True, converters={1: dataconverter},
                       squeeze=True)
