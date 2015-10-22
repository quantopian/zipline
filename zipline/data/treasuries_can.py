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
import six
from toolz import curry
from toolz.curried.operator import add as prepend

COLUMN_NAMES = {
    "V39063": '1month',
    "V39065": '3month',
    "V39066": '6month',
    "V39067": '1year',
    "V39051": '2year',
    "V39052": '3year',
    "V39053": '5year',
    "V39054": '7year',
    "V39055": '10year',
    # Bank of Canada refers to this as 'Long' Rate, approximately 30 years.
    "V39056": '30year',
}
BILL_IDS = ['V39063', 'V39065', 'V39066', 'V39067']
BOND_IDS = ['V39051', 'V39052', 'V39053', 'V39054', 'V39055', 'V39056']


@curry
def _format_url(instrument_type,
                instrument_ids,
                start_date,
                end_date,
                earliest_allowed_date):
    """
    Format a URL for loading data from Bank of Canada.
    """
    return (
        "http://www.bankofcanada.ca/stats/results/csv"
        "?lP=lookup_{instrument_type}_yields.php"
        "&sR={restrict}"
        "&se={instrument_ids}"
        "&dF={start}"
        "&dT={end}".format(
            instrument_type=instrument_type,
            instrument_ids='-'.join(map(prepend("L_"), instrument_ids)),
            restrict=earliest_allowed_date.strftime("%Y-%m-%d"),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
    )


format_bill_url = _format_url('tbill', BILL_IDS)
format_bond_url = _format_url('bond', BOND_IDS)


def load_frame(url, skiprows):
    """
    Load a DataFrame of data from a Bank of Canada site.
    """
    return pd.read_csv(
        url,
        skiprows=skiprows,
        skipinitialspace=True,
        na_values=["Bank holiday", "Not available"],
        parse_dates=["Date"],
        index_col="Date",
    ).dropna(how='all') \
     .tz_localize('UTC') \
     .rename(columns=COLUMN_NAMES)


def check_known_inconsistencies(bill_data, bond_data):
    """
    There are a couple quirks in the data provided by Bank of Canada.
    Check that no new quirks have been introduced in the latest download.
    """
    inconsistent_dates = bill_data.index.sym_diff(bond_data.index)
    known_inconsistencies = [
        # bill_data has an entry for 2010-02-15, which bond_data doesn't.
        # bond_data has an entry for 2006-09-04, which bill_data doesn't.
        # Both of these dates are bank holidays (Flag Day and Labor Day,
        # respectively).
        pd.Timestamp('2006-09-04', tz='UTC'),
        pd.Timestamp('2010-02-15', tz='UTC'),
        # 2013-07-25 comes back as "Not available" from the bills endpoint.
        # This date doesn't seem to be a bank holiday, but the previous
        # calendar implementation dropped this entry, so we drop it as well.
        # If someone cares deeply about the integrity of the Canadian trading
        # calendar, they may want to consider forward-filling here rather than
        # dropping the row.
        pd.Timestamp('2013-07-25', tz='UTC'),
    ]
    unexpected_inconsistences = inconsistent_dates.drop(known_inconsistencies)
    if len(unexpected_inconsistences):
        in_bills = bill_data.index.difference(bond_data.index).difference(
            known_inconsistencies
        )
        in_bonds = bond_data.index.difference(bill_data.index).difference(
            known_inconsistencies
        )
        raise ValueError(
            "Inconsistent dates for Canadian treasury bills vs bonds. \n"
            "Dates with bills but not bonds: {in_bills}.\n"
            "Dates with bonds but not bills: {in_bonds}.".format(
                in_bills=in_bills,
                in_bonds=in_bonds,
            )
        )


def earliest_possible_date():
    """
    The earliest date for which we can load data from this module.
    """
    today = pd.Timestamp('now', tz='UTC').normalize()
    # Bank of Canada only has the last 10 years of data at any given time.
    return today.replace(year=today.year - 10)


def get_treasury_data(start_date, end_date):
    bill_data = load_frame(
        format_bill_url(start_date, end_date, start_date),
        # We skip fewer rows here because we query for fewer bill fields,
        # which makes the header smaller.
        skiprows=18,
    )
    bond_data = load_frame(
        format_bond_url(start_date, end_date, start_date),
        skiprows=22,
    )
    check_known_inconsistencies(bill_data, bond_data)

    # dropna('any') removes the rows for which we only had data for one of
    # bills/bonds.
    out = pd.concat([bond_data, bill_data], axis=1).dropna(how='any')
    assert set(out.columns) == set(six.itervalues(COLUMN_NAMES))

    # Multiply by 0.01 to convert from percentages to expected output format.
    return out * 0.01
