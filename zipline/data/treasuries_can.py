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

import datetime
import requests

from . loader_utils import (
    source_to_records
)

from zipline.data.treasuries import (
    treasury_mappings, get_treasury_date, get_treasury_rate
)


_CURVE_MAPPINGS = {
    'date': (get_treasury_date, "Date"),
    '1month': (get_treasury_rate, "V39063"),
    '3month': (get_treasury_rate, "V39065"),
    '6month': (get_treasury_rate, "V39066"),
    '1year': (get_treasury_rate, "V39067"),
    '2year': (get_treasury_rate, "V39051"),
    '3year': (get_treasury_rate, "V39052"),
    '5year': (get_treasury_rate, "V39053"),
    '7year': (get_treasury_rate, "V39054"),
    '10year': (get_treasury_rate, "V39055"),
    # Bank of Canada refers to this as 'Long' Rate, approximately 30 years.
    '30year': (get_treasury_rate, "V39056"),
}

BILLS = ['V39063', 'V39065', 'V39066', 'V39067']
BONDS = ['V39051', 'V39052', 'V39053', 'V39054', 'V39055', 'V39056']


def get_treasury_source(start_date=None, end_date=None):

    today = datetime.date.today()
    # Bank of Canada only has 10 years of data and has this in the URL.
    restriction = datetime.date(today.year - 10, today.month, today.day)

    if not end_date:
        end_date = today

    if not start_date:
        start_date = restriction

    bill_url = (
        "http://www.bankofcanada.ca/stats/results/csv?"
        "lP=lookup_tbill_yields.php&sR={restrict}&se="
        "L_V39063-L_V39065-L_V39066-L_V39067&dF={start}&dT={end}"
        .format(restrict=restriction.strftime("%Y-%m-%d"),
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                )
    )

    bond_url = (
        "http://www.bankofcanada.ca/stats/results/csv?"
        "lP=lookup_bond_yields.php&sR={restrict}&se="
        "L_V39051-L_V39052-L_V39053-L_V39054-L_V39055-L_V39056"
        "&dF={start}&dT={end}"
        .format(restrict=restriction.strftime("%Y-%m-%d"),
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d")
                )
    )

    res_bill = requests.get(bill_url, stream=True)
    res_bond = requests.get(bond_url, stream=True)
    bill_iter = res_bill.iter_lines()
    bond_iter = res_bond.iter_lines()

    bill_row = ""
    while ",".join(BILLS) not in bill_row:
        bill_row = bill_iter.next()
        if 'Daily series:' in bill_row:
            bill_end_date = datetime.datetime.strptime(
                bill_row.split(' - ')[1].strip(),
                "%Y-%m-%d").date()
    bill_header = bill_row.split(",")

    bond_row = ""
    while ",".join(BONDS) not in bond_row:
        bond_row = bond_iter.next()
        if 'Daily series:' in bond_row:
            bond_end_date = datetime.datetime.strptime(
                bond_row.split(' - ')[1].strip(),
                "%Y-%m-%d").date()
    bond_header = bond_row.split(",")

    # Line up the two dates
    if bill_end_date > bond_end_date:
        bill_iter.next()
    elif bond_end_date > bill_end_date:
        bond_iter.next()

    for bill_row in bill_iter:
        bond_row = bond_iter.next()
        bill_dict = dict(zip(bill_header, bill_row.split(",")))
        bond_dict = dict(zip(bond_header, bond_row.split(",")))
        if ' Bank holiday' in bond_row.split(",") + bill_row.split(","):
            continue
        if ' Not available' in bond_row.split(",") + bill_row.split(","):
            continue

        bill_dict.update(bond_dict)
        yield bill_dict


def get_treasury_data():
    mappings = treasury_mappings(_CURVE_MAPPINGS)
    source = get_treasury_source()
    return source_to_records(mappings, source)
