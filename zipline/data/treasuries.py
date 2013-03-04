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


import requests

from StringIO import StringIO
from xml.dom.minidom import parse

from loader_utils import (
    guarded_conversion,
    safe_int,
    Mapping,
    date_conversion,
    source_to_records
)


def get_treasury_date(dstring):
    return date_conversion(dstring.split("T")[0], date_pattern='%Y-%m-%d',
                           to_utc=False)


def get_treasury_rate(string_val):
    val = guarded_conversion(float, string_val)
    if val is not None:
        val = round(val / 100.0, 4)
    return val

_CURVE_MAPPINGS = {
    'tid': (safe_int, "Id"),
    'date': (get_treasury_date, "NEW_DATE"),
    '1month': (get_treasury_rate, "BC_1MONTH"),
    '3month': (get_treasury_rate, "BC_3MONTH"),
    '6month': (get_treasury_rate, "BC_6MONTH"),
    '1year': (get_treasury_rate, "BC_1YEAR"),
    '2year': (get_treasury_rate, "BC_2YEAR"),
    '3year': (get_treasury_rate, "BC_3YEAR"),
    '5year': (get_treasury_rate, "BC_5YEAR"),
    '7year': (get_treasury_rate, "BC_7YEAR"),
    '10year': (get_treasury_rate, "BC_10YEAR"),
    '20year': (get_treasury_rate, "BC_20YEAR"),
    '30year': (get_treasury_rate, "BC_30YEAR"),
}


def treasury_mappings():
    return {key: Mapping(*value)
            for key, value
            in _CURVE_MAPPINGS.iteritems()}


def get_treasury_source():
    url = """\
http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData\
"""
    res = requests.get(url)

    content = StringIO(res.content)
    dom = parse(content)

    entries = dom.getElementsByTagName("entry")

    for entry in entries:
        properties = entry.getElementsByTagName("m:properties")
        datum = {node.nodeName.replace('d:', ''):
                 node.childNodes[0].nodeValue
                 if len(node.childNodes)
                 else None
                 for node in properties[0].childNodes
                 if node.nodeType == dom.ELEMENT_NODE}
        yield datum


def get_treasury_data():
    mappings = treasury_mappings()
    source = get_treasury_source()
    return source_to_records(mappings, source)
