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
import re

import requests

from collections import OrderedDict
import xml.etree.ElementTree as ET

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


class iter_to_stream(object):
    """
    Exposes an iterable as an i/o stream
    """
    def __init__(self, iterable):
        self.buffered = ""
        self.iter = iter(iterable)

    def read(self, size):
        result = ""
        while size > 0:
            data = self.buffered or next(self.iter, None)
            self.buffered = ""
            if data is None:
                break
            size -= len(data)
            if size < 0:
                data, self.buffered = data[:size], data[size:]
            result += data
        return result


def get_localname(element):
    qtag = ET.QName(element.tag).text
    return re.match("(\{.*\})(.*)", qtag).group(2)


def get_treasury_source():
    url = """\
http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData\
"""
    res = requests.get(url, stream=True)
    stream = iter_to_stream(res.iter_lines())

    elements = ET.iterparse(stream, ('end', 'start-ns', 'end-ns'))

    namespaces = OrderedDict()
    properties_xpath = ['']

    def updated_namespaces():
        if '' in namespaces and 'm' in namespaces:
            properties_xpath[0] = "{%s}content/{%s}properties" % (
                namespaces[''], namespaces['m']
            )
        else:
            properties_xpath[0] = ''

    for event, element in elements:
        if event == 'end':
            tag = get_localname(element)
            if tag == "entry":
                properties = element.find(properties_xpath[0])
                datum = {get_localname(node): node.text
                         for node in properties.getchildren()
                         if ET.iselement(node)}
                # clear the element after we've dealt with it:
                element.clear()
                yield datum

        elif event == 'start-ns':
            namespaces[element[0]] = element[1]
            updated_namespaces()

        elif event == 'end-ns':
            namespaces.popitem()
            updated_namespaces()


def get_treasury_data():
    mappings = treasury_mappings()
    source = get_treasury_source()
    return source_to_records(mappings, source)
