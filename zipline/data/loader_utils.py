#
# Copyright 2012 Quantopian, Inc.
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


"""
Various utilites used by different date loaders.

Could stand to be broken up more into components.
e.g. the mapping utilities.

"""

import datetime

import pytz

from collections import namedtuple

from functools import partial


def get_utc_from_exchange_time(naive):
    local = pytz.timezone('US/Eastern')
    local_dt = naive.replace(tzinfo=local)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt


def get_exchange_time_from_utc(utc_dt):
    """
    Takes in result from exchange time.
    """
    dt = utc_dt.replace(tzinfo=pytz.utc)
    local = pytz.timezone('US/Eastern')
    dt = dt.astimezone(local)

    return dt


def guarded_conversion(conversion, str_val):
    """
    Returns the result of applying the @conversion to @str_val
    """
    if str_val in (None, ""):
        return None
    return conversion(str_val)


def safe_int(str_val):
    """
    casts the @str_val to a float to handle the occassional
    decimal point in int fields from data providers.
    """
    f = float(str_val)
    i = int(f)
    return i


def date_conversion(date_str, date_pattern='%m/%d/%Y', to_utc=True):
    """
    Convert date strings from TickData (or other source) into epoch values.

    Specify to_utc=False if the input date is already UTC (or is naive).
    """
    dt = datetime.datetime.strptime(date_str, date_pattern)
    if to_utc:
        dt = get_utc_from_exchange_time(dt)
    else:
        dt = dt.replace(tzinfo=pytz.utc)
    return dt


# Mapping is a structure for how want to convert the source data into
# the form we insert into the database.
# - conversion, a function used to convert source input to our target value
# - source, the key(s) in the original source to pass to the conversion
#           method
#           If a single string, then it's a direct lookup into the
#           source row by that key
#           If an iterator, pass the source to as a list of keys,
#           in order, to the conversion function.
#           If empty, then the conversion method provides a 'default' value.
Mapping = namedtuple('Mapping', ['conversion', 'source'])


def apply_mapping(mapping, row):
    """
    Returns the value of a @mapping for a given @row.

    i.e. the @mapping.source values are extracted from @row and fed
    into the @mapping.conversion method.
    """
    if isinstance(mapping.source, str):
        # Do a 'direct' conversion of one key from the source row.
        return guarded_conversion(mapping.conversion, row[mapping.source])
    if mapping.source is None:
        # For hardcoded values.
        # conversion method will return a constant value
        return mapping.conversion()
    else:
        # Assume we are using multiple source values.
        # Feed the source values in order prescribed by mapping.source
        # to mapping.conversion.
        return mapping.conversion(*[row[source] for source in mapping.source])


def _row_cb(mapping, row):
    """
    Returns the dict created from our @mapping of the source @row.

    Not intended to be used directly, but rather to be the base of another
    function that supplies the mapping value.
    """
    return {
        target: apply_mapping(mapping, row)
        for target, mapping
        in mapping.iteritems()
    }


def make_row_cb(mapping):
    """
    Returns a func that can be applied to a dict that returns the
    application of the @mapping, which results in a dict.
    """
    return partial(_row_cb, mapping)


def source_to_records(mappings,
                      source,
                      source_wrapper=None,
                      records_wrapper=None):
    if source_wrapper:
        source = source_wrapper(source)

    callback = make_row_cb(mappings)

    records = (callback(row) for row in source)

    if records_wrapper:
        records = records_wrapper(records)

    return records
