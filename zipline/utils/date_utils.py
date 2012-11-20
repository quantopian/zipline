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


import pytz
import iso8601
from datetime import datetime, timedelta


# iso8061 utility
# ---------------------
def parse_iso8061(date_string):
    dt = iso8601.parse_date(date_string)
    dt = dt.replace(tzinfo=pytz.utc)
    return dt


# Epoch utilities
# ---------------------
UNIX_EPOCH = datetime(1970, 1, 1, 0, 0, tzinfo=pytz.utc)


def EPOCH(utc_datetime):
    """
    The key is to ensure all the dates you are using are in the utc timezone
    before you start converting. See http://pytz.sourceforge.net/ to learn how
    to do that properly. By normalizing to utc, you eliminate the ambiguity of
    daylight savings transitions. Then you can safely use timedelta to
    calculate distance from the unix epoch, and then convert to seconds or
    milliseconds.

    Note that the resulting unix timestamp is itself in the UTC timezone.
    If you wish to see the timestamp in a localized timezone, you will need
    to make another conversion.

    Also note that this will only work for dates after 1970.
    """
    assert isinstance(utc_datetime, datetime)
    # utc only please
    assert utc_datetime.tzinfo == pytz.utc

    # how long since the epoch?
    delta = utc_datetime - UNIX_EPOCH
    seconds = delta.total_seconds()
    ms = seconds * 1000
    return int(ms)


def UN_EPOCH(ms_since_epoch):
    delta = timedelta(milliseconds=ms_since_epoch)
    dt = UNIX_EPOCH + delta
    return dt


def iso8061_to_epoch(datestring):
    dt = parse_iso8061(datestring)
    return EPOCH(dt)


def epoch_now():
    dt = utcnow()
    return EPOCH(dt)


# UTC Datetime Subclasses
# -----------------------
def utcnow():
    return datetime.utcnow().replace(tzinfo=pytz.utc)


def days_since_epoch(ms_since_epoch):
    dt = UN_EPOCH(ms_since_epoch)
    delta = dt - UNIX_EPOCH
    return delta.days


def epoch_from_days(days_since_epoch):
    delta = timedelta(days=days_since_epoch)
    dt = UNIX_EPOCH + delta
    ms = EPOCH(dt)
    return ms


def tuple_to_date(date_tuple):
    year, month, day, hour, minute, second, micros = date_tuple
    dt = datetime(year, month, day, hour, minute, second)
    dt = dt.replace(microsecond=micros, tzinfo=pytz.utc)
    return dt
