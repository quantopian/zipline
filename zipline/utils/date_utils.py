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
from datetime import datetime, timedelta


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


# quarter utilities
# ---------------------
def get_quarter(dt):
    """
    convert the given datetime to an integer representing
    the number of calendar quarters since 0.
    """
    return (dt.year - 1) * 4 + (dt.month - 1) / 3 + 1


def dates_of_quarter(quarter_num):
    quarter_num -= 1
    year = quarter_num / 4 + 1
    quarter = quarter_num % 4 + 1

    if quarter == 1:
        start = datetime(year, 1, 1, 0, 0, tzinfo=pytz.utc)
        end = datetime(year, 3, 31, 23, 59, tzinfo=pytz.utc)
        return start, end

    elif quarter == 2:
        start = datetime(year, 4, 1, 0, 0, tzinfo=pytz.utc)
        end = datetime(year, 6, 30, 23, 59, tzinfo=pytz.utc)
        return start, end

    elif quarter == 3:
        start = datetime(year, 7, 1, 0, 0, tzinfo=pytz.utc)
        end = datetime(year, 9, 30, 23, 59, tzinfo=pytz.utc)
        return start, end

    elif quarter == 4:
        start = datetime(year, 10, 1, 0, 0, tzinfo=pytz.utc)
        end = datetime(year, 12, 31, 23, 59, tzinfo=pytz.utc)
        return start, end
