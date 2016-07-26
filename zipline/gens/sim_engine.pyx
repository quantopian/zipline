#
# Copyright 2015 Quantopian, Inc.
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

cimport numpy as np
import numpy as np
import pandas as pd
cimport cython
from cpython cimport bool

cdef np.int64_t _nanos_in_minute = 60000000000
NANOS_IN_MINUTE = _nanos_in_minute

cpdef enum:
    BAR = 0
    DAY_START = 1
    DAY_END = 2
    MINUTE_END = 3

cdef class MinuteSimulationClock:
    cdef object trading_days
    cdef bool minute_emission
    cdef np.int64_t[:] market_opens, market_closes
    cdef public dict minutes_by_day, minutes_to_day

    def __init__(self,
                 trading_days,
                 market_opens,
                 market_closes,
                 minute_emission=False):
        self.minute_emission = minute_emission
        self.market_opens = market_opens
        self.market_closes = market_closes
        self.trading_days = trading_days
        self.minutes_by_day = self.calc_minutes_by_day()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray[np.int64_t, ndim=1] market_minutes(self, np.intp_t i):
        cdef np.int64_t[:] market_opens, market_closes

        market_opens = self.market_opens
        market_closes = self.market_closes

        return np.arange(market_opens[i],
                         market_closes[i] + _nanos_in_minute,
                         _nanos_in_minute)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef dict calc_minutes_by_day(self):
        cdef dict minutes_by_day
        cdef int day_idx
        cdef object day

        minutes_by_day = {}
        for day_idx, day in enumerate(self.trading_days):
            minutes_by_day[day] = pd.to_datetime(
                self.market_minutes(day_idx), utc=True, box=True)
        return minutes_by_day

    def __iter__(self):
        minute_emission = self.minute_emission

        for day in self.trading_days:
            yield day, DAY_START

            minutes = self.minutes_by_day[day]

            for minute in minutes:
                yield minute, BAR
                if minute_emission:
                    yield minute, MINUTE_END

            yield minutes[-1], DAY_END



cdef class DailySimulationClock:
    cdef object trading_days

    def __init__(self, trading_days):
        self.trading_days = trading_days

    def __iter__(self):
        for i, day in enumerate(self.trading_days):
            yield day, DAY_START
            yield day, BAR
            yield day, DAY_END
