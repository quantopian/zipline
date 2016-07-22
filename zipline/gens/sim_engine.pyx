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
    SESSION_START = 1
    SESSION_END = 2
    MINUTE_END = 3
    BEFORE_TRADING_START_BAR = 4

cdef class MinuteSimulationClock:
    cdef object sessions
    cdef bool minute_emission
    cdef np.int64_t[:] market_opens, market_closes
    cdef object before_trading_start_minutes
    cdef dict minutes_by_session, minutes_to_session

    def __init__(self,
                 sessions,
                 market_opens,
                 market_closes,
                 before_trading_start_minutes,
                 minute_emission=False):
        self.minute_emission = minute_emission
        self.market_opens = market_opens
        self.market_closes = market_closes
        self.sessions = sessions
        self.minutes_by_session = self.calc_minutes_by_session()

        self.before_trading_start_minutes = before_trading_start_minutes

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
    cdef dict calc_minutes_by_session(self):
        cdef dict minutes_by_session
        cdef int session_idx
        cdef object session

        minutes_by_session = {}
        for session_idx, session in enumerate(self.sessions):
            minutes_by_session[session] = pd.to_datetime(
                self.market_minutes(session_idx), utc=True, box=True)
        return minutes_by_session

    def __iter__(self):
        minute_emission = self.minute_emission

        for idx, session in enumerate(self.sessions):
            yield session, SESSION_START

            bts_minute = self.before_trading_start_minutes[idx]
            regular_minutes = self.minutes_by_session[session]

            # we have to search anew every session, because there is no
            # guarantee that any two session start on the same minute
            bts_idx = regular_minutes.searchsorted(bts_minute)

            if bts_idx == len(regular_minutes):
                # before_trading_start is after the last close, so don't emit
                # it
                for minute in regular_minutes:
                    yield minute, BAR
                    if minute_emission:
                        yield minute, MINUTE_END
            else:
                # emit all the minutes before bts_minute
                for minute in regular_minutes[0:bts_idx]:
                    yield minute, BAR
                    if minute_emission:
                        yield minute, MINUTE_END

                yield bts_minute, BEFORE_TRADING_START_BAR

                # emit all the minutes after bts_minute
                for minute in regular_minutes[bts_idx:]:
                    yield minute, BAR
                    if minute_emission:
                        yield minute, MINUTE_END

            yield regular_minutes[-1], SESSION_END
