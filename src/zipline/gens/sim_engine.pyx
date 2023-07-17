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

cimport cython
cimport numpy as np
import numpy as np
import pandas as pd
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
    cdef bool minute_emission
    cdef np.int64_t[:] market_opens_nanos, market_closes_nanos, bts_nanos, \
        sessions_nanos
    cdef dict minutes_by_session

    def __init__(self,
                 sessions,
                 market_opens,
                 market_closes,
                 before_trading_start_minutes,
                 minute_emission=False):
        self.minute_emission = minute_emission

        self.market_opens_nanos = market_opens.values.astype(np.int64)
        self.market_closes_nanos = market_closes.values.astype(np.int64)
        self.sessions_nanos = sessions.values.astype(np.int64)
        self.bts_nanos = before_trading_start_minutes.values.astype(np.int64)

        self.minutes_by_session = self.calc_minutes_by_session()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef dict calc_minutes_by_session(self):
        cdef dict minutes_by_session
        cdef int session_idx
        cdef np.int64_t session_nano
        cdef np.ndarray[np.int64_t, ndim=1] minutes_nanos

        minutes_by_session = {}
        for session_idx, session_nano in enumerate(self.sessions_nanos):
            minutes_nanos = np.arange(
                self.market_opens_nanos[session_idx],
                self.market_closes_nanos[session_idx] + NANOS_IN_MINUTE,
                NANOS_IN_MINUTE
            )
            minutes_by_session[session_nano] = pd.to_datetime(
                minutes_nanos, utc=True
            )
        return minutes_by_session

    def __iter__(self):
        minute_emission = self.minute_emission

        cdef Py_ssize_t idx

        for idx, session_nano in enumerate(self.sessions_nanos):
            yield pd.Timestamp(session_nano, tz='UTC'), SESSION_START

            bts_minute = pd.Timestamp(self.bts_nanos[idx], tz='UTC')
            regular_minutes = self.minutes_by_session[session_nano]

            if bts_minute > regular_minutes[-1]:
                # before_trading_start is after the last close,
                # so don't emit it
                for minute, evt in self._get_minutes_for_list(
                    regular_minutes,
                    minute_emission
                ):
                    yield minute, evt
            else:
                # we have to search anew every session, because there is no
                # guarantee that any two session start on the same minute
                bts_idx = regular_minutes.searchsorted(bts_minute)

                # emit all the minutes before bts_minute
                for minute, evt in self._get_minutes_for_list(
                    regular_minutes[0:bts_idx],
                    minute_emission
                ):
                    yield minute, evt

                yield bts_minute, BEFORE_TRADING_START_BAR

                # emit all the minutes after bts_minute
                for minute, evt in self._get_minutes_for_list(
                    regular_minutes[bts_idx:],
                    minute_emission
                ):
                    yield minute, evt

            yield regular_minutes[-1], SESSION_END

    def _get_minutes_for_list(self, minutes, minute_emission):
        for minute in minutes:
            yield minute, BAR
            if minute_emission:
                yield minute, MINUTE_END
