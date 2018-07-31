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

from enum import Enum
import numpy as np
import pandas as pd
import itertools
from interface import Interface, implements

from zipline.extensions import extensible, register


@extensible
class Clock(Interface):

    def __iter__(self):
        raise NotImplementedError('__iter__ must be implemented')


_nanos_in_minute = np.int64(60000000000)
NANOS_IN_MINUTE = _nanos_in_minute


class SessionEvt(Enum):
    BAR = 0
    SESSION_START = 1
    SESSION_END = 2
    MINUTE_END = 3
    BEFORE_TRADING_START_BAR = 4


BAR = SessionEvt.BAR
SESSION_START = SessionEvt.SESSION_START
SESSION_END = SessionEvt.SESSION_END
MINUTE_END = SessionEvt.MINUTE_END
BEFORE_TRADING_START_BAR = SessionEvt.BEFORE_TRADING_START_BAR


class MinuteSimulationClock(implements(Clock)):

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

    def __iter__(self):
        for session_nano, market_open, market_close, bts_minute in zip(
            self.sessions_nanos,
            self.market_opens_nanos,
            self.market_closes_nanos,
            self.bts_nanos
        ):

            counter = pd.Timestamp(market_open, tz='UTC')
            market_close = pd.Timestamp(market_close, tz='UTC')
            bts_minute = pd.Timestamp(bts_minute, tz='UTC')

            yield pd.Timestamp(session_nano, tz='UTC'), SESSION_START

            if bts_minute < counter:
                yield bts_minute, BEFORE_TRADING_START_BAR

            while counter <= market_close:
                if counter == bts_minute:
                    yield counter, BEFORE_TRADING_START_BAR
                yield counter, BAR
                if self.minute_emission:
                    yield counter, MINUTE_END

                counter += pd.Timedelta(_nanos_in_minute)

            yield market_close, SESSION_END


@register(Clock, 'default')
def func():
    return MinuteSimulationClock
