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

from time import sleep

from logbook import Logger
import pandas as pd

from zipline.gens.sim_engine import (
    BAR,
    SESSION_START,
    SESSION_END,
    MINUTE_END,
    BEFORE_TRADING_START_BAR
)

log = Logger('Realtime Clock')


class RealtimeClock(object):
    """Realtime clock for live trading.

    This class is a drop-in replacement for
    :class:`zipline.gens.sim_engine.MinuteSimulationClock`.
    The key difference between the two is that the RealtimeClock's event
    emission is synchronized to the (broker's) wall time clock, while
    MinuteSimulationClock yields a new event on every iteration (regardless of
    wall clock).

    The :param:`time_skew` parameter represents the time difference between
    the Broker and the live trading machine's clock.
    """

    def __init__(self,
                 sessions,
                 execution_opens,
                 execution_closes,
                 before_trading_start_minutes,
                 minute_emission,
                 time_skew=pd.Timedelta("0s"),
                 is_broker_alive=None):
        self.sessions = sessions
        self.execution_opens = execution_opens
        self.execution_closes = execution_closes
        self.before_trading_start_minutes = before_trading_start_minutes
        self.minute_emission = minute_emission
        self.time_skew = time_skew
        self.is_broker_alive = is_broker_alive or (lambda: True)
        self._last_emit = None
        self._before_trading_start_bar_yielded = False

    def __iter__(self):
        yield self.sessions[0], SESSION_START

        while self.is_broker_alive():
            current_time = pd.to_datetime('now', utc=True)
            server_time = (current_time + self.time_skew).floor('1 min')

            if (server_time >= self.before_trading_start_minutes[0] and
                    not self._before_trading_start_bar_yielded):
                self._last_emit = server_time
                self._before_trading_start_bar_yielded = True
                yield server_time, BEFORE_TRADING_START_BAR
            elif server_time < self.execution_opens[0].tz_localize('UTC'):
                sleep(1)
            elif (self.execution_opens[0].tz_localize('UTC') <= server_time <
                  self.execution_closes[0].tz_localize('UTC')):
                if (self._last_emit is None or
                        server_time - self._last_emit >=
                        pd.Timedelta('1 minute')):
                    self._last_emit = server_time
                    yield server_time, BAR
                    if self.minute_emission:
                        yield server_time, MINUTE_END
                else:
                    sleep(1)
            elif server_time == self.execution_closes[0].tz_localize('UTC'):
                self._last_emit = server_time
                yield server_time, BAR
                if self.minute_emission:
                    yield server_time, MINUTE_END
                yield server_time, SESSION_END

                return
            elif server_time > self.execution_closes[0].tz_localize('UTC'):
                # Return with no yield if the algo is started in after hours
                return
            else:
                # We should never end up in this branch
                raise RuntimeError("Invalid state in RealtimeClock")
