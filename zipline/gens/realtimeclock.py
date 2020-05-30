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
    """
    Realtime clock for live trading.

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
                 is_broker_alive=None,
                 execution_id=None,
                 stop_execution_callback=None):
        today = pd.to_datetime('now', utc=True).date()
        beginning_of_today = pd.to_datetime(today, utc=True)

        self.sessions = sessions[(beginning_of_today <= sessions)]
        self.execution_opens = execution_opens[(beginning_of_today <= execution_opens)]
        self.execution_closes = execution_closes[(beginning_of_today <= execution_closes)]
        self.before_trading_start_minutes = before_trading_start_minutes[
            (beginning_of_today <= before_trading_start_minutes)]

        self.minute_emission = minute_emission
        self.time_skew = time_skew
        self.is_broker_alive = is_broker_alive or (lambda: True)
        self._last_emit = None
        self._before_trading_start_bar_yielded = False
        self._execution_id = execution_id
        self._stop_execution_callback = stop_execution_callback

    def __iter__(self):
        # yield from self.work_when_out_of_trading_hours()
        # return

        if not len(self.sessions):
            return

        for index, session in enumerate(self.sessions):
            self._before_trading_start_bar_yielded = False

            yield session, SESSION_START

            if self._stop_execution_callback:
                if self._stop_execution_callback(self._execution_id):
                    break

            while self.is_broker_alive():
                if self._stop_execution_callback:  # put it here too, to break inner loop as well
                    if self._stop_execution_callback(self._execution_id):
                        break
                current_time = pd.to_datetime('now', utc=True)
                server_time = (current_time + self.time_skew).floor('1 min')

                if (server_time >= self.before_trading_start_minutes[index] and
                        not self._before_trading_start_bar_yielded):
                    self._last_emit = server_time
                    self._before_trading_start_bar_yielded = True
                    yield server_time, BEFORE_TRADING_START_BAR
                elif (server_time < self.execution_opens[index].tz_localize('UTC') and index == 0) or \
                        (self.execution_closes[index - 1].tz_localize('UTC') <= server_time <
                         self.execution_opens[index].tz_localize('UTC')):
                    # sleep anywhere between yesterday's close and today's open
                    sleep(1)
                elif (self.execution_opens[index].tz_localize('UTC') <= server_time <
                      self.execution_closes[index].tz_localize('UTC')):
                    if (self._last_emit is None or
                            server_time - self._last_emit >=
                            pd.Timedelta('1 minute')):
                        self._last_emit = server_time
                        yield server_time, BAR
                        if self.minute_emission:
                            yield server_time, MINUTE_END
                    else:
                        sleep(1)
                elif server_time == self.execution_closes[index].tz_localize('UTC'):
                    self._last_emit = server_time
                    yield server_time, BAR
                    if self.minute_emission:
                        yield server_time, MINUTE_END
                    yield server_time, SESSION_END
                    break
                elif server_time > self.execution_closes[index].tz_localize('UTC'):
                    break
                else:
                    # We should never end up in this branch
                    raise RuntimeError("Invalid state in RealtimeClock")

    def work_when_out_of_trading_hours(self):
        """
        a debugging method to work while outside trading hours, so we are still able to make the engine work
        :return:
        """
        from datetime import timedelta
        num_days = 5
        from trading_calendars import get_calendar
        self.sessions = get_calendar("NYSE").sessions_in_range(
            str(pd.to_datetime('now', utc=True).date() - timedelta(days=num_days * 2)),
            str(pd.to_datetime('now', utc=True).date() + timedelta(days=num_days * 2))
        )

        # for day in range(num_days, 0, -1):
        for day in range(0, 1):
            # current_time = pd.to_datetime('now', utc=True)
            current_time = pd.to_datetime('2018/08/25', utc=True)
            # server_time = (current_time + self.time_skew).floor('1 min') - timedelta(days=day)
            server_time = (current_time + self.time_skew).floor('1 min') + timedelta(days=day)

            # yield self.sessions[-1 - day], SESSION_START
            yield self.sessions[day], SESSION_START
            yield server_time, BEFORE_TRADING_START_BAR
            should_end_day = True
            counter = 0
            num_minutes = 6 * 60
            minute_list = []
            for i in range(num_minutes + 1):
                minute_list.append(pd.to_datetime("13:31", utc=True) + timedelta(minutes=i))
            while self.is_broker_alive():
                # current_time = pd.to_datetime('now', utc=True)
                # server_time = (current_time + self.time_skew).floor('1 min')
                # server_time = minute_list[counter] - timedelta(days=day)
                server_time = minute_list[counter] + timedelta(days=day)
                if counter >= num_minutes and should_end_day:
                    if self.minute_emission:
                        yield server_time, MINUTE_END
                    yield server_time, SESSION_END
                    break

                if self._stop_execution_callback:
                    if self._stop_execution_callback(self._execution_id):
                        break
                if (self._last_emit is None or
                        server_time - self._last_emit >=
                        pd.Timedelta('1 minute')):
                    self._last_emit = server_time
                    yield server_time, BAR
                    counter += 1
                    if self.minute_emission:
                        yield server_time, MINUTE_END
                sleep(0.5)
