#
# Copyright 2016 Quantopian, Inc.
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

import logging
import pandas as pd

from zipline.utils.memoize import remember_last

log = logging.getLogger("Trading")


DEFAULT_CAPITAL_BASE = 1e5


class SimulationParameters:
    def __init__(
        self,
        start_session,
        end_session,
        trading_calendar,
        capital_base=DEFAULT_CAPITAL_BASE,
        emission_rate="daily",
        data_frequency="daily",
        arena="backtest",
    ):

        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp

        assert trading_calendar is not None, "Must pass in trading calendar!"
        assert start_session <= end_session, "Period start falls after period end."
        assert (
            start_session.tz_localize(None) <= trading_calendar.last_session
        ), "Period start falls after the last known trading day."
        assert (
            end_session.tz_localize(None) >= trading_calendar.first_session
        ), "Period end falls before the first known trading day."

        # chop off any minutes or hours on the given start and end dates,
        # as we only support session labels here (and we represent session
        # labels as midnight UTC).
        self._start_session = start_session.normalize()
        self._end_session = end_session.normalize()
        self._capital_base = capital_base

        self._emission_rate = emission_rate
        self._data_frequency = data_frequency

        # copied to algorithm's environment for runtime access
        self._arena = arena

        self._trading_calendar = trading_calendar

        if not trading_calendar.is_session(self._start_session.tz_localize(None)):
            # if the start date is not a valid session in this calendar,
            # push it forward to the first valid session
            self._start_session = trading_calendar.minute_to_session(
                self._start_session
            )

        if not trading_calendar.is_session(self._end_session.tz_localize(None)):
            # if the end date is not a valid session in this calendar,
            # pull it backward to the last valid session before the given
            # end date.
            self._end_session = trading_calendar.minute_to_session(
                self._end_session, direction="previous"
            )

        self._first_open = trading_calendar.session_first_minute(
            self._start_session.tz_localize(None)
        )
        self._last_close = trading_calendar.session_close(
            self._end_session.tz_localize(None)
        )

    @property
    def capital_base(self):
        return self._capital_base

    @property
    def emission_rate(self):
        return self._emission_rate

    @property
    def data_frequency(self):
        return self._data_frequency

    @data_frequency.setter
    def data_frequency(self, val):
        self._data_frequency = val

    @property
    def arena(self):
        return self._arena

    @arena.setter
    def arena(self, val):
        self._arena = val

    @property
    def start_session(self):
        return self._start_session

    @property
    def end_session(self):
        return self._end_session

    @property
    def first_open(self):
        return self._first_open

    @property
    def last_close(self):
        return self._last_close

    @property
    def trading_calendar(self):
        return self._trading_calendar

    @property
    @remember_last
    def sessions(self):
        return self._trading_calendar.sessions_in_range(
            self.start_session, self.end_session
        )

    def create_new(self, start_session, end_session, data_frequency=None):
        if data_frequency is None:
            data_frequency = self.data_frequency

        return SimulationParameters(
            start_session,
            end_session,
            self._trading_calendar,
            capital_base=self.capital_base,
            emission_rate=self.emission_rate,
            data_frequency=data_frequency,
            arena=self.arena,
        )

    def __repr__(self):
        return """
{class_name}(
    start_session={start_session},
    end_session={end_session},
    capital_base={capital_base},
    data_frequency={data_frequency},
    emission_rate={emission_rate},
    first_open={first_open},
    last_close={last_close},
    trading_calendar={trading_calendar}
)\
""".format(
            class_name=self.__class__.__name__,
            start_session=self.start_session,
            end_session=self.end_session,
            capital_base=self.capital_base,
            data_frequency=self.data_frequency,
            emission_rate=self.emission_rate,
            first_open=self.first_open,
            last_close=self.last_close,
            trading_calendar=self._trading_calendar,
        )
