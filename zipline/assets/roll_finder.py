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
from abc import ABCMeta, abstractmethod
from six import with_metaclass

from pandas import Timestamp


class RollFinder(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for calculating when futures contracts are the active
    contract.
    """
    @abstractmethod
    def _active_contract(self, oc, front, back, dt):
        raise NotImplementedError

    def get_contract_center(self, root_symbol, dt, offset):
        """
        Parameters
        ----------
        root_symbol : str
            The root symbol for the contract chain.
        dt : Timestamp
            The datetime for which to retrieve the current contract.
        offset : int
            The offset from the primary contract.
            0 is the primary, 1 is the secondary, etc.

        Returns
        -------
        Future
            The active future contract at the given dt.
        """
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        session = self.trading_calendar.minute_to_session_label(dt)
        front = oc.contract_before_auto_close(session.value)
        back = oc.contract_at_offset(front, 1, dt.value)
        if back is None:
            return front
        session = self.trading_calendar.minute_to_session_label(dt)
        primary = self._active_contract(oc, front, back, session)
        return oc.contract_at_offset(primary, offset, session.value)

    def get_rolls(self, root_symbol, start, end, offset):
        """
        Get the rolls, i.e. the session at which to hop from contract to
        contract in the chain.

        Parameters
        ----------
        root_symbol : str
            The root symbol for which to calculate rolls.
        start : Timestamp
            Start of the date range.
        end : Timestamp
            End of the date range.
        offset : int
            Offset from the primary.

        Returns
        -------
        rolls - list[tuple(sid, roll_date)]
            A list of rolls, where first value is the first active `sid`,
        and the `roll_date` on which to hop to the next contract.
            The last pair in the chain has a value of `None` since the roll
            is after the range.
        """
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        front = self.get_contract_center(root_symbol, end, 0)
        back = oc.contract_at_offset(front, 1, end.value)
        if back is not None:
            first = self._active_contract(oc, front, back, end)
        else:
            first = front
        for i, sid in enumerate(oc.contract_sids):
            if sid == first:
                break
        rolls = [(first, None)]
        sessions = self.trading_calendar.sessions_in_range(start, end)
        if first == front:
            i -= 1
        else:
            i -= 2
        auto_close_date = Timestamp(oc.auto_close_dates[i], tz='UTC')
        while auto_close_date > start and i > -1:
            session_loc = sessions.searchsorted(auto_close_date)
            front = oc.contract_sids[i]
            back = oc.contract_sids[i + 1]
            while session_loc > -1:
                session = sessions[session_loc]
                if back != self._active_contract(oc, front, back, session):
                    break
                session_loc -= 1
            roll_session = sessions[session_loc + 1]
            if roll_session > start:
                rolls.insert(0, (oc.contract_sids[i + offset],
                                 roll_session))
            i -= 1
            auto_close_date = Timestamp(oc.auto_close_dates[i],
                                        tz='UTC')
        return rolls


class CalendarRollFinder(RollFinder):
    """
    The CalendarRollFinder calculates contract rolls based purely on the
    contract's auto close date.
    """

    def __init__(self, trading_calendar, asset_finder):
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder

    def _active_contract(self, oc, front, back, dt):
        for i, sid in enumerate(oc.contract_sids):
            if sid == front:
                break
        auto_close_date = Timestamp(oc.auto_close_dates[i], tz='UTC')
        before_auto_close = dt < auto_close_date
        return front if before_auto_close else back


class VolumeRollFinder(RollFinder):
    """
    The CalendarRollFinder calculates contract rolls based on when
    volume activity transfers from one contract to another.
    """

    THRESHOLD = 0.10

    def __init__(self, trading_calendar, asset_finder, session_reader):
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder
        self.session_reader = session_reader

    def _active_contract(self, oc, front, back, dt):
        # FIXME: Possible vector for look ahead bias.
        front_vol = self.session_reader.get_value(front, dt, 'volume')
        back_vol = self.session_reader.get_value(back, dt, 'volume')
        return back if back_vol > front_vol else front
