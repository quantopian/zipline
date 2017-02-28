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
            end_session = self.trading_calendar.minute_to_session_label(end)
            first = self._active_contract(oc, front, back, end_session)
        else:
            first = front
        first_contract = oc.sid_to_contract[first]
        rolls = [((first_contract >> offset).contract.sid, None)]
        tc = self.trading_calendar
        sessions = tc.sessions_in_range(tc.minute_to_session_label(start),
                                        tc.minute_to_session_label(end))
        freq = sessions.freq
        if first == front:
            curr = first_contract << 1
        else:
            curr = first_contract << 2
        session = sessions[-1]

        while session > start and curr is not None:
            front = curr.contract.sid
            back = rolls[0][0]
            prev_c = curr.prev
            while session > start:
                prev = session - freq
                if prev_c is not None:
                    if prev < prev_c.contract.auto_close_date:
                        break
                if back != self._active_contract(oc, front, back, prev):
                    # TODO: Instead of listing each contract with its roll date
                    # as tuples, create a series which maps every day to the
                    # active contract on that day.
                    rolls.insert(0, ((curr >> offset).contract.sid, session))
                    break
                session = prev
            curr = curr.prev
            if curr is not None:
                session = curr.contract.auto_close_date
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
        contract = oc.sid_to_contract[front].contract
        auto_close_date = contract.auto_close_date
        auto_closed = dt >= auto_close_date
        return back if auto_closed else front


class VolumeRollFinder(RollFinder):
    """
    The CalendarRollFinder calculates contract rolls based on when
    volume activity transfers from one contract to another.
    """
    GRACE_DAYS = 7
    THRESHOLD = 0.10

    def __init__(self, trading_calendar, asset_finder, session_reader):
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder
        self.session_reader = session_reader

    def _active_contract(self, oc, front, back, dt):
        """
        Return the active contract based on the previous trading day's volume.

        In the rare case that a double volume switch occurs we treat the first
        switch as the roll. Take the following case for example:

        | +++++             _____
        |      +   __      /       <--- 'G'
        |       ++/++\++++/++
        |       _/    \__/   +
        |      /              +
        | ____/                +   <--- 'F'
        |_________|__|___|________
                  a  b   c         <--- Switches

        We should treat 'a' as the roll date rather than 'c' because from the
        perspective of 'a', if a switch happens and we are pretty close to the
        auto-close date, we would probably assume it is time to roll. This
        means that for every date after 'a', `data.current(cf, 'contract')`
        should return the 'G' contract.
        """
        tc = self.trading_calendar
        trading_day = tc.day
        prev = dt - trading_day
        get_value = self.session_reader.get_value
        front_vol = get_value(front, prev, 'volume')
        back_vol = get_value(back, prev, 'volume')
        front_contract = oc.sid_to_contract[front].contract

        if dt >= front_contract.auto_close_date or back_vol > front_vol:
            return back

        gap_start = \
            front_contract.auto_close_date - (trading_day * self.GRACE_DAYS)
        gap_end = prev - trading_day
        if dt < gap_start:
            return front

        # If we are within `self.GRACE_DAYS` of the front contract's auto close
        # date, and a volume flip happened during that period, return the back
        # contract as the active one.
        sessions = tc.sessions_in_range(
            tc.minute_to_session_label(gap_start),
            tc.minute_to_session_label(gap_end),
        )
        for session in sessions:
            front_vol = get_value(front, session, 'volume')
            back_vol = get_value(back, session, 'volume')
            if back_vol > front_vol:
                return back
        return front
