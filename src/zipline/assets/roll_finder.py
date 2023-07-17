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
from abc import ABC, abstractmethod

# Number of days over which to compute rolls when finding the current contract
# for a volume-rolling contract chain. For more details on why this is needed,
# see `VolumeRollFinder.get_contract_center`.
ROLL_DAYS_FOR_CURRENT_CONTRACT = 90


class RollFinder(ABC):
    """Abstract base class for calculating when futures contracts are the active
    contract.
    """

    @abstractmethod
    def _active_contract(self, oc, front, back, dt):
        raise NotImplementedError

    def _get_active_contract_at_offset(self, root_symbol, dt, offset):
        """For the given root symbol, find the contract that is considered active
        on a specific date at a specific offset.
        """
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        session = self.trading_calendar.minute_to_session(dt)
        front = oc.contract_before_auto_close(session.value)
        back = oc.contract_at_offset(front, 1, dt.value)
        if back is None:
            return front
        primary = self._active_contract(oc, front, back, session)
        return oc.contract_at_offset(primary, offset, session.value)

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
        return self._get_active_contract_at_offset(root_symbol, dt, offset)

    def get_rolls(self, root_symbol, start, end, offset):
        """Get the rolls, i.e. the session at which to hop from contract to
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
        front = self._get_active_contract_at_offset(root_symbol, end, 0)
        back = oc.contract_at_offset(front, 1, end.value)
        if back is not None:
            end_session = self.trading_calendar.minute_to_session(end)
            first = self._active_contract(oc, front, back, end_session)
        else:
            first = front
        first_contract = oc.sid_to_contract[first]
        rolls = [((first_contract >> offset).contract.sid, None)]
        tc = self.trading_calendar
        sessions = tc.sessions_in_range(
            tc.minute_to_session(start), tc.minute_to_session(end)
        )
        freq = sessions.freq
        if first == front:
            # This is a bit tricky to grasp. Once we have the active contract
            # on the given end date, we want to start walking backwards towards
            # the start date and checking for rolls. For this, we treat the
            # previous month's contract as the 'first' contract, and the
            # contract we just found to be active as the 'back'. As we walk
            # towards the start date, if the 'back' is no longer active, we add
            # that date as a roll.
            curr = first_contract << 1
        else:
            curr = first_contract << 2
        session = sessions[-1]

        start = start.tz_localize(None)

        while session > start and curr is not None:
            front = curr.contract.sid
            back = rolls[0][0]
            prev_c = curr.prev
            while session > start:
                prev = (session - freq).tz_localize(None)
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
                session = min(session, curr.contract.auto_close_date + freq)

        return rolls


class CalendarRollFinder(RollFinder):
    """The CalendarRollFinder calculates contract rolls based purely on the
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
    """The VolumeRollFinder calculates contract rolls based on when
    volume activity transfers from one contract to another.
    """

    GRACE_DAYS = 7

    def __init__(self, trading_calendar, asset_finder, session_reader):
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder
        self.session_reader = session_reader

    def _active_contract(self, oc, front, back, dt):
        r"""
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
        front_contract = oc.sid_to_contract[front].contract
        back_contract = oc.sid_to_contract[back].contract

        tc = self.trading_calendar
        trading_day = tc.day
        prev = dt - trading_day
        get_value = self.session_reader.get_value

        # If the front contract is past its auto close date it cannot be the
        # active contract, so return the back contract. Similarly, if the back
        # contract has not even started yet, just return the front contract.
        # The reason for using 'prev' to see if the contracts are alive instead
        # of using 'dt' is because we need to get each contract's volume on the
        # previous day, so we need to make sure that each contract exists on
        # 'prev' in order to call 'get_value' below.
        if dt > min(front_contract.auto_close_date, front_contract.end_date):
            return back
        elif front_contract.start_date > prev:
            return back
        elif dt > min(back_contract.auto_close_date, back_contract.end_date):
            return front
        elif back_contract.start_date > prev:
            return front

        front_vol = get_value(front, prev, "volume")
        back_vol = get_value(back, prev, "volume")
        if back_vol > front_vol:
            return back

        gap_start = max(
            back_contract.start_date,
            front_contract.auto_close_date - (trading_day * self.GRACE_DAYS),
        )
        gap_end = prev - trading_day
        if dt < gap_start:
            return front

        # If we are within `self.GRACE_DAYS` of the front contract's auto close
        # date, and a volume flip happened during that period, return the back
        # contract as the active one.
        sessions = tc.sessions_in_range(
            tc.minute_to_session(gap_start),
            tc.minute_to_session(gap_end),
        )
        for session in sessions:
            front_vol = get_value(front, session, "volume")
            back_vol = get_value(back, session, "volume")
            if back_vol > front_vol:
                return back
        return front

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
        # When determining the center contract on a specific day using volume
        # rolls, simply picking the contract with the highest volume could
        # cause flip-flopping between active contracts each day if the front
        # and back contracts are close in volume. Therefore, information about
        # the surrounding rolls is required. The `get_rolls` logic prevents
        # contracts from being considered active once they have rolled, so
        # incorporating that logic here prevents flip-flopping.
        day = self.trading_calendar.day
        end_date = min(
            dt + (ROLL_DAYS_FOR_CURRENT_CONTRACT * day),
            self.session_reader.last_available_dt.tz_localize(dt.tzinfo),
        )
        rolls = self.get_rolls(
            root_symbol=root_symbol,
            start=dt,
            end=end_date,
            offset=offset,
        )
        sid, acd = rolls[0]
        return self.asset_finder.retrieve_asset(sid)
