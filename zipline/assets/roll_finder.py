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
        raise NotImplemented

    @abstractmethod
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
        raise NotImplemented


class CalendarRollFinder(RollFinder):
    """
    The CalendarRollFinder calculates contract rolls based purely on the
    contract's auto close date.
    """

    def __init__(self, trading_calendar, asset_finder):
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder

    def get_contract_center(self, root_symbol, dt, offset):
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        session = self.trading_calendar.minute_to_session_label(dt)
        primary_candidate = oc.contract_before_auto_close(session.value)

        # Here is where a volume check would be.
        primary = primary_candidate
        return oc.contract_at_offset(primary, offset)

    def get_rolls(self, root_symbol, start, end, offset):
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        primary_at_end = self.get_contract_center(root_symbol, end, 0)
        for i, sid in enumerate(oc.contract_sids):
            if sid == primary_at_end:
                break
        i += offset
        first = oc.contract_sids[i]
        rolls = [(first, None)]
        i -= 1
        auto_close_date = Timestamp(oc.auto_close_dates[i - offset], tz='UTC')
        while auto_close_date > start and i > -1:
            rolls.insert(0, (oc.contract_sids[i - offset],
                             auto_close_date))
            i -= 1
            auto_close_date = Timestamp(oc.auto_close_dates[i - offset],
                                        tz='UTC')

        return rolls
