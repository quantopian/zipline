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
