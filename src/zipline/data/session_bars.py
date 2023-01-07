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
from abc import abstractmethod

from zipline.data.bar_reader import BarReader


class SessionBarReader(BarReader):
    """Reader for OHCLV pricing data at a session frequency."""

    @property
    def data_frequency(self):
        return "session"

    @property
    @abstractmethod
    def sessions(self):
        """

        Returns
        -------
        sessions : DatetimeIndex
           All session labels (unioning the range for all assets) which the
           reader can provide.
        """


class CurrencyAwareSessionBarReader(SessionBarReader):
    @abstractmethod
    def currency_codes(self, sids):
        """Get currencies in which prices are quoted for the requested sids.

        Assumes that a sid's prices are always quoted in a single currency.

        Parameters
        ----------
        sids : np.array[int64]
            Array of sids for which currencies are needed.

        Returns
        -------
        currency_codes : np.array[object]
            Array of currency codes for listing currencies of
            ``sids``. Implementations should return None for sids whose
            currency is unknown.
        """
