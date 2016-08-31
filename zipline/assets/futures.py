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

from pandas import Timestamp

from zipline.utils.input_validation import expect_types


class FutureChain(object):
    """
    Allows users to look up future contracts.

    Parameters
    ----------
    root_symbol : str
        The root symbol of a future chain.
    as_of_date : pandas.Timestamp
        Date at which the chain determination is rooted. I.e. the
        existing contract whose notice date is first after this date is
        the primary contract, etc.
    chain: list
        List of assets that represent the chain of contracts for the given
        root symbol at the given as_of_date.

    Attributes
    ----------
    root_symbol : str
        The root symbol of the future chain.
    as_of_date: Timestamp
        The current as-of date of this future chain.
    """
    @expect_types(root_symbol=str, as_of_date=Timestamp)
    def __init__(self, root_symbol, as_of_date, contracts):
        self._root_symbol = root_symbol
        self._as_of_date = as_of_date
        self._contracts = contracts

    def __repr__(self):
        return "FutureChain('%s', '%s')" % (
            self.root_symbol, self.as_of_date.strftime('%Y-%m-%d'))

    @property
    def root_symbol(self):
        """
        The root symbol for this future chain.

        Returns
        -------
        root_symbol: str
            The root symbol for this chain.
        """
        return self._root_symbol

    @property
    def as_of_date(self):
        """
        The as-of date of this future chain.

        Returns
        -------
        as_of_date: pd.Timestamp
            The as_of date for this chain.
        """
        return self._as_of_date

    @property
    def contracts(self):
        """
        Returns
        -------
        contracts: list
            The contracts wrapped by this chain.
        """
        return list(self._contracts)

    def __getitem__(self, key):
        return self._contracts[key]

    def __len__(self):
        return len(self._contracts)

    def __iter__(self):
        return iter(self._contracts)


# http://www.cmegroup.com/product-codes-listing/month-codes.html
CME_CODE_TO_MONTH = dict(zip('FGHJKMNQUVXZ', range(1, 13)))
MONTH_TO_CME_CODE = dict(zip(range(1, 13), 'FGHJKMNQUVXZ'))


def cme_code_to_month(code):
    """
    Convert a CME month code to a month index.

    The month codes are as follows:

    'F' -> 1  (January)
    'G' -> 2  (February)
    'H' -> 3  (March)
    'J' -> 4  (April)
    'K' -> 5  (May)
    'M' -> 6  (June)
    'N' -> 7  (July)
    'Q' -> 8  (August)
    'U' -> 9  (September)
    'V' -> 10 (October)
    'X' -> 11 (November)
    'Z' -> 12 (December)

    Parameters
    ----------
    code : str
        The month code to look up.

    Returns
    -------
    month : int
       The month number (starting at 1 for January) corresponding to the
       requested code.

    See Also
    --------
    month_to_cme_code
        Inverse of this function.
    """
    return CME_CODE_TO_MONTH[code]


def month_to_cme_code(month):
    """
    Convert a month to a CME code.

    The month codes are as follows:

    1 (January)   -> 'F'
    2 (February)  -> 'G'
    3 (March)     -> 'H'
    4 (April)     -> 'J'
    5 (May)       -> 'K'
    6 (June)      -> 'M'
    7 (July)      -> 'N'
    8 (August)    -> 'Q'
    9 (September) -> 'U'
    10 (October)  -> 'V'
    11 (November) -> 'X'
    12 (December) -> 'Z'

    Parameters
    ----------
    month : int
       The month number (starting at 1 for January) corresponding to the
       requested code.

    Returns
    -------
    code : str
        The month code to look up.

    See Also
    --------
    cme_code_to_month
        Inverse of this function.
    """
    return MONTH_TO_CME_CODE[month]
