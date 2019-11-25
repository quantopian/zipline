from functools import partial, total_ordering

from iso4217 import Currency as ISO4217Currency

import numpy as np

_ALL_CURRENCIES = {}


def strs_to_sids(strs, category_num):
    """TODO: Improve this.
    """
    out = np.full(len(strs), category_num << 50, dtype='i8')
    casted_buffer = np.ndarray(
        shape=out.shape,
        dtype='S6',
        buffer=out,
        strides=out.strides,
    )
    casted_buffer[:] = np.array(strs, dtype='S6')
    return out


def str_to_sid(str_, category_num):
    return strs_to_sids([str_], category_num)[0]


iso_currency_to_sid = partial(str_to_sid, category_num=3)


@total_ordering
class Currency(object):
    """A currency identifier, as defined by ISO-4217.

    Parameters
    ----------
    code : str
        ISO-4217 code for the currency.

    Attributes
    ----------
    code : str
        ISO-4217 currency code for the currency, e.g., 'USD'.
    name : str
        Plain english name for the currency, e.g., 'US Dollar'.
    """
    def __new__(cls, code):
        try:
            return _ALL_CURRENCIES[code]
        except KeyError:
            try:
                iso_currency = ISO4217Currency(code)
            except ValueError:
                raise ValueError(
                    "{!r} is not a valid currency code.".format(code)
                )
            obj = _ALL_CURRENCIES[code] = super(Currency, cls).__new__(cls)
            obj._currency = iso_currency
            obj._sid = iso_currency_to_sid(iso_currency.value)
            return obj

    @property
    def code(self):
        """ISO-4217 currency code for the currency.

        Returns
        -------
        code : str
        """
        return self._currency.value

    @property
    def name(self):
        """Plain english name for the currency.

        Returns
        -------
        name : str
        """
        return self._currency.currency_name

    @property
    def sid(self):
        """Unique integer identifier for this currency.
        """
        return self._sid

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.code == other.code

    def __hash__(self):
        return hash(self.code)

    def __lt__(self, other):
        return self.code < other.code

    def __repr__(self):
        return "{}({!r})".format(
            type(self).__name__,
            self.code
        )
