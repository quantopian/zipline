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
    """
    def __new__(cls, code):
        try:
            return _ALL_CURRENCIES[code]
        except KeyError:
            iso_currency = ISO4217Currency(code)
            obj = _ALL_CURRENCIES[code] = super(Currency, cls).__new__(cls)
            obj._currency = iso_currency
            obj._sid = iso_currency_to_sid(iso_currency.value)
            return obj

    @property
    def code(self):
        return self._currency.value

    @property
    def name(self):
        return self._currency.currency_name

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.code == other.code

    def __lt__(self, other):
        return self.code < other.code

    def __repr__(self):
        return "{}({!r})".format(
            type(self).__name__,
            self.code
        )
