from functools import total_ordering
from iso4217 import Currency as ISO4217Currency

_ALL_CURRENCIES = {}


# Special sentinel used to represent unknown or missing currencies.
MISSING_CURRENCY_CODE = 'XXX'


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
            # This isn't a real
            if code == MISSING_CURRENCY_CODE:
                name = "NO CURRENCY"
            else:
                try:
                    name = ISO4217Currency(code).currency_name
                except ValueError:
                    raise ValueError(
                        "{!r} is not a valid currency code.".format(code)
                    )

            obj = _ALL_CURRENCIES[code] = super(Currency, cls).__new__(cls)
            obj._code = code
            obj._name = name
            return obj

    @property
    def code(self):
        """ISO-4217 currency code for the currency.

        Returns
        -------
        code : str
        """
        return self._code

    @property
    def name(self):
        """Plain english name for the currency.

        Returns
        -------
        name : str
        """
        return self._name

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
