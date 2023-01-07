from zipline.utils.calendar_utils import get_calendar


class ExchangeInfo:
    """An exchange where assets are traded.

    Parameters
    ----------
    name : str or None
        The full name of the exchange, for example 'NEW YORK STOCK EXCHANGE' or
        'NASDAQ GLOBAL MARKET'.
    canonical_name : str
        The canonical name of the exchange, for example 'NYSE' or 'NASDAQ'. If
        None this will be the same as the name.
    country_code : str
        The country code where the exchange is located.

    Attributes
    ----------
    name : str or None
        The full name of the exchange, for example 'NEW YORK STOCK EXCHANGE' or
        'NASDAQ GLOBAL MARKET'.
    canonical_name : str
        The canonical name of the exchange, for example 'NYSE' or 'NASDAQ'. If
        None this will be the same as the name.
    country_code : str
        The country code where the exchange is located.
    calendar : TradingCalendar
        The trading calendar the exchange uses.
    """

    def __init__(self, name, canonical_name, country_code):
        self.name = name

        if canonical_name is None:
            canonical_name = name

        self.canonical_name = canonical_name
        self.country_code = country_code.upper()

    def __repr__(self):
        return "%s(%r, %r, %r)" % (
            type(self).__name__,
            self.name,
            self.canonical_name,
            self.country_code,
        )

    @property
    def calendar(self):
        """The trading calendar that this exchange uses."""
        return get_calendar(self.canonical_name)

    def __eq__(self, other):
        if not isinstance(other, ExchangeInfo):
            return NotImplemented

        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ("name", "canonical_name", "country_code")
        )

    def __ne__(self, other):
        eq = self == other
        if eq is NotImplemented:
            return NotImplemented
        return not eq
