"""
"""

from interface import implements, Interface


class FXRateReader(Interface):

    def get_rates(self, field, quote, bases, dates):
        """
        Get rates to convert ``bases`` into ``quote``.

        Parameters
        ----------
        field : str
            Currency field to load.
        quote : str
            Currency code of the currency into which we want to convert.
        bases : np.array[S3]
            Array of codes of the currencies from which we want to convert. A
            single currency may appear multiple times.
        dates : pd.DatetimeIndex
            Dates for which to load currencies.

        Returns
        -------
        rates : pd.DataFrame
            DataFrame indexed by (dates, bases) containing exchange rates
            mapping from base -> quote currency.
        """


class InMemoryFXRateReader(implements(FXRateReader)):
    """
    A simple in-memory FXRateReader.

    This is primarily used for testing.

    Parameters
    ----------
    data : dict
        Nested map from field name -> quote currency -> pd.DataFrame
        Leaf frames should be indexed by (dates, base currencies).
    """

    def __init__(self, data):
        self._data = data

    def get_rates(self, field, quote, bases, dates):
        return self._data[field][quote][bases].reindex(dates, method='ffill')


class ExplodingFXRateReader(implements(FXRateReader)):
    """An FXRateReader that raises an error when used.

    This is useful for testing contexts where FX rates aren't actually needed.
    """

    def get_rates(self, field, quote, bases, dates):
        raise AssertionError("FX rates requested unexpectedly!")
