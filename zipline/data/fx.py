"""Interface and definitions for foreign exchange rate readers.
"""
import six

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
        if six.PY3:
            # DataFrames in self._data contain str as column keys, which don't
            # compare equal to numpy bytes objects in Python 3. Convert to
            # unicode to make comparisons work as expected.
            cols = bases.astype('U3')
        else:
            # In py2, just use bases unchanged.
            cols = bases

        out = self._data[field][quote][cols].reindex(dates, method='ffill')

        # Ensure that result columns are the original input bases, even in py3.
        out.columns = bases

        return out


class ExplodingFXRateReader(implements(FXRateReader)):
    """An FXRateReader that raises an error when used.

    This is useful for testing contexts where FX rates aren't actually needed.
    """

    def get_rates(self, field, quote, bases, dates):
        raise AssertionError("FX rates requested unexpectedly!")
