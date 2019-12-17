"""Interface and definitions for foreign exchange rate readers.
"""
import six

from interface import implements

from .base import FXRateReader


class InMemoryFXRateReader(implements(FXRateReader)):
    """
    A simple in-memory FXRateReader.

    This is primarily used for testing.

    Parameters
    ----------
    data : dict
        Nested map from rate name -> quote currency -> pd.DataFrame
        Leaf frames should be indexed by (dates, base currencies).
    """

    def __init__(self, data):
        self._data = data

    def get_rates(self, rate, quote, bases, dts):
        """Get rates to convert ``bases`` into ``quote``.
        """
        if six.PY3:
            # DataFrames in self._data contain str as column keys, which don't
            # compare equal to numpy bytes objects in Python 3. Convert to
            # unicode to make comparisons work as expected.
            cols = bases.astype('U3')
        else:
            # In py2, just use bases unchanged.
            cols = bases

        df = self._data[rate][quote]
        self._check_dts(df.index, dts)

        out = df.loc[:, cols].reindex(dts, method='ffill')

        # Ensure that result columns are the original input bases, even in py3.
        out.columns = bases

        return out

    def _check_dts(self, stored, requested):
        """Validate that requested dates are in bounds for what we have stored.
        """
        request_start, request_end = requested[[0, -1]]
        data_start, data_end = stored[[0, -1]]

        if request_start < data_start:
            raise ValueError(
                "Requested fx rates starting at {}, but data starts at {}"
                .format(request_start, data_start)
            )

        if request_end > data_end:
            raise ValueError(
                "Requested fx rates ending at {}, but data ends at {}"
                .format(request_end, data_end)
            )
