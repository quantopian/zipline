"""Interface and definitions for foreign exchange rate readers."""
from interface import implements
import numpy as np

from .base import FXRateReader, DEFAULT_FX_RATE
from .utils import check_dts


class InMemoryFXRateReader(implements(FXRateReader)):
    """A simple in-memory FXRateReader.

    This is primarily used for testing.

    Parameters
    ----------
    data : dict
        Nested map from rate name -> quote currency -> pd.DataFrame
        Leaf frames should be indexed by (dates, base currencies).
    default_rate : str
        Rate to use when ``get_rates`` is called with a rate of 'default'.
    """

    def __init__(self, data, default_rate):
        self._data = data
        self._default_rate = default_rate

    def get_rates(self, rate, quote, bases, dts):
        """Get rates to convert ``bases`` into ``quote``.

        See :class:`zipline.data.fx.base.FXRateReader` for details.
        """
        if rate == DEFAULT_FX_RATE:
            rate = self._default_rate

        df = self._data[rate][quote]

        check_dts(dts)

        # Get raw values out of the frame.
        #
        # Logically, the operation here is:
        #
        # (df
        #  .reindex(dts, side='right')
        #  .reindex_axis(cols, axis='columns')
        #  .values)
        #
        # But pandas' performance on the above is not great, and we call this
        # method a lot, so we implement our own indexing logic.

        values = df.values
        row_ixs = df.index.searchsorted(dts.tz_localize(None), side="right") - 1
        col_ixs = df.columns.get_indexer(bases)

        out = values[:, col_ixs][row_ixs]

        # Handle dates before start and unknown bases.
        out[row_ixs == -1] = np.nan
        out[:, col_ixs == -1] = np.nan

        return out
