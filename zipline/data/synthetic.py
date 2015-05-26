"""
Synthetic data loaders for testing.
"""
from abc import abstractmethod

from numpy import (
    arange,
    empty,
    full,
)

from zipline.data.adjusted_array import adjusted_array
from zipline.data.baseloader import DataLoader


class SyntheticDataLoader(DataLoader):
    """
    DataLoader subclass that builds synthetic data based only on the shape of
    the desired output.  Keeps a log of all calls to load_columns() for use in
    testing.

    Subclasses should implement the following methods:

    make_column(dtype: np.dtype, nrows: int, ncols: int, idx: int) -> ndarray
    """
    def __init__(self, known_assets, adjustments):
        """
        Params
        ------
        known_assets: dict from dt -> list of sids.
        """
        self._log = []
        self._known_assets = known_assets
        self._adjustments = {}

    def load_adjusted_array(self, columns, dates, assets, lookback):
        """
        Load each column with self.make_column.
        """
        self._log.append((columns, dates, assets, lookback))

        nrows = len(dates)
        ncols = len(assets)
        return [
            adjusted_array(
                self.make_baseline(col.dtype, nrows, ncols),
                self._adjustments,
            )
            self.make_column(col.dtype, nrows, ncols, lookback)
            for col in columns
        ]

    @abstractmethod
    def make_baseline(self, dtype, nrows, ncols):
        """
        Returns an ndarray of dtype dtype and shape (nrows, ncols).

        idx is incremented and passed for each unique field loaded.
        """
        pass


class ConstantLoader(SyntheticDataLoader):
    """
    SyntheticDataLoader that returns a constant value for each sid/column.
    """

    def __init__(self, n, known_assets, adjustments):
        super(ConstantLoader, self).__init__(
            known_assets=known_assets,
            adjustments=adjustments,
        )
        self.n = n

    def make_column(self, dtype, nrows, ncols, lookback):
        baseline = full((nrows, ncols), self.n, dtype=dtype)
        return adjusted_array(baseline, self._adjustments).traverse(lookback)


class ARangeLoader(SyntheticDataLoader):
    """
    SyntheticDataLoader that returns np.aranges.
    """

    def make_column(self, dtype, nrows, ncols, idx):
        buf = empty(
            (nrows, ncols),
            dtype=dtype,
        )
        buf[:] = arange(1, ncols + 1)
        return buf
