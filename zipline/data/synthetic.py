"""
Synthetic data loaders for testing.
"""
from abc import abstractmethod

from numpy import (
    arange,
    empty,
    full,
)

from zipline.data.baseloader import DataLoader


class SyntheticDataLoader(DataLoader):
    """
    DataLoader subclass that builds synthetic data based only on the shape of
    the desired output.  Keeps a log of all calls to load_columns() for use in
    testing.

    Subclasses should implement the following methods:

    make_column(dtype: np.dtype, nrows: int, ncols: int, idx: int) -> ndarray
    """
    def __init__(self):
        self._log = []

    def load_chunk(self, columns, assets, dates):
        """
        Load each column with self.make_column.
        """
        self._log.append((columns, assets, dates))

        nrows = len(assets)
        ncols = len(dates)
        return [
            self.make_column(col.dtype, nrows, ncols, idx)
            for idx, col in enumerate(columns)
        ]

    @abstractmethod
    def make_column(self, dtype, nrows, ncols, idx):
        """
        Returns an ndarray of dtype dtype and shape (nrows, ncols).

        idx is incremented and passed for each unique field loaded.
        """
        pass


class ConstantLoader(SyntheticDataLoader):
    """
    SyntheticDataLoader that returns a constant value for each sid/column.
    """

    def __init__(self, n):
        super(ConstantLoader, self).__init__()
        self.n = n

    def make_column(self, dtype, nrows, ncols, idx):
        return full(
            (nrows, ncols),
            self.n,
            dtype=dtype,
        )


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
