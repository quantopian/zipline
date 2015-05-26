"""
Synthetic data loaders for testing.
"""
from abc import abstractmethod

from numpy import (
    arange,
    empty,
    full,
)

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)
from zipline.data.baseloader import DataLoader


class SyntheticDataLoader(DataLoader):
    """
    DataLoader subclass that builds synthetic data based only on the shape of
    the desired output.  Keeps a log of all calls to load_columns() for use in
    testing.
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

    def load_adjusted_array(self, columns, dates, assets):
        """
        Load each column with self.make_column.
        """
        self._log.append((columns, dates, assets))

        nrows = len(dates)
        ncols = len(assets)
        return [
            adjusted_array(
                self.make_baseline(col, nrows, ncols),
                NOMASK,
                self._adjustments,
            )
            for col in columns
        ]

    @abstractmethod
    def make_baseline(self, column, nrows, ncols):
        """
        Returns an ndarray of shape nrows, ncols for the given column.
        """
        pass


class ConstantLoader(SyntheticDataLoader):
    """
    SyntheticDataLoader that returns a constant value for each column.
    """

    def __init__(self, known_assets, adjustments, constants):
        super(ConstantLoader, self).__init__(
            known_assets=known_assets,
            adjustments=adjustments,
        )
        self._constants = constants

    def make_baseline(self, column, nrows, ncols):
        return full(
            (nrows, ncols),
            self._constants[column],
            dtype=column.dtype,
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
