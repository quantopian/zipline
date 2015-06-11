"""
Synthetic data loaders for testing.
"""
from abc import abstractmethod

from numpy import (
    full,
)

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)
from zipline.data.ffc.base import FFCLoader


class SyntheticDataLoader(FFCLoader):
    """
    DataLoader subclass that builds synthetic data based only on the shape of
    the desired output.  Keeps a log of all calls to load_columns() for use in
    testing.
    """
    def __init__(self, known_assets, adjustments):
        """
        known_assets: #TODO: Explain
        adjustments: #TODO:  Explain
        """
        self._log = []
        self._known_assets = known_assets
        self._adjustments = {}

    def _adjustments_for_dates(self, column, dates):
        adjustments = self._adjustments.get(column, {})
        out = {}
        for idx, dt in enumerate(dates):
            adjustments_for_dt = adjustments.get(dt, None)
            if adjustments_for_dt is not None:
                out[dt] = adjustments_for_dt
        return out

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
                self._adjustments_for_dates(col, dates),
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
