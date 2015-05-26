"""
Base class for FFC data loaders.
"""
from abc import (
    ABCMeta,
    abstractmethod,
)


from six import with_metaclass


class DataLoader(with_metaclass(ABCMeta)):

    @abstractmethod
    def load_adjusted_array(self, columns, dates, assets, lookback):
        pass
