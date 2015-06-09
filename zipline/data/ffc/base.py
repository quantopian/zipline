"""
Base class for FFC data loaders.
"""
from abc import (
    ABCMeta,
    abstractmethod,
)


from six import with_metaclass


class FFCLoader(with_metaclass(ABCMeta)):

    @abstractmethod
    def load_adjusted_array(self, columns, dates, assets):
        pass
