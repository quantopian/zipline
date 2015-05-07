"""
Base class for FFC data loaders.
"""
from abc import (
    ABCMeta,
    abstractmethod,
)


from six import with_metaclass


class FFCLoader(with_metaclass(ABCMeta)):
    """
    ABC for classes that can load data for use with zipline.modelling pipeline.

    TODO: DOCUMENT THIS MORE!
    """
    @abstractmethod
    def load_adjusted_array(self, columns, mask):
        pass
