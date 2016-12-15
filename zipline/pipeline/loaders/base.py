"""
Base class for Pipeline API data loaders.
"""
from abc import (
    ABCMeta,
    abstractmethod,
)


from six import with_metaclass


class PipelineLoader(with_metaclass(ABCMeta)):
    """
    ABC for classes that can load data for use with zipline.pipeline APIs.

    TODO: DOCUMENT THIS MORE!
    """
    @abstractmethod
    def load_adjusted_array(self, columns, dates, assets, mask):
        pass
