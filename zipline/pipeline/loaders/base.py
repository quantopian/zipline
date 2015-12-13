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

    Methods
    -------
    load_columns
    """
    @abstractmethod
    def load_columns(self, columns, dates, sids, mask):
        """
        Load data for the requested inputs.

        Parameters
        ----------
        columns : list[BoundColumn]
            The columns for which data has been requested.
        dates : pd.DatetimeIndex
            The date labels for the requested data.
        sids : pd.Int64Index
            Asset IDs for the requested data.
        mask : np.array[bool, ndim=2]
            Array of shape (len(dates), len(sids)) containing boolean values
            indicating whether each asset existed on each date.

        Returns
        -------
        loaded : dict[BoundColumn -> AdjustedArray]
            A dict containing an AdjustedArray for each requested column.
        """
        raise NotImplementedError('load_columns')
