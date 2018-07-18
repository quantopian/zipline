from abc import ABCMeta, abstractmethod
from six import with_metaclass

from zipline.extensions import extensible


@extensible
class DataPortal(with_metaclass(ABCMeta)):

    @abstractmethod
    def _reindex_extra_source(self, df, source_date_index):

        raise NotImplementedError('_reindex_extra_source')

    @abstractmethod
    def get_last_traded_dt(self, asset, dt, data_frequency):
        """
        Given an asset and dt, returns the last traded dt from the viewpoint
        of the given dt.

        If there is a trade on the dt, the answer is dt provided.
        """

        raise NotImplementedError('get_last_traded_dt')

    @abstractmethod
    def get_scalar_asset_spot_value(self, asset, field, dt, data_frequency):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ----------
        assets : Asset
            The asset or assets whose data is desired. This cannot be
            an arbitrary AssetConvertible.
        field : {'open', 'high', 'low', 'close', 'volume',
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        value : float, int, or pd.Timestamp
            The spot value of ``field`` for ``asset`` The return type is based
            on the ``field`` requested. If the field is one of 'open', 'high',
            'low', 'close', or 'price', the value will be a float. If the
            ``field`` is 'volume' the value will be a int. If the ``field`` is
            'last_traded' the value will be a Timestamp.
        """

        raise NotImplementedError('get_scalar_asset_spot_value')

    @abstractmethod
    def get_spot_value(self, assets, field, dt, data_frequency):
        """
        Public API method that returns a scalar value representing the value
        of the desired asset's field at either the given dt.

        Parameters
        ----------
        assets : Asset, ContinuousFuture, or iterable of same.
            The asset or assets whose data is desired.
        field : {'open', 'high', 'low', 'close', 'volume',
                 'price', 'last_traded'}
            The desired field of the asset.
        dt : pd.Timestamp
            The timestamp for the desired value.
        data_frequency : str
            The frequency of the data to query; i.e. whether the data is
            'daily' or 'minute' bars

        Returns
        -------
        value : float, int, or pd.Timestamp
            The spot value of ``field`` for ``asset`` The return type is based
            on the ``field`` requested. If the field is one of 'open', 'high',
            'low', 'close', or 'price', the value will be a float. If the
            ``field`` is 'volume' the value will be a int. If the ``field`` is
            'last_traded' the value will be a Timestamp.
        """

        raise NotImplementedError('get_spot_value')

    @abstractmethod
    def _get_minute_window_data(self, assets, field, minutes_for_window):
        """
        Internal method that gets a window of adjusted minute data for an asset
        and specified date range.  Used to support the history API method for
        minute bars.

        Missing bars are filled with NaN.

        Parameters
        ----------
        assets : iterable[Asset]
            The assets whose data is desired.

        field: string
            The specific field to return.  "open", "high", "close_price", etc.

        minutes_for_window: pd.DateTimeIndex
            The list of minutes representing the desired window.  Each minute
            is a pd.Timestamp.

        Returns
        -------
        A numpy array with requested values.
        """

        raise NotImplementedError('_get_minute_window_data')

    @abstractmethod
    def _get_history_daily_window_data(self,
                                       assets,
                                       days_for_window,
                                       end_dt,
                                       field_to_use,
                                       data_frequency):

        raise NotImplementedError('_get_history_daily_window_data')
