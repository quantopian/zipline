from pandas import Timestamp, Timedelta
from pandas.tseries.tools import normalize_date


class FutureChain(object):
    """ Allows users to look up future contracts.

    Parameters
    ----------
    asset_finder : AssetFinder
        An AssetFinder for future contract lookups, in particular the
        AssetFinder of the TradingAlgorithm instance.
    get_datetime : function
        A function that returns the simulation datetime, in particular
        the get_datetime method of the TradingAlgorithm instance.
    root_symbol : str
        The root symbol of a future chain.
    as_of_date : pandas.Timestamp, optional
        Date at which the chain determination is rooted. I.e. the
        existing contract whose notice date is first after this date is
        the primary contract, etc. If not provided, the current
        simulation date is used as the as_of_date.

    Attributes
    ----------
    root_symbol : str
        The root symbol of the future chain.
    as_of_date
        The current as-of date of this future chain.

    Methods
    -------
    as_of(dt)
    offset(time_delta)

    Raises
    ------
    RootSymbolNotFound
        Raised when the FutureChain is initialized with a root symbol for which
        a future chain could not be found.
    """
    def __init__(self, asset_finder, get_datetime, root_symbol,
                 as_of_date=None):
        self.root_symbol = root_symbol

        # Reference to the algo's AssetFinder for contract lookups
        self._asset_finder = asset_finder
        # Reference to the algo's get_datetime to know the current dt
        self._algorithm_get_datetime = get_datetime

        # If an as_of_date is provided, self._as_of_date uses that
        # value, otherwise None. This attribute backs the as_of_date property.
        if as_of_date:
            self._as_of_date = normalize_date(as_of_date)
        else:
            self._as_of_date = None

        # Attribute to cache the most up-to-date chain, and the dt when it was
        # last updated.
        self._current_chain = []
        self._last_updated = None

        # Get the initial chain, since self._last_updated is None.
        self._maybe_update_current_chain()

    def __repr__(self):
        # NOTE: The string returned cannot be used to instantiate this
        # exact FutureChain, since we don't want to display the asset
        # finder and get_datetime function to the user.
        if self._as_of_date:
            return "FutureChain(root_symbol='%s', as_of_date='%s')" % (
                self.root_symbol, self.as_of_date)
        else:
            return "FutureChain(root_symbol='%s')" % self.root_symbol

    def _get_datetime(self):
        """
        Returns the normalized simulation datetime.

        Returns
        -------
        pandas.Timestamp
            The normalized datetime of FutureChain's TradingAlgorithm.
        """
        return normalize_date(
            Timestamp(self._algorithm_get_datetime(), tz='UTC')
        )

    @property
    def as_of_date(self):
        """
        The current as-of date of this future chain.

        Returns
        -------
        pandas.Timestamp
            The user-provided as_of_date if given, otherwise the
            current datetime of the simulation.
        """
        if self._as_of_date is not None:
            return self._as_of_date
        else:
            return self._get_datetime()

    def _maybe_update_current_chain(self):
        """ Updates the current chain if it's out of date, then returns
            it.

            Returns
            -------
            list
                The up-to-date current chain, a list of Future objects.
        """
        dt = self._get_datetime()

        if (self._last_updated is None) or (self._last_updated != dt):
            self._current_chain = self._asset_finder.lookup_future_chain(
                self.root_symbol,
                self.as_of_date,
                dt
            )
            self._last_updated = dt

        return self._current_chain

    def __getitem__(self, key):
        return self._maybe_update_current_chain()[key]

    def __len__(self):
        return len(self._maybe_update_current_chain())

    def __iter__(self):
        return iter(self._maybe_update_current_chain())

    def as_of(self, dt):
        """ Get the future chain for this root symbol as of a specific date.

        Parameters
        ----------
        dt : datetime.datetime or pandas.Timestamp or str, optional
            The as_of_date for the new chain.

        Returns
        -------
        FutureChain

        """
        return FutureChain(
            asset_finder=self._asset_finder,
            get_datetime=self._algorithm_get_datetime,
            root_symbol=self.root_symbol,
            as_of_date=dt
        )

    def offset(self, time_delta):
        """ Get the future chain for this root symbol with a given
        offset from the current as_of_date.

        Parameters
        ----------
        time_delta : datetime.timedelta or pandas.Timedelta or str
            The offset from the current as_of_date for the new chain.

        Returns
        -------
        FutureChain

        """
        return self.as_of(self.as_of_date + Timedelta(time_delta))
