import abc
from numpy import vectorize
from functools import partial
import pandas as pd
from six import with_metaclass
from collections import namedtuple
from itertools import groupby

from zipline.utils.enum import enum
from zipline.utils.numpy_utils import vectorized_is_element
from zipline.assets import Asset


Restriction = namedtuple(
    'Restriction', ['asset', 'effective_date', 'state']
)


RESTRICTION_STATES = enum(
    'ALLOWED',
    'FROZEN',
)


class Restrictions(with_metaclass(abc.ABCMeta)):
    """
    Abstract restricted list interface
    """

    @abc.abstractmethod
    def is_restricted(self, assets, dt):
        """
        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?

        Parameters
        ----------
        asset : Asset of iterable of Assets
            The asset(s) for which we are querying a restriction
        dt : pd.Timestamp
            The timestamp of the restriction query

        Returns
        -------
        is_restricted : bool or pd.Series[bool] indexed by asset
            Is the asset or assets restricted on this dt?

        """
        raise NotImplementedError('is_restricted')


class NoopRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions
    """
    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return False
        return pd.Series(index=pd.Index(assets), data=[False]*len(assets))


class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset

    Parameters
    ----------
    restricted_list : iterable of assets
        The assets to be restricted
    """

    def __init__(self, restricted_list):
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        """
        An asset is restricted for all dts if it is in the static list
        """
        if isinstance(assets, Asset):
            return assets in self._restricted_set
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, self._restricted_set)
        )


class HistoricalRestrictions(Restrictions):
    """
    Historical restrictions stored in memory with effective dates for each
    asset

    Parameters
    ----------
    restrictions : iterable of namedtuple Restriction
        The restrictions, each defined by an asset, effective date and state
    """

    def __init__(self, restrictions):
        # A dict mapping each asset to its restrictions, which are sorted by
        # ascending order of effective_date
        self._restrictions_by_asset = {
            asset: sorted(
                restrictions_for_asset, key=lambda x: x.effective_date
            )
            for asset, restrictions_for_asset
            in groupby(restrictions, lambda x: x.asset)
        }

    def is_restricted(self, assets, dt):
        """
        Returns whether or not an asset or iterable of assets is restricted
        on a dt
        """
        if isinstance(assets, Asset):
            return self._is_restricted_for_asset(assets, dt)

        is_restricted = partial(self._is_restricted_for_asset, dt=dt)
        return pd.Series(
            index=pd.Index(assets),
            data=vectorize(is_restricted, otypes=[bool])(assets)
        )

    def _is_restricted_for_asset(self, asset, dt):
        state = RESTRICTION_STATES.ALLOWED
        for r in self._restrictions_by_asset.get(asset, ()):
            if r.effective_date > dt:
                break
            state = r.state
        return state == RESTRICTION_STATES.FROZEN


class SecurityListRestrictions(Restrictions):
    """
    Restrictions based on a security list

    Parameters
    ----------
    restrictions : zipline.utils.security_list.SecurityList
        The restrictions defined by a SecurityList
    """

    def __init__(self, security_list_by_dt):
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets, dt):
        securities_in_list = self.current_securities(dt)
        if isinstance(assets, Asset):
            return assets in securities_in_list
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, securities_in_list)
        )
