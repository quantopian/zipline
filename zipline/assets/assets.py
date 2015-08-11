# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta
from functools import partial
from numbers import Integral
from operator import getitem, itemgetter
import warnings

from logbook import Logger
import numpy as np
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types
import sqlalchemy as sa
from toolz import compose

from zipline.errors import (
    MultipleSymbolsFound,
    RootSymbolNotFound,
    SidNotFound,
    SymbolNotFound,
    MapAssetIdentifierIndexError,
)
from zipline.assets import (
    Asset, Equity, Future,
)
from zipline.assets.asset_writer import (
    FUTURE_TABLE_FIELDS,
    EQUITY_TABLE_FIELDS,
)

log = Logger('assets.py')


class AssetFinder(object):

    def __init__(self, engine, allow_sid_assignment=True, fuzzy_char=None):

        self.fuzzy_char = fuzzy_char
        self.allow_sid_assignment = allow_sid_assignment

        self.engine = engine
        metadata = sa.MetaData(bind=engine)
        self.equities = equities = sa.Table(
            'equities',
            metadata,
            autoload_with=engine,
        )
        self.futures_exchanges = sa.Table(
            'futures_exchanges',
            metadata,
            autoload_with=engine,
        )
        self.futures_root_symbols = sa.Table(
            'futures_root_symbols',
            metadata,
            autoload_with=engine,
        )
        self.futures_contracts = futures_contracts = sa.Table(
            'futures_contracts',
            metadata,
            autoload_with=engine,
        )
        self.asset_router = sa.Table(
            'asset_router',
            metadata,
            autoload_with=engine,
        )

        # Create the equity and future queries once.
        _equity_sid = equities.c.sid
        _equity_by_sid = sa.select(
            tuple(map(partial(getitem, equities.c), EQUITY_TABLE_FIELDS)),
        )

        def select_equity_by_sid(sid):
            return _equity_by_sid.where(_equity_sid == int(sid))

        self.select_equity_by_sid = select_equity_by_sid

        _future_sid = futures_contracts.c.sid
        _future_by_sid = sa.select(
            tuple(map(
                partial(getitem, futures_contracts.c),
                FUTURE_TABLE_FIELDS,
            )),
        )

        def select_future_by_sid(sid):
            return _future_by_sid.where(_future_sid == int(sid))

        self.select_future_by_sid = select_future_by_sid
        # Cache for lookup of assets by sid, the objects in the asset lookp may
        # be shared with the results from equity and future lookup caches.
        #
        # The top level cache exists to minimize lookups on the asset type
        # routing.
        #
        # The caches are read through, i.e. accessing an asset through
        # retrieve_asset, _retrieve_equity etc. will populate the cache on
        # first retrieval.
        self._asset_cache = {}
        self._equity_cache = {}
        self._future_cache = {}

        self._asset_type_cache = {}

        # Populated on first call to `lifetimes`.
        self._asset_lifetimes = None

    def asset_type_by_sid(self, sid):
        """
        Retrieve the asset type of a given sid.
        """
        try:
            return self._asset_type_cache[sid]
        except KeyError:
            pass

        asset_type = sa.select((self.asset_router.c.asset_type,)).where(
            self.asset_router.c.sid == int(sid),
        ).scalar()

        if asset_type is not None:
            self._asset_type_cache[sid] = asset_type
        return asset_type

    def retrieve_asset(self, sid, default_none=False):
        """
        Retrieve the Asset object of a given sid.
        """
        if isinstance(sid, Asset):
            return sid

        try:
            asset = self._asset_cache[sid]
        except KeyError:
            asset_type = self.asset_type_by_sid(sid)
            if asset_type == 'equity':
                asset = self._retrieve_equity(sid)
            elif asset_type == 'future':
                asset = self._retrieve_futures_contract(sid)
            else:
                asset = None

            self._asset_cache[sid] = asset

        if asset is not None:
            return asset
        elif default_none:
            return None
        else:
            raise SidNotFound(sid=sid)

    def _retrieve_equity(self, sid):
        """
        Retrieve the Equity object of a given sid.
        """
        try:
            return self._equity_cache[sid]
        except KeyError:
            pass

        data = self.select_equity_by_sid(sid).execute().fetchone()
        # Convert 'data' from a RowProxy object to a dict, to allow assignment
        data = dict(data.items())
        if data:
            if data['start_date']:
                data['start_date'] = pd.Timestamp(data['start_date'], tz='UTC')

            if data['end_date']:
                data['end_date'] = pd.Timestamp(data['end_date'], tz='UTC')

            if data['first_traded']:
                data['first_traded'] = pd.Timestamp(
                    data['first_traded'], tz='UTC')

            equity = Equity(**data)
        else:
            equity = None

        self._equity_cache[sid] = equity
        return equity

    def _retrieve_futures_contract(self, sid):
        """
        Retrieve the Future object of a given sid.
        """
        try:
            return self._future_cache[sid]
        except KeyError:
            pass

        data = self.select_future_by_sid(sid).execute().fetchone()
        # Convert 'data' from a RowProxy object to a dict, to allow assignment
        data = dict(data.items())
        if data:
            if data['start_date']:
                data['start_date'] = pd.Timestamp(data['start_date'], tz='UTC')

            if data['end_date']:
                data['end_date'] = pd.Timestamp(data['end_date'], tz='UTC')

            if data['first_traded']:
                data['first_traded'] = pd.Timestamp(
                    data['first_traded'], tz='UTC')

            if data['notice_date']:
                data['notice_date'] = pd.Timestamp(
                    data['notice_date'], tz='UTC')

            if data['expiration_date']:
                data['expiration_date'] = pd.Timestamp(
                    data['expiration_date'], tz='UTC')

            future = Future(**data)
        else:
            future = None

        self._future_cache[sid] = future
        return future

    def lookup_symbol_resolve_multiple(self, symbol, as_of_date=None):
        """
        Return matching Asset of name symbol in database.

        If multiple Assets are found and as_of_date is not set,
        raises MultipleSymbolsFound.

        If no Asset was active at as_of_date raises SymbolNotFound.
        """
        if as_of_date is not None:
            as_of_date = pd.Timestamp(normalize_date(as_of_date))

        equities_cols = self.equities.c
        if as_of_date:
            ad_value = as_of_date.value

            # If one SID exists for symbol, return that symbol
            candidates = sa.select((equities_cols.sid,)).where(
                (equities_cols.symbol == symbol) &
                (equities_cols.start_date <= ad_value) &
                (equities_cols.end_date >= ad_value),
            ).execute().fetchall()
            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0]['sid'])

            # If no SID exists for symbol, return SID with the
            # highest-but-not-over end_date
            elif not candidates:
                sid = sa.select((equities_cols.sid,)).where(
                    (equities_cols.symbol == symbol) &
                    (equities_cols.start_date <= ad_value),
                ).order_by(
                    equities_cols.end_date.desc(),
                ).scalar()
                if sid:
                    return self._retrieve_equity(sid)

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            elif len(candidates) > 1:
                sid = sa.select((equities_cols.sid,)).where(
                    (equities_cols.symbol == symbol) &
                    (equities_cols.start_date <= ad_value),
                ).order_by(
                    equities_cols.start_date.desc(),
                    equities_cols.end_date.desc(),
                ).scalar()
                if sid:
                    return self._retrieve_equity(sid)

            raise SymbolNotFound(symbol=symbol)

        else:
            sids = sa.select((equities_cols.sid,)).where(
                equities_cols.symbol == sid,
            ).execute().fetchall()
            if len(sids) == 1:
                return self._retrieve_equity(sids[0]['sid'])
            elif not sids:
                raise SymbolNotFound(symbol=symbol)
            else:
                raise MultipleSymbolsFound(
                    symbol=symbol,
                    options=list(map(
                        compose(self._retrieve_equity, itemgetter('sid')),
                        sids,
                    ))
                )

    def lookup_symbol(self, symbol, as_of_date, fuzzy=None):
        """
        If a fuzzy string is provided, then we try various symbols based on
        the provided symbol.  This is to facilitate mapping from a broker's
        symbol to ours in cases where mapping to the broker's symbol loses
        information. For example, if we have CMCS_A, but a broker has CMCSA,
        when the broker provides CMCSA, it can also provide fuzzy='_',
        so we can find a match by inserting an underscore.
        """
        symbol = symbol.upper()
        ad_value = normalize_date(as_of_date).value

        if fuzzy is None:
            try:
                return self.lookup_symbol_resolve_multiple(symbol, as_of_date)
            except SymbolNotFound:
                return None

        equities_cols = self.equities.c
        candidates = sa.select((equities_cols.sid,)).where(
            (equities_cols.fuzzy == fuzzy) &
            (equities_cols.start_date <= ad_value) &
            (equities_cols.end_date >= ad_value),
        ).execute().fetchall()

        # If one SID exists for symbol, return that symbol
        if len(candidates) == 1:
            return self._retrieve_equity(candidates[0]['sid'])

        # If multiple SIDs exist for symbol, return latest start_date with
        # end_date as a tie-breaker
        elif candidates:
            sid = sa.select((equities_cols.sid,)).where(
                (equities_cols.symbol == symbol) &
                (equities_cols.start_date <= ad_value),
            ).order_by(
                equities_cols.start_date.desc(),
                equities_cols.end_date.desc(),
            ).scalar()
            if sid:
                return self._retrieve_equity(sid)

        raise SymbolNotFound(symbol=symbol)

    def lookup_future_chain(self, root_symbol, as_of_date, knowledge_date):
        """ Return the futures chain for a given root symbol.

        Parameters
        ----------
        root_symbol : str
            Root symbol of the desired future.
        as_of_date : pd.Timestamp or pd.NaT
            Date at which the chain determination is rooted. I.e. the
            existing contract whose notice date is first after this
            date is the primary contract, etc. If NaT is given, the
            chain is unbounded, and all contracts for this root symbol
            are returned.
        knowledge_date : pd.Timestamp or pd.NaT
            Date for determining which contracts exist for inclusion in
            this chain. Contracts exist only if they have a start_date
            on or before this date. If NaT is given and as_of_date is
            is not NaT, the value of as_of_date is used for
            knowledge_date.

        Returns
        -------
        list
            A list of Future objects, the chain for the given
            parameters.

        Raises
        ------
        RootSymbolNotFound
            Raised when a future chain could not be found for the given
            root symbol.
        """

        fc_cols = self.futures_contracts.c

        if as_of_date is pd.NaT:
            # If the as_of_date is NaT, get all contracts for this
            # root symbol.
            sids = list(map(
                itemgetter('sid'),
                sa.select((fc_cols.sid,)).where(
                    (fc_cols.root_symbol == root_symbol),
                ).order_by(
                    fc_cols.notice_date.asc(),
                ).execute().fetchall()))
        else:
            as_of_date = as_of_date.value
            if knowledge_date is pd.NaT:
                # If knowledge_date is NaT, default to using as_of_date
                knowledge_date = as_of_date.value
            else:
                knowledge_date = knowledge_date.value

        sids = list(map(
            itemgetter('sid'),
            sa.select((fc_cols.sid,)).where(
                (fc_cols.root_symbol == root_symbol) &
                (fc_cols.notice_date >= as_of_date) &
                (fc_cols.start_date <= knowledge_date),
            ).order_by(
                fc_cols.notice_date.asc(),
            ).execute().fetchall()
        ))

        if not sids:
            # Check if root symbol exists.
            count = sa.select((sa.func.count(fc_cols.sid),)).where(
                fc_cols.root_symbol == root_symbol,
            ).scalar()
            if count == 0:
                raise RootSymbolNotFound(root_symbol=root_symbol)

        return map(self._retrieve_futures_contract, sids)

    @property
    def sids(self):
        return tuple(map(
            itemgetter('sid'),
            sa.select(self.asset_router.c.sid,).execute().fetchall(),
        ))

    def _lookup_generic_scalar(self,
                               asset_convertible,
                               as_of_date,
                               matches,
                               missing):
        """
        Convert asset_convertible to an asset.

        On success, append to matches.
        On failure, append to missing.
        """
        try:
            if isinstance(asset_convertible, Asset):
                matches.append(asset_convertible)

            elif isinstance(asset_convertible, Integral):
                result = self.retrieve_asset(int(asset_convertible))
                if result is None:
                    raise SymbolNotFound(symbol=asset_convertible)
                matches.append(result)

            elif isinstance(asset_convertible, string_types):
                # Throws SymbolNotFound on failure to match.
                matches.append(
                    self.lookup_symbol_resolve_multiple(
                        asset_convertible,
                        as_of_date,
                    )
                )
            else:
                raise NotAssetConvertible(
                    "Input was %s, not AssetConvertible."
                    % asset_convertible
                )

        except SymbolNotFound:
            missing.append(asset_convertible)
            return None

    def lookup_generic(self,
                       asset_convertible_or_iterable,
                       as_of_date):
        """
        Convert a AssetConvertible or iterable of AssetConvertibles into
        a list of Asset objects.

        This method exists primarily as a convenience for implementing
        user-facing APIs that can handle multiple kinds of input.  It should
        not be used for internal code where we already know the expected types
        of our inputs.

        Returns a pair of objects, the first of which is the result of the
        conversion, and the second of which is a list containing any values
        that couldn't be resolved.
        """
        matches = []
        missing = []

        # Interpret input as scalar.
        if isinstance(asset_convertible_or_iterable, AssetConvertible):
            self._lookup_generic_scalar(
                asset_convertible=asset_convertible_or_iterable,
                as_of_date=as_of_date,
                matches=matches,
                missing=missing,
            )
            try:
                return matches[0], missing
            except IndexError:
                if hasattr(asset_convertible_or_iterable, '__int__'):
                    raise SidNotFound(sid=asset_convertible_or_iterable)
                else:
                    raise SymbolNotFound(symbol=asset_convertible_or_iterable)

        # Interpret input as iterable.
        try:
            iterator = iter(asset_convertible_or_iterable)
        except TypeError:
            raise NotAssetConvertible(
                "Input was not a AssetConvertible "
                "or iterable of AssetConvertible."
            )

        for obj in iterator:
            self._lookup_generic_scalar(obj, as_of_date, matches, missing)
        return matches, missing

    def map_identifier_index_to_sids(self, index, as_of_date):
        """
        This method is for use in sanitizing a user's DataFrame or Panel
        inputs.

        Takes the given index of identifiers, checks their types, builds assets
        if necessary, and returns a list of the sids that correspond to the
        input index.

        Parameters
        ----------
        index : Iterable
            An iterable containing ints, strings, or Assets
        as_of_date : pandas.Timestamp
            A date to be used to resolve any dual-mapped symbols

        Returns
        -------
        List
            A list of integer sids corresponding to the input index
        """
        # This method assumes that the type of the objects in the index is
        # consistent and can, therefore, be taken from the first identifier
        first_identifier = index[0]

        # Ensure that input is AssetConvertible (integer, string, or Asset)
        if not isinstance(first_identifier, AssetConvertible):
            raise MapAssetIdentifierIndexError(obj=first_identifier)

        # If sids are provided, no mapping is necessary
        if isinstance(first_identifier, Integral):
            return index

        # If symbols or Assets are provided, construction and mapping is
        # necessary
        self.consume_identifiers(index)

        # Look up all Assets for mapping
        matches = []
        missing = []
        for identifier in index:
            self._lookup_generic_scalar(identifier, as_of_date,
                                        matches, missing)

        # Handle missing assets
        if len(missing) > 0:
            warnings.warn("Missing assets for identifiers: " + missing)

        # Return a list of the sids of the found assets
        return [asset.sid for asset in matches]

    def _compute_asset_lifetimes(self):
        """
        Compute and cache a recarry of asset lifetimes.
        """
        equities_cols = self.equities.c
        buf = np.array(
            tuple(map(
                float,
                sa.select((
                    equities_cols.sid,
                    equities_cols.start_date,
                    equities_cols.end_date,
                )).execute(),
            )),
            dype='<f8',  # use doubles so we get NaNs
        )
        lifetimes = np.recarray(
            buf=buf,
            shape=(len(buf),),
            dtype=[
                ('sid', '<f8'),
                ('start', '<f8'),
                ('end', '<f8')
            ],
        )
        start = lifetimes.start
        end = lifetimes.end
        start[np.isnan(start)] = 0  # convert missing starts to 0
        end[np.isnan(end)] = np.iinfo(int).max  # convert missing end to INTMAX
        # Cast the results back down to int.
        return lifetimes.astype([
            ('sid', '<i8'),
            ('start', '<i8'),
            ('end', '<i8'),
        ])

    def lifetimes(self, dates):
        """
        Compute a DataFrame representing asset lifetimes for the specified date
        range.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The dates for which to compute lifetimes.

        Returns
        -------
        lifetimes : pd.DataFrame
            A frame of dtype bool with `dates` as index and an Int64Index of
            assets as columns.  The value at `lifetimes.loc[date, asset]` will
            be True iff `asset` existed on `data`.

        See Also
        --------
        numpy.putmask
        """
        # This is a less than ideal place to do this, because if someone adds
        # assets to the finder after we've touched lifetimes we won't have
        # those new assets available.  Mutability is not my favorite
        # programming feature.
        if self._asset_lifetimes is None:
            self._asset_lifetimes = self._compute_asset_lifetimes()
        lifetimes = self._asset_lifetimes

        raw_dates = dates.asi8[:, None]
        mask = (lifetimes.start <= raw_dates) & (raw_dates <= lifetimes.end)
        return pd.DataFrame(mask, index=dates, columns=lifetimes.sid)


class AssetConvertible(with_metaclass(ABCMeta)):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Asset, six.string_types, and Integral
    """
    pass


AssetConvertible.register(Integral)
AssetConvertible.register(Asset)
# Use six.string_types for Python2/3 compatibility
for _type in string_types:
    AssetConvertible.register(_type)


class NotAssetConvertible(ValueError):
    pass
