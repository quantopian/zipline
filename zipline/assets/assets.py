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
from numbers import Integral
from operator import itemgetter
import warnings

from logbook import Logger
import numpy as np
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types, viewkeys
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
    split_delimited_symbol,
)

log = Logger('assets.py')

# A set of fields that need to be converted to strings before building an
# Asset to avoid unicode fields
_asset_str_fields = frozenset({
    'symbol',
    'asset_name',
    'exchange',
})

# A set of fields that need to be converted to timestamps in UTC
_asset_timestamp_fields = frozenset({
    'start_date',
    'end_date',
    'first_traded',
    'notice_date',
    'expiration_date',
    'auto_close_date',
})


def _convert_asset_timestamp_fields(dict):
    """
    Takes in a dict of Asset init args and converts dates to pd.Timestamps
    """
    for key in (_asset_timestamp_fields & viewkeys(dict)):
        value = pd.Timestamp(dict[key], tz='UTC')
        dict[key] = None if pd.isnull(value) else value


class AssetFinder(object):

    # Token used as a substitute for pickling objects that contain a
    # reference to an AssetFinder
    PERSISTENT_TOKEN = "<AssetFinder>"

    def __init__(self, engine):

        self.engine = engine
        metadata = sa.MetaData(bind=engine)

        table_names = ['equities', 'futures_exchanges', 'futures_root_symbols',
                       'futures_contracts', 'asset_router']
        metadata.reflect(only=table_names)
        for table_name in table_names:
            setattr(self, table_name, metadata.tables[table_name])

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

            # Cache the asset if it has been retrieved
            if asset is not None:
                self._asset_cache[sid] = asset

        if asset is not None:
            return asset
        elif default_none:
            return None
        else:
            raise SidNotFound(sid=sid)

    def retrieve_all(self, sids, default_none=False):
        return [self.retrieve_asset(sid, default_none) for sid in sids]

    def _retrieve_equity(self, sid):
        """
        Retrieve the Equity object of a given sid.
        """
        return self._retrieve_asset(
            sid, self._equity_cache, self.equities, Equity,
        )

    def _retrieve_futures_contract(self, sid):
        """
        Retrieve the Future object of a given sid.
        """
        return self._retrieve_asset(
            sid, self._future_cache, self.futures_contracts, Future,
        )

    @staticmethod
    def _select_asset_by_sid(asset_tbl, sid):
        return sa.select([asset_tbl]).where(asset_tbl.c.sid == int(sid))

    @staticmethod
    def _select_asset_by_symbol(asset_tbl, symbol):
        return sa.select([asset_tbl]).where(asset_tbl.c.symbol == symbol)

    def _retrieve_asset(self, sid, cache, asset_tbl, asset_type):
        try:
            return cache[sid]
        except KeyError:
            pass

        data = self._select_asset_by_sid(asset_tbl, sid).execute().fetchone()
        # Convert 'data' from a RowProxy object to a dict, to allow assignment
        data = dict(data.items())
        if data:
            _convert_asset_timestamp_fields(data)

            asset = asset_type(**data)
        else:
            asset = None

        cache[sid] = asset
        return asset

    def get_fuzzy_candidates(self, fuzzy_symbol):
        candidates = sa.select(
            (self.equities.c.sid,)
        ).where(self.equities.c.fuzzy_symbol == fuzzy_symbol).order_by(
            self.equities.c.start_date.desc(),
            self.equities.c.end_date.desc()
        ).execute().fetchall()
        return candidates

    def get_fuzzy_candidates_in_range(self, fuzzy_symbol, ad_value):
        candidates = sa.select(
            (self.equities.c.sid,)
        ).where(
            sa.and_(
                self.equities.c.fuzzy_symbol == fuzzy_symbol,
                self.equities.c.start_date <= ad_value,
                self.equities.c.end_date >= ad_value
            )
        ).order_by(
            self.equities.c.start_date.desc(),
            self.equities.c.end_date.desc(),
        ).execute().fetchall()
        return candidates

    def get_split_candidates_in_range(self,
                                      company_symbol,
                                      share_class_symbol,
                                      ad_value):
        candidates = sa.select(
            (self.equities.c.sid,)
        ).where(
            sa.and_(
                self.equities.c.company_symbol == company_symbol,
                self.equities.c.share_class_symbol == share_class_symbol,
                self.equities.c.start_date <= ad_value,
                self.equities.c.end_date >= ad_value
            )
        ).order_by(
            self.equities.c.start_date.desc(),
            self.equities.c.end_date.desc(),
        ).execute().fetchall()
        return candidates

    def get_split_candidates(self, company_symbol, share_class_symbol):
        candidates = sa.select(
            (self.equities.c.sid,)
        ).where(
            sa.and_(
                self.equities.c.company_symbol == company_symbol,
                self.equities.c.share_class_symbol == share_class_symbol
            )
        ).order_by(
            self.equities.c.start_date.desc(),
            self.equities.c.end_date.desc(),
        ).execute().fetchall()
        return candidates

    def resolve_no_matching_candidates(self,
                                       company_symbol,
                                       share_class_symbol,
                                       ad_value):
        candidates = sa.select((self.equities.c.sid,)).where(
            sa.and_(
                self.equities.c.company_symbol == company_symbol,
                self.equities.c.share_class_symbol ==
                share_class_symbol,
                self.equities.c.start_date <= ad_value),
            ).order_by(
                self.equities.c.end_date.desc(),
            ).execute().fetchall()
        return candidates

    def get_best_candidate(self, candidates):
        return self._retrieve_equity(candidates[0]['sid'])

    def get_equities_from_candidates(self, candidates):
        return list(map(
            compose(self._retrieve_equity, itemgetter('sid')),
            candidates,
        ))

    def lookup_symbol(self, symbol, as_of_date, fuzzy=False):
        """
        Return matching Equity of name symbol in database.

        If multiple Equities are found and as_of_date is not set,
        raises MultipleSymbolsFound.

        If no Equity was active at as_of_date raises SymbolNotFound.
        """

        company_symbol, share_class_symbol, fuzzy_symbol = \
            split_delimited_symbol(symbol)
        if as_of_date:
            # Format inputs
            as_of_date = pd.Timestamp(normalize_date(as_of_date))
            ad_value = as_of_date.value

            if fuzzy:
                # Search for a single exact match on the fuzzy column
                candidates = self.get_fuzzy_candidates_in_range(fuzzy_symbol,
                                                                ad_value)

                # If exactly one SID exists for fuzzy_symbol, return that sid
                if len(candidates) == 1:
                    return self.get_best_candidate(candidates)

            # Search for exact matches of the split-up company_symbol and
            # share_class_symbol
            candidates = self.get_split_candidates_in_range(company_symbol,
                                                            share_class_symbol,
                                                            ad_value)

            # If exactly one SID exists for symbol, return that symbol
            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if candidates:
                return self.get_best_candidate(candidates)

            # If no SID exists for symbol, return SID with the
            # highest-but-not-over end_date
            elif not candidates:
                candidates = self.resolve_no_matching_candidates(
                    company_symbol,
                    share_class_symbol,
                    ad_value
                )
                if candidates:
                    return self.get_best_candidate(candidates)

            raise SymbolNotFound(symbol=symbol)

        else:
            # If this is a fuzzy look-up, check if there is exactly one match
            # for the fuzzy symbol
            if fuzzy:
                candidates = self.get_fuzzy_candidates(fuzzy_symbol)
                if len(candidates) == 1:
                    return self.get_best_candidate(candidates)

            candidates = self.get_split_candidates(company_symbol,
                                                   share_class_symbol)
            if len(candidates) == 1:
                return self.get_best_candidate(candidates)
            elif not candidates:
                raise SymbolNotFound(symbol=symbol)
            else:
                raise MultipleSymbolsFound(
                    symbol=symbol,
                    options=self.get_equities_from_candidates(candidates)
                )

    def lookup_future_symbol(self, symbol):
        """ Return the Future object for a given symbol.

        Parameters
        ----------
        symbol : str
            The symbol of the desired contract.

        Returns
        -------
        Future
            A Future object.

        Raises
        ------
        SymbolNotFound
            Raised when no contract named 'symbol' is found.

        """

        data = self._select_asset_by_symbol(self.futures_contracts, symbol)\
                   .execute().fetchone()

        # If no data found, raise an exception
        if not data:
            raise SymbolNotFound(symbol=symbol)

        # If we find a contract, check whether it's been cached
        try:
            return self._future_cache[data['sid']]
        except KeyError:
            pass

        # Build the Future object from its parameters
        data = dict(data.items())
        _convert_asset_timestamp_fields(data)
        future = Future(**data)

        # Cache the Future object.
        self._future_cache[data['sid']] = future

        return future

    def lookup_future_chain(self, root_symbol, as_of_date):
        """ Return the futures chain for a given root symbol.

        Parameters
        ----------
        root_symbol : str
            Root symbol of the desired future.

        as_of_date : pd.Timestamp or pd.NaT
            Date at which the chain determination is rooted. I.e. the
            existing contract whose notice date/expiration date is first
            after this date is the primary contract, etc. If NaT is
            given, the chain is unbounded, and all contracts for this
            root symbol are returned.

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

            sids = list(map(
                itemgetter('sid'),
                sa.select((fc_cols.sid,)).where(
                    (fc_cols.root_symbol == root_symbol) &

                    # Filter to contracts that are still valid. If both
                    # exist, use the one that comes first in time (i.e.
                    # the lower value). If either notice_date or
                    # expiration_date is NaT, use the other. If both are
                    # NaT, the contract cannot be included in any chain.
                    sa.case(
                        [
                            (
                                fc_cols.notice_date == pd.NaT.value,
                                fc_cols.expiration_date >= as_of_date
                            ),
                            (
                                fc_cols.expiration_date == pd.NaT.value,
                                fc_cols.notice_date >= as_of_date
                            )
                        ],
                        else_=(
                            sa.func.min(
                                fc_cols.notice_date,
                                fc_cols.expiration_date
                            ) >= as_of_date
                        )
                    )
                ).order_by(
                    # Sort using expiration_date if valid. If it's NaT,
                    # use notice_date instead.
                    sa.case(
                        [
                            (
                                fc_cols.expiration_date == pd.NaT.value,
                                fc_cols.notice_date
                            )
                        ],
                        else_=fc_cols.expiration_date
                    ).asc()
                ).execute().fetchall()
            ))

        if not sids:
            # Check if root symbol exists.
            count = sa.select((sa.func.count(fc_cols.sid),)).where(
                fc_cols.root_symbol == root_symbol,
            ).scalar()
            if count == 0:
                raise RootSymbolNotFound(root_symbol=root_symbol)

        return list(map(self._retrieve_futures_contract, sids))

    @property
    def sids(self):
        return tuple(map(
            itemgetter('sid'),
            sa.select((self.asset_router.c.sid,)).execute().fetchall(),
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
        if isinstance(asset_convertible, Asset):
            matches.append(asset_convertible)

        elif isinstance(asset_convertible, Integral):
            try:
                result = self.retrieve_asset(int(asset_convertible))
            except SidNotFound:
                missing.append(asset_convertible)
                return None
            matches.append(result)

        elif isinstance(asset_convertible, string_types):
            try:
                matches.append(
                    self.lookup_symbol(asset_convertible, as_of_date)
                )
            except SymbolNotFound:
                missing.append(asset_convertible)
                return None
        else:
            raise NotAssetConvertible(
                "Input was %s, not AssetConvertible."
                % asset_convertible
            )

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

        # Look up all Assets for mapping
        matches = []
        missing = []
        for identifier in index:
            self._lookup_generic_scalar(identifier, as_of_date,
                                        matches, missing)

        # Handle missing assets
        if len(missing) > 0:
            warnings.warn("Missing assets for identifiers: %s" % missing)

        # Return a list of the sids of the found assets
        return [asset.sid for asset in matches]

    def _compute_asset_lifetimes(self):
        """
        Compute and cache a recarry of asset lifetimes.
        """
        equities_cols = self.equities.c
        buf = np.array(
            tuple(
                sa.select((
                    equities_cols.sid,
                    equities_cols.start_date,
                    equities_cols.end_date,
                )).execute(),
            ), dtype='<f8',  # use doubles so we get NaNs
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

    def lifetimes(self, dates, include_start_date):
        """
        Compute a DataFrame representing asset lifetimes for the specified date
        range.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            The dates for which to compute lifetimes.
        include_start_date : bool
            Whether or not to count the asset as alive on its start_date.

            This is useful in a backtesting context where `lifetimes` is being
            used to signify "do I have data for this asset as of the morning of
            this date?"  For many financial metrics, (e.g. daily close), data
            isn't available for an asset until the end of the asset's first
            day.

        Returns
        -------
        lifetimes : pd.DataFrame
            A frame of dtype bool with `dates` as index and an Int64Index of
            assets as columns.  The value at `lifetimes.loc[date, asset]` will
            be True iff `asset` existed on `date`.  If `include_start_date` is
            False, then lifetimes.loc[date, asset] will be false when date ==
            asset.start_date.

        See Also
        --------
        numpy.putmask
        zipline.pipeline.engine.SimplePipelineEngine._compute_root_mask
        """
        # This is a less than ideal place to do this, because if someone adds
        # assets to the finder after we've touched lifetimes we won't have
        # those new assets available.  Mutability is not my favorite
        # programming feature.
        if self._asset_lifetimes is None:
            self._asset_lifetimes = self._compute_asset_lifetimes()
        lifetimes = self._asset_lifetimes

        raw_dates = dates.asi8[:, None]
        if include_start_date:
            mask = lifetimes.start <= raw_dates
        else:
            mask = lifetimes.start < raw_dates
        mask &= (raw_dates <= lifetimes.end)

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


class AssetFinderCachedEquities(AssetFinder):
    """
    An extension to AssetFinder that loads all equities from equities table
    into memory and overrides the methods that lookup_symbol uses to look up
    those equities.
    """
    def __init__(self, engine):
        super(AssetFinderCachedEquities, self).__init__(engine)
        self.fuzzy_symbol_hashed_equities = {}
        self.company_share_class_hashed_equities = {}
        self.hashed_equities = sa.select(self.equities.c).execute().fetchall()
        self.load_hashed_equities()

    def load_hashed_equities(self):
        """
        Populates two maps - fuzzy symbol to list of equities having that
        fuzzy symbol and company symbol/share class symbol to list of
        equities having that combination of company symbol/share class symbol.
        """
        for equity in self.hashed_equities:
            company_symbol = equity['company_symbol']
            share_class_symbol = equity['share_class_symbol']
            fuzzy_symbol = equity['fuzzy_symbol']
            asset = self.convert_row_to_equity(equity)
            self.company_share_class_hashed_equities.setdefault(
                (company_symbol, share_class_symbol),
                []
            ).append(asset)
            self.fuzzy_symbol_hashed_equities.setdefault(
                fuzzy_symbol, []
            ).append(asset)

    def convert_row_to_equity(self, equity):
        """
        Converts a SQLAlchemy equity row to an Equity object.
        """
        data = dict(equity.items())
        _convert_asset_timestamp_fields(data)
        asset = Equity(**data)
        return asset

    def get_fuzzy_candidates(self, fuzzy_symbol):
        if fuzzy_symbol in self.fuzzy_symbol_hashed_equities:
            return self.fuzzy_symbol_hashed_equities[fuzzy_symbol]
        return []

    def get_fuzzy_candidates_in_range(self, fuzzy_symbol, ad_value):
        equities = self.get_fuzzy_candidates(fuzzy_symbol)
        fuzzy_candidates = []
        for equity in equities:
            if (equity.start_date.value <=
                    ad_value <=
                    equity.end_date.value):
                fuzzy_candidates.append(equity)
        return fuzzy_candidates

    def get_split_candidates(self, company_symbol, share_class_symbol):
        if (company_symbol, share_class_symbol) in \
                self.company_share_class_hashed_equities:
            return self.company_share_class_hashed_equities[(
                company_symbol, share_class_symbol)]
        return []

    def get_split_candidates_in_range(self,
                                      company_symbol,
                                      share_class_symbol,
                                      ad_value):
        equities = self.get_split_candidates(
            company_symbol, share_class_symbol
        )
        best_candidates = []
        for equity in equities:
            if (equity.start_date.value <=
                    ad_value <=
                    equity.end_date.value):
                best_candidates.append(equity)
        if best_candidates:
            best_candidates = sorted(
                best_candidates,
                key=lambda x: (x.start_date, x.end_date),
                reverse=True
            )
        return best_candidates

    def resolve_no_matching_candidates(self,
                                       company_symbol,
                                       share_class_symbol,
                                       ad_value):
        equities = self.get_split_candidates(
            company_symbol, share_class_symbol
        )
        partial_candidates = []
        for equity in equities:
            if equity.start_date.value <= ad_value:
                partial_candidates.append(equity)
        if partial_candidates:
            partial_candidates = sorted(
                partial_candidates,
                key=lambda x: x.end_date,
                reverse=True
            )
        return partial_candidates

    def get_best_candidate(self, candidates):
        return candidates[0]

    def get_equities_from_candidates(self, candidates):
        return candidates
