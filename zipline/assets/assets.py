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
import sqlite3
from sqlite3 import Row
import warnings

import numpy as np
from logbook import Logger
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types

from zipline.errors import (
    ConsumeAssetMetaDataError,
    MultipleSymbolsFound,
    RootSymbolNotFound,
    SidNotFound,
    SymbolNotFound,
    MapAssetIdentifierIndexError,
)
from zipline.assets import (
    Asset, Equity, Future
)
from zipline.assets import (
    NullAssetDBWriterLegacy,
    AssetDBWriterLegacyFromList,
    AssetDBWriterLegacyFromDictionary,
    AssetDBWriterLegacyFromDataFrame,
    AssetDBWriterLegacyFromReadable
)
from zipline.assets.asset_writer import (
    FUTURE_TABLE_FIELDS,
    EQUITY_TABLE_FIELDS
)

log = Logger('assets.py')

# Create the query once from the fields, so that the join is not done
# repeatedly.
FUTURE_BY_SID_QUERY = 'select {0} from futures_contracts where sid=?'.format(
    ", ".join(FUTURE_TABLE_FIELDS))

EQUITY_BY_SID_QUERY = 'select {0} from equities where sid=?'.format(
    ", ".join(EQUITY_TABLE_FIELDS))


def create_relevant_writer(metadata):
    """ Create an instance of AssetDBWriter relevant to
        processing metadata.
        Will be deprecated in future versions of zipline.
    """

    if isinstance(metadata, dict):
        return AssetDBWriterLegacyFromDictionary(metadata)
    elif isinstance(metadata, pd.DataFrame):
        return AssetDBWriterLegacyFromDataFrame(metadata)
    elif isinstance(metadata, list):
        return AssetDBWriterLegacyFromList(metadata)
    elif hasattr(metadata, 'read'):
        return AssetDBWriterLegacyFromReadable(metadata)
    elif metadata is None:
        return NullAssetDBWriterLegacy(metadata)
    else:
        raise ConsumeAssetMetaDataError(obj=metadata)


class AssetFinder(object):

    def __init__(self, metadata=None, allow_sid_assignment=True,
                 fuzzy_char=None, db_path=':memory:', create_table=True,
                 asset_writer=None):

        self.fuzzy_char = fuzzy_char
        self.allow_sid_assignment = allow_sid_assignment

        self.conn = sqlite3.connect(db_path)

        # AssetFinder can optionally accept an instance of
        # the AssetDBWriter class. If no writer is supplied,
        # we create a relevant writer based on the supplied metadata.
        # Note that this strucutre is for backward compatibility.
        # Ultimately AssetDBWriter and AssetFinder will be completely
        # separate, and AssetFinder will not instantiate AssetDBWriter.
        if asset_writer is None:
            _asset_writer = create_relevant_writer(metadata)
        else:
            _asset_writer = asset_writer

        # Create tables and read in metadata.
        if create_table:
            _asset_writer.init_db(self.conn)
            if metadata is not None:
                _asset_writer.write_all(self.conn,
                                        self.fuzzy_char,
                                        self.allow_sid_assignment)

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

    def clear_metadata(self):
        """
        Used for testing.
        Will be deprecated in future versions of zipline.
        """
        # Close the database connection
        self.conn.close()
        # Create new database connection in memory.
        self.conn = sqlite3.connect(':memory:')
        # Initialize the database tables using the same connection
        # as used by the AssetFinder.
        _asset_writer = NullAssetDBWriterLegacy({})
        _asset_writer.init_db(self.conn)

    def asset_type_by_sid(self, sid):
        """
        Retrieve the asset type of a given sid.
        """
        try:
            return self._asset_type_cache[sid]
        except KeyError:
            pass

        c = self.conn.cursor()
        # Python 3 compatibility required forcing to int for sid = 0.
        t = (int(sid),)
        query = 'SELECT asset_type FROM asset_router WHERE sid=:sid'
        c.execute(query, t)
        data = c.fetchone()
        if data is None:
            return

        asset_type = data[0]
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

        c = self.conn.cursor()
        c.row_factory = Row
        t = (int(sid),)
        c.execute(EQUITY_BY_SID_QUERY, t)
        data = dict(c.fetchone())
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

        c = self.conn.cursor()
        t = (int(sid),)
        c.row_factory = Row
        c.execute(FUTURE_BY_SID_QUERY, t)
        data = dict(c.fetchone())
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

        If no Asset was active at as_of_date, and allow_expired is False
        raises SymbolNotFound.
        """
        if as_of_date is not None:
            as_of_date = pd.Timestamp(normalize_date(as_of_date))

        c = self.conn.cursor()

        if as_of_date:
            # If one SID exists for symbol, return that symbol
            t = (symbol, as_of_date.value, as_of_date.value)
            query = ("SELECT sid FROM equities "
                     "WHERE symbol=? "
                     "AND start_date<=? "
                     "AND end_date>=?")
            c.execute(query, t)
            candidates = c.fetchall()

            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0][0])

            # If no SID exists for symbol, return SID with the
            # highest-but-not-over end_date
            if len(candidates) == 0:
                t = (symbol, as_of_date.value)
                query = ("SELECT sid FROM equities "
                         "WHERE symbol=? "
                         "AND start_date<=? "
                         "ORDER BY end_date DESC "
                         "LIMIT 1")
                c.execute(query, t)
                data = c.fetchone()

                if data:
                    return self._retrieve_equity(data[0])

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if len(candidates) > 1:
                t = (symbol, as_of_date.value)
                query = ("SELECT sid FROM equities "
                         "WHERE symbol=? " +
                         "AND start_date<=? " +
                         "ORDER BY start_date DESC, end_date DESC " +
                         "LIMIT 1")
                c.execute(query, t)
                data = c.fetchone()

                if data:
                    return self._retrieve_equity(data[0])

            raise SymbolNotFound(symbol=symbol)

        else:
            t = (symbol,)
            query = ("SELECT sid FROM equities WHERE symbol=?")
            c.execute(query, t)
            data = c.fetchall()

            if len(data) == 1:
                return self._retrieve_equity(data[0][0])
            elif not data:
                raise SymbolNotFound(symbol=symbol)
            else:
                options = []
                for row in data:
                    sid = row[0]
                    asset = self._retrieve_equity(sid)
                    options.append(asset)
                raise MultipleSymbolsFound(symbol=symbol,
                                           options=options)

    def lookup_symbol(self, symbol, as_of_date, fuzzy=False):
        """
        If a fuzzy string is provided, then we try various symbols based on
        the provided symbol.  This is to facilitate mapping from a broker's
        symbol to ours in cases where mapping to the broker's symbol loses
        information. For example, if we have CMCS_A, but a broker has CMCSA,
        when the broker provides CMCSA, it can also provide fuzzy='_',
        so we can find a match by inserting an underscore.
        """
        symbol = symbol.upper()
        as_of_date = normalize_date(as_of_date)

        if not fuzzy:
            try:
                return self.lookup_symbol_resolve_multiple(symbol, as_of_date)
            except SymbolNotFound:
                return None
        else:
            c = self.conn.cursor()
            fuzzy = symbol.replace(self.fuzzy_char, '')
            t = (fuzzy, as_of_date.value, as_of_date.value)
            query = ("SELECT sid FROM equities "
                     "WHERE fuzzy=? " +
                     "AND start_date<=? " +
                     "AND end_date>=?")
            c.execute(query, t)
            candidates = c.fetchall()

            # If one SID exists for symbol, return that symbol
            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0][0])

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if len(candidates) > 1:
                t = (symbol, as_of_date.value)
                query = ("SELECT sid FROM equities "
                         "WHERE symbol=? " +
                         "AND start_date<=? " +
                         "ORDER BY start_date desc, end_date desc" +
                         "LIMIT 1")
                c.execute(query, t)
                data = c.fetchone()
                if data:
                    return self._retrieve_equity(data[0])

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
        c = self.conn.cursor()
        if as_of_date is pd.NaT:
            # If the as_of_date is NaT, get all contracts for this
            # root symbol.
            t = {'root_symbol': root_symbol}
            c.execute("""
            select sid from futures
            where root_symbol=:root_symbol
            order by notice_date asc
            """, t)
        else:
            if knowledge_date is pd.NaT:
                # If knowledge_date is NaT, default to using as_of_date
                t = {'root_symbol': root_symbol,
                     'as_of_date': as_of_date.value,
                     'knowledge_date': as_of_date.value}
            else:
                t = {'root_symbol': root_symbol,
                     'as_of_date': as_of_date.value,
                     'knowledge_date': knowledge_date.value}

            c.execute("""
            select sid from futures
            where root_symbol=:root_symbol
            and :as_of_date < notice_date
            and start_date <= :knowledge_date
            order by notice_date asc
            """, t)
        sids = [r[0] for r in c.fetchall()]
        if not sids:
            # Check if root symbol exists.
            c.execute("""
            SELECT COUNT(sid) FROM futures_contracts
            WHERE root_symbol=:root_symbol
            """, t)
            count = c.fetchone()[0]
            if count == 0:
                raise RootSymbolNotFound(root_symbol=root_symbol)
            else:
                # If symbol exists, return empty future chain.
                return []
        return [self._retrieve_futures_contract(sid) for sid in sids]

    @property
    def sids(self):
        c = self.conn.cursor()
        query = 'SELECT sid FROM asset_router'
        c.execute(query)
        return [r[0] for r in c.fetchall()]

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

    def consume_identifiers(self, identifiers):
        """
        Consumes the provided identifiers, passing them to
        the asset writer to be added to the database.
        Will be deprecated in future versions of zipline.

        Parameters
        ----------
        identifiers
            The data to be consumed.
        """
        _asset_writer = AssetDBWriterLegacyFromList(identifiers)
        _asset_writer.consume_identifiers(self.conn,
                                          self.fuzzy_char,
                                          self.allow_sid_assignment)

    def consume_metadata(self, metadata):
        """
        Consumes the provided metadata, passing it to
        the asset writer to be added to the database.
        Will be deprecated in future versions of zipline.

        Parameters
        ----------
        metadata
            The data to be consumed.
        """
        _asset_writer = create_relevant_writer(metadata)
        _asset_writer.write_all(self.conn,
                                fuzzy_char=self.fuzzy_char,
                                allow_sid_assignment=self.allow_sid_assignment)

    def insert_metadata(self, identifier, **kwargs):
        """
        Insert information for a single identifier.
        Will be deprecated in future versions of zipline.
        """
        metadata = {}
        metadata[identifier] = kwargs
        _asset_writer = create_relevant_writer(metadata)
        _asset_writer.write_all(self.conn,
                                fuzzy_char=self.fuzzy_char,
                                allow_sid_assignment=self.allow_sid_assignment)

    def _compute_asset_lifetimes(self):
        """
        Compute and cache a recarry of asset lifetimes.

        FUTURE OPTIMIZATION: We're looping over a big array, which means this
        probably should be in C/Cython.
        """
        with self.conn as transaction:
            results = transaction.execute(
                'SELECT sid, start_date, end_date from equities'
            ).fetchall()

            lifetimes = np.recarray(
                shape=(len(results),),
                dtype=[('sid', 'i8'), ('start', 'i8'), ('end', 'i8')],
            )

            # TODO: This is **WAY** slower than it could be because we have to
            # check for None everywhere.  If we represented "no start date" as
            # 0, and "no end date" as MAX_INT in our metadata, this would be
            # significantly faster.
            NO_START = 0
            NO_END = np.iinfo(int).max
            for idx, (sid, start, end) in enumerate(results):
                lifetimes[idx] = (
                    sid,
                    start if start is not None else NO_START,
                    end if end is not None else NO_END,
                )
        return lifetimes

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
