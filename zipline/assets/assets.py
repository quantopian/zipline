#
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
import numpy as np
import sqlite3
from sqlite3 import Row
import warnings

from logbook import Logger
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types

from zipline.errors import (
    ConsumeAssetMetaDataError,
    InvalidAssetType,
    MultipleSymbolsFound,
    RootSymbolNotFound,
    SidAssignmentError,
    SidNotFound,
    SymbolNotFound,
    MapAssetIdentifierIndexError,
)
from zipline.assets._assets import (
    Asset, Equity, Future
)

log = Logger('assets.py')

# Expected fields for an Asset's metadata
ASSET_FIELDS = [
    'sid',
    'asset_type',
    'symbol',
    'root_symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange',
    'notice_date',
    'expiration_date',
    'contract_multiplier',
    # The following fields are for compatibility with other systems
    'file_name',  # Used as symbol
    'company_name',  # Used as asset_name
    'start_date_nano',  # Used as start_date
    'end_date_nano',  # Used as end_date
]


# Expected fields for an Asset's metadata
ASSET_TABLE_FIELDS = [
    'sid',
    'symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange',
]


# Expected fields for an Asset's metadata
FUTURE_TABLE_FIELDS = ASSET_TABLE_FIELDS + [
    'root_symbol',
    'notice_date',
    'expiration_date',
    'contract_multiplier',
]

EQUITY_TABLE_FIELDS = ASSET_TABLE_FIELDS


# Create the query once from the fields, so that the join is not done
# repeatedly.
FUTURE_BY_SID_QUERY = 'select {0} from futures where sid=?'.format(
    ", ".join(FUTURE_TABLE_FIELDS))

EQUITY_BY_SID_QUERY = 'select {0} from equities where sid=?'.format(
    ", ".join(EQUITY_TABLE_FIELDS))


class AssetFinder(object):

    def __init__(self,
                 metadata=None,
                 allow_sid_assignment=True,
                 fuzzy_char=None,
                 db_path=':memory:',
                 create_table=True):

        self.fuzzy_char = fuzzy_char

        # This flag controls if the AssetFinder is allowed to generate its own
        # sids. If False, metadata that does not contain a sid will raise an
        # exception when building assets.
        self.allow_sid_assignment = allow_sid_assignment

        if allow_sid_assignment:
            self.end_date_to_assign = normalize_date(
                pd.Timestamp('now', tz='UTC'))

        self.conn = sqlite3.connect(db_path)
        self.conn.text_factory = str
        self.cursor = self.conn.cursor()

        # The AssetFinder also holds a nested-dict of all metadata for
        # reference when building Assets
        self.metadata_cache = {}

        # Create table and read in metadata.
        # Should we use flags like 'r', 'w', instead?
        # What we need to support is:
        # - A 'throwaway' mode where the metadata is read each run.
        # - A 'write' mode where the data is written to the provided db_path
        # - A 'read' mode where the asset finder uses a prexisting db.
        if create_table:
            self.create_db_tables()
            if metadata is not None:
                self.consume_metadata(metadata)

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

    def create_db_tables(self):
        c = self.conn.cursor()

        c.execute("""
        CREATE TABLE equities(
        sid integer,
        symbol text,
        asset_name text,
        start_date integer,
        end_date integer,
        first_traded integer,
        exchange text,
        fuzzy text
        )""")

        c.execute('CREATE INDEX equities_sid on equities(sid)')
        c.execute('CREATE INDEX equities_symbol on equities(symbol)')
        c.execute('CREATE INDEX equities_fuzzy on equities(fuzzy)')

        c.execute("""
        CREATE TABLE futures(
        sid integer,
        symbol text,
        asset_name text,
        start_date integer,
        end_date integer,
        first_traded integer,
        exchange text,
        root_symbol text,
        notice_date integer,
        expiration_date integer,
        contract_multiplier real
        )""")

        c.execute('CREATE INDEX futures_sid on futures(sid)')
        c.execute('CREATE INDEX futures_root_symbol on equities(symbol)')

        c.execute("""
        CREATE TABLE asset_router
        (sid integer,
        asset_type text)
        """)

        c.execute('CREATE INDEX asset_router_sid on asset_router(sid)')

        self.conn.commit()

    def asset_type_by_sid(self, sid):
        try:
            return self._asset_type_cache[sid]
        except KeyError:
            pass

        c = self.conn.cursor()
        # Python 3 compatibility required forcing to int for sid = 0.
        t = (int(sid),)
        query = 'select asset_type from asset_router where sid=:sid'
        c.execute(query, t)
        data = c.fetchone()
        if data is None:
            return

        asset_type = data[0]
        self._asset_type_cache[sid] = asset_type

        return asset_type

    def retrieve_asset(self, sid, default_none=False):
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
            query = ("select sid from equities "
                     "where symbol=? "
                     "and start_date<=? "
                     "and end_date>=?")
            c.execute(query, t)
            candidates = c.fetchall()

            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0][0])

            # If no SID exists for symbol, return SID with the
            # highest-but-not-over end_date
            if len(candidates) == 0:
                t = (symbol, as_of_date.value)
                query = ("select sid from equities "
                         "where symbol=? "
                         "and start_date<=? "
                         "order by end_date desc "
                         "limit 1")
                c.execute(query, t)
                data = c.fetchone()

                if data:
                    return self._retrieve_equity(data[0])

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if len(candidates) > 1:
                t = (symbol, as_of_date.value)
                query = ("select sid from equities "
                         "where symbol=? " +
                         "and start_date<=? " +
                         "order by start_date desc, end_date desc " +
                         "limit 1")
                c.execute(query, t)
                data = c.fetchone()

                if data:
                    return self._retrieve_equity(data[0])

            raise SymbolNotFound(symbol=symbol)

        else:
            t = (symbol,)
            query = ("select sid from equities where symbol=?")
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
            query = ("select sid from equities "
                     "where fuzzy=? " +
                     "and start_date<=? " +
                     "and end_date>=?")
            c.execute(query, t)
            candidates = c.fetchall()

            # If one SID exists for symbol, return that symbol
            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0][0])

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if len(candidates) > 1:
                t = (symbol, as_of_date.value)
                query = ("select sid from equities "
                         "where symbol=? " +
                         "and start_date<=? " +
                         "order by start_date desc, end_date desc" +
                         "limit 1")
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
        as_of_date : pd.Timestamp
            Date at which the chain determination is rooted. I.e. the
            existing contract whose notice date is first after this
            date is the primary contract, etc.
        knowledge_date : pd.Timestamp
            Date for determining which contracts exist for inclusion in
            this chain. Contracts exist only if they have a start_date
            on or before this date.

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
            select count(sid) from futures where root_symbol=:root_symbol
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
        query = 'select sid from asset_router'
        c.execute(query)
        return [r[0] for r in c.fetchall()]

    @property
    def assets(self):
        return self.cache.values()

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
        __________
        index : Iterable
            An iterable containing ints, strings, or Assets
        as_of_date : pandas.Timestamp
            A date to be used to resolve any dual-mapped symbols

        Returns
        _______
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

    def _insert_metadata(self, identifier, **kwargs):
        """
        Inserts the given metadata kwargs to the entry for the given
        identifier. Matching fields in the existing entry will be overwritten.
        :param identifier: The identifier for which to insert metadata
        :param kwargs: The keyed metadata to insert
        """
        if identifier in self.metadata_cache:
            # Multiple pass insertion no longer supported.
            # This could and probably should raise an Exception, but is
            # currently just a short-circuit for compatibility with existing
            # testing structure in the test_algorithm module which creates
            # multiple sources which all insert redundant metadata.
            return

        entry = {}

        for key, value in kwargs.items():
            # Do not accept invalid fields
            if key not in ASSET_FIELDS:
                continue
            # Do not accept Nones
            if value is None:
                continue
            # Do not accept empty strings
            if value == '':
                continue
            # Do not accept nans from dataframes
            if isinstance(value, float) and np.isnan(value):
                continue
            entry[key] = value

        # Check if the sid is declared
        try:
            entry['sid']
        except KeyError:
            # If the identifier is not a sid, assign one
            if hasattr(identifier, '__int__'):
                entry['sid'] = identifier.__int__()
            else:
                if self.allow_sid_assignment:
                    # Assign the sid the value of its insertion order.
                    # This assumes that we are assigning values to all assets.
                    entry['sid'] = len(self.metadata_cache)
                else:
                    raise SidAssignmentError(identifier=identifier)

        # If the file_name is in the kwargs, it will be used as the symbol
        try:
            entry['symbol'] = entry.pop('file_name')
        except KeyError:
            pass

        # If the identifier coming in was a string and there is no defined
        # symbol yet, set the symbol to the incoming identifier
        try:
            entry['symbol']
            pass
        except KeyError:
            if isinstance(identifier, string_types):
                entry['symbol'] = identifier

        # If the company_name is in the kwargs, it may be the asset_name
        try:
            company_name = entry.pop('company_name')
            try:
                entry['asset_name']
            except KeyError:
                entry['asset_name'] = company_name
        except KeyError:
            pass

        # If dates are given as nanos, pop them
        try:
            entry['start_date'] = entry.pop('start_date_nano')
        except KeyError:
            pass
        try:
            entry['end_date'] = entry.pop('end_date_nano')
        except KeyError:
            pass
        try:
            entry['notice_date'] = entry.pop('notice_date_nano')
        except KeyError:
            pass
        try:
            entry['expiration_date'] = entry.pop('expiration_date_nano')
        except KeyError:
            pass

        # Process dates to Timestamps
        try:
            entry['start_date'] = pd.Timestamp(entry['start_date'], tz='UTC')
        except KeyError:
            # Set a default start_date of the EPOCH, so that all date queries
            # work when a start date is not provided.
            entry['start_date'] = pd.Timestamp(0, tz='UTC')
        try:
            # Set a default end_date of 'now', so that all date queries
            # work when a end date is not provided.
            entry['end_date'] = pd.Timestamp(entry['end_date'], tz='UTC')
        except KeyError:
            entry['end_date'] = self.end_date_to_assign
        try:
            entry['notice_date'] = pd.Timestamp(entry['notice_date'],
                                                tz='UTC')
        except KeyError:
            pass
        try:
            entry['expiration_date'] = pd.Timestamp(entry['expiration_date'],
                                                    tz='UTC')
        except KeyError:
            pass

        # Build an Asset of the appropriate type, default to Equity
        asset_type = entry.pop('asset_type', 'equity')
        if asset_type.lower() == 'equity':
            try:
                fuzzy = entry['symbol'].replace(self.fuzzy_char, '') \
                    if self.fuzzy_char else None
            except KeyError:
                fuzzy = None
            asset = Equity(**entry)
            c = self.conn.cursor()
            t = (asset.sid,
                 asset.symbol,
                 asset.asset_name,
                 asset.start_date.value if asset.start_date else None,
                 asset.end_date.value if asset.end_date else None,
                 asset.first_traded.value if asset.first_traded else None,
                 asset.exchange,
                 fuzzy)
            c.execute("""INSERT INTO equities(
            sid,
            symbol,
            asset_name,
            start_date,
            end_date,
            first_traded,
            exchange,
            fuzzy)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)""", t)

            t = (asset.sid,
                 'equity')
            c.execute("""INSERT INTO asset_router(sid, asset_type)
            VALUES(?, ?)""", t)

        elif asset_type.lower() == 'future':
            asset = Future(**entry)
            c = self.conn.cursor()
            t = (asset.sid,
                 asset.symbol,
                 asset.asset_name,
                 asset.start_date.value if asset.start_date else None,
                 asset.end_date.value if asset.end_date else None,
                 asset.first_traded.value if asset.first_traded else None,
                 asset.exchange,
                 asset.root_symbol,
                 asset.notice_date.value if asset.notice_date else None,
                 asset.expiration_date.value
                 if asset.expiration_date else None,
                 asset.contract_multiplier)
            c.execute("""INSERT INTO futures(
            sid,
            symbol,
            asset_name,
            start_date,
            end_date,
            first_traded,
            exchange,
            root_symbol,
            notice_date,
            expiration_date,
            contract_multiplier)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", t)

            t = (asset.sid,
                 'future')
            c.execute("""INSERT INTO asset_router(sid, asset_type)
            VALUES(?, ?)""", t)
        else:
            raise InvalidAssetType(asset_type=asset_type)

        self.metadata_cache[identifier] = entry

    def consume_identifiers(self, identifiers):
        """
        Consumes the given identifiers in to the metadata cache of this
        AssetFinder.
        """
        for identifier in identifiers:
            # Handle case where full Assets are passed in
            # For example, in the creation of a DataFrameSource, the source's
            # 'sid' args may be full Assets
            if isinstance(identifier, Asset):
                sid = identifier.sid
                metadata = identifier.to_dict()
                metadata['asset_type'] = identifier.__class__.__name__
                self.insert_metadata(identifier=sid, **metadata)
            else:
                self.insert_metadata(identifier)

    def consume_metadata(self, metadata):
        """
        Consumes the provided metadata in to the metadata cache. The
        existing values in the cache will be overwritten when there
        is a conflict.
        :param metadata: The metadata to be consumed
        """
        # Handle dicts
        if isinstance(metadata, dict):
            self._insert_metadata_dict(metadata)
        # Handle DataFrames
        elif isinstance(metadata, pd.DataFrame):
            self._insert_metadata_dataframe(metadata)
        # Handle readables
        elif hasattr(metadata, 'read'):
            self._insert_metadata_readable(metadata)
        else:
            raise ConsumeAssetMetaDataError(obj=metadata)

    def clear_metadata(self):
        """
        Used for testing.
        """
        self.metadata_cache = {}

        self.conn = sqlite3.connect(':memory:')
        self.create_db_tables()

    def insert_metadata(self, identifier, **kwargs):
        self._insert_metadata(identifier, **kwargs)
        self.conn.commit()

    def _insert_metadata_dataframe(self, dataframe):
        for identifier, row in dataframe.iterrows():
            self._insert_metadata(identifier, **row)
        self.conn.commit()

    def _insert_metadata_dict(self, dict):
        for identifier, entry in dict.items():
            self._insert_metadata(identifier, **entry)
        self.conn.commit()

    def _insert_metadata_readable(self, readable):
        for row in readable.read():
            # Parse out the row of the readable object
            metadata_dict = {}
            for field in ASSET_FIELDS:
                try:
                    row_value = row[field]
                    # Avoid passing placeholders
                    if row_value and (row_value != 'None'):
                        metadata_dict[field] = row[field]
                except KeyError:
                    continue
                except IndexError:
                    continue
            # Locate the identifier, fail if not found
            if 'sid' in metadata_dict:
                identifier = metadata_dict['sid']
            elif 'symbol' in metadata_dict:
                identifier = metadata_dict['symbol']
            else:
                raise ConsumeAssetMetaDataError(obj=row)
            self._insert_metadata(identifier, **metadata_dict)
        self.conn.commit()


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
