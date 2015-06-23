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
from itertools import chain
from numbers import Integral
import numpy as np
import operator

from logbook import Logger
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types

from zipline.errors import (
    ConsumeAssetMetaDataError,
    IdentifierNotFound,
    InvalidAssetType,
    MultipleSymbolsFound,
    SidAssignmentError,
    SidNotFound,
    SymbolNotFound,
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


class AssetFinder(object):

    def __init__(self, metadata=None, allow_sid_assignment=True):

        # Any particular instance of AssetFinder should be
        # consistent throughout its lifetime, so we grab a reference
        # to our cache now. That way, if the cache is refreshed later,
        # our instance will continue to use the old one.
        self.cache = {}
        self.sym_cache = {}
        self.identifier_cache = {}
        self.fuzzy_match = {}

        # This flag controls if the AssetFinder is allowed to generate its own
        # sids. If False, metadata that does not contain a sid will raise an
        # exception when building assets.
        self.allow_sid_assignment = allow_sid_assignment

        # The AssetFinder also holds a nested-dict of all metadata for
        # reference when building Assets
        self.metadata_cache = {}
        if metadata is not None:
            self.consume_metadata(metadata)

        self.populate_cache()

    def _next_free_sid(self):
        if len(self.cache) > 0:
            return max(self.cache.keys()) + 1
        return 0

    def _assign_sid(self, identifier):
        if hasattr(identifier, '__int__'):
            return identifier.__int__()
        if not self.allow_sid_assignment:
            raise SidAssignmentError(identifier=identifier)
        if isinstance(identifier, string_types):
            return self._next_free_sid()

    def retrieve_asset(self, sid, default_none=False):
        if isinstance(sid, Asset):
            return sid
        asset = self.cache.get(sid)
        if asset is not None:
            return asset
        elif default_none:
            return None
        else:
            raise SidNotFound(sid=sid)

    def retrieve_asset_by_identifier(self, identifier):
        if isinstance(identifier, Asset):
            return identifier
        asset = self.identifier_cache.get(identifier)
        if asset is not None:
            return asset
        else:
            raise IdentifierNotFound(identifier=identifier)

    @staticmethod
    def _lookup_symbol_in_infos(infos, as_of_date):
        """
        Search a list of symbols matching a given asset for the most recent
        known symbol as of as_of_date.

        Returns a pair of (Asset, bool), representing the best match we
        found for as_of_date, and whether or not that match was actually
        trading at as_of_date.

        If no entry in infos started before as_of_date, return (None, False).
        """
        # Sort entries by end_date before iterating.  If asset start and end
        # dates were always disjoint, then we could sort by either start or
        # end_date and get the same sorting.
        infos = sorted(infos, key=operator.attrgetter('end_date'))

        # Find the newest asset that started before as_of_date.
        candidates = [i for i in infos
                      if (i.start_date is None or i.start_date <= as_of_date)
                      and (i.end_date is None or as_of_date <= i.end_date)]

        # If one SID exists for symbol, return that symbol
        if len(candidates) == 1:
            return candidates[0], True

        # If no SID exists for symbol, return SID with the
        # highest-but-not-over end_date
        if len(candidates) == 0:
            candidates = [i for i in infos
                          if i.end_date < as_of_date]
            return (candidates[-1], False) if candidates else (None, False)

        # If multiple SIDs exist for symbol, return latest start_date with
        # end_date as a tie-breaker
        if len(candidates) > 1:
            best_candidate = sorted(
                candidates,
                key=lambda x: (x.start_date, x.end_date)
            )[-1]
            return best_candidate, True

    def lookup_symbol_resolve_multiple(self, symbol, as_of_date=None):
        """
        Return matching Asset of name symbol in database.

        If multiple Assets are found and as_of_date is not set,
        raises MultipleSymbolsFound.

        If no Asset was active at as_of_date, and allow_expired is False
        raises SymbolNotFound.
        """
        if as_of_date is not None:
            as_of_date = normalize_date(as_of_date)

        if symbol not in self.sym_cache:
            raise SymbolNotFound(symbol=symbol)

        infos = self.sym_cache[symbol]
        if as_of_date is None:
            if len(infos) == 1:
                return infos[0]
            else:
                raise MultipleSymbolsFound(symbol=symbol,
                                           options=str(infos))

        # Try to find symbol matching as_of_date
        asset, _ = self._lookup_symbol_in_infos(infos, as_of_date)
        if asset is None:
            raise SymbolNotFound(symbol=symbol)
        return asset

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
        as_of_date = normalize_date(as_of_date)

        if not fuzzy:
            try:
                return self.lookup_symbol_resolve_multiple(symbol, as_of_date)
            except SymbolNotFound:
                return None
        else:
            try:
                return self.fuzzy_match[(symbol, fuzzy, as_of_date)]
            except KeyError:
                # if symbol is CMCSA and fuzzy is '_', then
                # try CMCSA, then CMCS_A, then CMC_SA, etc.
                for fuzzy_symbol in chain(
                        (symbol,),
                        (symbol[:i] + fuzzy + symbol[i:]
                         for i in range(len(symbol) - 1, 0, -1))):

                    infos = self.sym_cache.get(fuzzy_symbol)
                    if infos:
                        info, date_match = self._lookup_symbol_in_infos(
                            infos,
                            as_of_date,
                        )

                        if info is not None and date_match:
                            self.fuzzy_match[(symbol, fuzzy, as_of_date)] = \
                                info
                            return info
                else:
                    self.fuzzy_match[(symbol, fuzzy, as_of_date)] = None

    def populate_cache(self):
        """
        Populates the asset cache with all values in the assets
        collection.
        """

        # Wipe caches before repopulating
        self.cache = {}
        self.sym_cache = {}
        self.identifier_cache = {}
        self.fuzzy_match = {}

        counter = 0
        for identifier, row in self.metadata_cache.items():
            self.spawn_asset(identifier=identifier, **row)
            counter += 1

    def spawn_asset(self, identifier, **kwargs):

        # Check if the sid is declared
        try:
            kwargs['sid']
            pass
        except KeyError:
            # If the identifier is not a sid, assign one
            kwargs['sid'] = self._assign_sid(identifier)
            # Update the metadata object with the new sid
            self.insert_metadata(identifier=identifier, sid=kwargs['sid'])

        # If the file_name is in the kwargs, it will be used as the symbol
        try:
            kwargs['symbol'] = kwargs.pop('file_name')
        except KeyError:
            pass

        # If the identifier coming in was a string and there is no defined
        # symbol yet, set the symbol to the incoming identifier
        try:
            kwargs['symbol']
            pass
        except KeyError:
            if isinstance(identifier, string_types):
                kwargs['symbol'] = identifier

        # If the company_name is in the kwargs, it may be the asset_name
        try:
            company_name = kwargs.pop('company_name')
            try:
                kwargs['asset_name']
            except KeyError:
                kwargs['asset_name'] = company_name
        except KeyError:
            pass

        # If dates are given as nanos, pop them
        try:
            kwargs['start_date'] = kwargs.pop('start_date_nano')
        except KeyError:
            pass
        try:
            kwargs['end_date'] = kwargs.pop('end_date_nano')
        except KeyError:
            pass
        try:
            kwargs['notice_date'] = kwargs.pop('notice_date_nano')
        except KeyError:
            pass
        try:
            kwargs['expiration_date'] = kwargs.pop('expiration_date_nano')
        except KeyError:
            pass

        # Process dates to Timestamps
        try:
            kwargs['start_date'] = pd.Timestamp(kwargs['start_date'], tz='UTC')
        except KeyError:
            pass
        try:
            kwargs['end_date'] = pd.Timestamp(kwargs['end_date'], tz='UTC')
        except KeyError:
            pass
        try:
            kwargs['notice_date'] = pd.Timestamp(kwargs['notice_date'],
                                                 tz='UTC')
        except KeyError:
            pass
        try:
            kwargs['expiration_date'] = pd.Timestamp(kwargs['expiration_date'],
                                                     tz='UTC')
        except KeyError:
            pass

        # Build an Asset of the appropriate type, default to Equity
        asset_type = kwargs.pop('asset_type', 'equity')
        if asset_type.lower() == 'equity':
            asset = Equity(**kwargs)
        elif asset_type.lower() == 'future':
            asset = Future(**kwargs)
        else:
            raise InvalidAssetType(asset_type=asset_type)

        self.cache[asset.sid] = asset
        self.identifier_cache[identifier] = asset
        if asset.symbol is not '':
            self.sym_cache.setdefault(asset.symbol, []).append(asset)

        return asset

    @property
    def sids(self):
        return self.cache.keys()

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

    def insert_metadata(self, identifier, **kwargs):
        """
        Inserts the given metadata kwargs to the entry for the given
        identifier. Matching fields in the existing entry will be overwritten.
        :param identifier: The identifier for which to insert metadata
        :param kwargs: The keyed metadata to insert
        """
        entry = self.metadata_cache.get(identifier, {})

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
        self.metadata_cache = {}

    def _insert_metadata_dataframe(self, dataframe):
        for identifier, row in dataframe.iterrows():
            self.insert_metadata(identifier, **row)

    def _insert_metadata_dict(self, dict):
        for identifier, entry in dict.items():
            self.insert_metadata(identifier, **entry)

    def _insert_metadata_readable(self, readable):
        for row in readable.read():
            # Parse out the row of the readable object
            metadata_dict = {}
            for field in ASSET_FIELDS:
                try:
                    row_value = row[field]
                    # Avoid passing placeholders
                    if row_value and (row_value is not 'None'):
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
            self.insert_metadata(identifier, **metadata_dict)


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
for type in string_types:
    AssetConvertible.register(type)


class NotAssetConvertible(ValueError):
    pass
