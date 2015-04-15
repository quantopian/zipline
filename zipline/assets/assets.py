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
from six import with_metaclass

from zipline.errors import (
    SymbolNotFound,
    MultipleSymbolsFound,
    SidNotFound,
    ConsumeAssetMetaDataError
)

from zipline.assets._assets import (
    Asset, Equity, Future
)

# asset_type values
# These values should match the AssetType enum in zipline.assets._assets
EQUITY = 1
FUTURE = 2

log = Logger('assets.py')


class AssetFinder(object):

    shared_caches = {'by_sid': {}, 'by_symbol': {}, 'fuzzy_match': {}}

    def __init__(self,
                 metadata,
                 trading_calendar,
                 force_populate=False):

        # Any particular instance of AssetFinder should be
        # consistent throughout its lifetime, so we grab a reference
        # to our cache now. That way, if the cache is refreshed later,
        # our instance will continue to use the old one.
        self.cache = self.shared_caches['by_sid']
        self.sym_cache = self.shared_caches['by_symbol']
        self.fuzzy_match = self.shared_caches['fuzzy_match']

        self.trading_calendar = trading_calendar
        self.metadata = metadata
        self.populate_cache(force_populate)

    def _next_free_sid(self):
        if len(self.cache) > 0:
            return max(self.cache.keys()) + 1
        return 0

    def _assign_sid(self, identifier):
        if hasattr(identifier, '__int__'):
            return identifier.__int__()
        if isinstance(identifier, str):
            return self._next_free_sid()

    def retrieve_asset(self, sid):
        asset = self.cache.get(sid)
        if asset is not None:
            return asset
        else:
            raise SidNotFound(sid=sid)

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

    def populate_cache(self, force=False):
        """
        Populates the asset cache with all values in the assets
        collection.
        """
        if not force and any(c for c in self.shared_caches.items()
                             if c is not None):
            return

        # Wipe caches before repopulating
        self.cache = {}
        self.sym_cache = {}
        self.fuzzy_match = {}

        counter = 0
        for identifier in self.metadata:
            row = self.metadata.retrieve_metadata(identifier=identifier)
            self.spawn_asset(identifier=identifier, **row)
            counter += 1

        self.shared_caches.update(
            {'by_sid': self.cache,
             'by_symbol': self.sym_cache,
             'fuzzy_match': {}}
        )

    def spawn_asset(self, identifier, **kwargs):

        # Check if the identifier is an int
        if isinstance(identifier, int) and ('sid' not in kwargs):
            kwargs['sid'] = identifier

        # Make sure that the sid exists in the kwargs
        if 'sid' not in kwargs.keys():
            kwargs['sid'] = self._assign_sid(identifier)

        # If the identifier coming in was a string and there is no defined
        # symbol yet, set the symbol to the incoming sid
        if isinstance(identifier, str) and 'symbol' not in kwargs.keys():
            kwargs['symbol'] = identifier

        asset_type = kwargs.pop('asset_type', None)
        asset = None

        # Process dates
        if 'start_date' in kwargs:
            kwargs['start_date'] = self.trading_calendar.\
                canonicalize_datetime(pd.Timestamp(kwargs['start_date']))
        if 'end_date' in kwargs:
            kwargs['end_date'] = self.trading_calendar.\
                canonicalize_datetime(pd.Timestamp(kwargs['end_date']))
        if 'notice_date' in kwargs:
            kwargs['notice_date'] = self.trading_calendar.\
                canonicalize_datetime(pd.Timestamp(kwargs['notice_date']))
        if 'expiration_date' in kwargs:
            kwargs['expiration_date'] = self.trading_calendar.\
                canonicalize_datetime(pd.Timestamp(kwargs['expiration_date']))

        if asset_type in (EQUITY, None):
            asset = Equity(**kwargs)
        if asset_type == FUTURE:
            asset = Future(**kwargs)

        self.cache[asset.sid] = asset
        if 'symbol' in kwargs.keys():
            self.sym_cache.setdefault(asset.symbol, []).append(asset)

        # Update the metadata object with the new sid
        self.metadata.insert_metadata(identifier=identifier, sid=asset.sid)
        return asset

    @classmethod
    def clear_cache(cls):
        cls.shared_caches.update(
            {'by_sid': None, 'by_symbol': None, 'fuzzy_match': None}
        )

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

            elif isinstance(asset_convertible, str):
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
                raise SidNotFound(sid=asset_convertible_or_iterable)

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


class AssetConvertible(with_metaclass(ABCMeta)):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Security, str, and Integral
    """
    pass
AssetConvertible.register(Integral)
AssetConvertible.register(str)
AssetConvertible.register(Asset)


class NotAssetConvertible(ValueError):
    pass


class AssetMetaData(object):

    cache = {}
    fields = ("sid",
              "asset_type",
              "symbol",
              "asset_name",
              "start_date",
              "end_date",
              "first_traded",
              "exchange",
              "notice_date",
              "expiration_date",
              "contract_multiplier")

    def __init__(self, data=None, identifiers=None):
        if data is not None:
            self.consume_metadata(data)
        if identifiers is not None:
            self.consume_identifiers(identifiers)

    def __iter__(self):
        return self.cache.__iter__()

    def _insert_dataframe(self, dataframe):
        for identifier, row in dataframe.iterrows():
            self.insert_metadata(identifier, **row)

    def _insert_dict(self, dict):
        for identifier, entry in dict.items():
            self.insert_metadata(identifier, **entry)

    def read(self):
        return self.cache.items()

    def retrieve_metadata(self, identifier):
        return self.cache.get(identifier)

    def insert_metadata(self, identifier, **kwargs):
        entry = self.retrieve_metadata(identifier)
        if entry is None:
            entry = {}

        for key, value in kwargs.items():
            # Do not accept invalid fields
            if key not in self.fields:
                continue
            # Do not accept Nones
            if value is None:
                continue
            # Do not accept nans from dataframes
            if isinstance(value, float) and np.isnan(value):
                continue
            entry[key] = value

        self.cache[identifier] = entry

    def consume_identifiers(self, identifiers):
        for identifier in identifiers:
            self.insert_metadata(identifier)

    def consume_metadata(self, metadata):
        """
        Consumes the provided metadata in to this AssetMetaData object. The
        existing values in this AssetMetaData will be overwritten when there
        is a conflict.
        :param metadata: The metadata to be consumed
        """
        if isinstance(metadata, AssetMetaData):
            for identifier in metadata:
                self.insert_metadata(identifier)
        elif isinstance(metadata, pd.DataFrame):
            self._insert_dataframe(metadata)
        elif isinstance(metadata, dict):
            self._insert_dict(metadata)
        else:
            raise ConsumeAssetMetaDataError(obj=metadata)

    def consume_data_source(self, source):
        if hasattr(source, 'identifiers'):
            for identifier in source.identifiers:
                if self.retrieve_metadata(identifier) is None:
                    self.insert_metadata(identifier=identifier)

    def erase(self):
        self.cache = {}
