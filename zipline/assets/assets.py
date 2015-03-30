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
import operator

from logbook import Logger
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass

from zipline.errors import (
    SymbolNotFound,
    MultipleSymbolsFound
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
                 table,
                 force_populate=False,
    ):

        self.table = table
        self.populate_cache(table, force_populate)

        # Any particular instance of AssetFinder should be
        # consistent throughout its lifetime, so we grab a reference
        # to our cache now. That way, if the cache is refreshed later,
        # our instance will continue to use the old one.
        self.cache = self.shared_caches['by_sid']
        self.sym_cache = self.shared_caches['by_symbol']
        self.fuzzy_match = self.shared_caches['fuzzy_match']

    def retrieve_asset(self, sid):
        return self.cache.get(sid)

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
                      if i.start_date <= as_of_date <= i.end_date]

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

    def populate_cache(self, table, force=False):
        """
        Populates the asset cache with all values in the assets
        collection.
        """
        if not force and any(c for c in self.shared_caches.itervalues()
                             if c is not None):
            return

        all_assets = table.read()

        cache = {}
        sym_cache = {}
        for row in all_assets:
            info = Asset(sid=row['sid'],
                            symbol=row['file_name'],
                            exchange=row['exchange'],
                            asset_name=row['company_name'],
                            start_date=pd.Timestamp(row['start_date_nano'],
                                                    unit='ns',
                                                    tz='UTC'),
                            end_date=pd.Timestamp(row['end_date_nano'],
                                                  unit='ns',
                                                  tz='UTC'))
            cache[info.sid] = info
            sym_cache.setdefault(info.symbol, []).append(info)
        log.info('Read %d items into cache' % len(cache))

        self.shared_caches.update(
            {'by_sid': cache, 'by_symbol': sym_cache, 'fuzzy_match': {}}
        )

    @classmethod
    def clear_cache(cls):
        cls.shared_caches.update(
            {'by_sid': None, 'by_symbol': None, 'fuzzy_match': None}
        )

    @property
    def sids(self):
        return self.cache.keys()

    def _lookup_generic_scalar(self,
                               asset_convertible,
                               reference_date,
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
                    raise SymbolNotFound()
                matches.append(result)

            elif isinstance(asset_convertible, basestring):
                # Throws SymbolNotFound on failure to match.
                matches.append(
                    self.lookup_symbol_resolve_multiple(
                        asset_convertible,
                        reference_date,
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
                       reference_date):
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
                asset_convertible_or_iterable,
                reference_date,
                matches,
                missing,
            )
            return matches[0], missing

        # Interpret input as iterable.
        try:
            iterator = iter(asset_convertible_or_iterable)
        except TypeError:
            raise NotAssetConvertible(
                "Input was not a AssetConvertible "
                "or iterable of AssetConvertible."
            )

        for obj in iterator:
            self._lookup_generic_scalar(obj, reference_date, matches, missing)
        return matches, missing


class AssetConvertible(with_metaclass(ABCMeta)):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Security, basestring, and Integral
    """
    pass
AssetConvertible.register(Integral)
AssetConvertible.register(basestring)
AssetConvertible.register(Asset)


class NotAssetConvertible(ValueError):
    pass
