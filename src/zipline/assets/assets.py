# Copyright 2016 Quantopian, Inc.
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

# import array
# import binascii
# import struct
from abc import ABC
from collections import deque, namedtuple
from functools import partial
from numbers import Integral
from operator import attrgetter, itemgetter

import logging
import numpy as np
import pandas as pd
import sqlalchemy as sa
from toolz import (
    compose,
    concat,
    concatv,
    curry,
    groupby,
    merge,
    partition_all,
    sliding_window,
    valmap,
)

from zipline.errors import (
    EquitiesNotFound,
    FutureContractsNotFound,
    MultipleSymbolsFound,
    MultipleSymbolsFoundForFuzzySymbol,
    MultipleValuesFoundForField,
    MultipleValuesFoundForSid,
    NoValueForSid,
    SameSymbolUsedAcrossCountries,
    SidsNotFound,
    SymbolNotFound,
    ValueNotFoundForField,
)
from zipline.utils.functional import invert
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import as_column
from zipline.utils.preprocess import preprocess
from zipline.utils.sqlite_utils import coerce_string_to_eng, group_into_chunks

from . import Asset, Equity, Future
from .asset_db_schema import ASSET_DB_VERSION
from .asset_writer import (
    SQLITE_MAX_VARIABLE_NUMBER,
    asset_db_table_names,
    check_version_info,
    split_delimited_symbol,
    symbol_columns as SYMBOL_COLUMNS,
)
from .continuous_futures import (
    ADJUSTMENT_STYLES,
    CHAIN_PREDICATES,
    ContinuousFuture,
    OrderedContracts,
)
from .exchange_info import ExchangeInfo

log = logging.getLogger("assets.py")

# A set of fields that need to be converted to strings before building an
# Asset to avoid unicode fields
# _asset_str_fields = frozenset(
#     {
#         "symbol",
#         "asset_name",
#         "exchange",
#     }
# )

# A set of fields that need to be converted to timestamps in UTC
_asset_timestamp_fields = frozenset(
    {
        "start_date",
        "end_date",
        "first_traded",
        "notice_date",
        "expiration_date",
        "auto_close_date",
    }
)

OwnershipPeriod = namedtuple("OwnershipPeriod", "start end sid value")


def merge_ownership_periods(mappings):
    """Given a dict of mappings where the values are lists of
    OwnershipPeriod objects, returns a dict with the same structure with
    new OwnershipPeriod objects adjusted so that the periods have no
    gaps.

    Orders the periods chronologically, and pushes forward the end date
    of each period to match the start date of the following period. The
    end date of the last period pushed forward to the max Timestamp.
    """
    return valmap(
        lambda v: tuple(
            OwnershipPeriod(
                a.start,
                b.start,
                a.sid,
                a.value,
            )
            for a, b in sliding_window(
                2,
                concatv(
                    sorted(v),
                    # concat with a fake ownership object to make the last
                    # end date be max timestamp
                    [
                        OwnershipPeriod(
                            pd.Timestamp.max,
                            None,
                            None,
                            None,
                        )
                    ],
                ),
            )
        ),
        mappings,
    )


def _build_ownership_map_from_rows(rows, key_from_row, value_from_row):
    mappings = {}
    for row in rows:
        mappings.setdefault(key_from_row(row), [],).append(
            OwnershipPeriod(
                # TODO FIX TZ MESS
                # pd.Timestamp(row.start_date, unit="ns", tz="utc"),
                # pd.Timestamp(row.end_date, unit="ns", tz="utc"),
                pd.Timestamp(row.start_date, unit="ns", tz=None),
                pd.Timestamp(row.end_date, unit="ns", tz=None),
                row.sid,
                value_from_row(row),
            ),
        )

    return merge_ownership_periods(mappings)


def build_ownership_map(conn, table, key_from_row, value_from_row):
    """Builds a dict mapping to lists of OwnershipPeriods, from a db table."""
    return _build_ownership_map_from_rows(
        conn.execute(sa.select(table.c)).fetchall(),
        key_from_row,
        value_from_row,
    )


def build_grouped_ownership_map(conn, table, key_from_row, value_from_row, group_key):
    """Builds a dict mapping group keys to maps of keys to lists of
    OwnershipPeriods, from a db table.
    """

    grouped_rows = groupby(
        group_key,
        conn.execute(sa.select(table.c)).fetchall(),
    )
    return {
        key: _build_ownership_map_from_rows(
            rows,
            key_from_row,
            value_from_row,
        )
        for key, rows in grouped_rows.items()
    }


@curry
def _filter_kwargs(names, dict_):
    """Filter out kwargs from a dictionary.

    Parameters
    ----------
    names : set[str]
        The names to select from ``dict_``.
    dict_ : dict[str, any]
        The dictionary to select from.

    Returns
    -------
    kwargs : dict[str, any]
        ``dict_`` where the keys intersect with ``names`` and the values are
        not None.
    """
    return {k: v for k, v in dict_.items() if k in names and v is not None}


_filter_future_kwargs = _filter_kwargs(Future._kwargnames)
_filter_equity_kwargs = _filter_kwargs(Equity._kwargnames)


def _convert_asset_timestamp_fields(dict_):
    """Takes in a dict of Asset init args and converts dates to pd.Timestamps"""
    for key in _asset_timestamp_fields & dict_.keys():
        # TODO FIX TZ MESS
        # value = pd.Timestamp(dict_[key], tz="UTC")
        value = pd.Timestamp(dict_[key], tz=None)
        dict_[key] = None if pd.isnull(value) else value
    return dict_


SID_TYPE_IDS = {
    # Asset would be 0,
    ContinuousFuture: 1,
}

CONTINUOUS_FUTURE_ROLL_STYLE_IDS = {
    "calendar": 0,
    "volume": 1,
}

CONTINUOUS_FUTURE_ADJUSTMENT_STYLE_IDS = {
    None: 0,
    "div": 1,
    "add": 2,
}


def _encode_continuous_future_sid(root_symbol, offset, roll_style, adjustment_style):
    # Generate a unique int identifier
    values = (
        SID_TYPE_IDS[ContinuousFuture],
        offset,
        *[ord(x) for x in root_symbol.upper()],
        CONTINUOUS_FUTURE_ROLL_STYLE_IDS[roll_style],
        CONTINUOUS_FUTURE_ADJUSTMENT_STYLE_IDS[adjustment_style],
    )
    return int("".join([str(x) for x in values]))


Lifetimes = namedtuple("Lifetimes", "sid start end")


class AssetFinder:
    """An AssetFinder is an interface to a database of Asset metadata written by
    an ``AssetDBWriter``.

    This class provides methods for looking up assets by unique integer id or
    by symbol.  For historical reasons, we refer to these unique ids as 'sids'.

    Parameters
    ----------
    engine : str or SQLAlchemy.engine
        An engine with a connection to the asset database to use, or a string
        that can be parsed by SQLAlchemy as a URI.
    future_chain_predicates : dict
        A dict mapping future root symbol to a predicate function which accepts
    a contract as a parameter and returns whether or not the contract should be
    included in the chain.

    See Also
    --------
    :class:`zipline.assets.AssetDBWriter`
    """

    @preprocess(engine=coerce_string_to_eng(require_exists=True))
    def __init__(self, engine, future_chain_predicates=CHAIN_PREDICATES):
        self.engine = engine
        metadata_obj = sa.MetaData()
        metadata_obj.reflect(engine, only=asset_db_table_names)
        for table_name in asset_db_table_names:
            setattr(self, table_name, metadata_obj.tables[table_name])

        # Check the version info of the db for compatibility
        with engine.connect() as conn:
            check_version_info(conn, self.version_info, ASSET_DB_VERSION)

        # Cache for lookup of assets by sid, the objects in the asset lookup
        # may be shared with the results from equity and future lookup caches.
        #
        # The top level cache exists to minimize lookups on the asset type
        # routing.
        #
        # The caches are read through, i.e. accessing an asset through
        # retrieve_asset will populate the cache on first retrieval.
        self._asset_cache = {}
        self._asset_type_cache = {}
        self._caches = (self._asset_cache, self._asset_type_cache)

        self._future_chain_predicates = (
            future_chain_predicates if future_chain_predicates is not None else {}
        )
        self._ordered_contracts = {}

        # Populated on first call to `lifetimes`.
        self._asset_lifetimes = {}

    @lazyval
    def exchange_info(self):
        with self.engine.connect() as conn:
            es = conn.execute(sa.select(self.exchanges.c)).fetchall()
        return {
            name: ExchangeInfo(name, canonical_name, country_code)
            for name, canonical_name, country_code in es
        }

    @lazyval
    def symbol_ownership_map(self):
        out = {}
        for mappings in self.symbol_ownership_maps_by_country_code.values():
            for key, ownership_periods in mappings.items():
                out.setdefault(key, []).extend(ownership_periods)

        return out

    @lazyval
    def symbol_ownership_maps_by_country_code(self):
        with self.engine.connect() as conn:
            query = sa.select(
                self.equities.c.sid,
                self.exchanges.c.country_code,
            ).where(self.equities.c.exchange == self.exchanges.c.exchange)
            sid_to_country_code = dict(conn.execute(query).fetchall())

            return build_grouped_ownership_map(
                conn,
                table=self.equity_symbol_mappings,
                key_from_row=(lambda row: (row.company_symbol, row.share_class_symbol)),
                value_from_row=lambda row: row.symbol,
                group_key=lambda row: sid_to_country_code[row.sid],
            )

    @lazyval
    def country_codes(self):
        return tuple(self.symbol_ownership_maps_by_country_code)

    @staticmethod
    def _fuzzify_symbol_ownership_map(ownership_map):
        fuzzy_mappings = {}
        for (cs, scs), owners in ownership_map.items():
            fuzzy_owners = fuzzy_mappings.setdefault(
                cs + scs,
                [],
            )
            fuzzy_owners.extend(owners)
            fuzzy_owners.sort()
        return fuzzy_mappings

    @lazyval
    def fuzzy_symbol_ownership_map(self):
        return self._fuzzify_symbol_ownership_map(self.symbol_ownership_map)

    @lazyval
    def fuzzy_symbol_ownership_maps_by_country_code(self):
        return valmap(
            self._fuzzify_symbol_ownership_map,
            self.symbol_ownership_maps_by_country_code,
        )

    @lazyval
    def equity_supplementary_map(self):
        with self.engine.connect() as conn:
            return build_ownership_map(
                conn,
                table=self.equity_supplementary_mappings,
                key_from_row=lambda row: (row.field, row.value),
                value_from_row=lambda row: row.value,
            )

    @lazyval
    def equity_supplementary_map_by_sid(self):
        with self.engine.connect() as conn:
            return build_ownership_map(
                conn,
                table=self.equity_supplementary_mappings,
                key_from_row=lambda row: (row.field, row.sid),
                value_from_row=lambda row: row.value,
            )

    def lookup_asset_types(self, sids):
        """Retrieve asset types for a list of sids.

        Parameters
        ----------
        sids : list[int]

        Returns
        -------
        types : dict[sid -> str or None]
            Asset types for the provided sids.
        """
        found = {}
        missing = set()

        for sid in sids:
            try:
                found[sid] = self._asset_type_cache[sid]
            except KeyError:
                missing.add(sid)

        if not missing:
            return found

        router_cols = self.asset_router.c

        with self.engine.connect() as conn:
            for assets in group_into_chunks(missing):
                query = sa.select(router_cols.sid, router_cols.asset_type).where(
                    self.asset_router.c.sid.in_(map(int, assets))
                )
                for sid, type_ in conn.execute(query).fetchall():
                    missing.remove(sid)
                    found[sid] = self._asset_type_cache[sid] = type_

                for sid in missing:
                    found[sid] = self._asset_type_cache[sid] = None

        return found

    def group_by_type(self, sids):
        """Group a list of sids by asset type.

        Parameters
        ----------
        sids : list[int]

        Returns
        -------
        types : dict[str or None -> list[int]]
            A dict mapping unique asset types to lists of sids drawn from sids.
            If we fail to look up an asset, we assign it a key of None.
        """
        return invert(self.lookup_asset_types(sids))

    def retrieve_asset(self, sid, default_none=False):
        """
        Retrieve the Asset for a given sid.
        """
        try:
            asset = self._asset_cache[sid]
            if asset is None and not default_none:
                raise SidsNotFound(sids=[sid])
            return asset
        except KeyError:
            return self.retrieve_all((sid,), default_none=default_none)[0]

    def retrieve_all(self, sids, default_none=False):
        """Retrieve all assets in `sids`.

        Parameters
        ----------
        sids : iterable of int
            Assets to retrieve.
        default_none : bool
            If True, return None for failed lookups.
            If False, raise `SidsNotFound`.

        Returns
        -------
        assets : list[Asset or None]
            A list of the same length as `sids` containing Assets (or Nones)
            corresponding to the requested sids.

        Raises
        ------
        SidsNotFound
            When a requested sid is not found and default_none=False.
        """
        sids = list(sids)
        hits, missing, failures = {}, set(), []
        for sid in sids:
            try:
                asset = self._asset_cache[sid]
                if not default_none and asset is None:
                    # Bail early if we've already cached that we don't know
                    # about an asset.
                    raise SidsNotFound(sids=[sid])
                hits[sid] = asset
            except KeyError:
                missing.add(sid)

        # All requests were cache hits.  Return requested sids in order.
        if not missing:
            return [hits[sid] for sid in sids]

        update_hits = hits.update

        # Look up cache misses by type.
        type_to_assets = self.group_by_type(missing)

        # Handle failures
        failures = {failure: None for failure in type_to_assets.pop(None, ())}
        update_hits(failures)
        self._asset_cache.update(failures)

        if failures and not default_none:
            raise SidsNotFound(sids=list(failures))

        # We don't update the asset cache here because it should already be
        # updated by `self.retrieve_equities`.
        update_hits(self.retrieve_equities(type_to_assets.pop("equity", ())))
        update_hits(self.retrieve_futures_contracts(type_to_assets.pop("future", ())))

        # We shouldn't know about any other asset types.
        if type_to_assets:
            raise AssertionError("Found asset types: %s" % list(type_to_assets.keys()))

        return [hits[sid] for sid in sids]

    def retrieve_equities(self, sids):
        """Retrieve Equity objects for a list of sids.

        Users generally shouldn't need to this method (instead, they should
        prefer the more general/friendly `retrieve_assets`), but it has a
        documented interface and tests because it's used upstream.

        Parameters
        ----------
        sids : iterable[int]

        Returns
        -------
        equities : dict[int -> Equity]

        Raises
        ------
        EquitiesNotFound
            When any requested asset isn't found.
        """
        return self._retrieve_assets(sids, self.equities, Equity)

    def _retrieve_equity(self, sid):
        return self.retrieve_equities((sid,))[sid]

    def retrieve_futures_contracts(self, sids):
        """Retrieve Future objects for an iterable of sids.

        Users generally shouldn't need to this method (instead, they should
        prefer the more general/friendly `retrieve_assets`), but it has a
        documented interface and tests because it's used upstream.

        Parameters
        ----------
        sids : iterable[int]

        Returns
        -------
        equities : dict[int -> Equity]

        Raises
        ------
        EquitiesNotFound
            When any requested asset isn't found.
        """
        return self._retrieve_assets(sids, self.futures_contracts, Future)

    @staticmethod
    def _select_assets_by_sid(asset_tbl, sids):
        return sa.select(asset_tbl).where(asset_tbl.c.sid.in_(map(int, sids)))

    @staticmethod
    def _select_asset_by_symbol(asset_tbl, symbol):
        return sa.select(asset_tbl).where(asset_tbl.c.symbol == symbol)

    def _select_most_recent_symbols_chunk(self, sid_group):
        """Retrieve the most recent symbol for a set of sids.

        Parameters
        ----------
        sid_group : iterable[int]
            The sids to lookup. The length of this sequence must be less than
            or equal to SQLITE_MAX_VARIABLE_NUMBER because the sids will be
            passed in as sql bind params.

        Returns
        -------
        sel : Selectable
            The sqlalchemy selectable that will query for the most recent
            symbol for each sid.

        Notes
        -----
        This is implemented as an inner select of the columns of interest
        ordered by the end date of the (sid, symbol) mapping. We then group
        that inner select on the sid with no aggregations to select the last
        row per group which gives us the most recently active symbol for all
        of the sids.
        """
        cols = self.equity_symbol_mappings.c

        # These are the columns we actually want.
        data_cols = (cols.sid,) + tuple(cols[name] for name in SYMBOL_COLUMNS)

        # Also select the max of end_date so that all non-grouped fields take
        # on the value associated with the max end_date.
        # to_select = data_cols + (sa.func.max(cols.end_date),)
        func_rank = (
            sa.func.rank()
            .over(order_by=cols.end_date.desc(), partition_by=cols.sid)
            .label("rnk")
        )
        to_select = data_cols + (func_rank,)

        subquery = (
            sa.select(*to_select)
            .where(cols.sid.in_(map(int, sid_group)))
            .subquery("sq")
        )
        query = (
            sa.select(subquery.columns)
            .filter(subquery.c.rnk == 1)
            .select_from(subquery)
        )
        return query

    def _lookup_most_recent_symbols(self, sids):
        with self.engine.connect() as conn:
            return {
                row.sid: {c: row[c] for c in SYMBOL_COLUMNS}
                for row in concat(
                    conn.execute(self._select_most_recent_symbols_chunk(sid_group))
                    .mappings()
                    .fetchall()
                    for sid_group in partition_all(SQLITE_MAX_VARIABLE_NUMBER, sids)
                )
            }

    def _retrieve_asset_dicts(self, sids, asset_tbl, querying_equities):
        if not sids:
            return

        if querying_equities:

            def mkdict(
                row,
                exchanges=self.exchange_info,
                symbols=self._lookup_most_recent_symbols(sids),
            ):
                d = dict(row)
                d["exchange_info"] = exchanges[d.pop("exchange")]
                # we are not required to have a symbol for every asset, if
                # we don't have any symbols we will just use the empty string
                return merge(d, symbols.get(row["sid"], {}))

        else:

            def mkdict(row, exchanges=self.exchange_info):
                d = dict(row)
                d["exchange_info"] = exchanges[d.pop("exchange")]
                return d

        for assets in group_into_chunks(sids):
            # Load misses from the db.
            query = self._select_assets_by_sid(asset_tbl, assets)

            with self.engine.connect() as conn:
                for row in conn.execute(query).mappings().fetchall():
                    yield _convert_asset_timestamp_fields(mkdict(row))

    def _retrieve_assets(self, sids, asset_tbl, asset_type):
        """Internal function for loading assets from a table.

        This should be the only method of `AssetFinder` that writes Assets into
        self._asset_cache.

        Parameters
        ---------
        sids : iterable of int
            Asset ids to look up.
        asset_tbl : sqlalchemy.Table
            Table from which to query assets.
        asset_type : type
            Type of asset to be constructed.

        Returns
        -------
        assets : dict[int -> Asset]
            Dict mapping requested sids to the retrieved assets.
        """
        # Fastpath for empty request.
        if not sids:
            return {}

        cache = self._asset_cache
        hits = {}

        querying_equities = issubclass(asset_type, Equity)
        filter_kwargs = (
            _filter_equity_kwargs if querying_equities else _filter_future_kwargs
        )

        rows = self._retrieve_asset_dicts(sids, asset_tbl, querying_equities)
        for row in rows:
            sid = row["sid"]
            asset = asset_type(**filter_kwargs(row))
            hits[sid] = cache[sid] = asset

        # If we get here, it means something in our code thought that a
        # particular sid was an equity/future and called this function with a
        # concrete type, but we couldn't actually resolve the asset.  This is
        # an error in our code, not a user-input error.
        misses = tuple(set(sids) - hits.keys())
        if misses:
            if querying_equities:
                raise EquitiesNotFound(sids=misses)
            else:
                raise FutureContractsNotFound(sids=misses)
        return hits

    def _lookup_symbol_strict(self, ownership_map, multi_country, symbol, as_of_date):
        """Resolve a symbol to an asset object without fuzzy matching.

        Parameters
        ----------
        ownership_map : dict[(str, str), list[OwnershipPeriod]]
            The mapping from split symbols to ownership periods.
        multi_country : bool
            Does this mapping span multiple countries?
        symbol : str
            The symbol to look up.
        as_of_date : datetime or None
            If multiple assets have held this sid, which day should the
            resolution be checked against? If this value is None and multiple
            sids have held the ticker, then a MultipleSymbolsFound error will
            be raised.

        Returns
        -------
        asset : Asset
            The asset that held the given symbol.

        Raises
        ------
        SymbolNotFound
            Raised when the symbol or symbol as_of_date pair do not map to
            any assets.
        MultipleSymbolsFound
            Raised when multiple assets held the symbol. This happens if
            multiple assets held the symbol at disjoint times and
            ``as_of_date`` is None, or if multiple assets held the symbol at
            the same time and``multi_country`` is True.

        Notes
        -----
        The resolution algorithm is as follows:

        - Split the symbol into the company and share class component.
        - Do a dictionary lookup of the
          ``(company_symbol, share_class_symbol)`` in the provided ownership
          map.
        - If there is no entry in the dictionary, we don't know about this
          symbol so raise a ``SymbolNotFound`` error.
        - If ``as_of_date`` is None:
          - If more there is more than one owner, raise
            ``MultipleSymbolsFound``
          - Otherwise, because the list mapped to a symbol cannot be empty,
            return the single asset.
        - Iterate through all of the owners:
          - If the ``as_of_date`` is between the start and end of the ownership
            period:
            - If multi_country is False, return the found asset.
            - Otherwise, put the asset in a list.
        - At the end of the loop, if there are no candidate assets, raise a
          ``SymbolNotFound``.
        - If there is exactly one candidate, return it.
        - Othewise, raise ``MultipleSymbolsFound`` because the ticker is not
          unique across countries.
        """
        # split the symbol into the components, if there are no
        # company/share class parts then share_class_symbol will be empty
        company_symbol, share_class_symbol = split_delimited_symbol(symbol)
        try:
            owners = ownership_map[company_symbol, share_class_symbol]
            assert owners, "empty owners list for %r" % symbol
        except KeyError as exc:
            # no equity has ever held this symbol
            raise SymbolNotFound(symbol=symbol) from exc

        if not as_of_date:
            # exactly one equity has ever held this symbol, we may resolve
            # without the date
            if len(owners) == 1:
                return self.retrieve_asset(owners[0].sid)

            options = {self.retrieve_asset(owner.sid) for owner in owners}

            if multi_country:
                country_codes = map(attrgetter("country_code"), options)

                if len(set(country_codes)) > 1:
                    raise SameSymbolUsedAcrossCountries(
                        symbol=symbol, options=dict(zip(country_codes, options))
                    )

            # more than one equity has held this ticker, this
            # is ambiguous without the date
            raise MultipleSymbolsFound(symbol=symbol, options=options)

        options = []
        country_codes = []
        for start, end, sid, _ in owners:
            if start <= as_of_date < end:
                # find the equity that owned it on the given asof date
                asset = self.retrieve_asset(sid)

                # if this asset owned the symbol on this asof date and we are
                # only searching one country, return that asset
                if not multi_country:
                    return asset
                else:
                    options.append(asset)
                    country_codes.append(asset.country_code)

        if not options:
            # no equity held the ticker on the given asof date
            raise SymbolNotFound(symbol=symbol)

        # if there is one valid option given the asof date, return that option
        if len(options) == 1:
            return options[0]

        # if there's more than one option given the asof date, a country code
        # must be passed to resolve the symbol to an asset
        raise SameSymbolUsedAcrossCountries(
            symbol=symbol, options=dict(zip(country_codes, options))
        )

    def _lookup_symbol_fuzzy(self, ownership_map, multi_country, symbol, as_of_date):
        symbol = symbol.upper()
        company_symbol, share_class_symbol = split_delimited_symbol(symbol)
        try:
            owners = ownership_map[company_symbol + share_class_symbol]
            assert owners, "empty owners list for %r" % symbol
        except KeyError as exc:
            # no equity has ever held a symbol matching the fuzzy symbol
            raise SymbolNotFound(symbol=symbol) from exc

        if not as_of_date:
            if len(owners) == 1:
                # only one valid match
                return self.retrieve_asset(owners[0].sid)

            options = []
            for _, _, sid, sym in owners:
                if sym == symbol:
                    # there are multiple options, look for exact matches
                    options.append(self.retrieve_asset(sid))

            if len(options) == 1:
                # there was only one exact match
                return options[0]

            # there is more than one exact match for this fuzzy symbol
            raise MultipleSymbolsFoundForFuzzySymbol(
                symbol=symbol,
                options=self.retrieve_all(owner.sid for owner in owners),
            )

        options = {}
        for start, end, sid, sym in owners:
            if start <= as_of_date < end:
                # see which fuzzy symbols were owned on the asof date.
                options[sid] = sym

        if not options:
            # no equity owned the fuzzy symbol on the date requested
            raise SymbolNotFound(symbol=symbol)

        sid_keys = list(options.keys())
        # If there was only one owner, or there is a fuzzy and non-fuzzy which
        # map to the same sid, return it.
        if len(options) == 1:
            return self.retrieve_asset(sid_keys[0])

        exact_options = []
        for sid, sym in options.items():
            # Possible to have a scenario where multiple fuzzy matches have the
            # same date. Want to find the one where symbol and share class
            # match.
            if (company_symbol, share_class_symbol) == split_delimited_symbol(sym):
                asset = self.retrieve_asset(sid)
                if not multi_country:
                    return asset
                else:
                    exact_options.append(asset)

        if len(exact_options) == 1:
            return exact_options[0]

        # multiple equities held tickers matching the fuzzy ticker but
        # there are no exact matches
        raise MultipleSymbolsFoundForFuzzySymbol(
            symbol=symbol,
            options=self.retrieve_all(owner.sid for owner in owners),
        )

    def _choose_fuzzy_symbol_ownership_map(self, country_code):
        if country_code is None:
            return self.fuzzy_symbol_ownership_map

        return self.fuzzy_symbol_ownership_maps_by_country_code.get(
            country_code,
        )

    def _choose_symbol_ownership_map(self, country_code):
        if country_code is None:
            return self.symbol_ownership_map

        return self.symbol_ownership_maps_by_country_code.get(country_code)

    def lookup_symbol(self, symbol, as_of_date, fuzzy=False, country_code=None):
        """Lookup an equity by symbol.

        Parameters
        ----------
        symbol : str
            The ticker symbol to resolve.
        as_of_date : datetime.datetime or None
            Look up the last owner of this symbol as of this datetime.
            If ``as_of_date`` is None, then this can only resolve the equity
            if exactly one equity has ever owned the ticker.
        fuzzy : bool, optional
            Should fuzzy symbol matching be used? Fuzzy symbol matching
            attempts to resolve differences in representations for
            shareclasses. For example, some people may represent the ``A``
            shareclass of ``BRK`` as ``BRK.A``, where others could write
            ``BRK_A``.
        country_code : str or None, optional
            The country to limit searches to. If not provided, the search will
            span all countries which increases the likelihood of an ambiguous
            lookup.

        Returns
        -------
        equity : Equity
            The equity that held ``symbol`` on the given ``as_of_date``, or the
            only equity to hold ``symbol`` if ``as_of_date`` is None.

        Raises
        ------
        SymbolNotFound
            Raised when no equity has ever held the given symbol.
        MultipleSymbolsFound
            Raised when no ``as_of_date`` is given and more than one equity
            has held ``symbol``. This is also raised when ``fuzzy=True`` and
            there are multiple candidates for the given ``symbol`` on the
            ``as_of_date``. Also raised when no ``country_code`` is given and
            the symbol is ambiguous across multiple countries.
        """
        if symbol is None:
            raise TypeError(
                "Cannot lookup asset for symbol of None for "
                "as of date %s." % as_of_date
            )

        if fuzzy:
            f = self._lookup_symbol_fuzzy
            mapping = self._choose_fuzzy_symbol_ownership_map(country_code)
        else:
            f = self._lookup_symbol_strict
            mapping = self._choose_symbol_ownership_map(country_code)

        if mapping is None:
            raise SymbolNotFound(symbol=symbol)
        return f(
            mapping,
            country_code is None,
            symbol,
            as_of_date,
        )

    def lookup_symbols(self, symbols, as_of_date, fuzzy=False, country_code=None):
        """Lookup a list of equities by symbol.

        Equivalent to::

            [finder.lookup_symbol(s, as_of, fuzzy) for s in symbols]

        but potentially faster because repeated lookups are memoized.

        Parameters
        ----------
        symbols : sequence[str]
            Sequence of ticker symbols to resolve.
        as_of_date : pd.Timestamp
            Forwarded to ``lookup_symbol``.
        fuzzy : bool, optional
            Forwarded to ``lookup_symbol``.
        country_code : str or None, optional
            The country to limit searches to. If not provided, the search will
            span all countries which increases the likelihood of an ambiguous
            lookup.

        Returns
        -------
        equities : list[Equity]
        """
        if not symbols:
            return []

        multi_country = country_code is None
        if fuzzy:
            f = self._lookup_symbol_fuzzy
            mapping = self._choose_fuzzy_symbol_ownership_map(country_code)
        else:
            f = self._lookup_symbol_strict
            mapping = self._choose_symbol_ownership_map(country_code)

        if mapping is None:
            raise SymbolNotFound(symbol=symbols[0])

        memo = {}
        out = []
        append_output = out.append
        for sym in symbols:
            if sym in memo:
                append_output(memo[sym])
            else:
                equity = memo[sym] = f(
                    mapping,
                    multi_country,
                    sym,
                    as_of_date,
                )
                append_output(equity)
        return out

    def lookup_future_symbol(self, symbol):
        """Lookup a future contract by symbol.

        Parameters
        ----------
        symbol : str
            The symbol of the desired contract.

        Returns
        -------
        future : Future
            The future contract referenced by ``symbol``.

        Raises
        ------
        SymbolNotFound
            Raised when no contract named 'symbol' is found.

        """
        with self.engine.connect() as conn:
            data = (
                conn.execute(
                    self._select_asset_by_symbol(self.futures_contracts, symbol)
                )
                .mappings()
                .fetchone()
            )

        # If no data found, raise an exception
        if not data:
            raise SymbolNotFound(symbol=symbol)
        return self.retrieve_asset(data["sid"])

    def lookup_by_supplementary_field(self, field_name, value, as_of_date):
        try:
            owners = self.equity_supplementary_map[
                field_name,
                value,
            ]
            assert owners, "empty owners list for field %r (sid: %r)" % (
                field_name,
                value,
            )
        except KeyError as exc:
            # no equity has ever held this value
            raise ValueNotFoundForField(field=field_name, value=value) from exc

        if not as_of_date:
            if len(owners) > 1:
                # more than one equity has held this value, this is ambigious
                # without the date
                raise MultipleValuesFoundForField(
                    field=field_name,
                    value=value,
                    options=set(
                        map(
                            compose(self.retrieve_asset, attrgetter("sid")),
                            owners,
                        )
                    ),
                )
            # exactly one equity has ever held this value, we may resolve
            # without the date
            return self.retrieve_asset(owners[0].sid)

        for start, end, sid, _ in owners:
            if start <= as_of_date < end:
                # find the equity that owned it on the given asof date
                return self.retrieve_asset(sid)

        # no equity held the value on the given asof date
        raise ValueNotFoundForField(field=field_name, value=value)

    def get_supplementary_field(self, sid, field_name, as_of_date):
        """Get the value of a supplementary field for an asset.

        Parameters
        ----------
        sid : int
            The sid of the asset to query.
        field_name : str
            Name of the supplementary field.
        as_of_date : pd.Timestamp, None
            The last known value on this date is returned. If None, a
            value is returned only if we've only ever had one value for
            this sid. If None and we've had multiple values,
            MultipleValuesFoundForSid is raised.

        Raises
        ------
        NoValueForSid
            If we have no values for this asset, or no values was known
            on this as_of_date.
        MultipleValuesFoundForSid
            If we have had multiple values for this asset over time, and
            None was passed for as_of_date.
        """
        try:
            periods = self.equity_supplementary_map_by_sid[
                field_name,
                sid,
            ]
            assert periods, "empty periods list for field %r and sid %r" % (
                field_name,
                sid,
            )
        except KeyError:
            raise NoValueForSid(field=field_name, sid=sid) from KeyError

        if not as_of_date:
            if len(periods) > 1:
                # This equity has held more than one value, this is ambigious
                # without the date
                raise MultipleValuesFoundForSid(
                    field=field_name,
                    sid=sid,
                    options={p.value for p in periods},
                )
            # this equity has only ever held this value, we may resolve
            # without the date
            return periods[0].value

        for start, end, _, value in periods:
            if start <= as_of_date < end:
                return value

        # Could not find a value for this sid on the as_of_date.
        raise NoValueForSid(field=field_name, sid=sid)

    def _get_contract_sids(self, root_symbol):
        fc_cols = self.futures_contracts.c
        with self.engine.connect() as conn:
            return (
                conn.execute(
                    sa.select(
                        fc_cols.sid,
                    )
                    .where(
                        (fc_cols.root_symbol == root_symbol)
                        & (fc_cols.start_date != pd.NaT.value)
                    )
                    .order_by(fc_cols.sid)
                )
                .scalars()
                .fetchall()
            )

    def _get_root_symbol_exchange(self, root_symbol):
        fc_cols = self.futures_root_symbols.c
        fields = (fc_cols.exchange,)

        with self.engine.connect() as conn:
            exchange = conn.execute(
                sa.select(*fields).where(fc_cols.root_symbol == root_symbol)
            ).scalar()

        if exchange is not None:
            return exchange
        else:
            raise SymbolNotFound(symbol=root_symbol)

    def get_ordered_contracts(self, root_symbol):
        try:
            return self._ordered_contracts[root_symbol]
        except KeyError:
            contract_sids = self._get_contract_sids(root_symbol)
            contracts = deque(self.retrieve_all(contract_sids))
            chain_predicate = self._future_chain_predicates.get(root_symbol, None)
            oc = OrderedContracts(root_symbol, contracts, chain_predicate)
            self._ordered_contracts[root_symbol] = oc
            return oc

    def create_continuous_future(self, root_symbol, offset, roll_style, adjustment):
        if adjustment not in ADJUSTMENT_STYLES:
            raise ValueError(
                f"Invalid adjustment style {adjustment!r}. Allowed adjustment styles are "
                f"{list(ADJUSTMENT_STYLES)}."
            )

        oc = self.get_ordered_contracts(root_symbol)
        exchange = self._get_root_symbol_exchange(root_symbol)

        sid = _encode_continuous_future_sid(root_symbol, offset, roll_style, None)
        mul_sid = _encode_continuous_future_sid(root_symbol, offset, roll_style, "div")
        add_sid = _encode_continuous_future_sid(root_symbol, offset, roll_style, "add")

        cf_template = partial(
            ContinuousFuture,
            root_symbol=root_symbol,
            offset=offset,
            roll_style=roll_style,
            start_date=oc.start_date,
            end_date=oc.end_date,
            exchange_info=self.exchange_info[exchange],
        )

        cf = cf_template(sid=sid)
        mul_cf = cf_template(sid=mul_sid, adjustment="mul")
        add_cf = cf_template(sid=add_sid, adjustment="add")

        self._asset_cache[cf.sid] = cf
        self._asset_cache[mul_cf.sid] = mul_cf
        self._asset_cache[add_cf.sid] = add_cf

        return {None: cf, "mul": mul_cf, "add": add_cf}[adjustment]

    def _make_sids(tblattr):
        def _(self):
            with self.engine.connect() as conn:
                return tuple(
                    conn.execute(sa.select(getattr(self, tblattr).c.sid))
                    .scalars()
                    .fetchall()
                )

        return _

    sids = property(
        _make_sids("asset_router"),
        doc="All the sids in the asset finder.",
    )
    equities_sids = property(
        _make_sids("equities"),
        doc="All of the sids for equities in the asset finder.",
    )
    futures_sids = property(
        _make_sids("futures_contracts"),
        doc="All of the sids for futures consracts in the asset finder.",
    )
    del _make_sids

    def _lookup_generic_scalar(self, obj, as_of_date, country_code, matches, missing):
        """
        Convert asset_convertible to an asset.

        On success, append to matches.
        On failure, append to missing.
        """
        result = self._lookup_generic_scalar_helper(
            obj,
            as_of_date,
            country_code,
        )
        if result is not None:
            matches.append(result)
        else:
            missing.append(obj)

    def _lookup_generic_scalar_helper(self, obj, as_of_date, country_code):
        if isinstance(obj, (Asset, ContinuousFuture)):
            return obj

        if isinstance(obj, Integral):
            try:
                return self.retrieve_asset(int(obj))
            except SidsNotFound:
                return None

        if isinstance(obj, str):
            # Try to look up as an equity first.
            try:
                return self.lookup_symbol(
                    symbol=obj, as_of_date=as_of_date, country_code=country_code
                )
            except SymbolNotFound:
                # Fall back to lookup as a Future
                try:
                    # TODO: Support country_code for future_symbols?
                    return self.lookup_future_symbol(obj)
                except SymbolNotFound:
                    return None

        raise NotAssetConvertible("Input was %s, not AssetConvertible." % obj)

    def lookup_generic(self, obj, as_of_date, country_code):
        """Convert an object into an Asset or sequence of Assets.

        This method exists primarily as a convenience for implementing
        user-facing APIs that can handle multiple kinds of input.  It should
        not be used for internal code where we already know the expected types
        of our inputs.

        Parameters
        ----------
        obj : int, str, Asset, ContinuousFuture, or iterable
            The object to be converted into one or more Assets.
            Integers are interpreted as sids. Strings are interpreted as
            tickers. Assets and ContinuousFutures are returned unchanged.
        as_of_date : pd.Timestamp or None
            Timestamp to use to disambiguate ticker lookups. Has the same
            semantics as in `lookup_symbol`.
        country_code : str or None
            ISO-3166 country code to use to disambiguate ticker lookups. Has
            the same semantics as in `lookup_symbol`.

        Returns
        -------
        matches, missing : tuple
            ``matches`` is the result of the conversion. ``missing`` is a list
             containing any values that couldn't be resolved. If ``obj`` is not
             an iterable, ``missing`` will be an empty list.
        """
        matches = []
        missing = []

        # Interpret input as scalar.
        if isinstance(obj, (AssetConvertible, ContinuousFuture)):
            self._lookup_generic_scalar(
                obj=obj,
                as_of_date=as_of_date,
                country_code=country_code,
                matches=matches,
                missing=missing,
            )
            try:
                return matches[0], missing
            except IndexError:
                if hasattr(obj, "__int__"):
                    raise SidsNotFound(sids=[obj]) from IndexError
                else:
                    raise SymbolNotFound(symbol=obj) from IndexError

        # Interpret input as iterable.
        try:
            iterator = iter(obj)
        except TypeError:
            raise NotAssetConvertible(
                "Input was not a AssetConvertible or iterable of AssetConvertible."
            ) from TypeError

        for obj in iterator:
            self._lookup_generic_scalar(
                obj=obj,
                as_of_date=as_of_date,
                country_code=country_code,
                matches=matches,
                missing=missing,
            )

        return matches, missing

    def _compute_asset_lifetimes(self, **kwargs):
        """Compute and cache a recarray of asset lifetimes"""
        sids = starts = ends = []
        equities_cols = self.equities.c
        exchanges_cols = self.exchanges.c
        if len(kwargs) == 1:
            if "country_codes" in kwargs.keys():
                condt = exchanges_cols.country_code.in_(kwargs["country_codes"])
            if "exchange_names" in kwargs.keys():
                condt = exchanges_cols.exchange.in_(kwargs["exchange_names"])

            with self.engine.connect() as conn:
                results = conn.execute(
                    sa.select(
                        equities_cols.sid,
                        equities_cols.start_date,
                        equities_cols.end_date,
                    ).where(
                        (exchanges_cols.exchange == equities_cols.exchange) & (condt)
                    )
                ).fetchall()
            if results:
                sids, starts, ends = zip(*results)

        sid = np.array(sids, dtype="i8")
        start = np.array(starts, dtype="f8")
        end = np.array(ends, dtype="f8")
        start[np.isnan(start)] = 0  # convert missing starts to 0
        end[np.isnan(end)] = np.iinfo(int).max  # convert missing end to INTMAX
        return Lifetimes(sid, start.astype("i8"), end.astype("i8"))

    def lifetimes(self, dates, include_start_date, country_codes):
        """Compute a DataFrame representing asset lifetimes for the specified date
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
        country_codes : iterable[str]
            The country codes to get lifetimes for.

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
        if isinstance(country_codes, str):
            raise TypeError(
                "Got string {!r} instead of an iterable of strings in "
                "AssetFinder.lifetimes.".format(country_codes),
            )

        # normalize to a cache-key so that we can memoize results.
        country_codes = frozenset(country_codes)

        lifetimes = self._asset_lifetimes.get(country_codes)
        if lifetimes is None:
            self._asset_lifetimes[
                country_codes
            ] = lifetimes = self._compute_asset_lifetimes(country_codes=country_codes)

        raw_dates = as_column(dates.asi8)
        if include_start_date:
            mask = lifetimes.start <= raw_dates
        else:
            mask = lifetimes.start < raw_dates
        mask &= raw_dates <= lifetimes.end

        return pd.DataFrame(mask, index=dates, columns=lifetimes.sid)

    def equities_sids_for_country_code(self, country_code):
        """Return all of the sids for a given country.

        Parameters
        ----------
        country_code : str
            An ISO 3166 alpha-2 country code.

        Returns
        -------
        tuple[int]
            The sids whose exchanges are in this country.
        """
        sids = self._compute_asset_lifetimes(country_codes=[country_code]).sid
        return tuple(sids.tolist())

    def equities_sids_for_exchange_name(self, exchange_name):
        """Return all of the sids for a given exchange_name.

        Parameters
        ----------
        exchange_name : str

        Returns
        -------
        tuple[int]
            The sids whose exchanges are in this country.
        """
        sids = self._compute_asset_lifetimes(exchange_names=[exchange_name]).sid
        return tuple(sids.tolist())


class AssetConvertible(ABC):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Asset, str, and Integral
    """

    pass


AssetConvertible.register(Integral)
AssetConvertible.register(Asset)
AssetConvertible.register(str)


class NotAssetConvertible(ValueError):
    pass


class PricingDataAssociable(ABC):
    """ABC for types that can be associated with pricing data.

    Includes Asset, Future, ContinuousFuture
    """

    pass


PricingDataAssociable.register(Asset)
PricingDataAssociable.register(Future)
PricingDataAssociable.register(ContinuousFuture)


def was_active(reference_date_value, asset):
    """Whether or not `asset` was active at the time corresponding to
    `reference_date_value`.

    Parameters
    ----------
    reference_date_value : int
        Date, represented as nanoseconds since EPOCH, for which we want to know
        if `asset` was alive.  This is generally the result of accessing the
        `value` attribute of a pandas Timestamp.
    asset : Asset
        The asset object to check.

    Returns
    -------
    was_active : bool
        Whether or not the `asset` existed at the specified time.
    """
    return asset.start_date.value <= reference_date_value <= asset.end_date.value


def only_active_assets(reference_date_value, assets):
    """Filter an iterable of Asset objects down to just assets that were alive at
    the time corresponding to `reference_date_value`.

    Parameters
    ----------
    reference_date_value : int
        Date, represented as nanoseconds since EPOCH, for which we want to know
        if `asset` was alive.  This is generally the result of accessing the
        `value` attribute of a pandas Timestamp.
    assets : iterable[Asset]
        The assets to filter.

    Returns
    -------
    active_assets : list
        List of the active assets from `assets` on the requested date.
    """
    return [a for a in assets if was_active(reference_date_value, a)]
