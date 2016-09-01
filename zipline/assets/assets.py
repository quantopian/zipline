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

from abc import ABCMeta
from collections import namedtuple
from numbers import Integral
from operator import itemgetter, attrgetter

from logbook import Logger
import numpy as np
import pandas as pd
from pandas import isnull
from six import with_metaclass, string_types, viewkeys, iteritems
import sqlalchemy as sa
from toolz import (
    compose,
    concat,
    concatv,
    curry,
    merge,
    partition_all,
    sliding_window,
    valmap,
)
from toolz.curried import operator as op

from zipline.errors import (
    EquitiesNotFound,
    FutureContractsNotFound,
    MapAssetIdentifierIndexError,
    MultipleSymbolsFound,
    RootSymbolNotFound,
    SidsNotFound,
    SymbolNotFound,
)
from . import (
    Asset, Equity, Future,
)
from .asset_writer import (
    check_version_info,
    split_delimited_symbol,
    asset_db_table_names,
    symbol_columns,
    SQLITE_MAX_VARIABLE_NUMBER,
)
from .asset_db_schema import (
    ASSET_DB_VERSION
)
from zipline.utils.control_flow import invert
from zipline.utils.memoize import lazyval, weak_lru_cache
from zipline.utils.numpy_utils import as_column
from zipline.utils.preprocess import preprocess
from zipline.utils.sqlite_utils import group_into_chunks, coerce_string_to_eng

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

SymbolOwnership = namedtuple('SymbolOwnership', 'start end sid symbol')


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
    """
    Takes in a dict of Asset init args and converts dates to pd.Timestamps
    """
    for key in _asset_timestamp_fields & viewkeys(dict_):
        value = pd.Timestamp(dict_[key], tz='UTC')
        dict_[key] = None if isnull(value) else value
    return dict_


class AssetFinder(object):
    """
    An AssetFinder is an interface to a database of Asset metadata written by
    an ``AssetDBWriter``.

    This class provides methods for looking up assets by unique integer id or
    by symbol.  For historical reasons, we refer to these unique ids as 'sids'.

    Parameters
    ----------
    engine : str or SQLAlchemy.engine
        An engine with a connection to the asset database to use, or a string
        that can be parsed by SQLAlchemy as a URI.

    See Also
    --------
    :class:`zipline.assets.AssetDBWriter`
    """
    # Token used as a substitute for pickling objects that contain a
    # reference to an AssetFinder.
    PERSISTENT_TOKEN = "<AssetFinder>"

    @preprocess(engine=coerce_string_to_eng)
    def __init__(self, engine):
        self.engine = engine
        metadata = sa.MetaData(bind=engine)
        metadata.reflect(only=asset_db_table_names)
        for table_name in asset_db_table_names:
            setattr(self, table_name, metadata.tables[table_name])

        # Check the version info of the db for compatibility
        check_version_info(engine, self.version_info, ASSET_DB_VERSION)

        # Cache for lookup of assets by sid, the objects in the asset lookup
        # may be shared with the results from equity and future lookup caches.
        #
        # The top level cache exists to minimize lookups on the asset type
        # routing.
        #
        # The caches are read through, i.e. accessing an asset through
        # retrieve_asset will populate the cache on first retrieval.
        self._caches = (self._asset_cache, self._asset_type_cache) = {}, {}

        # Populated on first call to `lifetimes`.
        self._asset_lifetimes = None

    def _reset_caches(self):
        """
        Reset our asset caches.

        You probably shouldn't call this method.
        """
        # This method exists as a workaround for the in-place mutating behavior
        # of `TradingAlgorithm._write_and_map_id_index_to_sids`.  No one else
        # should be calling this.
        for cache in self._caches:
            cache.clear()
        self.reload_symbol_maps()

    def reload_symbol_maps(self):
        """Clear the in memory symbol lookup maps.

        This will make any changes to the underlying db available to the
        symbol maps.
        """
        # clear the lazyval caches, the next access will requery
        try:
            del type(self).symbol_ownership_map[self]
        except KeyError:
            pass
        try:
            del type(self).fuzzy_symbol_ownership_map[self]
        except KeyError:
            pass

    @lazyval
    def symbol_ownership_map(self):
        rows = sa.select(self.equity_symbol_mappings.c).execute().fetchall()

        mappings = {}
        for row in rows:
            mappings.setdefault(
                (row.company_symbol, row.share_class_symbol),
                [],
            ).append(
                SymbolOwnership(
                    pd.Timestamp(row.start_date, unit='ns', tz='utc'),
                    pd.Timestamp(row.end_date, unit='ns', tz='utc'),
                    row.sid,
                    row.symbol,
                ),
            )

        return valmap(
            lambda v: tuple(
                SymbolOwnership(
                    a.start,
                    b.start,
                    a.sid,
                    a.symbol,
                ) for a, b in sliding_window(
                    2,
                    concatv(
                        sorted(v),
                        # concat with a fake ownership object to make the last
                        # end date be max timestamp
                        [SymbolOwnership(
                            pd.Timestamp.max.tz_localize('utc'),
                            None,
                            None,
                            None,
                        )],
                    ),
                )
            ),
            mappings,
            factory=lambda: mappings,
        )

    @lazyval
    def fuzzy_symbol_ownership_map(self):
        fuzzy_mappings = {}
        for (cs, scs), owners in iteritems(self.symbol_ownership_map):
            fuzzy_owners = fuzzy_mappings.setdefault(
                cs + scs,
                [],
            )
            fuzzy_owners.extend(owners)
            fuzzy_owners.sort()
        return fuzzy_mappings

    def lookup_asset_types(self, sids):
        """
        Retrieve asset types for a list of sids.

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

        for assets in group_into_chunks(missing):
            query = sa.select((router_cols.sid, router_cols.asset_type)).where(
                self.asset_router.c.sid.in_(map(int, assets))
            )
            for sid, type_ in query.execute().fetchall():
                missing.remove(sid)
                found[sid] = self._asset_type_cache[sid] = type_

            for sid in missing:
                found[sid] = self._asset_type_cache[sid] = None

        return found

    def group_by_type(self, sids):
        """
        Group a list of sids by asset type.

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
        return self.retrieve_all((sid,), default_none=default_none)[0]

    def retrieve_all(self, sids, default_none=False):
        """
        Retrieve all assets in `sids`.

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
        update_hits(self.retrieve_equities(type_to_assets.pop('equity', ())))
        update_hits(
            self.retrieve_futures_contracts(type_to_assets.pop('future', ()))
        )

        # We shouldn't know about any other asset types.
        if type_to_assets:
            raise AssertionError(
                "Found asset types: %s" % list(type_to_assets.keys())
            )

        return [hits[sid] for sid in sids]

    def retrieve_equities(self, sids):
        """
        Retrieve Equity objects for a list of sids.

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
        """
        Retrieve Future objects for an iterable of sids.

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
        return sa.select([asset_tbl]).where(
            asset_tbl.c.sid.in_(map(int, sids))
        )

    @staticmethod
    def _select_asset_by_symbol(asset_tbl, symbol):
        return sa.select([asset_tbl]).where(asset_tbl.c.symbol == symbol)

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
        symbol_cols = self.equity_symbol_mappings.c
        inner = sa.select(
            (symbol_cols.sid,) +
            tuple(map(
                op.getitem(symbol_cols),
                symbol_columns,
            )),
        ).where(
            symbol_cols.sid.in_(map(int, sid_group)),
        ).order_by(
            symbol_cols.end_date.asc(),
        )
        return sa.select(inner.c).group_by(inner.c.sid)

    def _lookup_most_recent_symbols(self, sids):
        symbols = {
            row.sid: {c: row[c] for c in symbol_columns}
            for row in concat(
                self.engine.execute(
                    self._select_most_recent_symbols_chunk(sid_group),
                ).fetchall()
                for sid_group in partition_all(
                    SQLITE_MAX_VARIABLE_NUMBER,
                    sids
                ),
            )
        }

        if len(symbols) != len(sids):
            raise EquitiesNotFound(
                sids=set(sids) - set(symbols),
                plural=True,
            )
        return symbols

    def _retrieve_asset_dicts(self, sids, asset_tbl, querying_equities):
        if not sids:
            return

        if querying_equities:
            def mkdict(row,
                       symbols=self._lookup_most_recent_symbols(sids)):
                return merge(row, symbols[row['sid']])
        else:
            mkdict = dict

        for assets in group_into_chunks(sids):
            # Load misses from the db.
            query = self._select_assets_by_sid(asset_tbl, assets)

            for row in query.execute().fetchall():
                yield _convert_asset_timestamp_fields(mkdict(row))

    def _retrieve_assets(self, sids, asset_tbl, asset_type):
        """
        Internal function for loading assets from a table.

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
            _filter_equity_kwargs
            if querying_equities else
            _filter_future_kwargs
        )

        rows = self._retrieve_asset_dicts(sids, asset_tbl, querying_equities)
        for row in rows:
            sid = row['sid']
            asset = asset_type(**filter_kwargs(row))
            hits[sid] = cache[sid] = asset

        # If we get here, it means something in our code thought that a
        # particular sid was an equity/future and called this function with a
        # concrete type, but we couldn't actually resolve the asset.  This is
        # an error in our code, not a user-input error.
        misses = tuple(set(sids) - viewkeys(hits))
        if misses:
            if querying_equities:
                raise EquitiesNotFound(sids=misses)
            else:
                raise FutureContractsNotFound(sids=misses)
        return hits

    def _lookup_symbol_strict(self, symbol, as_of_date):
        # split the symbol into the components, if there are no
        # company/share class parts then share_class_symbol will be empty
        company_symbol, share_class_symbol = split_delimited_symbol(symbol)
        try:
            owners = self.symbol_ownership_map[
                company_symbol,
                share_class_symbol,
            ]
            assert owners, 'empty owners list for %r' % symbol
        except KeyError:
            # no equity has ever held this symbol
            raise SymbolNotFound(symbol=symbol)

        if not as_of_date:
            if len(owners) > 1:
                # more than one equity has held this ticker, this is ambigious
                # without the date
                raise MultipleSymbolsFound(
                    symbol=symbol,
                    options=set(map(
                        compose(self.retrieve_asset, attrgetter('sid')),
                        owners,
                    )),
                )

            # exactly one equity has ever held this symbol, we may resolve
            # without the date
            return self.retrieve_asset(owners[0].sid)

        for start, end, sid, _ in owners:
            if start <= as_of_date < end:
                # find the equity that owned it on the given asof date
                return self.retrieve_asset(sid)

        # no equity held the ticker on the given asof date
        raise SymbolNotFound(symbol=symbol)

    def _lookup_symbol_fuzzy(self, symbol, as_of_date):
        symbol = symbol.upper()
        company_symbol, share_class_symbol = split_delimited_symbol(symbol)
        try:
            owners = self.fuzzy_symbol_ownership_map[
                company_symbol + share_class_symbol
            ]
            assert owners, 'empty owners list for %r' % symbol
        except KeyError:
            # no equity has ever held a symbol matching the fuzzy symbol
            raise SymbolNotFound(symbol=symbol)

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

            # there are more than one exact match for this fuzzy symbol
            raise MultipleSymbolsFound(
                symbol=symbol,
                options=set(options),
            )

        options = {}
        for start, end, sid, sym in owners:
            if start <= as_of_date < end:
                # see which fuzzy symbols were owned on the asof date.
                options[sid] = sym

        if not options:
            # no equity owned the fuzzy symbol on the date requested
            raise SymbolNotFound(symbol=symbol)

        sid_keys = options.keys()
        # If there was only one owner, or there is a fuzzy and non-fuzzy which
        # map to the same sid, return it.
        if len(options) == 1:
            return self.retrieve_asset(sid_keys[0])

        for sid, sym in options.items():
            if sym == symbol:
                # look for an exact match on the asof date
                return self.retrieve_asset(sid)

        # multiple equities held tickers matching the fuzzy ticker but
        # there are no exact matches
        raise MultipleSymbolsFound(
            symbol=symbol,
            options=[self.retrieve_asset(s) for s in sid_keys],
        )

    def lookup_symbol(self, symbol, as_of_date, fuzzy=False):
        """Lookup an equity by symbol.

        Parameters
        ----------
        symbol : str
            The ticker symbol to resolve.
        as_of_date : datetime or None
            Look up the last owner of this symbol as of this datetime.
            If ``as_of_date`` is None, then this can only resolve the equity
            if exactly one equity has ever owned the ticker.
        fuzzy : bool, optional
            Should fuzzy symbol matching be used? Fuzzy symbol matching
            attempts to resolve differences in representations for
            shareclasses. For example, some people may represent the ``A``
            shareclass of ``BRK`` as ``BRK.A``, where others could write
            ``BRK_A``.

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
            ``as_of_date``.
        """
        if symbol is None:
            raise TypeError("Cannot lookup asset for symbol of None for "
                            "as of date %s." % as_of_date)

        if fuzzy:
            return self._lookup_symbol_fuzzy(symbol, as_of_date)
        return self._lookup_symbol_strict(symbol, as_of_date)

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

        data = self._select_asset_by_symbol(self.futures_contracts, symbol)\
                   .execute().fetchone()

        # If no data found, raise an exception
        if not data:
            raise SymbolNotFound(symbol=symbol)
        return self.retrieve_asset(data['sid'])

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
            sids = self._get_future_sids_for_root_symbol(
                root_symbol,
                as_of_date.value
            )

        if not sids:
            # Check if root symbol exists.
            count = sa.select((sa.func.count(fc_cols.sid),)).where(
                fc_cols.root_symbol == root_symbol,
            ).scalar()
            if count == 0:
                raise RootSymbolNotFound(root_symbol=root_symbol)

        contracts = self.retrieve_futures_contracts(sids)
        return [contracts[sid] for sid in sids]

    @weak_lru_cache(100)
    def _get_future_sids_for_root_symbol(self, root_symbol, as_of_date_ns):
        fc_cols = self.futures_contracts.c

        return list(map(
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
                            fc_cols.expiration_date >= as_of_date_ns
                        ),
                        (
                            fc_cols.expiration_date == pd.NaT.value,
                            fc_cols.notice_date >= as_of_date_ns
                        )
                    ],
                    else_=(
                        sa.func.min(
                            fc_cols.notice_date,
                            fc_cols.expiration_date
                        ) >= as_of_date_ns
                    )
                )
            ).order_by(
                # If both dates exist sort using minimum of
                # expiration_date and notice_date
                # else if one is NaT use the other.
                sa.case(
                    [
                        (
                            fc_cols.expiration_date == pd.NaT.value,
                            fc_cols.notice_date
                        ),
                        (
                            fc_cols.notice_date == pd.NaT.value,
                            fc_cols.expiration_date
                        )
                    ],
                    else_=(
                        sa.func.min(
                            fc_cols.notice_date,
                            fc_cols.expiration_date
                        )
                    )
                ).asc()
            ).execute().fetchall()
        ))

    def lookup_expired_futures(self, start, end):
        if not isinstance(start, pd.Timestamp):
            start = pd.Timestamp(start)
        start = start.value
        if not isinstance(end, pd.Timestamp):
            end = pd.Timestamp(end)
        end = end.value

        fc_cols = self.futures_contracts.c

        nd = sa.func.nullif(fc_cols.notice_date, pd.tslib.iNaT)
        ed = sa.func.nullif(fc_cols.expiration_date, pd.tslib.iNaT)
        date = sa.func.coalesce(sa.func.min(nd, ed), ed, nd)

        sids = list(map(
            itemgetter('sid'),
            sa.select((fc_cols.sid,)).where(
                (date >= start) & (date < end)).order_by(
                sa.func.coalesce(ed, nd).asc()
            ).execute().fetchall()
        ))

        return sids

    def _make_sids(tblattr):
        def _(self):
            return tuple(map(
                itemgetter('sid'),
                sa.select((
                    getattr(self, tblattr).c.sid,
                )).execute().fetchall(),
            ))

        return _

    sids = property(
        _make_sids('asset_router'),
        doc='All the sids in the asset finder.',
    )
    equities_sids = property(
        _make_sids('equities'),
        doc='All of the sids for equities in the asset finder.',
    )
    futures_sids = property(
        _make_sids('futures_contracts'),
        doc='All of the sids for futures consracts in the asset finder.',
    )
    del _make_sids

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
            except SidsNotFound:
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
                    raise SidsNotFound(sids=[asset_convertible_or_iterable])
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

        if missing:
            raise ValueError("Missing assets for identifiers: %s" % missing)

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

        raw_dates = as_column(dates.asi8)
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


def was_active(reference_date_value, asset):
    """
    Whether or not `asset` was active at the time corresponding to
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
    return (
        asset.start_date.value
        <= reference_date_value
        <= asset.end_date.value
    )


def only_active_assets(reference_date_value, assets):
    """
    Filter an iterable of Asset objects down to just assets that were alive at
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
