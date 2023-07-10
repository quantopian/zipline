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
import re
from collections import namedtuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from toolz import first

from zipline.assets.asset_db_schema import (
    ASSET_DB_VERSION,
    asset_db_table_names,
    asset_router,
)
from zipline.assets.asset_db_schema import equities as equities_table
from zipline.assets.asset_db_schema import (
    equity_supplementary_mappings as equity_supplementary_mappings_table,
)
from zipline.assets.asset_db_schema import equity_symbol_mappings
from zipline.assets.asset_db_schema import exchanges as exchanges_table
from zipline.assets.asset_db_schema import futures_contracts as futures_contracts_table
from zipline.assets.asset_db_schema import futures_root_symbols, metadata, version_info
from zipline.errors import AssetDBVersionError
from zipline.utils.compat import ExitStack
from zipline.utils.preprocess import preprocess
from zipline.utils.range import from_tuple, intersecting_ranges
from zipline.utils.sqlite_utils import coerce_string_to_eng

# Define a namedtuple for use with the load_data and _load_data methods
AssetData = namedtuple(
    "AssetData",
    (
        "equities",
        "equities_mappings",
        "futures",
        "exchanges",
        "root_symbols",
        "equity_supplementary_mappings",
    ),
)

SQLITE_MAX_VARIABLE_NUMBER = 999

symbol_columns = frozenset(
    {
        "symbol",
        "company_symbol",
        "share_class_symbol",
    }
)
mapping_columns = symbol_columns | {"start_date", "end_date"}

_index_columns = {
    "equities": "sid",
    "equity_supplementary_mappings": "sid",
    "futures": "sid",
    "exchanges": "exchange",
    "root_symbols": "root_symbol",
}


def _normalize_index_columns_in_place(
    equities, equity_supplementary_mappings, futures, exchanges, root_symbols
):
    """
    Update dataframes in place to set indentifier columns as indices.

    For each input frame, if the frame has a column with the same name as its
    associated index column, set that column as the index.

    Otherwise, assume the index already contains identifiers.

    If frames are passed as None, they're ignored.
    """
    for frame, column_name in (
        (equities, "sid"),
        (equity_supplementary_mappings, "sid"),
        (futures, "sid"),
        (exchanges, "exchange"),
        (root_symbols, "root_symbol"),
    ):
        if frame is not None and column_name in frame:
            frame.set_index(column_name, inplace=True)


def _default_none(df, column):
    return None


def _no_default(df, column):
    if not df.empty:
        raise ValueError("no default value for column %r" % column)


# Default values for the equities DataFrame
_equities_defaults = {
    "symbol": _default_none,
    "asset_name": _default_none,
    "start_date": lambda df, col: 0,
    "end_date": lambda df, col: np.iinfo(np.int64).max,
    "first_traded": _default_none,
    "auto_close_date": _default_none,
    # the full exchange name
    "exchange": _no_default,
}

# the defaults for ``equities`` in ``write_direct``
_direct_equities_defaults = _equities_defaults.copy()
del _direct_equities_defaults["symbol"]

# Default values for the futures DataFrame
_futures_defaults = {
    "symbol": _default_none,
    "root_symbol": _default_none,
    "asset_name": _default_none,
    "start_date": lambda df, col: 0,
    "end_date": lambda df, col: np.iinfo(np.int64).max,
    "first_traded": _default_none,
    "exchange": _default_none,
    "notice_date": _default_none,
    "expiration_date": _default_none,
    "auto_close_date": _default_none,
    "tick_size": _default_none,
    "multiplier": lambda df, col: 1,
}

# Default values for the exchanges DataFrame
_exchanges_defaults = {
    "canonical_name": lambda df, col: df.index,
    "country_code": lambda df, col: "??",
}

# Default values for the root_symbols DataFrame
_root_symbols_defaults = {
    "sector": _default_none,
    "description": _default_none,
    "exchange": _default_none,
}

# Default values for the equity_supplementary_mappings DataFrame
_equity_supplementary_mappings_defaults = {
    "value": _default_none,
    "field": _default_none,
    "start_date": lambda df, col: 0,
    "end_date": lambda df, col: np.iinfo(np.int64).max,
}

# Default values for the equity_symbol_mappings DataFrame
_equity_symbol_mappings_defaults = {
    "sid": _no_default,
    "company_symbol": _default_none,
    "share_class_symbol": _default_none,
    "symbol": _default_none,
    "start_date": lambda df, col: 0,
    "end_date": lambda df, col: np.iinfo(np.int64).max,
}

# Fuzzy symbol delimiters that may break up a company symbol and share class
_delimited_symbol_delimiters_regex = re.compile(r"[./\-_]")
_delimited_symbol_default_triggers = frozenset({np.nan, None, ""})


def split_delimited_symbol(symbol):
    """
    Takes in a symbol that may be delimited and splits it in to a company
    symbol and share class symbol. Also returns the fuzzy symbol, which is the
    symbol without any fuzzy characters at all.

    Parameters
    ----------
    symbol : str
        The possibly-delimited symbol to be split

    Returns
    -------
    company_symbol : str
        The company part of the symbol.
    share_class_symbol : str
        The share class part of a symbol.
    """
    # return blank strings for any bad fuzzy symbols, like NaN or None
    if symbol in _delimited_symbol_default_triggers:
        return "", ""

    symbol = symbol.upper()

    split_list = re.split(
        pattern=_delimited_symbol_delimiters_regex,
        string=symbol,
        maxsplit=1,
    )

    # Break the list up in to its two components, the company symbol and the
    # share class symbol
    company_symbol = split_list[0]
    if len(split_list) > 1:
        share_class_symbol = split_list[1]
    else:
        share_class_symbol = ""

    return company_symbol, share_class_symbol


def _generate_output_dataframe(data_subset, defaults):
    """
    Generates an output dataframe from the given subset of user-provided
    data, the given column names, and the given default values.

    Parameters
    ----------
    data_subset : DataFrame
        A DataFrame, usually from an AssetData object,
        that contains the user's input metadata for the asset type being
        processed
    defaults : dict
        A dict where the keys are the names of the columns of the desired
        output DataFrame and the values are a function from dataframe and
        column name to the default values to insert in the DataFrame if no user
        data is provided

    Returns
    -------
    DataFrame
        A DataFrame containing all user-provided metadata, and default values
        wherever user-provided metadata was missing
    """
    # The columns provided.
    cols = set(data_subset.columns)
    desired_cols = set(defaults)

    # Drop columns with unrecognised headers.
    data_subset.drop(cols - desired_cols, axis=1, inplace=True)

    # Get those columns which we need but
    # for which no data has been supplied.
    for col in desired_cols - cols:
        # write the default value for any missing columns
        data_subset[col] = defaults[col](data_subset, col)

    return data_subset


def _check_asset_group(group):
    # workaround until fixed: https://github.com/pandas-dev/pandas/issues/47985
    if group.empty:
        return group
    row = group.sort_values("end_date").iloc[-1]
    row.start_date = group.start_date.min()
    row.end_date = group.end_date.max()
    row.drop(list(symbol_columns), inplace=True)
    return row


def _format_range(r):
    return (
        str(pd.Timestamp(r.start, unit="ns")),
        str(pd.Timestamp(r.stop, unit="ns")),
    )


def _check_symbol_mappings(df, exchanges, asset_exchange):
    """Check that there are no cases where multiple symbols resolve to the same
    asset at the same time in the same country.

    Parameters
    ----------
    df : pd.DataFrame
        The equity symbol mappings table.
    exchanges : pd.DataFrame
        The exchanges table.
    asset_exchange : pd.Series
        A series that maps sids to the exchange the asset is in.

    Raises
    ------
    ValueError
        Raised when there are ambiguous symbol mappings.
    """
    mappings = df.set_index("sid")[list(mapping_columns)].copy()
    try:
        mappings["country_code"] = exchanges["country_code"][
            asset_exchange.loc[df["sid"]]
        ].values
    except KeyError:
        mappings["country_code"] = exchanges.set_index("exchange")["country_code"].loc[
            asset_exchange.loc[df["sid"]].values
        ]

    ambiguous = {}

    def check_intersections(persymbol):
        intersections = list(
            intersecting_ranges(
                map(
                    from_tuple,
                    zip(persymbol.start_date, persymbol.end_date),
                )
            )
        )
        if intersections:
            data = persymbol[["start_date", "end_date"]].astype("datetime64[ns]")
            # indent the dataframe string, also compute this early because
            # ``persymbol`` is a view and ``astype`` doesn't copy the index
            # correctly in pandas 0.22
            msg_component = "\n  ".join(str(data).splitlines())
            ambiguous[persymbol.name] = intersections, msg_component

    mappings.groupby(["symbol", "country_code"], group_keys=False).apply(
        check_intersections
    )

    if ambiguous:
        raise ValueError(
            "Ambiguous ownership for %d symbol%s, multiple assets held the"
            " following symbols:\n%s"
            % (
                len(ambiguous),
                "" if len(ambiguous) == 1 else "s",
                "\n".join(
                    "%s (%s):\n  intersections: %s\n  %s"
                    % (
                        symbol,
                        country_code,
                        tuple(map(_format_range, intersections)),
                        cs,
                    )
                    for (symbol, country_code), (intersections, cs) in sorted(
                        ambiguous.items(),
                        key=first,
                    )
                ),
            )
        )


def _split_symbol_mappings(df, exchanges):
    """Split out the symbol: sid mappings from the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with multiple rows for each symbol: sid pair.
    exchanges : pd.DataFrame
        The exchanges table.

    Returns
    -------
    asset_info : pd.DataFrame
        The asset info with one row per asset.
    symbol_mappings : pd.DataFrame
        The dataframe of just symbol: sid mappings. The index will be
        the sid, then there will be three columns: symbol, start_date, and
        end_date.
    """
    mappings = df[list(mapping_columns)]
    with pd.option_context("mode.chained_assignment", None):
        mappings["sid"] = mappings.index
    mappings.reset_index(drop=True, inplace=True)

    # take the most recent sid->exchange mapping based on end date
    asset_exchange = (
        df[["exchange", "end_date"]]
        .sort_values("end_date")
        .groupby(level=0)["exchange"]
        .nth(-1)
    )

    _check_symbol_mappings(mappings, exchanges, asset_exchange)
    return (
        df.groupby(level=0, group_keys=False).apply(_check_asset_group),
        mappings,
    )


def _dt_to_epoch_ns(dt_series: pd.Series) -> pd.Index:
    """Convert a timeseries into an Int64Index of nanoseconds since the epoch.

    Parameters
    ----------
    dt_series : pd.Series
        The timeseries to convert.

    Returns
    -------
    idx : pd.Index
        The index converted to nanoseconds since the epoch.
    """
    index = pd.to_datetime(dt_series.values)
    if index.tzinfo is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    return index.view(np.int64)


def check_version_info(conn, version_table, expected_version: int):
    """
    Checks for a version value in the version table.

    Parameters
    ----------
    conn : Connection
        The connection to use to perform the check.
    version_table : sa.Table
        The version table of the asset database
    expected_version : int
        The expected version of the asset database

    Raises
    ------
    AssetDBVersionError
        If the version is in the table and not equal to ASSET_DB_VERSION.
    """
    # Read the version out of the table
    version_from_table = conn.execute(sa.select(version_table.c.version)).scalar()

    # A db without a version is considered v0
    if version_from_table is None:
        version_from_table = 0

    # Raise an error if the versions do not match
    if version_from_table != expected_version:
        raise AssetDBVersionError(
            db_version=version_from_table, expected_version=expected_version
        )


def write_version_info(conn, version_table, version_value):
    """
    Inserts the version value in to the version table.

    Parameters
    ----------
    conn : sa.Connection
        The connection to use to execute the insert.
    version_table : sa.Table
        The version table of the asset database
    version_value : int
        The version to write in to the database

    """
    if conn.engine.name == "postgresql":
        conn.execute(sa.text("ALTER SEQUENCE version_info_id_seq RESTART WITH 1"))
    conn.execute(version_table.insert().values(version=version_value))


class AssetDBWriter:
    """Class used to write data to an assets db.

    Parameters
    ----------
    engine : Engine or str
        An SQLAlchemy engine or path to a SQL database.
    """

    DEFAULT_CHUNK_SIZE = SQLITE_MAX_VARIABLE_NUMBER

    @preprocess(engine=coerce_string_to_eng(require_exists=False))
    def __init__(self, engine):
        self.engine = engine

    def _real_write(
        self,
        equities,
        equity_symbol_mappings,
        equity_supplementary_mappings,
        futures,
        exchanges,
        root_symbols,
        chunk_size,
    ):
        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            self.init_db(conn)

            if exchanges is not None:
                self._write_df_to_table(
                    exchanges_table,
                    exchanges,
                    conn,
                    chunk_size,
                )

            if root_symbols is not None:
                self._write_df_to_table(
                    futures_root_symbols,
                    root_symbols,
                    conn,
                    chunk_size,
                )

            if equity_supplementary_mappings is not None:
                self._write_df_to_table(
                    equity_supplementary_mappings_table,
                    equity_supplementary_mappings,
                    conn,
                    chunk_size,
                )

            if futures is not None:
                self._write_assets(
                    "future",
                    futures,
                    conn,
                    chunk_size,
                )

            if equities is not None:
                self._write_assets(
                    "equity",
                    equities,
                    conn,
                    chunk_size,
                    mapping_data=equity_symbol_mappings,
                )

    def write_direct(
        self,
        equities=None,
        equity_symbol_mappings=None,
        equity_supplementary_mappings=None,
        futures=None,
        exchanges=None,
        root_symbols=None,
        chunk_size=DEFAULT_CHUNK_SIZE,
    ):
        """Write asset metadata to a sqlite database in the format that it is
        stored in the assets db.

        Parameters
        ----------
        equities : pd.DataFrame, optional
            The equity metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this equity.
              asset_name : str
                  The full name for this asset.
              start_date : datetime
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              auto_close_date : datetime, optional
                  The date on which to close any positions in this asset.
              exchange : str
                  The exchange where this asset is traded.

            The index of this dataframe should contain the sids.
        futures : pd.DataFrame, optional
            The future contract metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this futures contract.
              root_symbol : str
                  The root symbol, or the symbol with the expiration stripped
                  out.
              asset_name : str
                  The full name for this asset.
              start_date : datetime, optional
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              exchange : str
                  The exchange where this asset is traded.
              notice_date : datetime
                  The date when the owner of the contract may be forced
                  to take physical delivery of the contract's asset.
              expiration_date : datetime
                  The date when the contract expires.
              auto_close_date : datetime
                  The date when the broker will automatically close any
                  positions in this contract.
              tick_size : float
                  The minimum price movement of the contract.
              multiplier: float
                  The amount of the underlying asset represented by this
                  contract.
        exchanges : pd.DataFrame, optional
            The exchanges where assets can be traded. The columns of this
            dataframe are:

              exchange : str
                  The full name of the exchange.
              canonical_name : str
                  The canonical name of the exchange.
              country_code : str
                  The ISO 3166 alpha-2 country code of the exchange.
        root_symbols : pd.DataFrame, optional
            The root symbols for the futures contracts. The columns for this
            dataframe are:

              root_symbol : str
                  The root symbol name.
              root_symbol_id : int
                  The unique id for this root symbol.
              sector : string, optional
                  The sector of this root symbol.
              description : string, optional
                  A short description of this root symbol.
              exchange : str
                  The exchange where this root symbol is traded.
        equity_supplementary_mappings : pd.DataFrame, optional
            Additional mappings from values of abitrary type to assets.
        chunk_size : int, optional
            The amount of rows to write to the SQLite table at once.
            This defaults to the default number of bind params in sqlite.
            If you have compiled sqlite3 with more bind or less params you may
            want to pass that value here.

        """
        if equities is not None:
            equities = _generate_output_dataframe(
                equities,
                _direct_equities_defaults,
            )
            if equity_symbol_mappings is None:
                raise ValueError(
                    "equities provided with no symbol mapping data",
                )

            equity_symbol_mappings = _generate_output_dataframe(
                equity_symbol_mappings,
                _equity_symbol_mappings_defaults,
            )
            _check_symbol_mappings(
                equity_symbol_mappings,
                exchanges,
                equities["exchange"],
            )

        if equity_supplementary_mappings is not None:
            equity_supplementary_mappings = _generate_output_dataframe(
                equity_supplementary_mappings,
                _equity_supplementary_mappings_defaults,
            )

        if futures is not None:
            futures = _generate_output_dataframe(_futures_defaults, futures)

        if exchanges is not None:
            exchanges = _generate_output_dataframe(
                exchanges.set_index("exchange"),
                _exchanges_defaults,
            )

        if root_symbols is not None:
            root_symbols = _generate_output_dataframe(
                root_symbols,
                _root_symbols_defaults,
            )

        # Set named identifier columns as indices, if provided.
        _normalize_index_columns_in_place(
            equities=equities,
            equity_supplementary_mappings=equity_supplementary_mappings,
            futures=futures,
            exchanges=exchanges,
            root_symbols=root_symbols,
        )

        self._real_write(
            equities=equities,
            equity_symbol_mappings=equity_symbol_mappings,
            equity_supplementary_mappings=equity_supplementary_mappings,
            futures=futures,
            exchanges=exchanges,
            root_symbols=root_symbols,
            chunk_size=chunk_size,
        )

    def write(
        self,
        equities=None,
        futures=None,
        exchanges=None,
        root_symbols=None,
        equity_supplementary_mappings=None,
        chunk_size=DEFAULT_CHUNK_SIZE,
    ):
        """Write asset metadata to a sqlite database.

        Parameters
        ----------
        equities : pd.DataFrame, optional
            The equity metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this equity.
              asset_name : str
                  The full name for this asset.
              start_date : datetime
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              auto_close_date : datetime, optional
                  The date on which to close any positions in this asset.
              exchange : str
                  The exchange where this asset is traded.

            The index of this dataframe should contain the sids.
        futures : pd.DataFrame, optional
            The future contract metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this futures contract.
              root_symbol : str
                  The root symbol, or the symbol with the expiration stripped
                  out.
              asset_name : str
                  The full name for this asset.
              start_date : datetime, optional
                  The date when this asset was created.
              end_date : datetime, optional
                  The last date we have trade data for this asset.
              first_traded : datetime, optional
                  The first date we have trade data for this asset.
              exchange : str
                  The exchange where this asset is traded.
              notice_date : datetime
                  The date when the owner of the contract may be forced
                  to take physical delivery of the contract's asset.
              expiration_date : datetime
                  The date when the contract expires.
              auto_close_date : datetime
                  The date when the broker will automatically close any
                  positions in this contract.
              tick_size : float
                  The minimum price movement of the contract.
              multiplier: float
                  The amount of the underlying asset represented by this
                  contract.
        exchanges : pd.DataFrame, optional
            The exchanges where assets can be traded. The columns of this
            dataframe are:

              exchange : str
                  The full name of the exchange.
              canonical_name : str
                  The canonical name of the exchange.
              country_code : str
                  The ISO 3166 alpha-2 country code of the exchange.
        root_symbols : pd.DataFrame, optional
            The root symbols for the futures contracts. The columns for this
            dataframe are:

              root_symbol : str
                  The root symbol name.
              root_symbol_id : int
                  The unique id for this root symbol.
              sector : string, optional
                  The sector of this root symbol.
              description : string, optional
                  A short description of this root symbol.
              exchange : str
                  The exchange where this root symbol is traded.
        equity_supplementary_mappings : pd.DataFrame, optional
            Additional mappings from values of abitrary type to assets.
        chunk_size : int, optional
            The amount of rows to write to the SQLite table at once.
            This defaults to the default number of bind params in sqlite.
            If you have compiled sqlite3 with more bind or less params you may
            want to pass that value here.

        See Also
        --------
        zipline.assets.asset_finder
        """
        if exchanges is None:
            exchange_names = [
                df["exchange"]
                for df in (equities, futures, root_symbols)
                if df is not None
            ]
            if exchange_names:
                exchanges = pd.DataFrame(
                    {
                        "exchange": pd.concat(exchange_names).unique(),
                    }
                )

        data = self._load_data(
            equities if equities is not None else pd.DataFrame(),
            futures if futures is not None else pd.DataFrame(),
            exchanges if exchanges is not None else pd.DataFrame(),
            root_symbols if root_symbols is not None else pd.DataFrame(),
            (
                equity_supplementary_mappings
                if equity_supplementary_mappings is not None
                else pd.DataFrame()
            ),
        )
        self._real_write(
            equities=data.equities,
            equity_symbol_mappings=data.equities_mappings,
            equity_supplementary_mappings=data.equity_supplementary_mappings,
            futures=data.futures,
            root_symbols=data.root_symbols,
            exchanges=data.exchanges,
            chunk_size=chunk_size,
        )

    def _write_df_to_table(self, tbl, df, txn, chunk_size):
        df = df.copy()
        for column, dtype in df.dtypes.items():
            if dtype.kind == "M":
                df[column] = _dt_to_epoch_ns(df[column])

        if txn.dialect.name == "postgresql":
            txn.execute(sa.text(f"ALTER TABLE {tbl.name} DISABLE TRIGGER ALL;"))

        df.to_sql(
            tbl.name,
            txn,
            index=True,
            index_label=first(tbl.primary_key.columns).name,
            if_exists="append",
            chunksize=chunk_size,
        )

    def _write_assets(self, asset_type, assets, txn, chunk_size, mapping_data=None):
        if asset_type == "future":
            tbl = futures_contracts_table
            if mapping_data is not None:
                raise TypeError("no mapping data expected for futures")

        elif asset_type == "equity":
            tbl = equities_table
            if mapping_data is None:
                raise TypeError("mapping data required for equities")
            # write the symbol mapping data.
            self._write_df_to_table(
                equity_symbol_mappings,
                mapping_data,
                txn,
                chunk_size,
            )

        else:
            raise ValueError(
                "asset_type must be in {'future', 'equity'}, got: %s" % asset_type,
            )

        self._write_df_to_table(tbl, assets, txn, chunk_size)

        pd.DataFrame(
            {
                asset_router.c.sid.name: assets.index.values,
                asset_router.c.asset_type.name: asset_type,
            }
        ).to_sql(
            asset_router.name,
            txn,
            if_exists="append",
            index=False,
            chunksize=chunk_size,
        )

    def _all_tables_present(self, txn):
        """
        Checks if any tables are present in the current assets database.

        Parameters
        ----------
        txn : Transaction
            The open transaction to check in.

        Returns
        -------
        has_tables : bool
            True if any tables are present, otherwise False.
        """
        # conn = txn.connect()
        for table_name in asset_db_table_names:
            return sa.inspect(txn).has_table(table_name)
            # if txn.dialect.has_table(conn, table_name):
            # return True
        # return False

    def init_db(self, txn=None):
        """Connect to database and create tables.

        Parameters
        ----------
        txn : sa.engine.Connection, optional
            The transaction block to execute in. If this is not provided, a new
            transaction will be started with the engine provided.

        Returns
        -------
        metadata : sa.MetaData
            The metadata that describes the new assets db.
        """
        with ExitStack() as stack:
            if txn is None:
                txn = stack.enter_context(self.engine.begin())

            tables_already_exist = self._all_tables_present(txn)

            # Create the SQL tables if they do not already exist.
            metadata.create_all(txn, checkfirst=True)

            if tables_already_exist:
                check_version_info(txn, version_info, ASSET_DB_VERSION)
            else:
                write_version_info(txn, version_info, ASSET_DB_VERSION)

    def _normalize_equities(self, equities, exchanges):
        # HACK: If 'company_name' is provided, map it to asset_name
        if "company_name" in equities.columns and "asset_name" not in equities.columns:
            equities["asset_name"] = equities["company_name"]

        # remap 'file_name' to 'symbol' if provided
        if "file_name" in equities.columns:
            equities["symbol"] = equities["file_name"]

        equities_output = _generate_output_dataframe(
            data_subset=equities,
            defaults=_equities_defaults,
        )

        # Split symbols to company_symbols and share_class_symbols
        tuple_series = equities_output["symbol"].apply(split_delimited_symbol)
        split_symbols = pd.DataFrame(
            tuple_series.tolist(),
            columns=["company_symbol", "share_class_symbol"],
            index=tuple_series.index,
        )
        equities_output = pd.concat((equities_output, split_symbols), axis=1)

        # Upper-case all symbol data
        for col in symbol_columns:
            equities_output[col] = equities_output[col].str.upper()

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        for col in ("start_date", "end_date", "first_traded", "auto_close_date"):
            equities_output[col] = _dt_to_epoch_ns(equities_output[col])

        return _split_symbol_mappings(equities_output, exchanges)

    def _normalize_futures(self, futures):
        futures_output = _generate_output_dataframe(
            data_subset=futures,
            defaults=_futures_defaults,
        )
        for col in ("symbol", "root_symbol"):
            futures_output[col] = futures_output[col].str.upper()

        for col in (
            "start_date",
            "end_date",
            "first_traded",
            "notice_date",
            "expiration_date",
            "auto_close_date",
        ):
            futures_output[col] = _dt_to_epoch_ns(futures_output[col])

        return futures_output

    def _normalize_equity_supplementary_mappings(self, mappings):
        mappings_output = _generate_output_dataframe(
            data_subset=mappings,
            defaults=_equity_supplementary_mappings_defaults,
        )

        for col in ("start_date", "end_date"):
            mappings_output[col] = _dt_to_epoch_ns(mappings_output[col])

        return mappings_output

    def _load_data(
        self, equities, futures, exchanges, root_symbols, equity_supplementary_mappings
    ):
        """
        Returns a standard set of pandas.DataFrames:
        equities, futures, exchanges, root_symbols
        """
        # Set named identifier columns as indices, if provided.
        _normalize_index_columns_in_place(
            equities=equities,
            equity_supplementary_mappings=equity_supplementary_mappings,
            futures=futures,
            exchanges=exchanges,
            root_symbols=root_symbols,
        )

        futures_output = self._normalize_futures(futures)

        equity_supplementary_mappings_output = (
            self._normalize_equity_supplementary_mappings(
                equity_supplementary_mappings,
            )
        )

        exchanges_output = _generate_output_dataframe(
            data_subset=exchanges,
            defaults=_exchanges_defaults,
        )

        equities_output, equities_mappings = self._normalize_equities(
            equities,
            exchanges_output,
        )

        root_symbols_output = _generate_output_dataframe(
            data_subset=root_symbols,
            defaults=_root_symbols_defaults,
        )

        return AssetData(
            equities=equities_output,
            equities_mappings=equities_mappings,
            futures=futures_output,
            exchanges=exchanges_output,
            root_symbols=root_symbols_output,
            equity_supplementary_mappings=equity_supplementary_mappings_output,
        )
