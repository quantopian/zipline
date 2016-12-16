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
from collections import namedtuple
import re

from contextlib2 import ExitStack
import numpy as np
import pandas as pd
import sqlalchemy as sa
from toolz import first

from zipline.errors import AssetDBVersionError
from zipline.assets.asset_db_schema import (
    ASSET_DB_VERSION,
    asset_db_table_names,
    asset_router,
    equities as equities_table,
    equity_symbol_mappings,
    equity_supplementary_mappings as equity_supplementary_mappings_table,
    futures_contracts as futures_contracts_table,
    futures_exchanges,
    futures_root_symbols,
    metadata,
    version_info,
)

from zipline.utils.preprocess import preprocess
from zipline.utils.range import from_tuple, intersecting_ranges
from zipline.utils.sqlite_utils import coerce_string_to_eng

# Define a namedtuple for use with the load_data and _load_data methods
AssetData = namedtuple(
    'AssetData', (
        'equities',
        'equities_mappings',
        'futures',
        'exchanges',
        'root_symbols',
        'equity_supplementary_mappings',
    ),
)

SQLITE_MAX_VARIABLE_NUMBER = 999

symbol_columns = frozenset({
    'symbol',
    'company_symbol',
    'share_class_symbol',
})
mapping_columns = symbol_columns | {'start_date', 'end_date'}

# Default values for the equities DataFrame
_equities_defaults = {
    'symbol': None,
    'asset_name': None,
    'start_date': 0,
    'end_date': 2 ** 62 - 1,
    'first_traded': None,
    'auto_close_date': None,
    # the canonical exchange name, like "NYSE"
    'exchange': None,
    # optional, something like "New York Stock Exchange"
    'exchange_full': None,
}

# Default values for the futures DataFrame
_futures_defaults = {
    'symbol': None,
    'root_symbol': None,
    'asset_name': None,
    'start_date': 0,
    'end_date': 2 ** 62 - 1,
    'first_traded': None,
    'exchange': None,
    'notice_date': None,
    'expiration_date': None,
    'auto_close_date': None,
    'tick_size': None,
    'multiplier': 1,
}

# Default values for the exchanges DataFrame
_exchanges_defaults = {
    'timezone': None,
}

# Default values for the root_symbols DataFrame
_root_symbols_defaults = {
    'root_symbol_id': None,
    'sector': None,
    'description': None,
    'exchange': None,
}

# Default values for the equity_supplementary_mappings DataFrame
_equity_supplementary_mappings_defaults = {
    'sid': None,
    'value': None,
    'field': None,
    'start_date': 0,
    'end_date': 2 ** 62 - 1,
}


# Fuzzy symbol delimiters that may break up a company symbol and share class
_delimited_symbol_delimiters_regex = re.compile(r'[./\-_]')
_delimited_symbol_default_triggers = frozenset({np.nan, None, ''})


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
        return '', ''

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
        share_class_symbol = ''

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
        output DataFrame and the values are the default values to insert in the
        DataFrame if no user data is provided

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
    data_subset.drop(cols - desired_cols,
                     axis=1,
                     inplace=True)

    # Get those columns which we need but
    # for which no data has been supplied.
    for col in desired_cols - cols:
        # write the default value for any missing columns
        data_subset[col] = defaults[col]

    return data_subset


def _check_asset_group(group):
    row = group.sort_values('end_date').iloc[-1]
    row.start_date = group.start_date.min()
    row.end_date = group.end_date.max()
    row.drop(list(symbol_columns), inplace=True)
    return row


def _format_range(r):
    return (
        str(pd.Timestamp(r.start, unit='ns')),
        str(pd.Timestamp(r.stop, unit='ns')),
    )


def _split_symbol_mappings(df):
    """Split out the symbol: sid mappings from the raw data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with multiple rows for each symbol: sid pair.

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
    ambigious = {}
    for symbol in mappings.symbol.unique():
        persymbol = mappings[mappings.symbol == symbol]
        intersections = list(intersecting_ranges(map(
            from_tuple,
            zip(persymbol.start_date, persymbol.end_date),
        )))
        if intersections:
            ambigious[symbol] = (
                intersections,
                persymbol[['start_date', 'end_date']].astype('datetime64[ns]'),
            )

    if ambigious:
        raise ValueError(
            'Ambiguous ownership for %d symbol%s, multiple assets held the'
            ' following symbols:\n%s' % (
                len(ambigious),
                '' if len(ambigious) == 1 else 's',
                '\n'.join(
                    '%s:\n  intersections: %s\n  %s' % (
                        symbol,
                        tuple(map(_format_range, intersections)),
                        # indent the dataframe string
                        '\n  '.join(str(df).splitlines()),
                    )
                    for symbol, (intersections, df) in sorted(
                        ambigious.items(),
                        key=first,
                    ),
                ),
            )
        )
    return (
        df.groupby(level=0).apply(_check_asset_group),
        df[list(mapping_columns)],
    )


def _dt_to_epoch_ns(dt_series):
    """Convert a timeseries into an Int64Index of nanoseconds since the epoch.

    Parameters
    ----------
    dt_series : pd.Series
        The timeseries to convert.

    Returns
    -------
    idx : pd.Int64Index
        The index converted to nanoseconds since the epoch.
    """
    index = pd.to_datetime(dt_series.values)
    if index.tzinfo is None:
        index = index.tz_localize('UTC')
    else:
        index = index.tz_convert('UTC')
    return index.view(np.int64)


def check_version_info(conn, version_table, expected_version):
    """
    Checks for a version value in the version table.

    Parameters
    ----------
    conn : sa.Connection
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
    version_from_table = conn.execute(
        sa.select((version_table.c.version,)),
    ).scalar()

    # A db without a version is considered v0
    if version_from_table is None:
        version_from_table = 0

    # Raise an error if the versions do not match
    if (version_from_table != expected_version):
        raise AssetDBVersionError(db_version=version_from_table,
                                  expected_version=expected_version)


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
    conn.execute(sa.insert(version_table, values={'version': version_value}))


class _empty(object):
    columns = ()


class AssetDBWriter(object):
    """Class used to write data to an assets db.

    Parameters
    ----------
    engine : Engine or str
        An SQLAlchemy engine or path to a SQL database.
    """
    DEFAULT_CHUNK_SIZE = SQLITE_MAX_VARIABLE_NUMBER

    @preprocess(engine=coerce_string_to_eng)
    def __init__(self, engine):
        self.engine = engine

    def write(self,
              equities=None,
              futures=None,
              exchanges=None,
              root_symbols=None,
              equity_supplementary_mappings=None,
              chunk_size=DEFAULT_CHUNK_SIZE):
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
              exchange : str, optional
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
              exchange : str, optional
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
                  The name of the exchange.
              timezone : str
                  The timezone of the exchange.
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
        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            self.init_db(conn)

            # Get the data to add to SQL.
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
            # Write the data to SQL.
            self._write_df_to_table(
                futures_exchanges,
                data.exchanges,
                conn,
                chunk_size,
            )
            self._write_df_to_table(
                futures_root_symbols,
                data.root_symbols,
                conn,
                chunk_size,
            )
            self._write_df_to_table(
                equity_supplementary_mappings_table,
                data.equity_supplementary_mappings,
                conn,
                chunk_size,
                idx=False,
            )
            self._write_assets(
                'future',
                data.futures,
                conn,
                chunk_size,
            )
            self._write_assets(
                'equity',
                data.equities,
                conn,
                chunk_size,
                mapping_data=data.equities_mappings,
            )

    def _write_df_to_table(
        self,
        tbl,
        df,
        txn,
        chunk_size,
        idx=True,
        idx_label=None,
    ):
        df.to_sql(
            tbl.name,
            txn.connection,
            index=idx,
            index_label=(
                idx_label
                if idx_label is not None else
                first(tbl.primary_key.columns).name
            ),
            if_exists='append',
            chunksize=chunk_size,
        )

    def _write_assets(self,
                      asset_type,
                      assets,
                      txn,
                      chunk_size,
                      mapping_data=None):
        if asset_type == 'future':
            tbl = futures_contracts_table
            if mapping_data is not None:
                raise TypeError('no mapping data expected for futures')

        elif asset_type == 'equity':
            tbl = equities_table
            if mapping_data is None:
                raise TypeError('mapping data required for equities')
            # write the symbol mapping data.
            self._write_df_to_table(
                equity_symbol_mappings,
                mapping_data,
                txn,
                chunk_size,
                idx_label='sid',
            )

        else:
            raise ValueError(
                "asset_type must be in {'future', 'equity'}, got: %s" %
                asset_type,
            )

        self._write_df_to_table(tbl, assets, txn, chunk_size)

        pd.DataFrame({
            asset_router.c.sid.name: assets.index.values,
            asset_router.c.asset_type.name: asset_type,
        }).to_sql(
            asset_router.name,
            txn.connection,
            if_exists='append',
            index=False,
            chunksize=chunk_size
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
        conn = txn.connect()
        for table_name in asset_db_table_names:
            if txn.dialect.has_table(conn, table_name):
                return True
        return False

    def init_db(self, txn=None):
        """Connect to database and create tables.

        Parameters
        ----------
        txn : sa.engine.Connection, optional
            The transaction to execute in. If this is not provided, a new
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

    def _normalize_equities(self, equities):
        # HACK: If 'company_name' is provided, map it to asset_name
        if ('company_name' in equities.columns and
                'asset_name' not in equities.columns):
            equities['asset_name'] = equities['company_name']

        # remap 'file_name' to 'symbol' if provided
        if 'file_name' in equities.columns:
            equities['symbol'] = equities['file_name']

        equities_output = _generate_output_dataframe(
            data_subset=equities,
            defaults=_equities_defaults,
        )

        # Split symbols to company_symbols and share_class_symbols
        tuple_series = equities_output['symbol'].apply(split_delimited_symbol)
        split_symbols = pd.DataFrame(
            tuple_series.tolist(),
            columns=['company_symbol', 'share_class_symbol'],
            index=tuple_series.index
        )
        equities_output = pd.concat((equities_output, split_symbols), axis=1)

        # Upper-case all symbol data
        for col in symbol_columns:
            equities_output[col] = equities_output[col].str.upper()

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        for col in ('start_date',
                    'end_date',
                    'first_traded',
                    'auto_close_date'):
            equities_output[col] = _dt_to_epoch_ns(equities_output[col])

        return _split_symbol_mappings(equities_output)

    def _normalize_futures(self, futures):
        futures_output = _generate_output_dataframe(
            data_subset=futures,
            defaults=_futures_defaults,
        )
        for col in ('symbol', 'root_symbol'):
            futures_output[col] = futures_output[col].str.upper()

        for col in ('start_date',
                    'end_date',
                    'first_traded',
                    'notice_date',
                    'expiration_date',
                    'auto_close_date'):
            futures_output[col] = _dt_to_epoch_ns(futures_output[col])

        return futures_output

    def _normalize_equity_supplementary_mappings(self, mappings):
        mappings_output = _generate_output_dataframe(
            data_subset=mappings,
            defaults=_equity_supplementary_mappings_defaults,
        )

        for col in ('start_date', 'end_date'):
            mappings_output[col] = _dt_to_epoch_ns(mappings_output[col])

        return mappings_output

    def _load_data(
        self,
        equities,
        futures,
        exchanges,
        root_symbols,
        equity_supplementary_mappings,
    ):
        """
        Returns a standard set of pandas.DataFrames:
        equities, futures, exchanges, root_symbols
        """
        # Check whether identifier columns have been provided.
        # If they have, set the index to this column.
        # If not, assume the index already cotains the identifier information.
        for df, id_col in [(equities, 'sid'),
                           (futures, 'sid'),
                           (exchanges, 'exchange'),
                           (root_symbols, 'root_symbol')]:
            if id_col in df.columns:
                df.set_index(id_col, inplace=True)

        equities_output, equities_mappings = self._normalize_equities(equities)
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
