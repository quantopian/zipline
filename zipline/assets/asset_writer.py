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
    generate_asset_db_metadata,
    asset_db_table_names,
    ASSET_DB_VERSION,
)

# Define a namedtuple for use with the load_data and _load_data methods
AssetData = namedtuple('AssetData', 'equities futures exchanges root_symbols')

SQLITE_MAX_VARIABLE_NUMBER = 999

# Default values for the equities DataFrame
_equities_defaults = {
    'symbol': None,
    'asset_name': None,
    'start_date': 0,
    'end_date': 2 ** 62 - 1,
    'first_traded': None,
    'auto_close_date': None,
    'exchange': None,
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

# Fuzzy symbol delimiters that may break up a company symbol and share class
_delimited_symbol_delimiter_regex = r'[./\-_]'
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
    ( str, str , str )
        A tuple of ( company_symbol, share_class_symbol, fuzzy_symbol)
    """
    # return blank strings for any bad fuzzy symbols, like NaN or None
    if symbol in _delimited_symbol_default_triggers:
        return ('', '', '')

    split_list = re.split(pattern=_delimited_symbol_delimiter_regex,
                          string=symbol,
                          maxsplit=1)

    # Break the list up in to its two components, the company symbol and the
    # share class symbol
    company_symbol = split_list[0]
    if len(split_list) > 1:
        share_class_symbol = split_list[1]
    else:
        share_class_symbol = ''

    # Strip all fuzzy characters from the symbol to get the fuzzy symbol
    fuzzy_symbol = re.sub(pattern=_delimited_symbol_delimiter_regex,
                          repl='',
                          string=symbol)

    return (company_symbol, share_class_symbol, fuzzy_symbol)


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
    need = desired_cols - cols

    # Combine the users supplied data with our required columns.
    output = pd.concat(
        (data_subset, pd.DataFrame(
            {k: defaults[k] for k in need},
            data_subset.index,
        )),
        axis=1,
        copy=False
    )

    return output


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


def check_version_info(version_table, expected_version):
    """
    Checks for a version value in the version table.

    Parameters
    ----------
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
    version_from_table = sa.select((version_table.c.version,)).scalar()

    # A db without a version is considered v0
    if version_from_table is None:
        version_from_table = 0

    # Raise an error if the versions do not match
    if (version_from_table != expected_version):
        raise AssetDBVersionError(db_version=version_from_table,
                                  expected_version=expected_version)


def write_version_info(version_table, version_value):
    """
    Inserts the version value in to the version table.

    Parameters
    ----------
    version_table : sa.Table
        The version table of the asset database
    version_value : int
        The version to write in to the database

    """
    sa.insert(version_table, values={'version': version_value}).execute()


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

    def __init__(self, engine):
        if isinstance(engine, str):
            engine = sa.create_engine('sqlite:///' + engine)
        self.engine = engine

    def write(self,
              equities=None,
              futures=None,
              exchanges=None,
              root_symbols=None,
              chunk_size=DEFAULT_CHUNK_SIZE):
        """Write asset metadata to a sqlite database.

        Parameters
        ----------
        equities : pd.DataFrame, optional
            The equity metadata. The columns for this dataframe are:

              symbol : str
                  The ticker symbol for this equity.
              fuzzy_symbol : str, optional
                  The fuzzy symbol for this equity. This is the symbol
                  without any delimiting characters like '.' or '_'.
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
        futures : pd.Dataframe, optional
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
        exchanges : pd.Dataframe, optional
            The exchanges where assets can be traded. The columns of this
            dataframe are:

              exchange : str
                  The name of the exchange.
              timezone : str
                  The timezone of the exchange.
        root_symbols : pd.Dataframe, optional
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
        chunk_size : int, optional
            The amount of rows to write to the SQLite table at once.
            This defaults to the default number of bind params in sqlite.
            If you have compiled sqlite3 with more bind or less params you may
            want to pass that value here.

        See Also
        --------
        zipline.assets.asset_finder
        """

        with self.engine.begin() as txn:
            # Create SQL tables if they do not exist.
            metadata = self.init_db(txn)

            # Get the data to add to SQL.
            data = self._load_data(
                equities if equities is not None else pd.DataFrame(),
                futures if futures is not None else pd.DataFrame(),
                exchanges if exchanges is not None else pd.DataFrame(),
                root_symbols if root_symbols is not None else pd.DataFrame(),
            )

            # Write the data to SQL.
            self._write_df_to_table(
                metadata.tables['futures_exchanges'],
                data.exchanges,
                txn,
                chunk_size,
            )
            self._write_df_to_table(
                metadata.tables['futures_root_symbols'],
                data.root_symbols,
                txn,
                chunk_size,
            )
            asset_router = metadata.tables['asset_router']
            self._write_assets(
                asset_router,
                metadata.tables['futures_contracts'],
                'future',
                data.futures,
                txn,
                chunk_size,
            )
            self._write_assets(
                asset_router,
                metadata.tables['equities'],
                'equity',
                data.equities,
                txn,
                chunk_size,
            )

    def _write_df_to_table(self, tbl, df, txn, chunk_size):
        df.to_sql(
            tbl.name,
            txn.connection,
            index_label=first(tbl.primary_key.columns).name,
            if_exists='append',
            chunksize=chunk_size,
        )

    def _write_assets(self,
                      asset_router,
                      tbl,
                      asset_type,
                      assets,
                      txn,
                      chunk_size):
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
            metadata = generate_asset_db_metadata(bind=txn)

            # Create the SQL tables if they do not already exist.
            metadata.create_all(checkfirst=True)

            version_info = metadata.tables['version_info']
            if tables_already_exist:
                check_version_info(version_info, ASSET_DB_VERSION)
            else:
                write_version_info(version_info, ASSET_DB_VERSION)

            return metadata

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
            columns=['company_symbol', 'share_class_symbol', 'fuzzy_symbol'],
            index=tuple_series.index
        )
        equities_output = equities_output.join(split_symbols)

        # Upper-case all symbol data
        for col in ('symbol',
                    'company_symbol',
                    'share_class_symbol',
                    'fuzzy_symbol'):
            equities_output[col] = equities_output[col].str.upper()

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        for col in ('start_date',
                    'end_date',
                    'first_traded',
                    'auto_close_date'):
            equities_output[col] = _dt_to_epoch_ns(equities_output[col])

        return equities_output

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

    def _load_data(self, equities, futures, exchanges, root_symbols):
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

        equities_output = self._normalize_equities(equities)
        futures_output = self._normalize_futures(futures)

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
            futures=futures_output,
            exchanges=exchanges_output,
            root_symbols=root_symbols_output,
        )
