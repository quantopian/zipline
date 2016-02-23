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

from abc import (
    ABCMeta,
    abstractmethod,
)
from collections import namedtuple

import re
import pandas as pd
import numpy as np
from six import with_metaclass
import sqlalchemy as sa

from zipline.errors import SidAssignmentError, AssetDBVersionError
from zipline.assets._assets import Asset
from zipline.assets.asset_db_schema import (
    generate_asset_db_metadata,
    asset_db_table_names,
    ASSET_DB_VERSION,
)

SQLITE_MAX_VARIABLE_NUMBER = 999

# Define a namedtuple for use with the load_data and _load_data methods
AssetData = namedtuple('AssetData', 'equities futures exchanges root_symbols')

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


class AssetDBWriter(with_metaclass(ABCMeta)):
    """
    Class used to write arbitrary data to SQLite database.
    Concrete subclasses will implement the logic for a specific
    input datatypes by implementing the _load_data method.

    Methods
    -------
    write_all(engine, allow_sid_assignment=True, constraints=False)
        Write the data supplied at initialization to the database.
    init_db(engine, constraints=False)
        Create the SQLite tables (called by write_all).
    load_data()
        Returns data in standard format.

    """
    CHUNK_SIZE = SQLITE_MAX_VARIABLE_NUMBER

    def __init__(self, equities=None, futures=None, exchanges=None,
                 root_symbols=None):

        if equities is None:
            equities = self.defaultval()
        self._equities = equities

        if futures is None:
            futures = self.defaultval()
        self._futures = futures

        if exchanges is None:
            exchanges = self.defaultval()
        self._exchanges = exchanges

        if root_symbols is None:
            root_symbols = self.defaultval()
        self._root_symbols = root_symbols

    @abstractmethod
    def defaultval(self):
        raise NotImplementedError

    def write_all(self,
                  engine,
                  allow_sid_assignment=True):
        """ Write pre-supplied data to SQLite.

        Parameters
        ----------
        engine : Engine
            An SQLAlchemy engine to a SQL database.
        allow_sid_assignment: bool, optional
            If True then the class can assign sids where necessary.
        constraints : bool, optional
            If True then create SQL ForeignKey and PrimaryKey constraints.

        """
        self.allow_sid_assignment = allow_sid_assignment

        # Begin an SQL transaction.
        with engine.begin() as txn:
            # Create SQL tables.
            self.init_db(txn)
            # Get the data to add to SQL.
            data = self.load_data()
            # Write the data to SQL.
            self._write_exchanges(data.exchanges, txn)
            self._write_root_symbols(data.root_symbols, txn)
            self._write_futures(data.futures, txn)
            self._write_equities(data.equities, txn)

    def _write_df_to_table(self, df, tbl, bind):
        df.to_sql(
            tbl.name,
            bind.connection,
            index_label=[col.name for col in tbl.primary_key.columns][0],
            if_exists='append',
            chunksize=self.CHUNK_SIZE,
        )

    def _write_assets(self, assets, asset_tbl, asset_type, bind):
        self._write_df_to_table(assets, asset_tbl, bind)

        pd.DataFrame({self.asset_router.c.sid.name: assets.index.values,
                      self.asset_router.c.asset_type.name: asset_type}).to_sql(
            self.asset_router.name,
            bind.connection,
            if_exists='append',
            index=False,
            chunksize=self.CHUNK_SIZE,
        )

    def _write_exchanges(self, exchanges, bind):
        self._write_df_to_table(exchanges, self.futures_exchanges, bind)

    def _write_root_symbols(self, root_symbols, bind):
        self._write_df_to_table(root_symbols, self.futures_root_symbols, bind)

    def _write_futures(self, futures, bind):
        self._write_assets(futures, self.futures_contracts, 'future', bind)

    def _write_equities(self, equities, bind):
        self._write_assets(equities, self.equities, 'equity', bind)

    def check_for_tables(self, engine):
        """
        Checks if any tables are present in the current assets database.

        Returns
        -------
        bool
            True if any tables are present, otherwise False.
        """
        conn = engine.connect()
        for table_name in asset_db_table_names:
            if engine.dialect.has_table(conn, table_name):
                return True
        return False

    def init_db(self, engine):
        """Connect to database and create tables.

        Parameters
        ----------
        engine : Engine
            An engine to a SQL database.
        constraints : bool, optional
            If True, create SQL ForeignKey and PrimaryKey constraints.
        """
        tables_already_exist = self.check_for_tables(engine)
        metadata = generate_asset_db_metadata(bind=engine)

        for table_name in asset_db_table_names:
            setattr(self, table_name, metadata.tables[table_name])

        # Create the SQL tables if they do not already exist.
        metadata.create_all(checkfirst=True)

        if tables_already_exist:
            check_version_info(self.version_info, ASSET_DB_VERSION)
        else:
            write_version_info(self.version_info, ASSET_DB_VERSION)

        return metadata

    def load_data(self):
        """
        Returns a standard set of pandas.DataFrames:
        equities, futures, exchanges, root_symbols
        """

        data = self._load_data()

        ###############################
        # Generate equities DataFrame #
        ###############################

        # HACK: If company_name is provided, map it to asset_name
        if ('company_name' in data.equities.columns
                and 'asset_name' not in data.equities.columns):
            data.equities['asset_name'] = data.equities['company_name']
        if 'file_name' in data.equities.columns:
            data.equities['symbol'] = data.equities['file_name']

        equities_output = _generate_output_dataframe(
            data_subset=data.equities,
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
        equities_output['symbol'] = \
            equities_output.symbol.str.upper()
        equities_output['company_symbol'] = \
            equities_output.company_symbol.str.upper()
        equities_output['share_class_symbol'] = \
            equities_output.share_class_symbol.str.upper()
        equities_output['fuzzy_symbol'] = \
            equities_output.fuzzy_symbol.str.upper()

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        for date_col in ('start_date', 'end_date', 'first_traded',
                         'auto_close_date'):
            equities_output[date_col] = \
                self.dt_to_epoch_ns(equities_output[date_col])

        ##############################
        # Generate futures DataFrame #
        ##############################

        futures_output = _generate_output_dataframe(
            data_subset=data.futures,
            defaults=_futures_defaults,
        )

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        for date_col in ('start_date', 'end_date', 'first_traded',
                         'notice_date', 'expiration_date', 'auto_close_date'):
            futures_output[date_col] = \
                self.dt_to_epoch_ns(futures_output[date_col])

        # Convert symbols and root_symbols to upper case.
        futures_output['symbol'] = futures_output.symbol.str.upper()
        futures_output['root_symbol'] = futures_output.root_symbol.str.upper()

        ################################
        # Generate exchanges DataFrame #
        ################################

        exchanges_output = _generate_output_dataframe(
            data_subset=data.exchanges,
            defaults=_exchanges_defaults,
        )

        ###################################
        # Generate root symbols DataFrame #
        ###################################

        root_symbols_output = _generate_output_dataframe(
            data_subset=data.root_symbols,
            defaults=_root_symbols_defaults,
        )

        return AssetData(equities=equities_output,
                         futures=futures_output,
                         exchanges=exchanges_output,
                         root_symbols=root_symbols_output)

    @staticmethod
    def dt_to_epoch_ns(dt_series):
        index = pd.to_datetime(dt_series.values)
        try:
            index = index.tz_localize('UTC')
        except TypeError:
            index = index.tz_convert('UTC')

        return index.view(np.int64)

    @abstractmethod
    def _load_data(self):
        """
        Subclasses should implement this method to return data in a standard
        format: a pandas.DataFrame for each of the following tables:
        equities, futures, exchanges, root_symbols.

        For each of these DataFrames the index columns should be the integer
        unique identifier for the table, which are sid, sid, exchange_id and
        root_symbol_id respectively.
        """

        raise NotImplementedError('load_data')


class AssetDBWriterFromList(AssetDBWriter):
    """
    Class used to write list data to SQLite database.
    """

    defaultval = list

    def _load_data(self):

        # 0) Instantiate empty dictionaries
        _equities, _futures, _exchanges, _root_symbols = {}, {}, {}, {}

        # 1) Populate dictionaries
        # Return the largest sid in our database, if one exists.
        id_counter = sa.select(
            [sa.func.max(self.asset_router.c.sid)]
        ).execute().scalar()
        # Base sid creation on largest sid in database, or 0 if
        # no sids exist.
        if id_counter is None:
            id_counter = 0
        else:
            id_counter += 1
        for output, data in [(_equities, self._equities),
                             (_futures, self._futures), ]:
            for identifier in data:
                if isinstance(identifier, Asset):
                    sid = identifier.sid
                    metadata = identifier.to_dict()
                    output[sid] = metadata
                elif hasattr(identifier, '__int__'):
                    output[identifier.__int__()] = {'symbol': None}
                else:
                    if self.allow_sid_assignment:
                        output[id_counter] = {'symbol': identifier}
                        id_counter += 1
                    else:
                        raise SidAssignmentError(identifier=identifier)

        exchange_counter = 0
        for identifier in self._exchanges:
            if hasattr(identifier, '__int__'):
                _exchanges[identifier.__int__()] = {}
            else:
                _exchanges[exchange_counter] = {'exchange': identifier}
                exchange_counter += 1

        root_symbol_counter = 0
        for identifier in self._root_symbols:
            if hasattr(identifier, '__int__'):
                _root_symbols[identifier.__int__()] = {}
            else:
                _root_symbols[root_symbol_counter] = \
                    {'root_symbol': identifier}
                root_symbol_counter += 1

        # 2) Convert dictionaries to pandas.DataFrames.
        _equities = pd.DataFrame.from_dict(_equities, orient='index')
        _futures = pd.DataFrame.from_dict(_futures, orient='index')
        _exchanges = pd.DataFrame.from_dict(_exchanges, orient='index')
        _root_symbols = pd.DataFrame.from_dict(_root_symbols, orient='index')

        # 3) Return the data inside a named tuple.
        return AssetData(equities=_equities,
                         futures=_futures,
                         exchanges=_exchanges,
                         root_symbols=_root_symbols)


class AssetDBWriterFromDictionary(AssetDBWriter):
    """
    Class used to write dictionary data to SQLite database.

    Expects to be initialised with dictionaries in the following format:

    {id_0: {attribute_1 : ...}, id_1: {attribute_2: ...}, ...}
    """

    defaultval = dict

    def _load_data(self):

        _equities = pd.DataFrame.from_dict(self._equities, orient='index')
        _futures = pd.DataFrame.from_dict(self._futures, orient='index')
        _exchanges = pd.DataFrame.from_dict(self._exchanges, orient='index')
        _root_symbols = pd.DataFrame.from_dict(self._root_symbols,
                                               orient='index')

        return AssetData(equities=_equities,
                         futures=_futures,
                         exchanges=_exchanges,
                         root_symbols=_root_symbols)


class AssetDBWriterFromDataFrame(AssetDBWriter):
    """
    Class used to write pandas.DataFrame data to SQLite database.
    """

    defaultval = pd.DataFrame

    def _load_data(self):

        # Check whether identifier columns have been provided.
        # If they have, set the index to this column.
        # If not, assume the index already cotains the identifier information.
        for df, id_col in [
            (self._equities, 'sid'),
            (self._futures, 'sid'),
            (self._exchanges, 'exchange'),
            (self._root_symbols, 'root_symbol'),
        ]:
            if id_col in df.columns:
                df.set_index([id_col], inplace=True)

        return AssetData(equities=self._equities,
                         futures=self._futures,
                         exchanges=self._exchanges,
                         root_symbols=self._root_symbols)
