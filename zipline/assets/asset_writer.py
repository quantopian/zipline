from abc import (
    ABCMeta,
    abstractmethod,
)

import pandas as pd
import numpy as np
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types

from zipline.errors import (
    ConsumeAssetMetaDataError,
    InvalidAssetType,
    SidAssignmentError,
)
from zipline.assets import (
    Asset, Equity, Future
)

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
    'root_symbol',
    'expiration_date',
    'contract_multiplier',
    # The following fields are for compatibility with other systems
    'file_name',  # Used as symbol
    'company_name',  # Used as asset_name
    'start_date_nano',  # Used as start_date
    'end_date_nano'  # Used as end_date
]

# Expected fields for an Asset's metadata
ASSET_TABLE_FIELDS = [
    'sid',
    'symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange'
]

# Expected fields for an Asset's metadata
FUTURE_TABLE_FIELDS = ASSET_TABLE_FIELDS + [
    'root_symbol_id',
    'notice_date',
    'expiration_date',
    'contract_multiplier',
]

EQUITY_TABLE_FIELDS = ASSET_TABLE_FIELDS

EXCHANGE_TABLE_FIELDS = [
    'exchange_id',
    'exchange',
    'timezone'
]

ROOT_SYMBOL_TABLE_FIELDS = [
    'root_symbol_id',
    'root_symbol',
    'sector',
    'description',
    'exchange_id'
]


class AssetDBWriter(with_metaclass(ABCMeta)):
    """
    Class used to write arbitrary data to SQLite database.
    Concrete subclasses will implement the logic for a specific
    input datatypes by implementing the load_data method.

    Methods
    -------
    write_all(db_conn, fuzzy_char=None, allow_sid_assignment=True,
              constraints=False)
        Write the data supplied at initialization to the database.
    init_db(db_conn, constraints=False)
        Create the SQLite tables (called by write_all).
    load_data(self)
        Returns data in standard format.

    """

    def write_all(self, db_conn, fuzzy_char=None, allow_sid_assignment=True,
                  constraints=False):
        """ Write pre-supplied data to SQLite.

        Parameters
        ----------
        db_conn: sqlite3.Connection
            A connection to a SQLite database.
        fuzzy_char: string
            A string for use in fuzzy matching.
        allow_sid_assignment: boolean
            If True then the class can assign sids where necessary.
        constraints: boolean
            If True, create SQL ForeignKey and Index constraints.

        """
        self.fuzzy_char = fuzzy_char
        self.allow_sid_assignment = allow_sid_assignment
        if allow_sid_assignment:
            ts = normalize_date(pd.Timestamp('now', tz='UTC'))
            # Store as seconds since UNIX Epoch for compatibility
            # with SQL.
            self.end_date_to_assign = (ts.value // 10 ** 9)

        # Store a nested-dict of all metadata for
        # reference when building Assets
        self.metadata_cache = {}

        # Create SQL tables
        self.init_db(db_conn, constraints)

        # Get the data to add to SQL
        equities, futures, exchanges, root_symbols = self.load_data()

        c = db_conn.cursor()
        # Write to the SQL tables, using the raw SQL driver instead
        # of the pandas.DataFrame.to_sql method, as the former
        # allows us to create an SQL transaction
        c.execute('BEGIN')
        # Everything between here and the db_conn.commit()
        # will be part of of the same SQL transaction.
        self._write_exchanges(exchanges, db_conn)
        self._write_root_symbols(root_symbols, db_conn)
        self._write_futures(futures, db_conn)
        self._write_equities(equities, db_conn)
        db_conn.commit()

    def _write_exchanges(self, exchanges, db_conn):

        data = [tuple(x) for x in exchanges.to_records()]

        c = db_conn.cursor()
        # The OR IGNORE syntax means we do not insert data
        # which would violate an SQL constraint.
        c.executemany("""
            INSERT OR IGNORE INTO futures_exchanges
            ('exchange_id', 'exchange', 'timezone')
            VALUES (?, ?, ?)
            """, data)

    def _write_root_symbols(self, root_symbols, db_conn):

        data = [tuple(x) for x in root_symbols.to_records()]

        c = db_conn.cursor()
        c.executemany("""
            INSERT OR IGNORE INTO futures_root_symbols
            ('root_symbol_id', 'root_symbol', 'sector',
             'description', 'exchange_id')
            VALUES(?, ?, ?, ?, ?)
        """, data)

    def _write_futures(self, futures, db_conn):

        # Retrieve the data to add to the futures_contracts table
        data = [tuple(x) for x in futures.to_records()]

        # Retrieve the data to add to the asset_router table
        sids = [x[0] for x in data]
        futs = ['future'] * len(sids)
        to_insert = zip(sids, futs)

        c = db_conn.cursor()
        c.executemany("""
            INSERT OR IGNORE INTO futures_contracts
            ('sid', 'symbol', 'root_symbol', 'asset_name',
             'start_date', 'end_date', 'first_traded', 'exchange',
             'notice_date', 'expiration_date', 'contract_multiplier')
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        c.executemany("""
            INSERT INTO asset_router
            ('sid', 'asset_type')
            VALUES(?,?)
        """, to_insert)

    def _write_equities(self, equities, db_conn):

        # Apply fuzzy matching.
        if self.fuzzy_char:
            equities['fuzzy'] = equities['symbol'].str.\
                replace(self.fuzzy_char, '')

        # Retrieve the data to add to the equitites table
        data = [tuple(x) for x in equities.to_records()]

        # Retrieve the data to add to the asset_router table
        sids = [x[0] for x in data]
        eqs = ['equity'] * len(sids)
        to_insert = zip(sids, eqs)

        c = db_conn.cursor()
        c.executemany("""
            INSERT OR IGNORE INTO equities
            ('sid', 'symbol', 'asset_name', 'start_date',
             'end_date', 'first_traded', 'exchange', 'fuzzy')
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        c.executemany("""
            INSERT INTO asset_router
            ('sid', 'asset_type')
            VALUES(?,?)
        """, to_insert)

    def init_db(self,
                db_conn,
                constraints=False):
        """Connect to database and create tables.

        Parameters
        ----------
        db_conn: sqlite3.Connection
            A connection to a SQLite database.
        constraints: boolean
            If True, create SQL ForeignKey and Index constraints.
        """

        c = db_conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS equities (
            sid INTEGER NOT NULL,
            symbol TEXT,
            asset_name TEXT,
            start_date INTEGER DEFAULT 0,
            end_date INTEGER,
            first_traded INTEGER,
            exchange TEXT,
            fuzzy TEXT
        )""")

        c.execute("""
            CREATE TABLE IF NOT EXISTS futures_exchanges (
            exchange_id INTEGER NOT NULL,
            exchange TEXT,
            timezone TEXT
        )""")

        c.execute("""
            CREATE TABLE IF NOT EXISTS futures_root_symbols (
            root_symbol_id INTEGER NOT NULL,
            root_symbol TEXT,
            sector TEXT,
            description TEXT,
            exchange_id INTEGER{fk}
        )""".format(fk=", FOREIGN KEY(exchange_id) REFERENCES "
                    "futures_exchanges(exchange_id)"
                    if constraints else ""))

        c.execute("""
            CREATE TABLE IF NOT EXISTS futures_contracts (
            sid INTEGER NOT NULL,
            symbol TEXT,
            root_symbol_id INTEGER,
            root_symbol TEXT,
            asset_name TEXT,
            start_date INTEGER DEFAULT 0,
            end_date INTEGER,
            first_traded INTEGER,
            exchange_id INTEGER,
            exchange TEXT,
            notice_date INTEGER,
            expiration_date INTEGER,
            contract_multiplier REAL{fk}
        )""".format(fk=", FOREIGN KEY(exchange_id) REFERENCES "
                    "futures_exchanges(exchange_id), "
                    "FOREIGN KEY(root_symbol_id) REFERENCES "
                    "futures_root_symbols(root_symbol_id)"
                    if constraints else ""))

        c.execute("""
            CREATE TABLE IF NOT EXISTS asset_router
            (sid integer,
            asset_type text
        )""")

        # Note: Would be optimal to use INTEGER PRIMARY KEY here, but SQLite
        # tables cannot be modified after creation. Using CREATE UNIQUE INDEX
        # only marginally less performant.
        if constraints:

            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_equities_sid '
                      'ON equities(sid)')
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_equities_symbol '
                      'ON equities(symbol)')
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_equities_fuzzy '
                      'ON equities(fuzzy)')
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_futures_exchanges_en '   # noqa
                      'ON futures_exchanges(exchange_id)')
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_futures_contracts_sid '  # noqa
                      'ON futures_contracts(sid)')
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_futures_root_symbols_id '  # noqa
                      'ON futures_root_symbols(root_symbol_id)')
            c.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_asset_router_sid  '
                      'ON asset_router(sid)')

        # Note: Also need a max_date table.

        db_conn.commit()

    @staticmethod
    def dict_subset(dict_, subset):
        res = {}
        for k in subset:
           res[k] = dict_[k]
        return res

    @abstractmethod
    def load_data(self):
        """
        Subclasses should implement this method to return data in a standard
        format: a pandas.DataFrame for each of the following tables:
        equities, futures, exchanges, root_symbols
        """

        raise NotImplementedError('load_data')


class AssetDBWriterFromList(AssetDBWriter):
    """
    Class used to write list data to SQLite database.
    """

    def __init__(self, equities=[], futures=[], exchanges=[], root_symbols=[]):

        self._equities = equities
        self._futures = futures
        self._exchanges = exchanges
        self._root_symbols = root_symbols

    def load_data(self):

        equities_data = pd.DataFrame(index=self._equities)

        futures_data = pd.DataFrame(index=self._futures)

        exchange_data = pd.DataFrame(index=self._exchanges)

        root_symbol_data = pd.DataFrame(index=self._root_symbols)

        # Assume the keys are the exchange_ids
        exchange_cols = ['exchange', 'timezone']
        exchanges = pd.DataFrame(columns=exchange_cols)

        # Assume the keys are the root_symbol_ids
        root_symbols_cols = ['root_symbol', 'sector',
                             'description', 'exchange_id']
        root_symbols = pd.DataFrame(columns=root_symbols_cols)

        # Assume the keys are the sids
        futures_cols = ['symbol', 'root_symbol', 'asset_name',
                        'start_date', 'end_date', 'first_traded', 'exchange',
                        'notice_date', 'expiration_date',
                        'contract_multiplier']
        futures = pd.DataFrame(columns=futures_cols)

        # Assume the keys are the sids
        equities_cols = ['symbol', 'asset_name', 'start_date',
                         'end_date', 'first_traded', 'exchange', 'fuzzy']
        equities = pd.DataFrame(columns=equities_cols)

        # Append any data the user has provided.
        exchanges = exchanges.append(exchange_data, verify_integrity=True)
        root_symbols = root_symbols.append(root_symbol_data,
                                           verify_integrity=True)
        futures = futures.append(futures_data, verify_integrity=True)
        equities = equities.append(equities_data, verify_integrity=True)

        return equities, futures, exchanges, root_symbols


class AssetDBWriterFromDictionary(AssetDBWriter):
    """
    Class used to write dictionary data to SQLite database.

    Expects a dictionary to be passed to load_data
    with the following format:

    {id_0: {start_date : ...}, id_1: {start_data: ...}, ...}
    """

    def __init__(self, equities={}, futures={}, exchanges={}, root_symbols={}):

        self._equities = equities
        self._futures = futures
        self._exchanges = exchanges
        self._root_symbols = root_symbols

    def load_data(self):
        """
        Convert our nested dictionaries to pandas DataFrames.
        """

        # ******** Generate equities data ********
        equities_defaults = {
            'symbol': None,
            'asset_name': None,
            'start_date': 0,
            'end_date': None,
            'first_traded': None,
            'exchange': None,
            'fuzzy': None,
        }
        equities_cols = {'symbol', 'asset_name', 'start_date',
                         'end_date', 'first_traded', 'exchange', 'fuzzy'}
        equities_data = pd.DataFrame.from_dict(self._equities, orient='index')
        cols = set(equities_data.columns)

        # Drop columns with unrecognised headers.
        equities_data.drop(cols - (cols & equities_cols), axis=1, inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = equities_cols - set(equities_data.columns)

        # Combine the users supplied data with our required columns.
        equities_data = pd.concat(
            (equities_data, pd.DataFrame(
                self.dict_subset(equities_defaults, need),
                equities_data.index,
            )),
            axis=1,
            copy=False
        )

        # ******** Generate futures data ********
        futures_defaults = {
            'symbol': None,
            'root_symbol': None,
            'asset_name': None,
            'start_date': 0,
            'end_date': None,
            'first_traded': None,
            'exchange': None,
            'notice_date': None,
            'expiration_date': None,
            'contract_multiplier': 1,
        }
        futures_cols = {'symbol', 'root_symbol', 'asset_name',
                        'start_date', 'end_date', 'first_traded', 'exchange',
                        'notice_date', 'expiration_date',
                        'contract_multiplier'}
        futures_data = pd.DataFrame.from_dict(self._futures, orient='index')
        cols = set(futures_data.columns)

        # Drop columns with unrecognised headers.
        futures_data.drop(cols - (cols & futures_cols), axis=1, inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = futures_cols - set(futures_data.columns)

        # Combine the users supplied data with our required columns.
        futures_data = pd.concat(
            (futures_data, pd.DataFrame(
                self.dict_subset(futures_defaults, need),
                futures_data.index,
            )),
            axis=1,
            copy=False
        )

        # ******** Generate exchanges data ********
        exchanges_defaults = {
            'exchange': None,
            'timezone': None,
        }
        exchanges_cols = {'exchange', 'timezone', }
        exchanges_data = pd.DataFrame.from_dict(self._exchanges,
                                                orient='index')
        cols = set(exchanges_data.columns)

        # Drop columns with unrecognised headers.
        exchanges_data.drop(cols - (cols & exchanges_cols), axis=1,
                            inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = exchanges_cols - set(exchanges_data.columns)

        # Combine the users supplied data with our required columns.
        exchanges_data = pd.concat(
            (exchanges_data, pd.DataFrame(
                self.dict_subset(exchanges_defaults, need),
                exchanges_data.index,
            )),
            axis=1,
            copy=False
        )

        # ******** Generate root symbols data ********
        root_symbols_defaults = {
            'root_symbol': None,
            'sector': None,
            'description': None,
            'exchange_id': None,
        }
        root_symbols_cols = {'root_symbol', 'sector',
                             'description', 'exchange_id'}
        root_symbols_data = pd.DataFrame.from_dict(self._root_symbols,
                                                   orient='index')
        cols = set(root_symbols_data.columns)

        # Drop columns with unrecognised headers.
        root_symbols_data.drop(cols - (cols & root_symbols_cols), axis=1,
                               inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = root_symbols_cols - set(root_symbols_data.columns)

        # Combine the users supplied data with our required columns.
        root_symbols_data = pd.concat(
            (root_symbols_data, pd.DataFrame(
                self.dict_subset(root_symbols_defaults, need),
                root_symbols_data.index,
            )),
            axis=1,
            copy=False
        )

        return equities_data, futures_data, exchanges_data, root_symbols_data


class AssetDBWriterFromDataFrame(AssetDBWriter):
    """
    Class used to write pandas.DataFrame data to SQLite database.
    """

    def __init__(self, equities=pd.DataFrame(), futures=pd.DataFrame(),
                 exchanges=pd.DataFrame(), root_symbols=pd.DataFrame()):

        self._equities = equities
        self._futures = futures
        self._exchanges = exchanges
        self._root_symbols = root_symbols

    def load_data(self):
        """
        Convert our nested  to pandas DataFrames.
        """

        # ******** Generate equities data ********
        equities_defaults = {
            'symbol': None,
            'asset_name': None,
            'start_date': 0,
            'end_date': None,
            'first_traded': None,
            'exchange': None,
            'fuzzy': None,
        }
        equities_cols = {'symbol', 'asset_name', 'start_date',
                         'end_date', 'first_traded', 'exchange', 'fuzzy'}
        equities_data = self._equities
        cols = set(equities_data.columns)

        # Drop columns with unrecognised headers.
        equities_data.drop(cols - (cols & equities_cols), axis=1, inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = equities_cols - set(equities_data.columns)

        # Combine the users supplied data with our required columns.
        equities_data = pd.concat(
            (equities_data, pd.DataFrame(
                self.dict_subset(equities_defaults, need),
                equities_data.index,
            )),
            axis=1,
            copy=False
        )

        # ******** Generate futures data ********
        futures_defaults = {
            'symbol': None,
            'root_symbol': None,
            'asset_name': None,
            'start_date': 0,
            'end_date': None,
            'first_traded': None,
            'exchange': None,
            'notice_date': None,
            'expiration_date': None,
            'contract_multiplier': 1,
        }
        futures_cols = {'symbol', 'root_symbol', 'asset_name',
                        'start_date', 'end_date', 'first_traded', 'exchange',
                        'notice_date', 'expiration_date',
                        'contract_multiplier'}
        futures_data = self._futures
        cols = set(futures_data.columns)

        # Drop columns with unrecognised headers.
        futures_data.drop(cols - (cols & futures_cols), axis=1, inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = futures_cols - set(futures_data.columns)

        # Combine the users supplied data with our required columns.
        futures_data = pd.concat(
            (futures_data, pd.DataFrame(
                self.dict_subset(futures_defaults, need),
                futures_data.index,
            )),
            axis=1,
            copy=False
        )

        # ******** Generate exchanges data ********
        exchanges_defaults = {
            'exchange': None,
            'timezone': None,
        }
        exchanges_cols = {'exchange', 'timezone', }
        exchanges_data = self._exchanges
        cols = set(exchanges_data.columns)

        # Drop columns with unrecognised headers.
        exchanges_data.drop(cols - (cols & exchanges_cols), axis=1,
                            inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = exchanges_cols - set(exchanges_data.columns)

        # Combine the users supplied data with our required columns.
        exchanges_data = pd.concat(
            (exchanges_data, pd.DataFrame(
                self.dict_subset(exchanges_defaults, need),
                exchanges_data.index,
            )),
            axis=1,
            copy=False
        )

        # ******** Generate root symbols data ********
        root_symbols_defaults = {
            'root_symbol': None,
            'sector': None,
            'description': None,
            'exchange_id': None,
        }
        root_symbols_cols = {'root_symbol', 'sector',
                             'description', 'exchange_id'}
        root_symbols_data = self._root_symbols
        cols = set(root_symbols_data.columns)

        # Drop columns with unrecognised headers.
        root_symbols_data.drop(cols - (cols & root_symbols_cols), axis=1,
                               inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = root_symbols_cols - set(root_symbols_data.columns)

        # Combine the users supplied data with our required columns.
        root_symbols_data = pd.concat(
            (root_symbols_data, pd.DataFrame(
                self.dict_subset(root_symbols_defaults, need),
                root_symbols_data.index,
            )),
            axis=1,
            copy=False
        )

        return equities_data, futures_data, exchanges_data, root_symbols_data


class AssetDBWriterLegacy(AssetDBWriter):
    """
    Overwrites some of the functionality of AssetDBWriter.
    Used for backward compatibility. Will be deprecated.

    Methods
    -------
    write_all(db_conn, fuzzy_char=None, allow_sid_assignment=True,
              constraints=False)
        Write the data supplied at initialization to the database.
    write_block(self, identifier, **kwargs)
        Inserts the given metadata kwargs to the entry for the given
        sid. Matching fields in the existing entry will be overwritten.
        Will be deprecated in future versions of zipline.
    init_db(db_conn, constraints=False)
        Create the SQLite tables (called by write_all).
    load_data(self, equities, futures, exchanges, root_symbols)
        Returns data in standard format.
    consume_identifiers(self, db_conn, fuzzy_char=None,
                        allow_sid_assignment=True,
                        constraints=False)
        Consume the identifiers supplied at initialization.
        Will be deprecated in future versions of zipline.
    """

    def __init__(self, data):

        self._data = data
        self.metadata_cache = {}

    def write_all(self,
                  db_conn,
                  fuzzy_char=None,
                  allow_sid_assignment=True,
                  constraints=False):
        """Top-level entry point for writing a new asset db.

        Parameters
        ----------
        db_conn: sqlite3.Connection
            A connection to our SQLite database.
        fuzzy_char: string
            A string to be used in fuzzy matching.
        allow_sid_assignment: boolean
            If True, allow the writer to assign sids where necessary.
        constraints: boolean
            If True, add SQL constraints to tables.
        """

        self.conn = db_conn

        self.fuzzy_char = fuzzy_char
        self.allow_sid_assignment = allow_sid_assignment

        # This flag controls if the AssetDBWriter is allowed to generate its
        # own sids. If False, metadata that does not contain a sid will raise
        # an exception when building assets.
        if allow_sid_assignment:
            self.end_date_to_assign = normalize_date(
                pd.Timestamp('now', tz='UTC'))

        # Create SQL tables.
        self.init_db(self.conn, constraints)

        # Write to SQL tables.
        for sid, metadata in self.load_data(self._data):
            self.write_block(sid, **metadata)

    def write_block(self, identifier, **kwargs):
        """
        Inserts the given metadata kwargs to the entry for the given
        sid. Matching fields in the existing entry will be overwritten.
        Will be deprecated in future versions of zipline.
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
            # Do not accept NaNs from dataframes
            if isinstance(value, float) and np.isnan(value):
                continue
            entry[key] = value

        # Check if the sid is declared
        try:
            entry['sid']
        except KeyError:
            # If the sid is not a sid, assign one
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
            c.execute("""
                INSERT INTO equities (
                sid,
                symbol,
                asset_name,
                start_date,
                end_date,
                first_traded,
                exchange,
                fuzzy)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """, t)

            t = (asset.sid,
                 'equity')
            c.execute("""
                INSERT INTO asset_router (
                sid, asset_type)
                VALUES(?, ?)
            """, t)

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
            c.execute("""
                INSERT INTO futures_contracts(
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
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, t)

            t = (asset.sid,
                 'future')
            c.execute("""
                INSERT INTO asset_router (
                sid,
                asset_type)
                VALUES(?, ?)
            """, t)
        else:
            raise InvalidAssetType(asset_type=asset_type)

        self.metadata_cache[identifier] = entry

        self.conn.commit()

    def consume_identifiers(self, db_conn, fuzzy_char=None,
                            allow_sid_assignment=True, constraints=False):
        """
        Consumes the given identifiers in to the metadata cache of this
        AssetDBWriter, and adds to database.
        Will be deprecated in future versions of zipline.
        """

        self.conn = db_conn
        self.fuzzy_char = fuzzy_char
        self.allow_sid_assignment = allow_sid_assignment

        # This flag controls if the AssetDBWriter is allowed to generate its
        # own sids. If False, metadata that does not contain a sid will raiset
        # an exception when building assets.
        if allow_sid_assignment:
            self.end_date_to_assign = normalize_date(
                pd.Timestamp('now', tz='UTC'))

        # Create SQL tables
        self.init_db(self.conn, constraints)

        for identifier in self._data:
            # Handle case where full Assets are passed in
            # For example, in the creation of a DataFrameSource, the source's
            # 'sid' args may be full Assets
            if isinstance(identifier, Asset):
                sid = identifier.sid
                metadata = identifier.to_dict()
                metadata['asset_type'] = identifier.__class__.__name__
                self.write_block(sid, **metadata)
            else:
                self.write_block(identifier)


class NullAssetDBWriterLegacy(AssetDBWriterLegacy):
    """
    An implementation of AssetDBWriterLegacy for use
    when no data is initially specified.
    """

    def load_data(self, __):
        for i in iter(()):
            yield


class AssetDBWriterLegacyFromList(AssetDBWriterLegacy):
    """
    Returns a generator yielding entries from sid_list.
    """
    def load_data(self, sid_list):

        for i in sid_list:
            yield i


class AssetDBWriterLegacyFromDictionary(AssetDBWriterLegacy):
    """ An implementation of AssetDBWriter for use
        with dictionaries.

        Expects a dictionary to be passed to load_data
        with the following format:

        {id_0: {start_date : ...}, id_1: {start_data: ...}, ...}
    """

    def load_data(self, dict):
        """
        Returns a generator yielding pairs of (identifier, metadata)
        """
        for identifier, metadata in dict.items():
            yield identifier, metadata


class AssetDBWriterLegacyFromDataFrame(AssetDBWriterLegacy):
    """ An implementation of AssetDBWriter for use
        with pandas DataFrames.

        Expects dataframe to be passed to load_data
        to have the following structure:
            * column names must be the metadata fields
            * index must be the different asset identifiers
            * array contents should be the metadata value
    """

    def load_data(self, dataframe):
        """
        Returns a generator yielding pairs of (identifier, metadata)
        """
        for identifier, row in dataframe.iterrows():
            yield identifier, row.to_dict()


class AssetDBWriterLegacyFromReadable(AssetDBWriterLegacy):
    """ An implementation of AssetDBWriter for use
        with objects with a 'read' property.

        The object's read method must return rows
        containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    """

    def load_data(self, readable):
        """
        Returns a generator yielding pairs of (identifier, metadata)
        """
        for row in readable.read():
            id_metadata = {}
            for field in ASSET_FIELDS:
                try:
                    row_value = row[field]
                    # Avoid passing placeholder strings
                    if row_value and (row_value != 'None'):
                        id_metadata[field] = row[field]
                except KeyError:
                    continue
                except IndexError:
                    continue
            if 'sid' in id_metadata:
                identifier = id_metadata['sid']
                del id_metadata['sid']
            elif 'symbol' in id_metadata:
                identifier = id_metadata['symbol']
                del id_metadata['symbol']
            else:
                raise ConsumeAssetMetaDataError(obj=row)
            yield identifier, id_metadata
