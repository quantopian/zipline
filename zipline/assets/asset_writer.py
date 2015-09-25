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

from zipline.errors import SidAssignmentError
from zipline.assets._assets import Asset

# Define a namedtuple for use with the load_data and _load_data methods
AssetData = namedtuple('AssetData', 'equities futures exchanges root_symbols')

# Expected fields for an Asset's metadata
ASSET_TABLE_FIELDS = frozenset({
    'sid',
    'symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange',
})

# Expected fields for a Future's metadata
FUTURE_TABLE_FIELDS = ASSET_TABLE_FIELDS | {
    'notice_date',
    'expiration_date',
    'auto_close_date',
    'contract_multiplier',
}

# Expected fields for an Equity's metadata
EQUITY_TABLE_FIELDS = ASSET_TABLE_FIELDS | {
    'company_symbol',
    'share_class_symbol',
    'fuzzy_symbol',
}

EXCHANGE_TABLE_FIELDS = frozenset({
    'exchange',
    'timezone',
})

ROOT_SYMBOL_TABLE_FIELDS = frozenset({
    'root_symbol',
    'root_symbol_id',
    'sector',
    'description',
    'exchange',
})

# Default values for the equities DataFrame
_equities_defaults = {
    'symbol': None,
    'asset_name': None,
    'start_date': 0,
    'end_date': 2 ** 62 - 1,
    'first_traded': None,
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
    'contract_multiplier': 1,
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
    desired_cols = {col for col in defaults.keys()}

    # Drop columns with unrecognised headers.
    data_subset.drop(cols - (cols & desired_cols),
                     axis=1,
                     inplace=True)

    # Get those columns which we need but
    # for which no data has been supplied.
    need = desired_cols - set(data_subset.columns)

    # Combine the users supplied data with our required columns.
    output = pd.concat(
        (data_subset, pd.DataFrame(
            _dict_subset(defaults, need),
            data_subset.index,
        )),
        axis=1,
        copy=False
    )

    return output


def _dict_subset(dict_, subset):
    res = {}
    for k in subset:
        res[k] = dict_[k]
    return res


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
    def write_all(self,
                  engine,
                  allow_sid_assignment=True,
                  constraints=True):
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
            self.init_db(txn, constraints)
            # Get the data to add to SQL.
            data = self.load_data()
            # Write the data to SQL.
            self._write_exchanges(data.exchanges, txn)
            self._write_root_symbols(data.root_symbols, txn)
            self._write_futures(data.futures, txn)
            self._write_equities(data.equities, txn)

    def _write_exchanges(self, exchanges, bind=None):
        recs = exchanges.reset_index().rename_axis(
            {'index': 'exchange'},
            1,
        ).to_dict('records')
        # In SQLAlchemy, insert().values([]) will insert NULLs,
        # hence we check first to avoid violating NOT NULL constraints.
        if recs:
            self.futures_exchanges.insert().values(recs).execute(bind=bind)

    def _write_root_symbols(self, root_symbols, bind=None):
        recs = root_symbols.reset_index().rename_axis(
            {'index': 'root_symbol'},
            1,
        ).to_dict('records')
        if recs:
            self.futures_root_symbols.insert().values(recs).execute(bind=bind)

    def _write_futures(self, futures, bind=None):
        recs = futures.reset_index().rename_axis(
            {'index': 'sid'},
            1,
        ).to_dict('records')
        for record in recs:
            self.futures_contracts.insert().values([record]).execute(bind=bind)
            self.asset_router.insert().values([(record['sid'], 'future')])\
                .execute(bind=bind)

    def _write_equities(self, equities, bind=None):
        recs = equities.reset_index().rename_axis(
            {'index': 'sid'},
            1,
        ).to_dict('records')
        for record in recs:
            self.equities.insert().values([record]).execute(bind=bind)
            self.asset_router.insert().values((record['sid'], 'equity'))\
                .execute(bind=bind)

    def init_db(self, engine, constraints=True):
        """Connect to database and create tables.

        Parameters
        ----------
        engine : Engine
            An engine to a SQL database.
        constraints : bool, optional
            If True, create SQL ForeignKey and PrimaryKey constraints.
        """
        self.sql_metadata = metadata = sa.MetaData(bind=engine)
        self.equities = sa.Table(
            'equities',
            metadata,
            sa.Column(
                'sid',
                sa.Integer,
                unique=True,
                nullable=False,
                primary_key=constraints,
            ),
            sa.Column('symbol', sa.Text),
            sa.Column('company_symbol', sa.Text, index=True),
            sa.Column('share_class_symbol', sa.Text),
            sa.Column('fuzzy_symbol', sa.Text, index=True),
            sa.Column('asset_name', sa.Text),
            sa.Column('start_date', sa.Integer, default=0),
            sa.Column('end_date', sa.Integer),
            sa.Column('first_traded', sa.Integer),
            sa.Column('exchange', sa.Text),
        )
        self.futures_exchanges = sa.Table(
            'futures_exchanges',
            metadata,
            sa.Column(
                'exchange',
                sa.Text,
                unique=True,
                nullable=False,
                primary_key=constraints,
            ),
            sa.Column('timezone', sa.Text),
        )
        self.futures_root_symbols = sa.Table(
            'futures_root_symbols',
            metadata,
            sa.Column(
                'root_symbol',
                sa.Text,
                unique=True,
                nullable=False,
                primary_key=constraints,
            ),
            sa.Column('root_symbol_id', sa.Integer),
            sa.Column('sector', sa.Text),
            sa.Column('description', sa.Text),
            sa.Column(
                'exchange',
                sa.Text,
                *((sa.ForeignKey(self.futures_exchanges.c.exchange),)
                  if constraints else ())
            ),
        )
        self.futures_contracts = sa.Table(
            'futures_contracts',
            metadata,
            sa.Column(
                'sid',
                sa.Integer,
                unique=True,
                nullable=False,
                primary_key=constraints,
            ),
            sa.Column('symbol', sa.Text),
            sa.Column(
                'root_symbol',
                sa.Text,
                *((sa.ForeignKey(self.futures_root_symbols.c.root_symbol),)
                  if constraints else ())
            ),
            sa.Column('asset_name', sa.Text),
            sa.Column('start_date', sa.Integer, default=0),
            sa.Column('end_date', sa.Integer),
            sa.Column('first_traded', sa.Integer),
            sa.Column(
                'exchange',
                sa.Text,
                *((sa.ForeignKey(self.futures_exchanges.c.exchange),)
                  if constraints else ())
            ),
            sa.Column('notice_date', sa.Integer),
            sa.Column('expiration_date', sa.Integer),
            sa.Column('auto_close_date', sa.Integer),
            sa.Column('contract_multiplier', sa.Float),
        )
        self.asset_router = sa.Table(
            'asset_router',
            metadata,
            sa.Column(
                'sid',
                sa.Integer,
                unique=True,
                nullable=False,
                primary_key=constraints),
            sa.Column('asset_type', sa.Text),
        )
        # Create the SQL tables if they do not already exist.
        metadata.create_all(checkfirst=True)
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
        if ('company_name' in data.equities.columns) \
                and ('asset_name' not in data.equities.columns):
            data.equities['asset_name'] = data.equities['company_name']
        if ('file_name' in data.equities.columns):
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
        equities_output['start_date'] = \
            equities_output['start_date'].apply(self.convert_datetime)
        equities_output['end_date'] = \
            equities_output['end_date'].apply(self.convert_datetime)
        equities_output['first_traded'] = \
            equities_output['first_traded'].apply(self.convert_datetime)

        ##############################
        # Generate futures DataFrame #
        ##############################

        futures_output = _generate_output_dataframe(
            data_subset=data.futures,
            defaults=_futures_defaults,
        )

        # Convert date columns to UNIX Epoch integers (nanoseconds)
        futures_output['start_date'] = \
            futures_output['start_date'].apply(self.convert_datetime)
        futures_output['end_date'] = \
            futures_output['end_date'].apply(self.convert_datetime)
        futures_output['first_traded'] = \
            futures_output['first_traded'].apply(self.convert_datetime)
        futures_output['notice_date'] = \
            futures_output['notice_date'].apply(self.convert_datetime)
        futures_output['expiration_date'] = \
            futures_output['expiration_date'].apply(self.convert_datetime)
        futures_output['auto_close_date'] = \
            futures_output['auto_close_date'].apply(self.convert_datetime)

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

    def convert_datetime(self, dt):
        """Convert a datetime variable to integer of nanoseconds
           since UNIX Epoch.

        Parameters
        ----------
        dt : datetime-coercible
            A string, int or pd.Timestamp instance representing a datetime, or
            None/NaN.

        Returns
        -------
        int
            nanoseconds since UNIX Epoch, or None if parameter 'dt' is null.
        """

        # Check for null parameter
        if pd.isnull(dt):
            return None

        # If no timezone is specified, assume UTC.
        # Otherwise, convert to UTC.
        try:
            dt = pd.Timestamp(dt).tz_localize('UTC')
        except TypeError:
            dt = pd.Timestamp(dt).tz_convert('UTC')

        # Get seconds from UNIX Epoch
        total_seconds_from_epoch = self._seconds_from_unix_time(dt)

        # Return nanoseconds since UNIX Epoch
        return int(total_seconds_from_epoch * 1000000000)

    def _seconds_from_unix_time(self, dt):
        """Return seconds between dt and UNIX Epoch.

        Parameters
        ----------
        dt: pandas.Timestamp
            The time for which to calculate seconds since UNIX Epoch.

        Returns
        -------
        float
            Seconds between dt and UNIX Epoch.

        """
        epoch = pd.to_datetime(0, utc=True)
        delta = dt - epoch
        return delta.total_seconds()

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

    def __init__(self, equities=None, futures=None, exchanges=None,
                 root_symbols=None):

        if equities is not None:
            self._equities = equities
        else:
            self._equities = []

        if futures is not None:
            self._futures = futures
        else:
            self._futures = []

        if exchanges is not None:
            self._exchanges = exchanges
        else:
            self._exchanges = []

        if root_symbols is not None:
            self._root_symbols = root_symbols
        else:
            self._root_symbols = []

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
                    metadata['asset_type'] = identifier.__class__.__name__
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

    def __init__(self, equities=None, futures=None, exchanges=None,
                 root_symbols=None):

        if equities is not None:
            self._equities = equities
        else:
            self._equities = {}

        if futures is not None:
            self._futures = futures
        else:
            self._futures = {}

        if exchanges is not None:
            self._exchanges = exchanges
        else:
            self._exchanges = {}

        if root_symbols is not None:
            self._root_symbols = root_symbols
        else:
            self._root_symbols = {}

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

    def __init__(self, equities=None, futures=None, exchanges=None,
                 root_symbols=None):

        if equities is not None:
            self._equities = equities
        else:
            self._equities = pd.DataFrame()

        if futures is not None:
            self._futures = futures
        else:
            self._futures = pd.DataFrame()

        if exchanges is not None:
            self._exchanges = exchanges
        else:
            self._exchanges = pd.DataFrame()

        if root_symbols is not None:
            self._root_symbols = root_symbols
        else:
            self._root_symbols = pd.DataFrame()

    def _load_data(self):

        # Check whether identifier columns have been provided.
        # If they have, set the index to this column.
        # If not, assume the index already cotains the identifier information.
        if 'sid' in self._equities.columns:
            self._equities.set_index(['sid'], inplace=True)
        if 'sid' in self._futures.columns:
            self._futures.set_index(['sid'], inplace=True)
        if 'exchange_id' in self._exchanges.columns:
            self._exchanges.set_index(['exchange'], inplace=True)
        if 'root_symbol_id' in self._root_symbols.columns:
            self._root_symbols.set_index(['root_symbol'], inplace=True)

        return AssetData(equities=self._equities,
                         futures=self._futures,
                         exchanges=self._exchanges,
                         root_symbols=self._root_symbols)
