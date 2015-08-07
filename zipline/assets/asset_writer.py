from abc import (
    ABCMeta,
    abstractmethod,
)

import pandas as pd
from six import with_metaclass
import sqlalchemy as sa

from zipline.errors import SidAssignmentError

ASSET_FIELDS = frozenset({
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
    'end_date_nano',  # Used as end_date
})

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


# Expected fields for an Asset's metadata
FUTURE_TABLE_FIELDS = ASSET_TABLE_FIELDS | {
    'root_symbol_id',
    'notice_date',
    'expiration_date',
    'contract_multiplier',
}

EQUITY_TABLE_FIELDS = ASSET_TABLE_FIELDS

EXCHANGE_TABLE_FIELDS = frozenset({
    'exchange_id',
    'exchange',
    'timezone'
})

ROOT_SYMBOL_TABLE_FIELDS = ({
    'root_symbol_id',
    'root_symbol',
    'sector',
    'description',
    'exchange_id'
})


class AssetData(object):
    """ Class to store collection of asset data.
    """
    def __init__(self, equities=None, futures=None,
                 exchanges=None, root_symbols=None):
        """
        Data supplied to this object should be of a consistent type.
        """

        self.equities = equities
        self.futures = futures
        self.exchanges = exchanges
        self.root_symbols = root_symbols


class AssetDBWriter(with_metaclass(ABCMeta)):
    """
    Class used to write arbitrary data to SQLite database.
    Concrete subclasses will implement the logic for a specific
    input datatypes by implementing the load_data method.

    Methods
    -------
    write_all(engine, fuzzy_char=None, allow_sid_assignment=True,
              constraints=False)
        Write the data supplied at initialization to the database.
    init_db(engine, constraints=False)
        Create the SQLite tables (called by write_all).
    load_data()
        Returns data in standard format.

    """
    def __init__(self):
        self.sql_metadata = None

    def write_all(self,
                  engine,
                  fuzzy_char=None,
                  allow_sid_assignment=True,
                  constraints=True):
        """ Write pre-supplied data to SQLite.

        Parameters
        ----------
        engine : Engine
            An engine to a SQL database.
        fuzzy_char : str, optional
            A string for use in fuzzy matching.
        allow_sid_assignment: bool, optional
            If True then the class can assign sids where necessary.
        constraints : bool, optional
            If True, create SQL ForeignKey and Index constraints.

        """
        self.allow_sid_assignment = allow_sid_assignment
        # Create SQL tables
        self.init_db(engine, constraints)
        # Get the data to add to SQL
        equities, futures, exchanges, root_symbols = self.load_data()
        with engine.begin() as txn:
            self._write_exchanges(exchanges, txn)
            self._write_root_symbols(root_symbols, txn)
            self._write_futures(futures, txn)
            self._write_equities(equities, fuzzy_char, txn)

    def _write_exchanges(self, exchanges, bind=None):
        self.futures_exchanges.insert().values(
            exchanges.reset_index().rename_axis(
                {'index': 'exchange_id'},
                1,
            ).to_dict('records'),
        ).execute(bind=bind)

    def _write_root_symbols(self, root_symbols, bind=None):
        self.futures_root_symbols.insert().values(
            root_symbols.reset_index().rename_axis(
                {'index': 'root_symbol_id'},
                1,
            ).to_dict('records'),
        ).execute(bind=bind)

    def _write_futures(self, futures, bind=None):
        recs = futures.reset_index().rename_axis(
            {'index': 'sid'},
            1,
        ).to_dict('records')
        if recs:
            self.futures_contracts.insert().values(recs).execute(bind=bind)
            ar_recs = [(rec['sid'], 'future') for rec in recs]
            self.asset_router.insert().values(ar_recs).execute(bind=bind)

    def _write_equities(self, equities, fuzzy_char, bind=None):
        # Apply fuzzy matching.
        if fuzzy_char:
            equities['fuzzy'] = equities['symbol'].str.replace(fuzzy_char, '')

        recs = equities.reset_index().rename_axis(
            {'index': 'sid'},
            1,
        ).to_dict('records')
        if recs:
            self.equities.insert().values(recs).execute(bind=bind)
            ar_recs = [(rec['sid'], 'equity') for rec in recs]
            self.asset_router.insert().values(ar_recs).execute(bind=bind)

    def init_db(self, engine, constraints=True):
        """Connect to database and create tables.

        Parameters
        ----------
        engine : Engine
            An engine to a SQL database.
        constraints : bool
            If True, create SQL ForeignKey and Index constraints.
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
            sa.Column('asset_name', sa.Text),
            sa.Column('start_date', sa.Integer, default=0),
            sa.Column('end_date', sa.Integer),
            sa.Column('first_traded', sa.Integer),
            sa.Column('exchange', sa.Text),
            sa.Column('fuzzy', sa.Text),
        )
        self.futures_exchanges = sa.Table(
            'futures_exchanges',
            metadata,
            sa.Column(
                'exchange_id',
                sa.Integer,
                unique=True,
                nullable=False,
                primary_key=constraints,
            ),
            sa.Column('exchange', sa.Text),
            sa.Column('timezone', sa.Text),
        )
        self.futures_root_symbols = sa.Table(
            'futures_root_symbols',
            metadata,
            sa.Column(
                'root_symbol_id',
                sa.Integer,
                unique=True,
                nullable=False,
                primary_key=constraints,
            ),
            sa.Column('root_symbol', sa.Text),
            sa.Column('sector', sa.Text),
            sa.Column('description', sa.Text),
            sa.Column(
                'exchange_id',
                sa.Integer,
                *((sa.ForeignKey(self.futures_exchanges.c.exchange_id),)
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
                'root_symbol_id',
                sa.Integer,
                *((sa.ForeignKey(self.futures_root_symbols.c.root_symbol_id),)
                  if constraints else ())
            ),
            sa.Column('root_symbol', sa.Text),
            sa.Column('asset_name', sa.Text),
            sa.Column('start_date', sa.Integer, default=0),
            sa.Column('end_date', sa.Integer),
            sa.Column('first_traded', sa.Integer),
            sa.Column(
                'exchange_id',
                sa.Integer,
                *((sa.ForeignKey(self.futures_exchanges.c.exchange_id),)
                  if constraints else ())
            ),
            sa.Column('exchange', sa.Text),
            sa.Column('notice_date', sa.Integer),
            sa.Column('expiration_date', sa.Integer),
            sa.Column('contract_multiplier', sa.Float),
        )
        self.asset_router = sa.Table(
            'asset_router',
            metadata,
            sa.Column('sid', sa.Integer, primary_key=constraints),
            sa.Column('asset_type', sa.Text),
        )
        metadata.create_all(checkfirst=True)
        return metadata

    def load_data(self):
        """
        Returns a standard set of pandas.DataFrames:
        equities, futures, exchanges, root_symbols
        """

        equities_data, futures_data, exchanges_data, root_symbols_data = \
            self._load_data()

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

        cols = set(data.equities.columns)

        # Drop columns with unrecognised headers.
        data.equities.drop(cols - (cols & equities_cols), axis=1,
                           inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = equities_cols - set(data.equities.columns)

        # Combine the users supplied data with our required columns.
        equities_output = pd.concat(
            (data.equities, pd.DataFrame(
                self.dict_subset(equities_defaults, need),
                data.equities.index,
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

        cols = set(data.futures.columns)

        # Drop columns with unrecognised headers.
        data.futures.drop(cols - (cols & futures_cols), axis=1,
                          inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = futures_cols - set(data.futures.columns)

        # Combine the users supplied data with our required columns.
        futures_output = pd.concat(
            (data.futures, pd.DataFrame(
                self.dict_subset(futures_defaults, need),
                data.futures.index,
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

        cols = set(data.exchanges.columns)

        # Drop columns with unrecognised headers.
        data.exchanges.drop(cols - (cols & exchanges_cols), axis=1,
                            inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = exchanges_cols - set(data.exchanges.columns)

        # Combine the users supplied data with our required columns.
        exchanges_output = pd.concat(
            (data.exchanges, pd.DataFrame(
                self.dict_subset(exchanges_defaults, need),
                data.exchanges.index,
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

        cols = set(data.root_symbols.columns)

        # Drop columns with unrecognised headers.
        data.root_symbols.drop(cols - (cols & root_symbols_cols), axis=1,
                               inplace=True)

        # Get those columns which we need but
        # for which no data has been supplied.
        need = root_symbols_cols - set(data.root_symbols.columns)

        # Combine the users supplied data with our required columns.
        root_symbols_output = pd.concat(
            (data.root_symbols, pd.DataFrame(
                self.dict_subset(root_symbols_defaults, need),
                data.root_symbols.index,
            )),
            axis=1,
            copy=False
        )

        return equities_data, futures_data, exchanges_data, root_symbols_data

    @staticmethod
    def dict_subset(dict_, subset):
        res = {}
        for k in subset:
            res[k] = dict_[k]
        return res

    @abstractmethod
    def _load_data(self):
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

    def _load_data(self):

        # 0) Instantiate empty dictionaries
        _equities, _futures, _exchanges, _root_symbols = {}, {}, {}, {}

        # 1) Populate dictionaries
        id_counter = 0
        for output, data in [(_equities, self._equities),
                             (_futures, self._futures), ]:
            for identifier in data:
                if hasattr(identifier, '__int__'):
                    output[identifier.__int__()] = {}
                else:
                    if self.allow_sid_assignment:
                        output[id_counter] = {'symbol': identifier}
                        id_counter += 1
                    else:
                        SidAssignmentError(identifier=identifier)

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

        # Convert dictionaries to pandas.DataFrames
        _equities = pd.DataFrame.from_dict(_equities, orient='index')
        _futures = pd.DataFrame.from_dict(_futures, orient='index')
        _exchanges = pd.DataFrame.from_dict(_exchanges, orient='index')
        _root_symbols = pd.DataFrame.from_dict(_root_symbols, orient='index')

        return asset_data(equities=_equities,
                          futures=_futures,
                          exchanges=_exchanges,
                          root_symbols=_root_symbols)


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

    def _load_data(self):

        _equities = pd.DataFrame.from_dict(self._equities, orient='index')
        _futures = pd.DataFrame.from_dict(self._futures, orient='index')
        _exchanges = pd.DataFrame.from_dict(self._exchanges, orient='index')
        _root_symbols = pd.DataFrame.from_dict(self._root_symbols,
                                               orient='index')

        return _equities, _futures, _exchanges, _root_symbols


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

    def _load_data(self):

        return self._equities, self._futures, self._exchanges,
        self._root_symbols
